"""
Grad-CAM explainability for PneumoBackstop ResNet18.
Run from project root:
    python generate_gradcam.py

Creates Grad-CAM overlays for representative TP, TN, FP and FN test examples.
The test set is used only for post-hoc explanation, not model/threshold selection.
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "checkpoints" / "best_resnet18.pth"
TEST_PREDICTIONS = ROOT / "outputs" / "final_model_evaluation" / "test_resnet18_predictions.csv"
OUTPUT_DIR = ROOT / "outputs" / "explainability" / "gradcam_resnet18"
IMAGE_SIZE = 224
THRESHOLD = 0.50
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_model(device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_index):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        score = logits[:, class_index].sum()
        score.backward()

        # Global-average-pool the gradients to obtain channel importance weights.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear", align_corners=False
        )
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return logits.detach(), cam

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def make_overlay(image_rgb, cam):
    heat = plt.get_cmap("jet")(cam)[..., :3]
    heat = (heat * 255).astype(np.uint8)
    blended = (0.55 * image_rgb.astype(np.float32) +
               0.45 * heat.astype(np.float32)).clip(0, 255).astype(np.uint8)
    return blended


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT}")
    if not TEST_PREDICTIONS.exists():
        raise FileNotFoundError(f"Missing predictions file: {TEST_PREDICTIONS}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    gradcam = GradCAM(model, model.layer4[-1])

    df = pd.read_csv(TEST_PREDICTIONS)
    df["predicted_label_id"] = (df["p_pneumonia"] >= THRESHOLD).astype(int)
    df["case"] = np.select(
        [
            (df["true_label_id"] == 1) & (df["predicted_label_id"] == 1),
            (df["true_label_id"] == 0) & (df["predicted_label_id"] == 0),
            (df["true_label_id"] == 0) & (df["predicted_label_id"] == 1),
            (df["true_label_id"] == 1) & (df["predicted_label_id"] == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="OTHER"
    )

    # Up to two examples from each confusion-matrix category.
    selected = []
    for category in ["TP", "TN", "FP", "FN"]:
        subset = df[df["case"] == category].copy()
        if not subset.empty:
            # Select examples near the decision boundary where possible.
            subset["distance_to_threshold"] = (
                subset["p_pneumonia"] - THRESHOLD
            ).abs()
            chosen = subset.sort_values("distance_to_threshold").head(2)
            selected.extend((category, row) for _, row in chosen.iterrows())

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    report_rows = []
    for index, (category, row) in enumerate(selected, start=1):
        image_path = Path(row["path"])
        with Image.open(image_path) as im:
            image = im.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        image_rgb = np.asarray(image)
        input_tensor = transform(image).unsqueeze(0).to(device)

        true_label = int(row["true_label_id"])
        predicted_label = int(row["predicted_label_id"])

        # Explain the class the model actually predicted.
        logits, cam = gradcam.generate(input_tensor, predicted_label)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        overlay = make_overlay(image_rgb, cam)

        filename = f"{index:02d}_{category}_{image_path.stem}.png"
        output_path = OUTPUT_DIR / filename

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original X-ray")
        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM heatmap")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        for ax in axes:
            ax.axis("off")

        true_name = "PNEUMONIA" if true_label == 1 else "NORMAL"
        pred_name = "PNEUMONIA" if predicted_label == 1 else "NORMAL"
        fig.suptitle(
            f"{category} | True: {true_name} | Predicted: {pred_name} | "
            f"P(pneumonia)={probabilities[1]:.3f}",
            fontsize=10
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        report_rows.append({
            "case": category,
            "image_path": str(image_path),
            "true_label": true_name,
            "predicted_label": pred_name,
            "p_pneumonia": float(probabilities[1]),
            "gradcam_file": str(output_path)
        })
        print(f"Saved {category}: {output_path}")

    gradcam.close()

    report_path = OUTPUT_DIR / "gradcam_case_summary.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"\nGenerated {len(report_rows)} Grad-CAM examples.")
    print("Case counts:")
    print(df["case"].value_counts().to_string())
    print("Summary:", report_path)
    print("\nInterpret heatmaps cautiously: they show regions influencing the model,")
    print("not proof that those regions correspond to clinically valid findings.")


if __name__ == "__main__":
    main()
