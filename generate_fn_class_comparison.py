"""
Class-specific Grad-CAM comparison for false-negative PneumoBackstop cases.

For each ResNet18 false negative, creates:
1. Original X-ray
2. Grad-CAM for NORMAL class (the model's predicted class)
3. NORMAL-class overlay
4. Grad-CAM for PNEUMONIA class (the missed class)
5. PNEUMONIA-class overlay

Run from the project root:
    python generate_fn_class_comparison.py
"""

from pathlib import Path
import csv

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
PREDICTIONS = ROOT / "outputs" / "final_model_evaluation" / "test_resnet18_predictions.csv"
OUTPUT_DIR = ROOT / "outputs" / "explainability" / "gradcam_class_comparison"
IMAGE_SIZE = 224
THRESHOLD = 0.50
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.fwd = target_layer.register_forward_hook(self._capture_activation)
        self.bwd = target_layer.register_full_backward_hook(self._capture_gradient)

    def _capture_activation(self, module, inputs, output):
        self.activations = output

    def _capture_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_index):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        class_score = logits[:, class_index].sum()
        class_score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear", align_corners=False
        )
        cam = cam[0, 0].detach().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 1e-8:
            cam /= cam.max()
        return logits.detach(), cam

    def close(self):
        self.fwd.remove()
        self.bwd.remove()


def load_model(device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def make_overlay(image_rgb, cam):
    heatmap_rgb = (plt.get_cmap("jet")(cam)[..., :3] * 255).astype(np.uint8)
    return (
        0.55 * image_rgb.astype(np.float32) +
        0.45 * heatmap_rgb.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    if not PREDICTIONS.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {PREDICTIONS}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    explainer = GradCAM(model, model.layer4[-1])

    df = pd.read_csv(PREDICTIONS)
    df["predicted_label_id"] = (df["p_pneumonia"] >= THRESHOLD).astype(int)
    false_negatives = df[
        (df["true_label_id"] == 1) & (df["predicted_label_id"] == 0)
    ].copy()

    if false_negatives.empty:
        raise RuntimeError("No false-negative cases found at threshold 0.50.")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    summary = []
    for index, (_, row) in enumerate(false_negatives.iterrows(), start=1):
        image_path = Path(row["path"])
        with Image.open(image_path) as im:
            image = im.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        image_rgb = np.asarray(image)
        tensor = transform(image).unsqueeze(0).to(device)

        # Both class-specific CAMs are generated from the same input image.
        logits, cam_normal = explainer.generate(tensor, class_index=0)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        _, cam_pneumonia = explainer.generate(tensor, class_index=1)

        normal_overlay = make_overlay(image_rgb, cam_normal)
        pneumonia_overlay = make_overlay(image_rgb, cam_pneumonia)

        fig, axes = plt.subplots(1, 5, figsize=(19, 4.5))
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original X-ray")
        axes[1].imshow(cam_normal, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("NORMAL class Grad-CAM")
        axes[2].imshow(normal_overlay)
        axes[2].set_title("NORMAL class overlay")
        axes[3].imshow(cam_pneumonia, cmap="jet", vmin=0, vmax=1)
        axes[3].set_title("PNEUMONIA class Grad-CAM")
        axes[4].imshow(pneumonia_overlay)
        axes[4].set_title("PNEUMONIA class overlay")

        fig.suptitle(
            f"False Negative {index}: {image_path.name} | "
            f"P(NORMAL)={probabilities[0]:.3f} | "
            f"P(PNEUMONIA)={probabilities[1]:.3f}",
            fontsize=11
        )
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()

        output_file = OUTPUT_DIR / f"FN_{index}_{image_path.stem}_class_comparison.png"
        fig.savefig(output_file, dpi=180, bbox_inches="tight")
        plt.close(fig)

        summary.append({
            "image_path": str(image_path),
            "true_label": "PNEUMONIA",
            "predicted_label": "NORMAL",
            "p_normal": float(probabilities[0]),
            "p_pneumonia": float(probabilities[1]),
            "normal_gradcam": "generated",
            "pneumonia_gradcam": "generated",
            "figure": str(output_file)
        })
        print(f"Saved class comparison: {output_file}", flush=True)

    explainer.close()

    summary_file = OUTPUT_DIR / "false_negative_class_comparison.csv"
    with summary_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nGenerated {len(summary)} class-specific comparisons.")
    print("Output directory:", OUTPUT_DIR)
    print("Summary:", summary_file)
    print(
        "\nInterpretation note: class-specific Grad-CAM shows regions contributing "
        "to each class score; it does not prove lesion localization or clinical validity."
    )


if __name__ == "__main__":
    main()
