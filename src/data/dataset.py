import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SIZE = 224
CLASS_TO_ID = {"NORMAL": 0, "PNEUMONIA": 1}

# ImageNet normalization values for RGB images.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class ChestXrayDataset(Dataset):
    def __init__(self, manifest_path, training=False):
        self.manifest_path = Path(manifest_path)
        self.training = training

        with self.manifest_path.open(
            "r", newline="", encoding="utf-8-sig"
        ) as file:
            self.records = list(csv.DictReader(file))

        if not self.records:
            raise ValueError(f"Empty manifest: {self.manifest_path}")

        for record in self.records:
            if record["label"] not in CLASS_TO_ID:
                raise ValueError(f"Unknown label: {record['label']}")
            if not Path(record["path"]).is_file():
                raise FileNotFoundError(record["path"])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        label = CLASS_TO_ID[record["label"]]

        with Image.open(record["path"]) as image:
            image = image.convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
            )

            if self.training:
                # Small, reproducible-type augmentation; random per sample.
                if random.random() < 0.5:
                    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

                angle = random.uniform(-5.0, 5.0)
                image = image.rotate(
                    angle, resample=Image.Resampling.BILINEAR
                )

            array = np.asarray(image, dtype=np.float32) / 255.0

        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = (tensor - MEAN) / STD

        return tensor, torch.tensor(label, dtype=torch.long)
