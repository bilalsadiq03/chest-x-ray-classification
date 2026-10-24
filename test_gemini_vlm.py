import csv
import os
from pathlib import Path
from google import genai
from google.genai import types

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Set GEMINI_API_KEY in this PowerShell session first.")

manifest = Path("outputs/splits/test.csv")
with manifest.open(newline="", encoding="utf-8-sig") as f:
    records = list(csv.DictReader(f))

if not records:
    raise RuntimeError("Test manifest is empty.")

# Choose one image without using its ground-truth label.
image_path = Path(records[0]["path"])
if not image_path.exists():
    raise FileNotFoundError(f"Image path not found: {image_path}")

client = genai.Client(api_key=api_key)

prompt = """
You are evaluating a chest X-ray image for a research prototype.
Describe only visual findings you can reasonably identify.
Assess whether the image has findings suggestive of pneumonia.

Return exactly these fields:
Assessment: PNEUMONIA_SUGGESTIVE, NOT_PNEUMONIA_SUGGESTIVE, or INDETERMINATE
Findings: a brief description of visible findings
Limitations: important uncertainty or image-quality limitations

Do not invent findings. This is an experimental model assessment,
not a clinical diagnosis. Do not assume the patient's history.
"""

with image_path.open("rb") as f:
    image_bytes = f.read()

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=[
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        prompt,
    ],
)

print("Gemini API test successful.")
print("Image:", image_path.name)
print("Model response:")
print(response.text or "[No text returned]")
