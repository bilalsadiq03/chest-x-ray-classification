import csv
import os
import time
from pathlib import Path

from google import genai
from google.genai import types

MANIFEST = Path("outputs/splits/test.csv")
OUTPUT = Path("outputs/gemini_vlm_assessments.csv")
MODEL = "gemini-3.6-flash"

FIELDS = [
    "path",
    "assessment",
    "findings",
    "limitations",
    "raw_response",
    "error",
    "model",
]

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Set it in this PowerShell session."
    )

if not MANIFEST.exists():
    raise FileNotFoundError(f"Missing test manifest: {MANIFEST}")

with MANIFEST.open(newline="", encoding="utf-8-sig") as f:
    records = list(csv.DictReader(f))

if not records:
    raise RuntimeError("The test manifest contains no images.")

client = genai.Client(api_key=api_key)

# Resume safely using results already written to disk.
completed = {}
if OUTPUT.exists():
    with OUTPUT.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            # Retry previous failures rather than treating them as completed.
            if row.get("path") and not row.get("error"):
                completed[row["path"]] = row

def save_result(row):
    # Rewrite atomically so a partial write is less likely to corrupt results.
    temp = OUTPUT.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for existing in completed.values():
            writer.writerow(existing)
        writer.writerow(row)
    temp.replace(OUTPUT)

def parse_response(text):
    result = {
        "assessment": "INDETERMINATE",
        "findings": "",
        "limitations": "",
    }
    current = None

    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()

        if lower.startswith("assessment:"):
            value = stripped.split(":", 1)[1].strip().upper()
            allowed = {
                "PNEUMONIA_SUGGESTIVE",
                "NOT_PNEUMONIA_SUGGESTIVE",
                "INDETERMINATE",
            }
            result["assessment"] = (
                value if value in allowed else "INDETERMINATE"
            )
            current = None
        elif lower.startswith("findings:"):
            result["findings"] = stripped.split(":", 1)[1].strip()
            current = "findings"
        elif lower.startswith("limitations:"):
            result["limitations"] = stripped.split(":", 1)[1].strip()
            current = "limitations"
        elif current and stripped:
            result[current] += "\n" + stripped

    return result

prompt = """
You are evaluating a chest X-ray for a research prototype.
This is an experimental image assessment, not a clinical diagnosis.

Assess whether visible findings are suggestive of pneumonia.
Describe only findings you can reasonably identify. Do not invent
findings or assume patient history. If uncertain, use INDETERMINATE.

Return exactly these fields:
Assessment: PNEUMONIA_SUGGESTIVE, NOT_PNEUMONIA_SUGGESTIVE, or INDETERMINATE
Findings: brief description of visible findings
Limitations: uncertainty or image-quality limitations

Do not provide treatment recommendations.
"""

total = len(records)
print(f"Test images: {total}")
print(f"Already completed successfully: {len(completed)}")
print(f"Model: {MODEL}")
print("Ground-truth labels are not included in the model prompt.")

for index, record in enumerate(records, start=1):
    image_path_text = record["path"]

    if image_path_text in completed:
        continue

    image_path = Path(image_path_text)
    if not image_path.exists():
        # Also handle manifests containing paths relative to the project root.
        image_path = Path.cwd() / image_path_text

    if not image_path.is_file():
        row = dict.fromkeys(FIELDS, "")
        row.update({
            "path": image_path_text,
            "assessment": "INDETERMINATE",
            "error": f"Image file not found: {image_path}",
            "model": MODEL,
        })
        save_result(row)
        print(f"[{index}/{total}] Missing image: {image_path_text}")
        continue

    suffix = image_path.suffix.lower()
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix)

    if not mime_type:
        row = dict.fromkeys(FIELDS, "")
        row.update({
            "path": image_path_text,
            "assessment": "INDETERMINATE",
            "error": f"Unsupported image extension: {suffix}",
            "model": MODEL,
        })
        save_result(row)
        continue

    try:
        image_bytes = image_path.read_bytes()
    except OSError as exc:
        row = dict.fromkeys(FIELDS, "")
        row.update({
            "path": image_path_text,
            "assessment": "INDETERMINATE",
            "error": str(exc),
            "model": MODEL,
        })
        save_result(row)
        continue

    last_error = None
    response_text = ""

    for attempt in range(1, 6):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_type,
                    ),
                    prompt,
                ],
            )
            response_text = response.text or ""
            if not response_text.strip():
                raise RuntimeError("Gemini returned an empty response.")
            last_error = None
            break
        except Exception as exc:
            last_error = str(exc)
            print(
                f"  Attempt {attempt}/5 failed: "
                f"{last_error[:200]}"
            )
            if attempt < 5:
                time.sleep(min(2 ** attempt, 30))

    row = dict.fromkeys(FIELDS, "")
    row.update({
        "path": image_path_text,
        "model": MODEL,
        "raw_response": response_text,
        "error": last_error or "",
        "assessment": "INDETERMINATE",
    })

    if response_text:
        parsed = parse_response(response_text)
        row.update(parsed)

    save_result(row)

    if not last_error:
        completed[image_path_text] = row

    print(
        f"[{index}/{total}] "
        f"{row['assessment']} | "
        f"{image_path.name} | "
        f"{'ERROR' if last_error else 'saved'}"
    )

print("\nVLM inference run finished.")
print("Results:", OUTPUT)
print("Successful assessments:", len(completed))
print("Total test images:", total)
print("Images not successfully processed:", total - len(completed))
print("Review error column before computing metrics.")
