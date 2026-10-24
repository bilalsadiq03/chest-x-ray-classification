import csv
import os
import time
from pathlib import Path

from google import genai
from google.genai import types

PREDICTIONS = Path("outputs/test_predictions.csv")
OLD_RESULTS = Path("outputs/gemini_vlm_assessments.csv")
OUTPUT = Path("outputs/gemini_second_look.csv")

# Stay below the observed five-requests-per-minute limit.
REQUEST_INTERVAL = 15
MODEL = "gemini-3.6-flash"

FIELDS = [
    "path", "assessment", "findings", "limitations",
    "raw_response", "error", "model"
]

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Set it in this PowerShell session."
    )

if not PREDICTIONS.exists():
    raise FileNotFoundError(PREDICTIONS)

with PREDICTIONS.open(newline="", encoding="utf-8-sig") as f:
    predictions = list(csv.DictReader(f))

# Select images classified as NORMAL by the CNN.
# True labels are deliberately not used for selection or sent to Gemini.
targets = [
    row for row in predictions
    if row["predicted_label"].strip().upper() == "NORMAL"
]

if not targets:
    raise RuntimeError("No CNN-predicted NORMAL cases were found.")

# Reuse previous successful assessments where possible.
existing = {}
if OLD_RESULTS.exists():
    with OLD_RESULTS.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("path") and not row.get("error"):
                existing[row["path"]] = row

if OUTPUT.exists():
    with OUTPUT.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("path") and not row.get("error"):
                existing[row["path"]] = row

client = genai.Client(api_key=api_key)

prompt = """
Assess this chest X-ray for a research prototype.
Determine whether visible findings are suggestive of pneumonia.
Do not assume patient history or invent findings. If uncertain,
use INDETERMINATE. This is not a clinical diagnosis.

Return exactly:
Assessment: PNEUMONIA_SUGGESTIVE, NOT_PNEUMONIA_SUGGESTIVE, or INDETERMINATE
Findings: brief description of visible findings
Limitations: uncertainty or image-quality limitations
"""

def parse_response(text):
    result = {
        "assessment": "INDETERMINATE",
        "findings": "",
        "limitations": "",
    }
    section = None

    for line in text.splitlines():
        s = line.strip()
        low = s.lower()

        if low.startswith("assessment:"):
            value = s.split(":", 1)[1].strip().upper()
            allowed = {
                "PNEUMONIA_SUGGESTIVE",
                "NOT_PNEUMONIA_SUGGESTIVE",
                "INDETERMINATE",
            }
            result["assessment"] = (
                value if value in allowed else "INDETERMINATE"
            )
            section = None
        elif low.startswith("findings:"):
            result["findings"] = s.split(":", 1)[1].strip()
            section = "findings"
        elif low.startswith("limitations:"):
            result["limitations"] = s.split(":", 1)[1].strip()
            section = "limitations"
        elif section and s:
            result[section] += "\n" + s

    return result

def save_all(rows):
    temp = OUTPUT.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(OUTPUT)

results = []
for target in targets:
    path = target["path"]
    if path in existing:
        results.append({k: existing[path].get(k, "") for k in FIELDS})

print("CNN-predicted NORMAL images:", len(targets))
print("Already assessed and reusable:", len(results))
print("Remaining API calls:", len(targets) - len(results))
print("Model:", MODEL)
print("Results will be saved to:", OUTPUT)

last_request_time = None

for i, target in enumerate(targets, start=1):
    path = target["path"]

    if any(row["path"] == path for row in results):
        continue

    image_path = Path(path)
    if not image_path.is_file():
        image_path = Path.cwd() / path

    row = dict.fromkeys(FIELDS, "")
    row.update({"path": path, "model": MODEL})

    if not image_path.is_file():
        row.update({
            "assessment": "INDETERMINATE",
            "error": f"Image not found: {image_path}",
        })
        results.append(row)
        save_all(results)
        print(f"[{i}/{len(targets)}] Missing image: {path}")
        continue

    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(image_path.suffix.lower())

    if not mime:
        row.update({
            "assessment": "INDETERMINATE",
            "error": f"Unsupported extension: {image_path.suffix}",
        })
        results.append(row)
        save_all(results)
        continue

    # Enforce a minimum interval between requests.
    if last_request_time is not None:
        wait = REQUEST_INTERVAL - (time.monotonic() - last_request_time)
        if wait > 0:
            time.sleep(wait)

    try:
        last_request_time = time.monotonic()
        response = client.models.generate_content(
            model=MODEL,
            contents=[
                types.Part.from_bytes(
                    data=image_path.read_bytes(),
                    mime_type=mime,
                ),
                prompt,
            ],
        )

        text = response.text or ""
        if not text.strip():
            raise RuntimeError("Empty response from Gemini.")

        row.update(parse_response(text))
        row["raw_response"] = text
        row["error"] = ""

    except Exception as exc:
        row.update({
            "assessment": "INDETERMINATE",
            "error": str(exc)[:1000],
        })
        results.append(row)
        save_all(results)

        message = str(exc)
        print(f"[{i}/{len(targets)}] API error: {message[:220]}")

        # Stop rather than repeatedly hammering a depleted quota.
        if "429" in message or "RESOURCE_EXHAUSTED" in message:
            print(
                "Rate limit/quota error. Saved progress; stopping safely. "
                "Wait for quota reset or check your Gemini usage limits."
            )
            break

        continue

    results.append(row)
    save_all(results)
    print(f"[{i}/{len(targets)}] {row['assessment']} | {image_path.name}")

print("\nSaved:", OUTPUT)
print("Successful assessments:", sum(not r["error"] for r in results))
print("Selected CNN-normal cases:", len(targets))
print("Run this script again to resume after a temporary interruption.")
