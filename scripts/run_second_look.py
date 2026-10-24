"""
scripts/run_second_look.py — Second-look VLM evaluation entry point.

Usage (from project root):
    $env:GEMINI_API_KEY = (Get-Content .env | Select-String GEMINI_API_KEY).Line.Split('=')[1].Trim('"')
    .venv\Scripts\python.exe scripts/run_second_look.py

The script:
1. Reads CNN test predictions from outputs/test_predictions.csv
2. Selects images predicted as NORMAL by the CNN
3. Reuses any existing successful assessments from outputs/gemini_second_look.csv
4. Sends remaining images to the Gemini API (respecting the rate limit)
5. Stops safely on quota exhaustion and saves progress

Run again after quota resets to resume from where it left off.
"""

import csv
import logging
import os
import sys
import time
from pathlib import Path

# ── project root on path ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from src.vlm.client import assess_image, AssessmentStatus, VLMResult

# ── configuration ────────────────────────────────────────────────────────────
PREDICTIONS     = Path("outputs/test_predictions.csv")
OUTPUT          = Path("outputs/gemini_second_look.csv")
REQUEST_INTERVAL = 15          # minimum seconds between API calls
MODEL           = os.environ.get("VLM_MODEL", "gemini-2.0-flash")

FIELDS = ["path", "assessment", "findings", "limitations",
          "raw_response", "error", "status", "model"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("outputs/logs/second_look.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── validation ───────────────────────────────────────────────────────────────
api_key = os.environ.get("GEMINI_API_KEY", "").strip()
if not api_key:
    logger.error("GEMINI_API_KEY is not set. Add it to .env or the shell environment.")
    sys.exit(1)

if not PREDICTIONS.exists():
    logger.error("Predictions file not found: %s — run recover_baseline.py first.", PREDICTIONS)
    sys.exit(1)

# ── load predictions ─────────────────────────────────────────────────────────
with PREDICTIONS.open(newline="", encoding="utf-8-sig") as f:
    all_predictions = list(csv.DictReader(f))

targets = [
    r for r in all_predictions
    if r["predicted_label"].strip().upper() == "NORMAL"
]
if not targets:
    logger.error("No CNN-predicted NORMAL images found in %s", PREDICTIONS)
    sys.exit(1)

# ── load existing successful results ─────────────────────────────────────────
VALID_ASSESSMENTS = {"PNEUMONIA_SUGGESTIVE", "NOT_PNEUMONIA_SUGGESTIVE", "INDETERMINATE"}

def load_existing(path: Path) -> dict[str, dict]:
    existing: dict[str, dict] = {}
    if not path.exists():
        return existing
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if (row.get("path")
                    and not row.get("error", "").strip()
                    and row.get("assessment", "").strip() in VALID_ASSESSMENTS):
                existing[row["path"]] = row
    return existing

existing = load_existing(OUTPUT)

# ── save helper ───────────────────────────────────────────────────────────────
def save_all(results: list[dict]) -> None:
    Path("outputs/logs").mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in FIELDS})
    tmp.replace(OUTPUT)

# ── gemini client ─────────────────────────────────────────────────────────────
from google import genai
client = genai.Client(api_key=api_key)

# ── main loop ─────────────────────────────────────────────────────────────────
results: list[dict] = []

# Pre-populate with reusable results
for t in targets:
    p = t["path"]
    if p in existing:
        row = {k: existing[p].get(k, "") for k in FIELDS}
        row["status"] = AssessmentStatus.SKIPPED.value
        results.append(row)

logger.info("CNN-predicted NORMAL: %d", len(targets))
logger.info("Already assessed (reusable): %d", len(results))
logger.info("Remaining API calls: %d", len(targets) - len(results))
logger.info("Model: %s", MODEL)

last_request_time: float | None = None

for i, target in enumerate(targets, start=1):
    path_str = target["path"]

    # Skip already done
    if any(r["path"] == path_str for r in results):
        continue

    image_path = Path(path_str)
    if not image_path.is_file():
        image_path = Path.cwd() / path_str

    if not image_path.is_file():
        results.append({
            "path": path_str, "assessment": "INDETERMINATE",
            "error": f"Image not found: {path_str}",
            "status": AssessmentStatus.ERROR.value,
            "model": MODEL,
            "findings": "", "limitations": "", "raw_response": "",
        })
        save_all(results)
        logger.warning("[%d/%d] Missing image: %s", i, len(targets), path_str)
        continue

    # Rate limiting
    if last_request_time is not None:
        wait = REQUEST_INTERVAL - (time.monotonic() - last_request_time)
        if wait > 0:
            time.sleep(wait)

    last_request_time = time.monotonic()
    result: VLMResult = assess_image(client, image_path, MODEL)

    row = {
        "path":         result.path,
        "assessment":   result.assessment,
        "findings":     result.findings,
        "limitations":  result.limitations,
        "raw_response": result.raw_response,
        "error":        result.error,
        "status":       result.status.value,
        "model":        result.model,
    }
    results.append(row)
    save_all(results)

    logger.info(
        "[%d/%d] %s | %s | %s",
        i, len(targets),
        result.status.value,
        result.assessment,
        image_path.name,
    )

    if result.status == AssessmentStatus.QUOTA_LIMITED:
        logger.warning(
            "Quota/rate-limit hit. Progress saved. "
            "Wait for quota to reset then run this script again."
        )
        break

# ── summary ───────────────────────────────────────────────────────────────────
successful   = sum(r["status"] in (AssessmentStatus.SUCCESS.value, AssessmentStatus.SKIPPED.value) for r in results)
quota_errors = sum(r["status"] == AssessmentStatus.QUOTA_LIMITED.value for r in results)
other_errors = sum(r["status"] == AssessmentStatus.ERROR.value for r in results)

logger.info("\n=== SECOND-LOOK RUN COMPLETE ===")
logger.info("Total selected:    %d", len(targets))
logger.info("Successful:        %d", successful)
logger.info("Quota-limited:     %d", quota_errors)
logger.info("Other errors:      %d", other_errors)
logger.info("Results saved to:  %s", OUTPUT)
logger.info("Run again to resume after quota resets.")
