"""
Gemini VLM client for PneumoBackstop second-look evaluation.

Key design decisions:
 - Every API result has an explicit status: SUCCESS, QUOTA_LIMITED, MALFORMED,
   ERROR, or SKIPPED.
 - On QUOTA_LIMITED, the caller should stop and not retry; the run saves all
   progress and can be resumed after the quota resets.
 - API keys are never logged; only masked prefixes appear in debug output.
 - Ground-truth labels are never passed to the model.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------

class AssessmentStatus(str, Enum):
    SUCCESS       = "SUCCESS"        # Parsed a valid structured response
    QUOTA_LIMITED = "QUOTA_LIMITED"  # HTTP 429 / RESOURCE_EXHAUSTED
    MALFORMED     = "MALFORMED"      # Got a response but could not parse it
    ERROR         = "ERROR"          # Other API/network error
    SKIPPED       = "SKIPPED"        # Already assessed; reused from cache

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

VALID_ASSESSMENTS = frozenset({
    "PNEUMONIA_SUGGESTIVE",
    "NOT_PNEUMONIA_SUGGESTIVE",
    "INDETERMINATE",
})

@dataclass
class VLMResult:
    path:         str
    status:       AssessmentStatus
    assessment:   str = "INDETERMINATE"
    findings:     str = ""
    limitations:  str = ""
    raw_response: str = ""
    error:        str = ""
    model:        str = ""

# ---------------------------------------------------------------------------
# Prompt (no ground-truth labels are included)
# ---------------------------------------------------------------------------

SECOND_LOOK_PROMPT = """\
Assess this chest X-ray for a research prototype.
Determine whether visible findings are suggestive of pneumonia.
Do not assume patient history or invent findings.
If uncertain, use INDETERMINATE. This is not a clinical diagnosis.

Return exactly three labelled lines (no markdown, no extra commentary):
Assessment: PNEUMONIA_SUGGESTIVE, NOT_PNEUMONIA_SUGGESTIVE, or INDETERMINATE
Findings: brief description of visible findings
Limitations: key sources of uncertainty or image-quality limitations
"""

# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_vlm_response(text: str) -> tuple[str, str, str, bool]:
    """
    Parse the structured VLM output into (assessment, findings, limitations, ok).

    Returns ok=False when the assessment field is missing or invalid.
    """
    result: dict[str, str] = {
        "assessment":  "INDETERMINATE",
        "findings":    "",
        "limitations": "",
    }
    section: Optional[str] = None
    found_assessment = False

    for raw_line in text.splitlines():
        line  = raw_line.strip()
        lower = line.lower()

        if lower.startswith("assessment:"):
            value = line.split(":", 1)[1].strip().upper()
            result["assessment"] = (
                value if value in VALID_ASSESSMENTS else "INDETERMINATE"
            )
            found_assessment = value in VALID_ASSESSMENTS
            section = None

        elif lower.startswith("findings:"):
            result["findings"] = line.split(":", 1)[1].strip()
            section = "findings"

        elif lower.startswith("limitations:"):
            result["limitations"] = line.split(":", 1)[1].strip()
            section = "limitations"

        elif section and line:
            result[section] += "\n" + line

    return (
        result["assessment"],
        result["findings"],
        result["limitations"],
        found_assessment,
    )

# ---------------------------------------------------------------------------
# Helper: detect quota errors
# ---------------------------------------------------------------------------

def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()

# ---------------------------------------------------------------------------
# Main assessment function
# ---------------------------------------------------------------------------

def assess_image(
    client,
    image_path: Path,
    model_name: str,
    max_retries: int = 3,
    base_delay: float = 4.0,
) -> VLMResult:
    """
    Send one image to the Gemini API and return a VLMResult.

    The function performs up to *max_retries* attempts on transient errors.
    It stops immediately on quota errors and returns QUOTA_LIMITED status.
    No ground-truth labels are ever passed to the model.
    """
    # Resolve MIME type
    mime_map = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".webp": "image/webp",
    }
    suffix   = image_path.suffix.lower()
    mime     = mime_map.get(suffix)
    path_str = str(image_path)

    if not mime:
        return VLMResult(
            path   = path_str,
            status = AssessmentStatus.ERROR,
            error  = f"Unsupported image extension: {suffix}",
            model  = model_name,
        )

    try:
        image_bytes = image_path.read_bytes()
    except OSError as exc:
        return VLMResult(
            path   = path_str,
            status = AssessmentStatus.ERROR,
            error  = f"Cannot read image: {exc}",
            model  = model_name,
        )

    from google.genai import types  # imported here to avoid hard dependency at module load

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model    = model_name,
                contents = [
                    types.Part.from_bytes(data=image_bytes, mime_type=mime),
                    SECOND_LOOK_PROMPT,
                ],
            )
            text = response.text or ""

            if not text.strip():
                raise RuntimeError("Gemini returned an empty response.")

            assessment, findings, limitations, ok = parse_vlm_response(text)
            status = AssessmentStatus.SUCCESS if ok else AssessmentStatus.MALFORMED

            return VLMResult(
                path         = path_str,
                status       = status,
                assessment   = assessment,
                findings     = findings,
                limitations  = limitations,
                raw_response = text,
                error        = "" if ok else "No valid Assessment field found.",
                model        = model_name,
            )

        except Exception as exc:
            last_error = str(exc)
            # Mask API key in error messages
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if api_key and api_key in last_error:
                last_error = last_error.replace(api_key, "***")

            if _is_quota_error(exc):
                logger.warning("Quota/rate-limit error on attempt %d: %s", attempt, last_error[:200])
                return VLMResult(
                    path   = path_str,
                    status = AssessmentStatus.QUOTA_LIMITED,
                    error  = last_error[:500],
                    model  = model_name,
                )

            logger.warning("Attempt %d/%d failed: %s", attempt, max_retries, last_error[:200])
            if attempt < max_retries:
                sleep = min(base_delay * (2 ** (attempt - 1)), 60.0)
                time.sleep(sleep)

    return VLMResult(
        path   = path_str,
        status = AssessmentStatus.ERROR,
        error  = last_error[:500],
        model  = model_name,
    )
