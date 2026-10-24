"""
Unit tests for VLM response parsing and status handling.

These tests are purely local — they do not call any API.
"""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vlm.client import parse_vlm_response, VALID_ASSESSMENTS, AssessmentStatus


class TestParseVlmResponse:
    """Tests for the VLM structured-response parser."""

    def test_valid_pneumonia_suggestive(self):
        text = (
            "Assessment: PNEUMONIA_SUGGESTIVE\n"
            "Findings: Increased opacity in lower lobe\n"
            "Limitations: Low image quality"
        )
        assessment, findings, limitations, ok = parse_vlm_response(text)
        assert assessment == "PNEUMONIA_SUGGESTIVE"
        assert "opacity" in findings
        assert ok is True

    def test_valid_not_pneumonia(self):
        text = (
            "Assessment: NOT_PNEUMONIA_SUGGESTIVE\n"
            "Findings: Lungs clear bilaterally\n"
            "Limitations: Lateral view not available"
        )
        assessment, findings, limitations, ok = parse_vlm_response(text)
        assert assessment == "NOT_PNEUMONIA_SUGGESTIVE"
        assert ok is True

    def test_valid_indeterminate(self):
        text = (
            "Assessment: INDETERMINATE\n"
            "Findings: Difficult to assess\n"
            "Limitations: Poor image quality"
        )
        assessment, findings, limitations, ok = parse_vlm_response(text)
        assert assessment == "INDETERMINATE"
        assert ok is True

    def test_invalid_assessment_falls_back_to_indeterminate(self):
        text = (
            "Assessment: UNCLEAR\n"
            "Findings: Something\n"
            "Limitations: Nothing"
        )
        assessment, findings, limitations, ok = parse_vlm_response(text)
        assert assessment == "INDETERMINATE"
        assert ok is False  # Value was not in the allowed set

    def test_missing_assessment_field(self):
        text = "Findings: Clear\nLimitations: None"
        assessment, findings, limitations, ok = parse_vlm_response(text)
        assert assessment == "INDETERMINATE"
        assert ok is False

    def test_empty_response(self):
        assessment, findings, limitations, ok = parse_vlm_response("")
        assert assessment == "INDETERMINATE"
        assert ok is False

    def test_multiline_findings(self):
        text = (
            "Assessment: NOT_PNEUMONIA_SUGGESTIVE\n"
            "Findings: Clear lung fields\n"
            "No significant infiltrates\n"
            "Limitations: Single projection only"
        )
        assessment, findings, limitations, ok = parse_vlm_response(text)
        assert ok is True
        assert "infiltrates" in findings or "Clear" in findings

    def test_case_insensitive_field_prefix(self):
        text = (
            "assessment: PNEUMONIA_SUGGESTIVE\n"
            "findings: Consolidation\n"
            "limitations: None"
        )
        assessment, findings, limitations, ok = parse_vlm_response(text)
        assert assessment == "PNEUMONIA_SUGGESTIVE"
        assert ok is True

    def test_all_valid_assessments_are_recognized(self):
        for value in VALID_ASSESSMENTS:
            text = f"Assessment: {value}\nFindings: x\nLimitations: y"
            assessment, findings, limitations, ok = parse_vlm_response(text)
            assert assessment == value
            assert ok is True

    def test_assessment_status_enum_values(self):
        assert AssessmentStatus.SUCCESS.value       == "SUCCESS"
        assert AssessmentStatus.QUOTA_LIMITED.value == "QUOTA_LIMITED"
        assert AssessmentStatus.MALFORMED.value     == "MALFORMED"
        assert AssessmentStatus.ERROR.value         == "ERROR"
        assert AssessmentStatus.SKIPPED.value       == "SKIPPED"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
