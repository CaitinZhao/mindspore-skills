from pathlib import Path


def test_skill_markers_present():
    skill_md = Path(__file__).resolve().parents[1] / "SKILL.md"
    text = skill_md.read_text(encoding="utf-8")
    assert "## Stage 0: Gather Context and Detect Stack" in text
    assert "## Stage 1: Find Similar Problem First" in text
    assert "## Stage 2: Analyze Failure" in text
    assert "## Stage 3: Validate and Close" in text
    assert "stack (`ms` or `pta`)" in text
