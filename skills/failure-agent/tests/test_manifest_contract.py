from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = SKILL_ROOT / "skill.yaml"
SHOWCASE = SKILL_ROOT / "reference" / "failure-showcase.md"


def _manifest_text() -> str:
    return MANIFEST.read_text(encoding="utf-8")


def test_manifest_contract_fields_present():
    text = _manifest_text()
    assert 'name: "failure-agent"' in text
    assert 'description: "Diagnose MindSpore and PTA (PyTorch + torch_npu) crashes, runtime errors, hangs, and communication failures with evidence-first triage, ordered knowledge lookup, and manual-only report candidates."' in text
    assert 'version: "0.5.13"' in text
    assert 'type: "manual"' in text
    assert 'path: "SKILL.md"' in text
    assert 'python: []' in text
    assert 'network: "none"' in text
    assert 'filesystem: "workspace-write"' in text


def test_manifest_declares_dual_stack_scope_and_honest_dependencies():
    text = _manifest_text()
    assert "MindSpore and PTA" in text
    assert 'name: "factory_root"' in text
    assert 'type: "path"' in text
    assert '"bash"' in text
    assert '"rg"' in text
    assert "report_schema" in text
    assert "out_dir_layout" in text


def test_reference_files_exist():
    reference_dir = SKILL_ROOT / "reference"
    for name in [
        "failure-showcase.md",
        "pta-diagnosis.md",
        "backend-diagnosis.md",
        "mindspore-api-reference.md",
        "mindspore-dianosis.md",
        "pta-diagnosis.md",
        "cann-api-reference.md",
    ]:
        assert (reference_dir / name).exists(), name


def test_failure_showcase_uses_known_issue_transition_shape():
    text = SHOWCASE.read_text(encoding="utf-8")
    assert "# Failure Showcase (Known-Issue Transition Reference)" in text
    assert "## Shared Failures" in text
    assert "## MindSpore-Focused Failures" in text
    assert "## PTA / torch_npu-Focused Failures" in text
    assert "- kind: known_issue" in text
    assert "- symptom: failure" in text
    assert "- id_hint:" in text
    assert "- severity:" in text
    assert "- lifecycle_state:" in text
    assert "- source_kind:" in text
    assert "- confidence_level:" in text
    assert "- tags:" in text
    assert "- affects_platforms:" in text
    assert "- detection_pattern:" in text
    assert "- description:" in text
    assert "- fix_summary:" in text
    assert "- validation:" in text
    assert "Do not auto-mutate Factory from this file." in text
