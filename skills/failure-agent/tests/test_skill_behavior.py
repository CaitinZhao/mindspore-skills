from pathlib import Path


SKILL_MD = Path(__file__).resolve().parents[1] / "SKILL.md"


def test_behavior_rules_require_evidence_and_validation():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "Collect evidence before diagnosis." in text
    assert "State assumptions and unknowns explicitly." in text
    assert "Every root-cause claim must include a validation check." in text
    assert "Do not treat a fix as confirmed until the user verifies it." in text
    assert "do not guess the Factory path; only use an explicitly provided `factory_root`" in text
    assert "do not fabricate Factory lookups when cards are unavailable" in text
    assert "do not claim a knowledge hit unless the signature actually matches" in text
    assert "do not auto-submit a Factory `report`" in text


def test_mindspore_scoping_and_deep_debug_rules_present():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "### MindSpore scoping summary" in text
    assert "`API/mode misuse`, `unsupported/missing op`, `graph compile/frontend issue`, `runtime/backend issue`, `distributed/communication issue`, or `numerical/precision symptom` when it appears as part of a runtime failure rather than standalone accuracy work" in text
    assert "state the selected layer or component" in text
    assert "cite 2-4 supporting facts from the evidence" in text
    assert "[mindspore-dianosis](reference/mindspore-dianosis.md)" in text


def test_dual_stack_routes_and_output_contract_present():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "`ms`: `Platform -> Scripts -> MindSpore Framework -> Backend`" in text
    assert "`pta`: `Platform -> Scripts -> torch_npu Framework -> CANN`" in text
    assert "knowledge-hit status: `known_issue`, `operator`, or `none`" in text
    assert "12. Knowledge candidate or `report` candidate (optional, manual only)" in text


def test_direct_read_contract_and_fallback_present():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "`factory_root` is provided and readable" in text
    assert "`factory_root/cards/known_issues/*.yaml`" in text
    assert "`factory_root/cards/operators/*.yaml`" in text
    assert "do not directly read `perf_features`, `algo_features`, `models`, or `reports` in this skill" in text
    assert "temporary fallback, not the primary knowledge source" in text


def test_reference_guides_align_with_current_factory_cards():
    root = SKILL_MD.parent / "reference"
    backend = (root / "backend-diagnosis.md").read_text(encoding="utf-8")
    errors = (root / "pta-diagnosis.md").read_text(encoding="utf-8")
    ms_api = (root / "mindspore-api-reference.md").read_text(encoding="utf-8")
    deep = (root / "mindspore-dianosis.md").read_text(encoding="utf-8")
    pta = (root / "pta-diagnosis.md").read_text(encoding="utf-8")
    assert "missing-cann-environment" in backend
    assert "ms-tbe-operator-compilation-error" in backend
    assert "distributed-communication-timeout" in pta
    assert "ms-context-empty" in ms_api
    assert "2.14748e+09" in ms_api
    assert "device generation" in deep
    assert "blocking vs default async execution" in deep
    assert "stack-version-mismatch" in pta
    assert "Expected all tensors to be on the same device" in pta
    assert "native_functions.yaml" in pta
    assert "ASCEND_LAUNCH_BLOCKING=1" in pta
