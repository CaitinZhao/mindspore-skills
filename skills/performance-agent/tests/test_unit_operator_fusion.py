"""Unit tests for analyze_operator_fusion.py."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import run_script


def test_fusion_with_flash_attention_pattern(tmp_path: Path):
    """Test fusion detection identifies FlashAttention opportunity."""
    hotspot_json = tmp_path / "hotspot.json"
    hotspot_json.write_text(json.dumps({
        "top_operators": [
            {"operator": "SelfAttention_MatMul", "share_percent": 25.0, "count": 100},
            {"operator": "Softmax_Compute", "share_percent": 12.0, "count": 100},
            {"operator": "Dropout_Generator", "share_percent": 8.0, "count": 100},
            {"operator": "LayerNorm", "share_percent": 5.0, "count": 100},
        ]
    }), encoding="utf-8")

    output_json = tmp_path / "fusion.json"
    run_script("analyze_operator_fusion.py",
               "--hotspot-json", str(hotspot_json),
               "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["fusion_analysis_available"] is True
    assert result["total_fusion_candidates"] >= 1
    types = [o["type"] for o in result["opportunities"]]
    assert "flash_attention" in types


def test_fusion_with_matmul_allreduce_tp(tmp_path: Path):
    """Test fusion detects MatmulAllReduce in TP scenario."""
    hotspot_json = tmp_path / "hotspot.json"
    hotspot_json.write_text(json.dumps({
        "top_operators": [
            {"operator": "MatMul_Dense", "share_percent": 30.0, "count": 50},
            {"operator": "AllReduce_Sum", "share_percent": 20.0, "count": 50},
        ]
    }), encoding="utf-8")

    comm_json = tmp_path / "comm.json"
    comm_json.write_text(json.dumps({
        "top_collectives": [{"name": "AllReduce", "time_ms": 100}]
    }), encoding="utf-8")

    output_json = tmp_path / "fusion.json"
    run_script("analyze_operator_fusion.py",
               "--hotspot-json", str(hotspot_json),
               "--communication-json", str(comm_json),
               "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    types = [o["type"] for o in result["opportunities"]]
    assert "matmul_allreduce" in types


def test_fusion_no_hotspot(tmp_path: Path):
    """Test fusion analysis with no input data."""
    output_json = tmp_path / "fusion.json"
    run_script("analyze_operator_fusion.py",
               "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["fusion_analysis_available"] is False
