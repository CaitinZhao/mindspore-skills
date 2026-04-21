#!/usr/bin/env python3
"""Single-command performance analysis pipeline for Ascend profiler data.

Usage:
    python run_analysis.py <profiler_data_dir> [--top-n 30] [--force]

Produces all artifacts under <profiler_data_dir>/out/ in one pass.
Uses a cached summary JSON to avoid re-scanning on repeat runs.

For _ascend_pt format with SQLite DB:
  - One GROUP BY scan for operator hotspots (~60s for 28M rows)
  - One GROUP BY scan for queue/device hotspots
  - Results cached in out/meta/_cache.json for instant re-runs
"""
import argparse
import csv
import hashlib
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_common import (
    infer_hardware, infer_stack_from_root, get_peak_tflops,
    write_json, write_text,
)


# ===========================================================================
# Format detection
# ===========================================================================

def detect_format(root: Path) -> str:
    name = root.name.lower()
    if name.endswith("_ascend_pt") or (root / "ASCEND_PROFILER_OUTPUT").exists():
        if _find_db(root):
            return "ascend_pt_db"
        return "ascend_pt_csv"
    if name.endswith("_ascend_ms"):
        return "ascend_ms"
    return "unknown"


def _find_db(root: Path) -> Optional[Path]:
    for p in root.rglob("ascend_pytorch_profiler.db"):
        return p
    for p in root.rglob("*.db"):
        if "profiler" in p.name.lower():
            return p
    return None


# ===========================================================================
# Single-pass DB analysis
# ===========================================================================

def analyze_db(db_path: Path, output_dir: Path, top_n: int = 30) -> dict:
    """Run all analysis in minimum number of table scans."""
    db = sqlite3.connect(str(db_path))
    cur = db.cursor()

    # 1. Load string map and enum map
    cur.execute("SELECT id, value FROM STRING_IDS")
    str_map = {r[0]: r[1] for r in cur.fetchall()}
    cur.execute("SELECT id, name FROM ENUM_API_TYPE")
    enum_map = {r[0]: r[1] for r in cur.fetchall()}

    # 2. Wall time (text MIN/MAX, single scan ~10s)
    print("  [a] Computing wall time...")
    t0 = time.time()
    cur.execute("SELECT MIN(startNs), MAX(endNs) FROM PYTORCH_API")
    r = cur.fetchone()
    wall_ns = int(r[1]) - int(r[0])
    print(f"      Done in {time.time()-t0:.1f}s (wall={wall_ns/1e9:.1f}s)")

    # 3. One GROUP BY for all op-level analysis (single scan ~60s)
    print("  [b] Scanning operator hotspots...")
    t0 = time.time()
    cur.execute(f"""
        SELECT name, type, COUNT(*),
               SUM(CAST(endNs AS INTEGER) - CAST(startNs AS INTEGER)),
               AVG(CAST(endNs AS INTEGER) - CAST(startNs AS INTEGER)),
               MAX(CAST(endNs AS INTEGER) - CAST(startNs AS INTEGER))
        FROM PYTORCH_API
        WHERE endNs > startNs
        GROUP BY name, type
        ORDER BY SUM(CAST(endNs AS INTEGER) - CAST(startNs AS INTEGER)) DESC
    """)
    all_rows = cur.fetchall()
    print(f"      Done in {time.time()-t0:.1f}s ({len(all_rows)} groups)")

    db.close()

    # --- Split into op rows (type=50001) and queue rows (type=50002) ---
    all_op_rows = [r for r in all_rows if r[1] == 50001]
    all_queue_rows = [r for r in all_rows if r[1] == 50002]

    # --- Process results ---
    total_op_ns = sum(int(r[3]) for r in all_op_rows)
    total_queue_ns = sum(int(r[3]) for r in all_queue_rows)

    # Top ops
    top_ops = []
    for r in all_op_rows[:top_n]:
        ns = int(r[3])
        top_ops.append({
            "name": str_map.get(r[0], f"ID:{r[0]}"),
            "type": enum_map.get(r[1], str(r[1])),
            "calls": r[2],
            "total_ns": ns,
            "total_ms": round(ns / 1e6, 1),
            "share_pct": round(ns / total_op_ns * 100, 2),
            "avg_us": round(int(r[4]) / 1e3, 1),
            "max_ms": round(int(r[5]) / 1e6, 2),
        })

    # Queue ops
    top_queue = []
    for r in all_queue_rows[:top_n]:
        ns = int(r[3])
        top_queue.append({
            "name": str_map.get(r[0], f"ID:{r[0]}").replace("Dequeue@", "").replace("Enqueue@", ""),
            "calls": r[2],
            "total_ns": ns,
            "total_ms": round(ns / 1e6, 1),
            "share_pct": round(ns / total_queue_ns * 100, 2),
            "avg_us": round(int(r[4]) / 1e3, 1),
            "max_ms": round(int(r[5]) / 1e6, 2),
        })

    # Sync points (from all_op_rows)
    syncs = []
    for r in all_op_rows:
        max_ms = int(r[5]) / 1e6
        if max_ms > 50:
            syncs.append({
                "name": str_map.get(r[0], f"ID:{r[0]}"),
                "calls": r[2],
                "max_ms": round(max_ms, 1),
                "avg_us": round(int(r[4]) / 1e3, 1),
            })
    syncs.sort(key=lambda x: x["max_ms"], reverse=True)

    # Fwd/Bwd split
    bw_ids = {sid for sid, v in str_map.items() if "Backward" in v}
    ag_ids = {sid for sid, v in str_map.items() if v.startswith("autograd::")}
    fw_ns = sum(int(r[3]) for r in all_op_rows if r[0] not in bw_ids and r[0] not in ag_ids)
    bw_ns = sum(int(r[3]) for r in all_op_rows if r[0] in bw_ids)
    ag_ns = sum(int(r[3]) for r in all_op_rows if r[0] in ag_ids)
    total_fba = fw_ns + bw_ns + ag_ns
    fwd_bwd = {
        "forward_ns": fw_ns, "forward_pct": round(fw_ns / total_fba * 100, 1),
        "backward_ns": bw_ns, "backward_pct": round(bw_ns / total_fba * 100, 1),
        "autograd_ns": ag_ns, "autograd_pct": round(ag_ns / total_fba * 100, 1),
        "total_ns": total_fba,
        "fw_bw_ratio": f"1:{bw_ns / fw_ns:.2f}" if fw_ns else "N/A",
    }

    # Model inference (from all rows including queue ops)
    all_counts = {str_map.get(r[0], f"ID:{r[0]}"): r[2] for r in all_rows}
    opt_steps = sum(v for k, v in all_counts.items() if "Optimizer.step" in k)
    loss_count = all_counts.get("aten::cross_entropy_loss", all_counts.get("aten::nll_loss", 0))
    fa_fwd = all_counts.get("Dequeue@aclnnFlashAttentionScore", 0)
    fa_bwd = all_counts.get("Dequeue@aclnnFlashAttentionScoreGrad", 0)
    silu_bwd = all_counts.get("Dequeue@aclnnSiluBackward", 0)
    linear_count = all_counts.get("aten::linear", 0)
    micro_batches = max(loss_count // max(opt_steps, 1), 1)
    layers = fa_fwd // max(loss_count, 1)
    linear_per_layer = linear_count // max(loss_count, 1) // 2

    model = {
        "optimizer": "AdamW",
        "optimizer_steps": opt_steps,
        "loss_calls": loss_count,
        "micro_batches_per_step": micro_batches,
        "gradient_accumulation": micro_batches,
        "inferred_layers": layers,
        "linear_per_layer_fwd": linear_per_layer,
        "mlp_type": "SwiGLU" if silu_bwd > 0 else "Standard",
        "attention_type": "FlashAttention" if fa_fwd > 0 else "Standard",
        "estimated_scale": _estimate_scale(layers, linear_per_layer),
    }

    # Dtype cast overhead
    cast_total_ns = 0
    cast_ops = []
    for r in all_op_rows:
        name = str_map.get(r[0], "")
        if any(p in name for p in ("_npu_dtype_cast", "npu_dtype_cast", "NpuDtypeCastBackward", "aclnnCast")):
            ns = int(r[3])
            cast_total_ns += ns
            cast_ops.append({"name": name, "calls": r[2], "total_ms": round(ns / 1e6, 1)})
    dtype_cast = {"total_ms": round(cast_total_ns / 1e6, 1), "ops": sorted(cast_ops, key=lambda x: x["total_ms"], reverse=True)}

    # Optimizer step timing
    # We'll derive from loss count and wall time
    per_step_s = (wall_ns / 1e9) / max(opt_steps, 1) if opt_steps else 0

    # Device utilization
    dequeue_ns = sum(r["total_ns"] for r in top_queue if "Dequeue" in str_map.get(
        next((rid for rid, v in str_map.items() if v == r["name"]), 0), ""))
    # Approximate: use all Dequeue time as kernel estimate
    kernel_ns = sum(r["total_ns"] for r in top_queue[:15])  # from queue data

    util = {
        "wall_s": round(wall_ns / 1e9, 1),
        "op_s": round(total_op_ns / 1e9, 1),
        "queue_s": round(total_queue_ns / 1e9, 1),
        "kernel_s": round(total_queue_ns / 1e9, 2),
        "device_util_pct": round(total_queue_ns / wall_ns * 100, 1) if wall_ns else 0,
        "kernel_util_pct": round(total_queue_ns / wall_ns * 100, 1) if wall_ns else 0,
        "host_device_ratio": round(total_op_ns / total_queue_ns, 1) if total_queue_ns else 0,
    }

    return {
        "wall_ns": wall_ns,
        "util": util,
        "top_ops": top_ops,
        "top_queue": top_queue,
        "syncs": syncs,
        "fwd_bwd": fwd_bwd,
        "model": model,
        "dtype_cast": dtype_cast,
        "per_step_s": per_step_s,
        "opt_steps": opt_steps,
    }


def _estimate_scale(layers: int, linear_per_layer: int) -> str:
    if layers <= 0: return "unknown"
    if layers <= 16: return "~1-3B"
    if layers <= 36: return "~7B"
    if layers <= 48: return "~13B"
    if layers <= 64: return "~30B"
    if layers <= 96: return "~70B"
    return "~100B+"


# ===========================================================================
# Bottleneck classification & suggestions (unchanged logic, compact)
# ===========================================================================

def classify_bottlenecks(util, top_ops, fwd_bwd, syncs, dtype_cast, model):
    bns = []
    dev = util.get("device_util_pct", 100)
    kern = util.get("kernel_util_pct", 100)

    if dev < 50:
        bns.append({"rank": 1, "domain": "host_framework_overhead",
            "severity": "CRITICAL" if dev < 40 else "HIGH",
            "title": f"Host Launch Overhead Dominates ({100-dev:.1f}% non-device time)",
            "confidence": "strong",
            "evidence": [f"Device utilization: {dev}%", f"Kernel utilization: {kern}%",
                         f"Host-device ratio: {util.get('host_device_ratio',0)}x",
                         f"Autograd engine: {fwd_bwd.get('autograd_pct',0)}%"],
            "suggestion_id": "HOST-01"})

    if kern < 30:
        bns.append({"rank": len(bns)+1, "domain": "low_mfu",
            "severity": "HIGH" if kern < 20 else "MEDIUM",
            "title": f"Device Severely Underutilized (kernel: {kern}%)",
            "confidence": "strong",
            "evidence": [f"Kernel time: {util.get('kernel_s',0)}s / wall: {util.get('wall_s',0)}s"],
            "suggestion_id": "COMP-02"})

    big_syncs = [s for s in syncs if s["max_ms"] > 100]
    if big_syncs:
        w = big_syncs[0]
        bns.append({"rank": len(bns)+1, "domain": "unnecessary_sync",
            "severity": "HIGH" if w["max_ms"] > 200 else "MEDIUM",
            "title": f"Sync Stall: {w['name']} (max {w['max_ms']}ms)",
            "confidence": "strong",
            "evidence": [f"{s['name']}: max {s['max_ms']}ms" for s in big_syncs[:3]],
            "suggestion_id": "NPU-AFFINITY-04"})

    cast_ms = dtype_cast.get("total_ms", 0)
    total_ms = fwd_bwd.get("total_ns", 0) / 1e6
    if cast_ms > total_ms * 0.03:
        bns.append({"rank": len(bns)+1, "domain": "dtype_cast_overhead",
            "severity": "MEDIUM",
            "title": f"Dtype Cast Overhead ({cast_ms:.0f}ms, {cast_ms/total_ms*100:.1f}%)",
            "confidence": "moderate",
            "evidence": [f"{op['name']}: {op['total_ms']}ms" for op in dtype_cast.get("ops",[])[:3]],
            "suggestion_id": "HOST-01-SECONDARY"})

    if top_ops and top_ops[0]["share_pct"] > 25:
        op = top_ops[0]
        bns.append({"rank": len(bns)+1, "domain": "operator_hotspot",
            "severity": "HIGH" if op["share_pct"] > 35 else "MEDIUM",
            "title": f"Operator Hotspot: {op['name']} ({op['share_pct']}%)",
            "confidence": "strong",
            "evidence": [f"{op['name']}: {op['total_ms']}ms, {op['calls']} calls"],
            "suggestion_id": "COMP-03"})

    for i, b in enumerate(bns):
        b["rank"] = i + 1
    return bns


SUGGESTIONS_MAP = {
    "HOST-01": ("HIGH", "Enable torch.compile (Graph Mode)",
                "Host-device ratio > 2x. Graph compilation fuses small ops, eliminates Python overhead.",
                "20-50% throughput improvement",
                'model = torch.compile(model, mode="max-autotune")'),
    "COMP-02": ("HIGH", "Improve Device Utilization",
                "Kernel utilization < 30%. Enable graph compilation (HOST-01) and consider larger batch size.",
                "2-3x MFU improvement",
                "print(f'Allocated: {torch_npu.npu.memory_allocated()/1e9:.1f} GB')"),
    "NPU-AFFINITY-04": ("HIGH", "Fix Synchronization Stall",
                "Large sync stalls causing device idle gaps.",
                "Save hundreds of ms per occurrence",
                "if global_step % 100 == 0:\n    norm = clip_grad_norm_(...)"),
    "HOST-01-SECONDARY": ("MEDIUM", "Reduce Dtype Cast Overhead",
                          "Significant FP32/FP16 cast overhead. Consider BF16 or verify AMP config.",
                          "5-10% throughput improvement",
                          "model = model.to(torch.bfloat16)"),
    "COMP-03": ("HIGH", "Optimize Hotspot Operator",
                "Single operator dominates compute time. Check for fused variant.",
                "10-30% step time reduction",
                "# Check for fused variant or custom kernel"),
}


def build_suggestions(bottlenecks, model):
    suggestions = []
    seen = set()
    for bn in bottlenecks:
        sid = bn.get("suggestion_id", "")
        if sid in SUGGESTIONS_MAP and sid not in seen:
            seen.add(sid)
            pri, title, desc, benefit, code = SUGGESTIONS_MAP[sid]
            suggestions.append({"id": sid, "priority": pri, "title": title,
                                "description": desc, "expected_benefit": benefit, "code_example": code})
    if model.get("optimizer") == "AdamW" and "NPU-AFFINITY-01" not in seen:
        suggestions.append({"id": "NPU-AFFINITY-01", "priority": "MEDIUM",
                            "title": "Replace AdamW with NpuFusedAdamW",
                            "description": "NPU fused optimizer batches parameter updates.",
                            "expected_benefit": "5-15% optimizer step reduction",
                            "code_example": "optimizer = torch_npu.optim.NpuFusedAdamW(model.parameters(), lr=1e-4)"})
    return suggestions


# ===========================================================================
# Report rendering
# ===========================================================================

def render_report(data, bottlenecks, suggestions, root, hardware, elapsed) -> str:
    util = data["util"]
    model = data["model"]
    top_ops = data["top_ops"]
    top_queue = data["top_queue"]
    fwd_bwd = data["fwd_bwd"]
    dev = util.get("device_util_pct", 100)
    kern = util.get("kernel_util_pct", 100)
    verdict = "host_bound" if dev < 50 else ("compute_bound" if kern > 70 else "balanced")

    L = [f"# Performance Diagnosis Report\n",
         f"**Generated:** {time.strftime('%Y-%m-%d %H:%M')} | **Elapsed:** {elapsed:.1f}s",
         f"**Data:** `{root.name}` | **Hardware:** {hardware or 'Unknown'} | **Verdict:** {verdict.upper()}\n",
         "## Executive Summary\n"]
    if verdict == "host_bound":
        L.append(f"**HOST-BOUND**: NPU utilized only **{dev}%** (kernel: **{kern}%**). "
                 f"Host-device ratio: **{util.get('host_device_ratio',0)}x**. Enable graph compilation.\n")
    if model.get("inferred_layers"):
        L.append(f"Model: ~{model['inferred_layers']}-layer Transformer ({model.get('estimated_scale','?')}), "
                 f"{model.get('mlp_type','?')} MLP, {model.get('gradient_accumulation','?')} grad accum.\n")

    L.append("## Timing\n")
    L.append(f"| Metric | Value |")
    L.append(f"|--------|-------|")
    L.append(f"| Wall time | {util.get('wall_s',0)}s |")
    L.append(f"| Optimizer steps | {data.get('opt_steps',0)} |")
    L.append(f"| Per step | {data.get('per_step_s',0):.2f}s |")
    L.append(f"| Device utilization | **{dev}%** |")
    L.append(f"| Kernel utilization | **{kern}%** |")
    L.append(f"| Host-device ratio | **{util.get('host_device_ratio',0)}x** |\n")

    if fwd_bwd:
        L.append(f"**FW/BW/Autograd:** {fwd_bwd['forward_pct']}% / {fwd_bwd['backward_pct']}% / {fwd_bwd['autograd_pct']}% "
                 f"(FW:BW = {fwd_bwd['fw_bw_ratio']})\n")

    if top_ops:
        L.append("## Top Operators\n")
        L.append("| # | Operator | Calls | Total | Share | Avg | Max |")
        L.append("|---|----------|-------|-------|-------|-----|-----|")
        for i, op in enumerate(top_ops[:15], 1):
            t = f"{op['total_ms']/1000:.2f}s" if op['total_ms'] > 1000 else f"{op['total_ms']:.0f}ms"
            L.append(f"| {i} | `{op['name']}` | {op['calls']:,} | {t} | {op['share_pct']}% | {op['avg_us']:.0f}us | {op['max_ms']:.0f}ms |")
        L.append("")

    if top_queue:
        L.append("## Top Device Kernels (from Dequeue ops)\n")
        L.append("| # | Kernel | Calls | Total | Share |")
        L.append("|---|--------|-------|-------|-------|")
        for i, k in enumerate(top_queue[:10], 1):
            t = f"{k['total_ms']/1000:.2f}s" if k['total_ms'] > 1000 else f"{k['total_ms']:.0f}ms"
            L.append(f"| {i} | `{k['name'][:50]}` | {k['calls']:,} | {t} | {k['share_pct']}% |")
        L.append("")

    if bottlenecks:
        L.append("## Bottlenecks\n")
        icons = {"CRITICAL": "X", "HIGH": "!", "MEDIUM": "?"}
        for bn in bottlenecks:
            ic = icons.get(bn['severity'], "-")
            L.append(f"### [{ic}] {bn['rank']}. {bn['title']}")
            L.append(f"- **Domain:** {bn['domain']} | **Confidence:** {bn['confidence']}")
            for ev in bn['evidence'][:3]:
                L.append(f"  - {ev}")
            L.append("")

    if suggestions:
        L.append("## Optimization Suggestions\n")
        for i, sg in enumerate(suggestions, 1):
            L.append(f"### {i}. [{sg['id']}] {sg['title']} ({sg['priority']})")
            L.append(f"**{sg['description']}** Expected: {sg['expected_benefit']}")
            L.append(f"```python\n{sg['code_example']}\n```\n")

    return "\n".join(L) + "\n"


# ===========================================================================
# Cache management
# ===========================================================================

def _cache_key(db_path: Path) -> str:
    return hashlib.md5(f"{db_path}-{db_path.stat().st_size}".encode()).hexdigest()[:12]


def _cache_path(output_dir: Path) -> Path:
    return output_dir / "meta" / "_cache.json"


def load_cache(output_dir: Path, db_path: Path) -> Optional[dict]:
    cp = _cache_path(output_dir)
    if not cp.exists():
        return None
    try:
        cache = json.loads(cp.read_text(encoding="utf-8"))
        if cache.get("_key") == _cache_key(db_path):
            return cache
    except Exception:
        pass
    return None


def save_cache(output_dir: Path, db_path: Path, data: dict) -> None:
    cp = _cache_path(output_dir)
    data["_key"] = _cache_key(db_path)
    write_json(cp, data)


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Single-command performance analysis")
    parser.add_argument("profiler_dir", help="Path to profiler data directory")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--force", action="store_true", help="Force re-analysis, ignore cache")
    args = parser.parse_args()

    root = Path(args.profiler_dir).resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        return 1

    t0 = time.time()
    fmt = detect_format(root)
    print(f"[1/6] Format: {fmt}")

    out_dir = root / "out"
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    hardware = infer_hardware(root)
    info_file = root / "profiler_info.json"
    profiler_info = json.loads(info_file.read_text(encoding="utf-8", errors="replace")) if info_file.exists() else {}

    # Check cache
    if fmt == "ascend_pt_db" and not args.force:
        db_path = _find_db(root)
        cached = load_cache(out_dir, db_path)
        if cached:
            print(f"[2/6] Using cached results (use --force to re-analyze)")
            data = cached
        else:
            print(f"[2/6] Analyzing DB (first run, will cache results)...")
            data = analyze_db(db_path, out_dir, args.top_n)
            save_cache(out_dir, db_path, data)
    elif fmt == "ascend_pt_db":
        db_path = _find_db(root)
        print(f"[2/6] Analyzing DB (--force)...")
        data = analyze_db(db_path, out_dir, args.top_n)
        save_cache(out_dir, db_path, data)
    else:
        print(f"[2/6] Format {fmt} not supported for automated analysis yet.")
        print(f"      Use the manual pipeline described in SKILL.md.")
        return 1

    print(f"[3/6] Classifying bottlenecks...")
    bottlenecks = classify_bottlenecks(data["util"], data["top_ops"], data["fwd_bwd"],
                                        data["syncs"], data["dtype_cast"], data["model"])

    print(f"[4/6] Generating suggestions...")
    suggestions = build_suggestions(bottlenecks, data["model"])

    print(f"[5/6] Writing artifacts...")
    dev = data["util"].get("device_util_pct", 0)
    kern = data["util"].get("kernel_util_pct", 0)
    verdict = "host_bound" if dev < 50 else ("compute_bound" if kern > 70 else "balanced")

    write_json(meta_dir / "performance-profile.json", {
        "schema_version": "performance-agent/0.1",
        "trace_root": str(root), "format": fmt,
        "stack": infer_stack_from_root(root),
        "torch_npu_version": profiler_info.get("torch_npu_version"),
        "cann_version": profiler_info.get("cann_version"),
        "workload_type": "training" if any("Backward" in op["name"] for op in data["top_ops"][:20]) else "inference",
        "primary_symptom": "host launch overhead" if verdict == "host_bound" else "throughput",
        "confidence": "strong",
        "hardware": {"detected_model": hardware, "peak_fp16_tflops": get_peak_tflops(hardware, "fp16")},
        "model_architecture": data["model"],
        "device_utilization": data["util"],
        "forward_backward_ratio": data["fwd_bwd"],
    })
    write_json(meta_dir / "bottlenecks.json", {"bottlenecks": bottlenecks})
    write_json(meta_dir / "optimization-suggestions.json", {"suggestions": suggestions})
    write_json(meta_dir / "performance-verdict.json", {
        "verdict": verdict,
        "dominant_bottleneck": bottlenecks[0]["domain"] if bottlenecks else "unknown",
        "severity": bottlenecks[0]["severity"] if bottlenecks else "UNKNOWN",
        "device_utilization_pct": dev,
        "primary_recommendation": suggestions[0]["title"] if suggestions else "N/A",
    })

    print(f"[6/6] Generating report...")
    report = render_report(data, bottlenecks, suggestions, root, hardware, time.time() - t0)
    write_text(out_dir / "report.md", report)

    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Report: {out_dir / 'report.md'}")
    print(f"Verdict: {verdict.upper()}")
    if bottlenecks:
        print(f"Top bottleneck: {bottlenecks[0]['title']}")
    if suggestions:
        print(f"Top suggestion: {suggestions[0]['title']}")
    print(f"{'='*55}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
