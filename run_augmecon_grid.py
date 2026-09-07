"""
AUGMECON grid driver for proposed_augmecon_gurobi.py (Mavrotas 2009).

Default loop is k = q … 0 (loose → tight) so the surplus bypass is valid:
if s* ≥ Δε at ε_k, the next floor(s*/Δε) tighter right-hand sides copy the
same efficient solution.

Each grid point is the two-stage lex subproblem (ρ → 0+):
  Stage A: max (Z1 − p ΣΔ)  s.t. Z2 + s = ε_k
  Stage B: max s            s.t. primary ≥ Stage-A incumbent

Example:
  python run_augmecon_grid.py --mip-gap 0.001 --threads 12 --mip-focus 2
  python run_augmecon_grid.py --payoff --payoff-only
  python run_augmecon_grid.py --extra-epsilon 52356
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from gurobipy import GRB

from proposed_augmecon_gurobi import (
    LEX_PRIMARY_TOL,
    R_MAX,
    R_MIN,
    RHO,
    apply_greedy_start,
    apply_mip_start,
    build_model,
    capture_assignment,
    load_instance,
    metrics_from_assignment,
    print_subproblem_metrics,
    q as DEFAULT_Q,
    solve_lex_subproblem,
)

ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "result" / "augmecon_grid.json"
OUT_CSV = ROOT / "result" / "augmecon_grid.csv"
PAYOFF_JSON = ROOT / "result" / "augmecon_payoff.json"
LOG_DIR = ROOT / "result" / "logs"


def _safe_out_path(raw: str | None, default: Path) -> Path:
    """Resolve a user path and reject writes outside the project root."""
    if raw is None:
        return default
    p = Path(raw).expanduser().resolve()
    root = ROOT.resolve()
    try:
        p.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"Refusing path outside project root: {p}") from exc
    return p

GRID_METRIC_KEYS = (
    "status",
    "infeasible",
    "epsilon_k",
    "value",
    "risk",
    "cost",
    "slack",
    "primary_obj",
    "primary_bound",
    "primary_gap_pct",
    "rho_term",
    "augmented_obj",
    "bypass",
    "time_s",
    "stage_a_time_s",
    "stage_b_time_s",
    "n_chosen",
)


def _summary_row(metrics: dict) -> dict:
    return {k: metrics.get(k) for k in GRID_METRIC_KEYS}


def _epsilon(k: int, r_min: float, step: float) -> float:
    return float(r_min) + int(k) * float(step)


def _write_outputs(payload: dict, json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(payload, jf, ensure_ascii=False, indent=2)
    rows = list(payload.get("points") or [])
    extra = list(payload.get("extra") or [])
    fieldnames = [
        "kind",
        "k",
        "skipped",
        "copied_from_k",
        "epsilon_k",
        "risk",
        "value",
        "slack",
        "time_s",
        "bypass",
        "primary_obj",
        "primary_bound",
        "primary_gap_pct",
        "status",
        "infeasible",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "kind": "grid",
                    "k": r.get("k"),
                    "skipped": r.get("skipped", False),
                    "copied_from_k": r.get("copied_from_k"),
                    "epsilon_k": r.get("epsilon_k"),
                    "risk": r.get("risk"),
                    "value": r.get("value"),
                    "slack": r.get("slack"),
                    "time_s": r.get("time_s"),
                    "bypass": r.get("bypass"),
                    "primary_obj": r.get("primary_obj"),
                    "primary_bound": r.get("primary_bound"),
                    "primary_gap_pct": r.get("primary_gap_pct"),
                    "status": r.get("status"),
                    "infeasible": r.get("infeasible"),
                }
            )
        for r in extra:
            w.writerow(
                {
                    "kind": r.get("label", "extra"),
                    "k": "",
                    "skipped": False,
                    "copied_from_k": "",
                    "epsilon_k": r.get("epsilon_k"),
                    "risk": r.get("risk"),
                    "value": r.get("value"),
                    "slack": r.get("slack"),
                    "time_s": r.get("time_s"),
                    "bypass": r.get("bypass"),
                    "primary_obj": r.get("primary_obj"),
                    "primary_bound": r.get("primary_bound"),
                    "primary_gap_pct": r.get("primary_gap_pct"),
                    "status": r.get("status"),
                    "infeasible": r.get("infeasible"),
                }
            )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def _z1_z2_from_incumbent(pack) -> tuple[float, float]:
    z1 = 0.0
    z2 = 0.0
    for u, v in pack.u_v_pairs:
        if pack.x[u, v].X > 0.5:
            z1 += float(pack.user_video_data[u][v]["value"])
            z2 += float(pack.videos_data[v]["r"])
    return z1, z2


def solve_payoff_table(
    pack,
    step: float,
    r_hint_max: float,
    warm_start: bool,
    include_push_info: bool,
) -> dict:
    """Lexicographic payoff for the Z2 range used by the ε-grid."""
    pack.drop_epsilon()
    if warm_start and pack.u_v_pairs:
        apply_greedy_start(pack, float(r_hint_max) * 10.0 + 1.0)

    print("=== Payoff P1: max (Z1 − p ΣΔ) ===")
    pack.model.ModelSense = GRB.MAXIMIZE
    pack.model.setObjective(pack.obj_primary, GRB.MAXIMIZE)
    pack.model.optimize()
    t1 = float(pack.model.Runtime)
    if pack.model.SolCount <= 0:
        raise RuntimeError("Payoff P1 (max primary) found no incumbent.")
    obj_star = float(pack.obj_primary.getValue())
    z1_max, z2_at_z1max = _z1_z2_from_incumbent(pack)
    print(
        f"P1 primary={obj_star:.6g}  Z1={z1_max:.6g}  "
        f"Z2={z2_at_z1max:.6g}  time={t1:.3f}s"
    )

    print("=== Payoff P1-lex: min Z2 s.t. primary ≥ incumbent ===")
    keep = pack.model.addConstr(
        pack.obj_primary >= obj_star - LEX_PRIMARY_TOL,
        name="payoff_keep_primary",
    )
    pack.model.setObjective(pack.risk_expr, GRB.MINIMIZE)
    pack.model.optimize()
    t1b = float(pack.model.Runtime)
    if pack.model.SolCount <= 0:
        pack.model.remove(keep)
        pack.model.update()
        raise RuntimeError("Payoff P1-lex (min Z2 on max-primary face) found no incumbent.")
    z1_at_nadir, z2_nadir = _z1_z2_from_incumbent(pack)
    pack.model.remove(keep)
    pack.model.update()
    print(f"P1-lex Z2_nadir={z2_nadir:.6g}  Z1={z1_at_nadir:.6g}  time={t1b:.3f}s")

    print("=== Payoff P2: min Z2 ===")
    pack.model.setObjective(pack.risk_expr, GRB.MINIMIZE)
    pack.model.optimize()
    t2 = float(pack.model.Runtime)
    if pack.model.SolCount <= 0:
        raise RuntimeError("Payoff P2 (min Z2) found no incumbent.")
    z1_at_z2min, z2_min = _z1_z2_from_incumbent(pack)
    print(f"P2 Z2_min={z2_min:.6g}  Z1={z1_at_z2min:.6g}  time={t2:.3f}s")

    print("=== Payoff P2-lex: max primary s.t. Z2 ≤ Z2_min ===")
    keep2 = pack.model.addConstr(
        pack.risk_expr <= z2_min + LEX_PRIMARY_TOL,
        name="payoff_keep_z2min",
    )
    pack.model.setObjective(pack.obj_primary, GRB.MAXIMIZE)
    pack.model.optimize()
    t2b = float(pack.model.Runtime)
    if pack.model.SolCount <= 0:
        pack.model.remove(keep2)
        pack.model.update()
        raise RuntimeError("Payoff P2-lex (max primary at Z2_min) found no incumbent.")
    z1_at_utopia, z2_check = _z1_z2_from_incumbent(pack)
    pack.model.remove(keep2)
    pack.model.update()
    print(f"P2-lex Z1={z1_at_utopia:.6g}  Z2={z2_check:.6g}  time={t2b:.3f}s")

    pack.model.ModelSense = GRB.MAXIMIZE
    pack.model.setObjective(pack.obj_primary, GRB.MAXIMIZE)
    payoff = {
        "z1_max": z1_max,
        "z2_at_z1_max": z2_at_z1max,
        "z2_nadir": z2_nadir,
        "z1_at_nadir": z1_at_nadir,
        "z2_min": z2_min,
        "z1_at_z2min": z1_at_z2min,
        "z1_at_z2min_lex": z1_at_utopia,
        "z2_at_z2min_lex": z2_check,
        "primary_at_z1_max": obj_star,
        "time_s": t1 + t1b + t2 + t2b,
        "include_push_info": include_push_info,
    }
    return payoff


def _warm_start_for_epsilon(
    pack,
    epsilon_k: float,
    prev_x: dict[tuple[int, int], int] | None,
    prev_risk: float | None,
    use_greedy: bool,
) -> None:
    if prev_x and prev_risk is not None and prev_risk <= float(epsilon_k) + 1e-8:
        apply_mip_start(pack, prev_x, epsilon_k)
        return
    if use_greedy and pack.u_v_pairs:
        apply_greedy_start(pack, epsilon_k)


def run_grid(args: argparse.Namespace) -> dict:
    q = int(args.q)
    if q <= 0:
        raise SystemExit("--q must be a positive integer.")
    k_min = 0 if args.k_min is None else int(args.k_min)
    k_max = q if args.k_max is None else int(args.k_max)
    if not (0 <= k_min <= k_max <= q):
        raise SystemExit(f"Require 0 ≤ k_min ≤ k_max ≤ q; got {k_min}, {k_max}, {q}.")

    users_data, videos_data, user_video_data = load_instance()
    r_min = float(R_MIN if args.r_min is None else args.r_min)
    r_max = float(R_MAX if args.r_max is None else args.r_max)
    log_dir = _safe_out_path(args.log_dir, LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    pack = build_model(
        users_data,
        videos_data,
        user_video_data,
        epsilon_k=r_max,
        r_range=max(r_max - r_min, 1.0),
        value_eps=args.value_eps,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        mip_focus=args.mip_focus,
        output_flag=0 if args.quiet else 1,
        log_file=str(log_dir / "augmecon_grid.log"),
    )

    payoff = None
    if args.payoff:
        payoff = solve_payoff_table(
            pack,
            step=1.0,
            r_hint_max=r_max,
            warm_start=not args.no_warm_start,
            include_push_info=False,
        )
        r_min = float(payoff["z2_min"])
        r_max = float(payoff["z2_nadir"])
        payoff_path = _safe_out_path(args.payoff_output, PAYOFF_JSON)
        payoff_path.parent.mkdir(parents=True, exist_ok=True)
        with open(payoff_path, "w", encoding="utf-8") as pf:
            json.dump(payoff, pf, ensure_ascii=False, indent=2)
        print(f"Wrote payoff table to {payoff_path}")
        print(f"Using lexicographic range Z2 in [{r_min:.6g}, {r_max:.6g}]")
        if args.payoff_only:
            pack.model.dispose()
            return {"payoff": payoff, "points": [], "extra": []}

    r_range = r_max - r_min
    if r_range <= 1e-12:
        raise SystemExit("Z2 range is degenerate; check payoff / --r-min / --r-max.")
    step = r_range / float(q)
    pack.r_range = r_range
    pack.add_epsilon(_epsilon(k_max, r_min, step))

    payload = {
        "method": "AUGMECON",
        "mode": args.mode,
        "q": q,
        "rho": RHO,
        "r_min": r_min,
        "r_max": r_max,
        "step": step,
        "mip_gap": args.mip_gap,
        "threads": args.threads,
        "mip_focus": args.mip_focus,
        "time_limit": args.time_limit,
        "direction": "descending" if not args.ascending else "ascending",
        "bypass_enabled": not args.no_bypass and not args.ascending,
        "payoff": payoff,
        "points": [],
        "extra": [],
    }

    json_path = _safe_out_path(args.output, OUT_JSON)
    csv_path = _safe_out_path(args.csv, OUT_CSV)

    prev_x: dict[tuple[int, int], int] | None = None
    prev_risk: float | None = None
    last_solved_k: int | None = None
    skip_until_exclusive: int | None = None

    ks = list(range(k_max, k_min - 1, -1))
    if args.ascending:
        ks = list(range(k_min, k_max + 1))

    for k in ks:
        eps_k = _epsilon(k, r_min, step)
        row = {
            "k": k,
            "skipped": False,
            "copied_from_k": None,
        }

        if (
            not args.no_bypass
            and not args.ascending
            and skip_until_exclusive is not None
            and k > skip_until_exclusive
            and prev_x is not None
        ):
            copied = metrics_from_assignment(
                pack,
                prev_x,
                eps_k,
                step,
                primary_obj=payload["points"][-1].get("primary_obj") if payload["points"] else None,
                primary_bound=payload["points"][-1].get("primary_bound") if payload["points"] else None,
                status="Bypass copy",
                include_push_info=False,
            )
            row.update(_summary_row(copied))
            row["skipped"] = True
            row["copied_from_k"] = last_solved_k
            row["time_s"] = 0.0
            row["stage_a_time_s"] = 0.0
            row["stage_b_time_s"] = 0.0
            payload["points"].append(row)
            print(
                f"[k={k}] bypass copy from k={last_solved_k}  "
                f"ε={eps_k:.4f}  Z1={row['value']:.4f}  Z2={row['risk']:.4f}  "
                f"s={row['slack']:.4f}"
            )
            _write_outputs(payload, json_path, csv_path)
            continue

        pack.model.setParam("LogFile", str(log_dir / f"augmecon_k{k}.log"))
        _warm_start_for_epsilon(
            pack,
            eps_k,
            prev_x,
            prev_risk,
            use_greedy=not args.no_warm_start,
        )
        metrics = solve_lex_subproblem(
            pack,
            eps_k,
            step,
            args.mode,
            include_push_info=False,
        )
        print_subproblem_metrics(metrics)
        row.update(_summary_row(metrics))
        payload["points"].append(row)

        if metrics.get("infeasible"):
            print(f"[k={k}] infeasible at ε={eps_k:g}; stopping tighter points.")
            _write_outputs(payload, json_path, csv_path)
            break

        prev_x = getattr(pack, "last_assignment", None)
        if not prev_x:
            prev_x = capture_assignment(pack)
        prev_risk = float(metrics["risk"])
        last_solved_k = k
        print(
            f"[k={k}] ε={eps_k:.4f}  Z1={metrics['value']:.4f}  "
            f"Z2={metrics['risk']:.4f}  s={metrics['slack']:.4f}  "
            f"bypass={metrics['bypass']}  time={metrics['time_s']:.2f}s"
        )

        if payload["bypass_enabled"] and int(metrics.get("bypass") or 0) > 0:
            skip_until_exclusive = k - int(metrics["bypass"]) - 1
            print(
                f"  bypass next {metrics['bypass']} tighter grid points "
                f"(next solve at k≤{skip_until_exclusive})"
            )
        else:
            skip_until_exclusive = None

        _write_outputs(payload, json_path, csv_path)

    for extra_eps in args.extra_epsilon or []:
        print(f"=== Extra AUGMECON subproblem ε={extra_eps:g} ===")
        pack.model.setParam("LogFile", str(log_dir / f"augmecon_eps_{extra_eps:g}.log"))
        _warm_start_for_epsilon(
            pack,
            float(extra_eps),
            prev_x,
            prev_risk,
            use_greedy=not args.no_warm_start,
        )
        metrics = solve_lex_subproblem(
            pack,
            float(extra_eps),
            step,
            args.mode,
            include_push_info=args.save_extra_push_info,
        )
        print_subproblem_metrics(metrics)
        extra_row = _summary_row(metrics)
        extra_row["label"] = f"extra_epsilon_{extra_eps:g}"
        extra_row["epsilon_k"] = float(extra_eps)
        if args.save_extra_push_info and "push_info" in metrics:
            extra_path = ROOT / "result" / f"augmecon_extra_eps_{extra_eps:g}.json"
            extra_path.parent.mkdir(parents=True, exist_ok=True)
            with open(extra_path, "w", encoding="utf-8") as ef:
                json.dump(metrics, ef, ensure_ascii=False, indent=2)
            print(f"Wrote extra assignment JSON to {extra_path}")
        payload["extra"].append(extra_row)
        _write_outputs(payload, json_path, csv_path)

    pack.model.dispose()
    return payload


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "AUGMECON grid: lex two-stage subproblems from k=q down to 0, "
            "with Mavrotas surplus bypass."
        )
    )
    p.add_argument("--q", type=int, default=DEFAULT_Q, help="Number of equal Z2 intervals (default 20).")
    p.add_argument("--k-min", type=int, default=None, help="Inclusive lower k (default 0).")
    p.add_argument("--k-max", type=int, default=None, help="Inclusive upper k (default q).")
    p.add_argument("--r-min", type=float, default=None, help="Z2_min (default Table-3 endpoint).")
    p.add_argument("--r-max", type=float, default=None, help="Z2_max / nadir (default Table-3 endpoint).")
    p.add_argument(
        "--payoff",
        action="store_true",
        help="Compute lexicographic payoff table and use it as [Z2_min, Z2_nadir].",
    )
    p.add_argument("--payoff-only", action="store_true", help="Stop after the payoff table.")
    p.add_argument("--payoff-output", type=str, default=None, help=f"Payoff JSON (default {PAYOFF_JSON}).")
    p.add_argument(
        "--mode",
        choices=("lex", "rho"),
        default="lex",
        help="lex = two-stage (default); rho = scalar ρ s/r ablation.",
    )
    p.add_argument("--mip-gap", type=float, default=0.001, help="Gurobi MIPGap (default 0.001 = 0.1%%).")
    p.add_argument("--time-limit", type=float, default=None, help="Gurobi TimeLimit per optimize() call.")
    p.add_argument("--threads", type=int, default=12, help="Gurobi Threads (default 12).")
    p.add_argument("--mip-focus", type=int, default=2, help="MIPFocus (default 2).")
    p.add_argument("--value-eps", type=float, default=0.0, help="Drop arcs with value <= eps.")
    p.add_argument("--no-warm-start", action="store_true", help="Disable greedy / previous-k MIP starts.")
    p.add_argument(
        "--no-bypass",
        action="store_true",
        help="Solve every k even when surplus would skip tighter points.",
    )
    p.add_argument(
        "--ascending",
        action="store_true",
        help="Loop k=0..q (bypass disabled; not the Mavrotas acceleration).",
    )
    p.add_argument(
        "--extra-epsilon",
        type=float,
        action="append",
        default=None,
        help="Additional RHS after the grid (repeatable). Example: --extra-epsilon 52356",
    )
    p.add_argument(
        "--save-extra-push-info",
        action="store_true",
        help="Write full per-user lists for --extra-epsilon runs (large JSON).",
    )
    p.add_argument("--output", type=str, default=None, help=f"Grid JSON (default {OUT_JSON}).")
    p.add_argument("--csv", type=str, default=None, help=f"Grid CSV (default {OUT_CSV}).")
    p.add_argument("--log-dir", type=str, default=None, help=f"Gurobi logs (default {LOG_DIR}).")
    p.add_argument("--quiet", action="store_true", help="Gurobi OutputFlag=0 (grid summaries still print).")
    args = p.parse_args()
    if args.payoff_only:
        args.payoff = True
    run_grid(args)


if __name__ == "__main__":
    main()
