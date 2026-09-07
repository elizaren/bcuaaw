"""
与 balanced_model_pulp.py 同构的 MIP，使用 Gurobi 原生 API 与稀疏弧建模。

相对 balanced_model_gurobi.py 的规则变更（来自 balanced_model_pulp.py）：
- 每用户增加硬约束：Σ_v x_uv ≥ MIN_PUSH_RATIO · K[u]（MIN_PUSH_RATIO=0.4）

目标：max Σ (alpha * v_uv - beta * r_v) x_uv - PENALTY * Σ_u (δ_T+δ_R+δ_A+δ_E)

约束（与 balanced_model_pulp.py 一致）：
- 每视频：Σ_u x_uv ≤ N[v]
- 每类别 k∈{1..4}：该类视频的全局推送次数 ≤ MAX_CATEGORY_PUSH_COUNT
- 每用户：MIN_PUSH_RATIO·K[u] ≤ Σ_v x ≤ K[u]；Σ t_v x ≤ T[u]+δ_T；Σ r_v x ≤ R[u]+δ_R；
         Σ ent_v x ≤ A[u]+δ_A；Σ edu_v x ≥ E[u]-δ_E
- 全局：Σ c_v x_uv ≤ TOTAL_COST

优化：省略 t_v>T[u]、c_v>TOTAL_COST、以及可选 v≤eps 的弧；可选贪心初值 + 对应 δ 的 MIP 起点。
贪心初值在时长 T 内填满后，若仍低于 0.4·K，则允许超出 T（由 δ_T 补齐）以尽量满足下限。

依赖：gurobipy + 有效 Gurobi 许可证。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

USER_CSV_PATH = Path(__file__).resolve().parent / "users.csv"
VIDEO_CSV_PATH = Path(__file__).resolve().parent / "videos.csv"
USER_VIDEO_CSV_PATH = Path(__file__).resolve().parent / "user_video_value.csv"
RESULT_JSON_PATH = Path(__file__).resolve().parent / "result/common_model_result2.json"

MAX_CATEGORY_PUSH_COUNT = 9275
TOTAL_COST = 43
PENALTY_COEFF = 0.01
ALPHA = 0.834
BETA = 0.166
MIN_PUSH_RATIO = 0.4


def grb_status_to_label(status: int) -> str:
    return {
        GRB.LOADED: "Loaded",
        GRB.OPTIMAL: "Optimal",
        GRB.INFEASIBLE: "Infeasible",
        GRB.INF_OR_UNBD: "Infeasible or unbounded",
        GRB.UNBOUNDED: "Unbounded",
        GRB.CUTOFF: "Cutoff",
        GRB.ITERATION_LIMIT: "Iteration limit",
        GRB.NODE_LIMIT: "Node limit",
        GRB.TIME_LIMIT: "Time limit",
        GRB.SOLUTION_LIMIT: "Solution limit",
        GRB.INTERRUPTED: "Interrupted",
        GRB.NUMERIC: "Numeric error",
        GRB.SUBOPTIMAL: "Suboptimal",
        GRB.INPROGRESS: "In progress",
        GRB.USER_OBJ_LIMIT: "User objective limit",
    }.get(status, f"Unknown({status})")


def load_users(path: Path | str = USER_CSV_PATH) -> dict[int, dict]:
    users: dict[int, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = int(row["user_id"])
            users[uid] = {
                "age": float(row["age"]),
                "I": int(row["I"]),
                "T": int(row["T"]),
                "R": float(row["R"]),
                "K": int(row["K"]),
                "A": int(row["A"]),
                "E": int(row["E"]),
            }
    return users


def load_videos(path: Path | str = VIDEO_CSV_PATH) -> dict[int, dict]:
    videos: dict[int, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = int(row["video_id"])
            videos[vid] = {
                "K": int(row["K"]),
                "t": float(row["t"]),
                "D": int(row["D"]),
                "entertainment_flag": int(row["entertainment_flag"]),
                "education_flag": int(row["education_flag"]),
                "r": float(row["r"]),
                "c": float(row["c"]),
                "N": int(row["N"]),
            }
    return videos


def load_user_video_data(path: Path | str = USER_VIDEO_CSV_PATH) -> dict[int, dict[int, dict]]:
    user_video_data: dict[int, dict[int, dict]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = int(row["user_id"])
            vid = int(row["video_id"])
            if uid not in user_video_data:
                user_video_data[uid] = {}
            user_video_data[uid][vid] = {"value": float(row["v"])}
    return user_video_data


def build_arc_list(
    videos_data: dict[int, dict],
    user_video_data: dict[int, dict[int, dict]],
    users_data: dict[int, dict],
    value_eps: float,
    total_cost_cap: float,
) -> list[tuple[int, int, float]]:
    """可行 (user_id, video_id, value)；省略 t>T[u]、c>TOTAL_COST、v≤eps。"""
    arcs: list[tuple[int, int, float]] = []
    for u, row in user_video_data.items():
        if u not in users_data:
            continue
        t_limit = float(users_data[u]["T"])
        for v, info in row.items():
            if v not in videos_data:
                continue
            val = info["value"]
            if val <= value_eps:
                continue
            vd = videos_data[v]
            if vd["t"] > t_limit:
                continue
            if float(vd["c"]) > total_cost_cap:
                continue
            arcs.append((u, v, val))
    return arcs


def heuristic_x_greedy(
    arcs: list[tuple[int, int, float]],
    users_data: dict[int, dict],
    videos_data: dict[int, dict],
    total_cost_cap: float,
) -> dict[tuple[int, int], int]:
    """
    按用户分组，弧按 value/duration 降序贪心。
    先在时长 T 内尽量填满（上限 K）；若推送数仍低于 0.4·K，则允许超出 T（δ_T 补齐）继续补足下限。
    满足每用户推送数上限 K、每视频次数 N、全局成本；尽量满足下限 0.4·K。
    不保证满足 R/A/E/类别上限；用于 MIP 初值，δ 由 compute_delta_starts 补齐。
    """
    by_user: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for u, v, val in arcs:
        by_user[u].append((u, v, val))

    remaining_video = {v: videos_data[v]["N"] for v in videos_data}
    user_time: dict[int, float] = defaultdict(float)
    user_count: dict[int, int] = defaultdict(int)
    spent_cost = 0.0
    start: dict[tuple[int, int], int] = {}

    def try_add(u: int, v: int, t_cap: float | None) -> bool:
        nonlocal spent_cost
        c_v = float(videos_data[v]["c"])
        t_v = float(videos_data[v]["t"])
        if spent_cost + c_v > total_cost_cap + 1e-12:
            return False
        if t_cap is not None and user_time[u] + t_v > t_cap + 1e-12:
            return False
        if remaining_video[v] <= 0:
            return False
        start[(u, v)] = 1
        user_time[u] += t_v
        user_count[u] += 1
        spent_cost += c_v
        remaining_video[v] -= 1
        return True

    for u in sorted(by_user.keys()):
        t_cap = float(users_data[u]["T"])
        k_cap = int(users_data[u]["K"])
        k_min = math.ceil(MIN_PUSH_RATIO * k_cap - 1e-12)
        cand = sorted(
            by_user[u],
            key=lambda z: z[2] / max(videos_data[z[1]]["t"], 1e-12),
            reverse=True,
        )
        for _, v, _val in cand:
            if user_count[u] >= k_cap:
                break
            try_add(u, v, t_cap)
        if user_count[u] < k_min:
            for _, v, _val in cand:
                if user_count[u] >= k_min:
                    break
                if (u, v) in start:
                    continue
                try_add(u, v, t_cap=None)
    return start


def compute_delta_starts(
    users_data: dict[int, dict],
    videos_data: dict[int, dict],
    x_on: dict[tuple[int, int], int],
    all_users: list[int],
) -> tuple[dict[int, float], dict[int, float], dict[int, int], dict[int, int]]:
    """给定 0/1 的 x，取满足用户侧软约束的最小非负 δ（与 balanced_model_pulp 形式一致）。"""
    dT: dict[int, float] = {}
    dR: dict[int, float] = {}
    dA: dict[int, int] = {}
    dE: dict[int, int] = {}

    agg: dict[int, list[int]] = defaultdict(list)
    for (u, v), z in x_on.items():
        if z:
            agg[u].append(v)

    for u in all_users:
        Tlim = float(users_data[u]["T"])
        Rlim = float(users_data[u]["R"])
        Alim = int(users_data[u]["A"])
        Elim = int(users_data[u]["E"])
        tot_t = tot_r = 0.0
        ent = edu = 0
        for v in agg[u]:
            vd = videos_data[v]
            tot_t += float(vd["t"])
            tot_r += float(vd["r"])
            ent += int(vd["entertainment_flag"])
            edu += int(vd["education_flag"])
        dT[u] = max(0.0, tot_t - Tlim)
        dR[u] = max(0.0, tot_r - Rlim)
        dA[u] = max(0, ent - Alim)
        dE[u] = max(0, Elim - edu)
    return dT, dR, dA, dE


def actual_gap_pct(incumbent: float, proven_bound: float) -> float | None:
    """Actual Gap = ((Best Bound - Incumbent) / |Incumbent|) × 100%."""
    if abs(incumbent) < 1e-12:
        return None
    return ((proven_bound - incumbent) / abs(incumbent)) * 100.0


def print_active_soft_slacks(
    all_users: list[int],
    delta_T: dict,
    delta_R: dict,
    delta_A: dict,
    delta_E: dict,
    tol: float = 1e-6,
    top_n: int = 5,
) -> None:
    """Print which soft-constraint slacks are active (δ > tol) in the incumbent."""
    groups = (
        ("time (δ_T)", delta_T),
        ("risk (δ_R)", delta_R),
        ("ent  (δ_A)", delta_A),
        ("edu  (δ_E)", delta_E),
    )
    print("--- Active soft constraints (δ > 0) ---")
    any_active = False
    total_slack_sum = 0.0
    for label, deltas in groups:
        active = [(u, float(deltas[u].X)) for u in all_users if float(deltas[u].X) > tol]
        active.sort(key=lambda t: t[1], reverse=True)
        n_active = len(active)
        slack_sum = sum(v for _, v in active)
        total_slack_sum += slack_sum
        if n_active:
            any_active = True
        print(f"  {label}: {n_active}/{len(all_users)} users, sum={slack_sum:.6g}")
        for u, v in active[:top_n]:
            print(f"    user {u}: δ={v:.6g}")
        if n_active > top_n:
            print(f"    ... and {n_active - top_n} more")
    if not any_active:
        print("  (none — all soft constraints satisfied with δ=0)")
    else:
        print(f"  total Σδ = {total_slack_sum:.6g}")
        print(f"  penalty term = {PENALTY_COEFF * total_slack_sum:.6g}")


def build_result_payload(
    status: int,
    objective: float | None,
    total_value: float,
    total_cost: float,
    total_risk: float,
    pushes_by_user: dict[int, list[int]],
    users_data: dict[int, dict],
    user_video_data: dict[int, dict[int, dict]],
    videos_data: dict[int, dict],
    incumbent_objective: float | None = None,
    proven_bound: float | None = None,
    actual_gap: float | None = None,
) -> dict:
    push_info: list[dict] = []
    for u in sorted(pushes_by_user.keys()):
        vids = pushes_by_user[u]
        video_order = sorted(vids, key=lambda vid: (-user_video_data[u][vid]["value"], vid))
        video_time = sum(videos_data[vid]["t"] for vid in video_order)
        videos_out = [
            {
                "video_id": vid,
                "value": user_video_data[u][vid]["value"],
                "risk": videos_data[vid]["r"],
                "category": videos_data[vid]["K"],
            }
            for vid in video_order
        ]
        push_info.append(
            {
                "user_id": u,
                "video_time_limit": int(users_data[u]["T"]),
                "video_time": video_time,
                "video_num": len(video_order),
                "video_order": video_order,
                "videos": videos_out,
            }
        )
    return {
        "status": grb_status_to_label(status),
        "objective": objective,
        "incumbent_objective": incumbent_objective,
        "proven_bound": proven_bound,
        "actual_gap": actual_gap,
        "value": total_value,
        "risk": total_risk,
        "cost": total_cost,
        "push_info": push_info,
    }


def solve(
    time_limit: float | None,
    mip_gap: float | None,
    threads: int | None,
    mip_focus: int | None,
    value_eps: float,
    warm_start: bool,
    dump_model: str | None,
    output_flag: int,
) -> None:
    users_data = load_users(USER_CSV_PATH)
    videos_data = load_videos(VIDEO_CSV_PATH)
    user_video_data = load_user_video_data(USER_VIDEO_CSV_PATH)

    total_cost_f = float(TOTAL_COST)
    arcs = build_arc_list(videos_data, user_video_data, users_data, value_eps, total_cost_f)
    u_v_pairs = [(u, v) for u, v, _ in arcs]

    all_users = sorted(users_data.keys())
    all_videos = sorted(videos_data.keys())

    model = gp.Model("CommonModelGurobi2")
    model.ModelSense = GRB.MAXIMIZE
    model.setParam("OutputFlag", output_flag)
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)
    if mip_gap is not None:
        model.setParam("MIPGap", mip_gap)
    if threads is not None:
        model.setParam("Threads", threads)
    if mip_focus is not None:
        model.setParam("MIPFocus", mip_focus)

    obj_coeff: dict[tuple[int, int], float] = {}
    for u, v, val in arcs:
        r_v = float(videos_data[v]["r"])
        obj_coeff[u, v] = ALPHA * val - BETA * r_v

    x = model.addVars(u_v_pairs, vtype=GRB.BINARY, name="x")

    delta_T = model.addVars(all_users, lb=0.0, vtype=GRB.CONTINUOUS, name="delta_T")
    delta_R = model.addVars(all_users, lb=0.0, vtype=GRB.CONTINUOUS, name="delta_R")
    delta_A = model.addVars(all_users, lb=0.0, vtype=GRB.INTEGER, name="delta_A")
    delta_E = model.addVars(all_users, lb=0.0, vtype=GRB.INTEGER, name="delta_E")

    obj_x = gp.quicksum(obj_coeff[uv] * x[uv] for uv in u_v_pairs)
    obj_pen = PENALTY_COEFF * gp.quicksum(
        delta_T[u] + delta_R[u] + delta_A[u] + delta_E[u] for u in all_users
    )
    model.setObjective(obj_x - obj_pen, GRB.MAXIMIZE)

    arcs_by_user: dict[int, list[tuple[int, int]]] = defaultdict(list)
    arcs_by_video: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for u, v, _ in arcs:
        arcs_by_user[u].append((u, v))
        arcs_by_video[v].append((u, v))

    for v in all_videos:
        if v not in arcs_by_video:
            continue
        model.addConstr(
            gp.quicksum(x[uu, vv] for uu, vv in arcs_by_video[v]) <= videos_data[v]["N"],
            name=f"cap_v{v}",
        )

    for k in range(1, 5):
        pairs_k = [(u, v) for u, v in u_v_pairs if int(videos_data[v]["K"]) == k]
        if pairs_k:
            model.addConstr(
                gp.quicksum(x[u, v] for u, v in pairs_k) <= MAX_CATEGORY_PUSH_COUNT,
                name=f"cat_{k}",
            )

    under_min = [
        u
        for u in all_users
        if len(arcs_by_user.get(u, [])) + 1e-12 < MIN_PUSH_RATIO * int(users_data[u]["K"])
    ]
    if under_min:
        print(
            f"Warning: {len(under_min)} users have fewer candidate arcs than "
            f"{MIN_PUSH_RATIO:g}*K; min-count may be infeasible."
        )

    for u in all_users:
        pairs_u = arcs_by_user.get(u, [])
        k_u = int(users_data[u]["K"])
        k_min = MIN_PUSH_RATIO * k_u
        model.addConstr(
            gp.quicksum(x[uu, vv] for uu, vv in pairs_u) <= k_u,
            name=f"user_count_{u}",
        )
        model.addConstr(
            gp.quicksum(x[uu, vv] for uu, vv in pairs_u) >= k_min,
            name=f"user_count_min_{u}",
        )
        if pairs_u:
            model.addConstr(
                gp.quicksum(videos_data[vv]["t"] * x[uu, vv] for uu, vv in pairs_u)
                <= float(users_data[u]["T"]) + delta_T[u],
                name=f"time_{u}",
            )
            model.addConstr(
                gp.quicksum(videos_data[vv]["r"] * x[uu, vv] for uu, vv in pairs_u)
                <= float(users_data[u]["R"]) + delta_R[u],
                name=f"risk_{u}",
            )
            model.addConstr(
                gp.quicksum(videos_data[vv]["entertainment_flag"] * x[uu, vv] for uu, vv in pairs_u)
                <= int(users_data[u]["A"]) + delta_A[u],
                name=f"ent_{u}",
            )
            model.addConstr(
                gp.quicksum(videos_data[vv]["education_flag"] * x[uu, vv] for uu, vv in pairs_u)
                >= int(users_data[u]["E"]) - delta_E[u],
                name=f"edu_{u}",
            )
        else:
            # 无候选弧时 x 全为 0，仅需教育下限松弛：0 >= E[u] - delta_E[u]
            model.addConstr(delta_E[u] >= int(users_data[u]["E"]), name=f"edu_{u}")

    if u_v_pairs:
        model.addConstr(
            gp.quicksum(videos_data[v]["c"] * x[u, v] for u, v in u_v_pairs) <= total_cost_f,
            name="total_cost",
        )

    if warm_start and u_v_pairs:
        h = heuristic_x_greedy(arcs, users_data, videos_data, total_cost_f)
        dT, dR, dA, dE = compute_delta_starts(users_data, videos_data, h, all_users)
        for uv, var in x.items():
            var.Start = float(h.get(uv, 0))
        for u in all_users:
            delta_T[u].Start = dT[u]
            delta_R[u].Start = dR[u]
            delta_A[u].Start = float(dA[u])
            delta_E[u].Start = float(dE[u])

    if dump_model:
        n = len(u_v_pairs) + 4 * len(all_users)
        if n > 500_000:
            print(f"Skipping --dump-model: ~{n} vars (threshold 500000).")
        else:
            model.write(dump_model)
            print(f"Wrote model to {dump_model}")

    model.optimize()

    status = model.Status
    print(f"Status: {grb_status_to_label(status)} (code {status})")

    if model.SolCount > 0:
        incumbent = float(model.ObjVal)
        proven_bound = float(model.ObjBound)
        gap_pct = actual_gap_pct(incumbent, proven_bound)

        print(f"Incumbent Objective: {incumbent:.6g}")
        print(f"Proven Bound: {proven_bound:.6g}")
        if gap_pct is None:
            print("Actual Gap: N/A (incumbent ≈ 0)")
        else:
            print(f"Actual Gap: {gap_pct:.6g}%")
        print(f"Binary x count: {len(u_v_pairs)}")

        chosen = [(u, v) for (u, v) in u_v_pairs if x[u, v].X > 0.5]
        total_value = sum(float(user_video_data[u][v]["value"]) for u, v in chosen)
        total_cost = sum(float(videos_data[v]["c"]) for _, v in chosen)
        total_risk = sum(float(videos_data[v]["r"]) for _, v in chosen)
        print(f"Total value (sum v_uv): {total_value:.10g}")
        print(f"Total push cost: {total_cost:.10g}")
        print(f"Total push risk: {total_risk:.10g}")

        print_active_soft_slacks(all_users, delta_T, delta_R, delta_A, delta_E)

        by_user: dict[int, list[int]] = defaultdict(list)
        for u, v in chosen:
            by_user[u].append(v)

        payload = build_result_payload(
            status=status,
            objective=incumbent,
            total_value=total_value,
            total_cost=total_cost,
            total_risk=total_risk,
            pushes_by_user=dict(by_user),
            users_data=users_data,
            user_video_data=user_video_data,
            videos_data=videos_data,
            incumbent_objective=incumbent,
            proven_bound=proven_bound,
            actual_gap=gap_pct,
        )
        RESULT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULT_JSON_PATH, "w", encoding="utf-8") as jf:
            json.dump(payload, jf, ensure_ascii=False, indent=2)
        print(f"Wrote results to {RESULT_JSON_PATH}")
    else:
        print("No incumbent solution.")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Gurobi-native balanced_model with min push 0.4*K (sparse arcs + soft slacks)."
    )
    p.add_argument("--time-limit", type=float, default=None, help="Gurobi TimeLimit (seconds).")
    p.add_argument("--mip-gap", type=float, default=None, help="MIPGap (e.g. 0.01 for 1%%).")
    p.add_argument("--threads", type=int, default=None, help="Gurobi Threads.")
    p.add_argument("--mip-focus", type=int, default=None, help="MIPFocus (0 default, 1 feas, 2 optimality bound, 3 bound).")
    p.add_argument(
        "--value-eps",
        type=float,
        default=0.0,
        help="Drop arcs with value <= eps (dense CSV 上常为 0 与 PuLP 一致).",
    )
    p.add_argument("--no-warm-start", action="store_true", help="Disable heuristic MIP start.")
    p.add_argument(
        "--dump-model",
        type=str,
        default=None,
        metavar="PATH",
        help="Write .lp / .mps (skipped if estimated vars > 500k).",
    )
    p.add_argument("--quiet", action="store_true", help="Gurobi OutputFlag=0.")
    args = p.parse_args()

    solve(
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        mip_focus=args.mip_focus,
        value_eps=args.value_eps,
        warm_start=not args.no_warm_start,
        dump_model=args.dump_model,
        output_flag=0 if args.quiet else 1,
    )


if __name__ == "__main__":
    main()
