"""
与 baseline_model_pulp.py 同构的 MIP，采用 Gurobi 原生 API 与稀疏弧建模。

相对 baseline_model_gurobi.py 的规则变更（来自 baseline_model_pulp.py）：
- 每用户增加硬约束：Σ_v x_uv ≥ MIN_PUSH_RATIO · K[u]（MIN_PUSH_RATIO=0.4）

目标：max Σ v·x

约束（与 baseline_model_pulp.py 一致）：
- 每用户：MIN_PUSH_RATIO·K[u] ≤ Σ_v x ≤ K[u]；Σ_v t·x ≤ T[u]
- 每视频：Σ_u x ≤ N[v]
- 全局：Σ_{u,v} c_v·x ≤ TOTAL_COST

优化：省略 t>T[u]、c_v>TOTAL_COST 及 v≤eps 的弧；可选贪心初值。
贪心初值先按 v/t 在时长 T 内填满；若仍低于 0.4·K，则回退该用户已选弧，
改按时长升序重填，尽量在硬约束 T 内满足下限（不允许超出 T）。

依赖：pip install gurobipy（需已配置 Gurobi 许可证）
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
RESULT_JSON_PATH = Path(__file__).resolve().parent / "result/greed_model_result2.json"

# 全系统推送总成本上限（与 baseline_model_pulp.py 中 TOTAL_COST 保持一致）
TOTAL_COST = 43
MIN_PUSH_RATIO = 0.4


def grb_status_to_label(status: int) -> str:
    """与 Gurobi 求解状态对应的可读字符串（写入 JSON / 终端）。"""
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
) -> dict:
    """
    组装 greed_model_result2.json。

    根字段：objective 为求解器目标值；value 为当前解上所有选中弧的 user-video 价值之和（应与 objective 一致）。
    push_info 仅包含有至少一次推送的用户；按 user_id 升序。
    对每条记录：
    - video_time_limit：users.csv 中该用户的 T（播放总时长上限）。
    - video_time：video_order 中各视频时长 t 之和。
    - video_num：实际推送视频个数。
    - video_order：该 user_id 被推送的 video_id 列表（价值降序，同价值则 video_id 升序）。
    - videos：与 video_order 同序的推送视频详情。
    """
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
        "value": total_value,
        "risk": total_risk,
        "cost": total_cost,
        "push_info": push_info,
    }


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
) -> list[tuple[int, int, float, float]]:
    """返回 (user_id, video_id, value, duration) 可行弧；省略 t>T[u]、c>TOTAL_COST 及 v≤eps 的弧。"""
    arcs: list[tuple[int, int, float, float]] = []
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
            t_v = videos_data[v]["t"]
            if t_v > t_limit:
                continue
            if float(videos_data[v]["c"]) > total_cost_cap:
                continue
            arcs.append((u, v, val, t_v))
    return arcs


def heuristic_start(
    arcs: list[tuple[int, int, float, float]],
    user_time_limit: dict[int, float],
    user_max_pushes: dict[int, int],
    video_cap: dict[int, int],
    total_cost_cap: float,
    videos_data: dict[int, dict],
) -> dict[tuple[int, int], int]:
    """
    按用户分组，对候选弧按 value/duration 降序贪心，满足 K[u]、T[u]、N[v] 及全局 Σ c·x ≤ total_cost_cap。
    若推送数仍低于 0.4·K，则回退该用户已选弧，改按时长升序重填，尽量在硬约束 T 内满足下限。
    不保证最优，仅作 MIP 初始解。
    """
    by_user: dict[int, list[tuple[int, int, float, float]]] = defaultdict(list)
    for u, v, val, t_v in arcs:
        by_user[u].append((u, v, val, t_v))

    remaining_video = {v: video_cap[v] for v in {vv for _, vv, _, _ in arcs}}
    user_time: dict[int, float] = defaultdict(float)
    spent_cost = 0.0
    start: dict[tuple[int, int], int] = {}

    def try_add(u: int, v: int, t_v: float, t_cap: float) -> bool:
        nonlocal spent_cost
        c_v = float(videos_data[v]["c"])
        if spent_cost + c_v > total_cost_cap + 1e-12:
            return False
        if user_time[u] + t_v > t_cap + 1e-12:
            return False
        if remaining_video[v] <= 0:
            return False
        start[(u, v)] = 1
        user_time[u] += t_v
        spent_cost += c_v
        remaining_video[v] -= 1
        return True

    def rollback_user(u: int, picked: list[tuple[int, float]]) -> None:
        nonlocal spent_cost
        for v, t_v in picked:
            start.pop((u, v), None)
            remaining_video[v] += 1
            spent_cost -= float(videos_data[v]["c"])
        user_time[u] = 0.0

    def fill_user(u: int, cand: list[tuple[int, int, float, float]], k_cap: int, t_cap: float) -> list[tuple[int, float]]:
        picked: list[tuple[int, float]] = []
        for _, v, _val, t_v in cand:
            if len(picked) >= k_cap:
                break
            if (u, v) in start:
                continue
            if try_add(u, v, t_v, t_cap):
                picked.append((v, t_v))
        return picked

    for u in sorted(by_user.keys()):
        t_cap = user_time_limit[u]
        k_cap = user_max_pushes[u]
        k_min = math.ceil(MIN_PUSH_RATIO * k_cap - 1e-12)
        cand_value = sorted(by_user[u], key=lambda z: z[2] / max(z[3], 1e-12), reverse=True)
        picked = fill_user(u, cand_value, k_cap, t_cap)
        if len(picked) < k_min:
            rollback_user(u, picked)
            cand_short = sorted(by_user[u], key=lambda z: (z[3], -z[2]))
            fill_user(u, cand_short, k_cap, t_cap)
    return start


def solve(
    time_limit: float | None,
    mip_gap: float | None,
    threads: int | None,
    value_eps: float,
    warm_start: bool,
    dump_model: str | None,
    output_flag: int,
) -> None:
    users_data = load_users(USER_CSV_PATH)
    videos_data = load_videos(VIDEO_CSV_PATH)
    user_video_data = load_user_video_data(USER_VIDEO_CSV_PATH)

    arcs = build_arc_list(videos_data, user_video_data, users_data, value_eps, float(TOTAL_COST))
    if not arcs:
        print("No arcs after filtering; nothing to solve.")
        return

    users = sorted(users_data.keys())
    videos = sorted(videos_data.keys())

    model = gp.Model("GreedyModelGurobiMinPush")
    model.ModelSense = GRB.MAXIMIZE
    model.setParam("OutputFlag", output_flag)
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)
    if mip_gap is not None:
        model.setParam("MIPGap", mip_gap)
    if threads is not None:
        model.setParam("Threads", threads)

    u_v_pairs = [(u, v) for u, v, _, _ in arcs]
    obj_coeff = {(u, v): val for u, v, val, _ in arcs}
    x = model.addVars(u_v_pairs, vtype=GRB.BINARY, obj=obj_coeff, name="x")

    arcs_by_user: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    arcs_by_video: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for u, v, val, t_v in arcs:
        arcs_by_user[u].append((u, v, t_v))
        arcs_by_video[v].append((u, v))

    under_min = [
        u
        for u in users
        if len(arcs_by_user.get(u, [])) + 1e-12 < MIN_PUSH_RATIO * int(users_data[u]["K"])
    ]
    if under_min:
        print(
            f"Warning: {len(under_min)} users have fewer candidate arcs than "
            f"{MIN_PUSH_RATIO:g}*K; min-count may be infeasible."
        )

    for u in users:
        pairs_u = arcs_by_user.get(u, [])
        k_u = int(users_data[u]["K"])
        k_min = MIN_PUSH_RATIO * k_u
        model.addConstr(
            gp.quicksum(x[u, vid] for _, vid, _ in pairs_u) <= k_u,
            name=f"count_u{u}",
        )
        model.addConstr(
            gp.quicksum(x[u, vid] for _, vid, _ in pairs_u) >= k_min,
            name=f"count_min_u{u}",
        )
        if pairs_u:
            model.addConstr(
                gp.quicksum(t_v * x[u, vid] for _, vid, t_v in pairs_u) <= float(users_data[u]["T"]),
                name=f"time_u{u}",
            )

    for v in videos:
        if v not in arcs_by_video:
            continue
        model.addConstr(
            gp.quicksum(x[uu, vv] for uu, vv in arcs_by_video[v]) <= videos_data[v]["N"],
            name=f"cap_v{v}",
        )

    model.addConstr(
        gp.quicksum(videos_data[v]["c"] * x[u, v] for u, v in u_v_pairs) <= float(TOTAL_COST),
        name="total_cost",
    )

    if warm_start:
        cap_by_video = {vid: videos_data[vid]["N"] for vid in videos_data}
        t_by_user = {uid: float(users_data[uid]["T"]) for uid in users_data}
        k_by_user = {uid: int(users_data[uid]["K"]) for uid in users_data}
        h = heuristic_start(arcs, t_by_user, k_by_user, cap_by_video, float(TOTAL_COST), videos_data)
        for (u, v), var in x.items():
            var.Start = float(h.get((u, v), 0))

    if dump_model:
        n = len(u_v_pairs)
        if n > 200_000:
            print(f"Skipping --dump-model: {n} variables (threshold 200000).")
        else:
            model.write(dump_model)
            print(f"Wrote model to {dump_model}")

    model.optimize()

    status = model.Status
    print(f"Status: {grb_status_to_label(status)} (code {status})")

    if model.SolCount > 0:
        print(f"Objective: {model.ObjVal:.6g}")
        if getattr(model, "IsMIP", 0):
            print(f"Best bound: {model.ObjBound:.6g}")

        chosen: list[tuple[int, int]] = [
            (u, v) for (u, v) in u_v_pairs if x[u, v].X > 0.5
        ]
        total_cost = sum(videos_data[v]["c"] for _, v in chosen)
        total_risk = sum(videos_data[v]["r"] for _, v in chosen)
        total_value = sum(user_video_data[u][v]["value"] for u, v in chosen)
        print(f"Total push cost: {total_cost:.10g}")
        print(f"Total push risk: {total_risk:.10g}")
        print(f"Total value (sum of v over pushes): {total_value:.10g}")

        by_user: dict[int, list[int]] = defaultdict(list)
        for u, v in chosen:
            by_user[u].append(v)

        payload = build_result_payload(
            status=status,
            objective=float(model.ObjVal),
            total_value=total_value,
            total_cost=total_cost,
            total_risk=total_risk,
            pushes_by_user=dict(by_user),
            users_data=users_data,
            user_video_data=user_video_data,
            videos_data=videos_data,
        )
        RESULT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULT_JSON_PATH, "w", encoding="utf-8") as jf:
            json.dump(payload, jf, ensure_ascii=False, indent=2)
        print(f"Wrote results to {RESULT_JSON_PATH}")
    else:
        print("No incumbent solution.")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Gurobi-native baseline_model with min push 0.4*K (sparse arcs)."
    )
    p.add_argument("--time-limit", type=float, default=None, help="Gurobi TimeLimit (seconds).")
    p.add_argument("--mip-gap", type=float, default=None, help="MIPGap (e.g. 0.01 for 1%%).")
    p.add_argument("--threads", type=int, default=None, help="Gurobi Threads.")
    p.add_argument(
        "--value-eps",
        type=float,
        default=0.0,
        help="Drop arcs with value <= eps (maximization: nonpositive arcs never optimal).",
    )
    p.add_argument("--no-warm-start", action="store_true", help="Disable heuristic MIP start.")
    p.add_argument(
        "--dump-model",
        type=str,
        default=None,
        metavar="PATH",
        help="Write .lp or .mps (only if variable count <= 200k).",
    )
    p.add_argument("--quiet", action="store_true", help="Gurobi OutputFlag=0.")
    args = p.parse_args()

    solve(
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        value_eps=args.value_eps,
        warm_start=not args.no_warm_start,
        dump_model=args.dump_model,
        output_flag=0 if args.quiet else 1,
    )


if __name__ == "__main__":
    main()
