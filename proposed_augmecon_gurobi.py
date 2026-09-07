"""
Proposed model as an AUGMECON subproblem (Mavrotas 2009).

Bi-objective pair: max Z1, min Z2, with the same (2)–(13) soft slacks as
proposed_gurobi.py. Default ALPHA=1, BETA=0 so the primary scalar is
Z1 − p ΣΔ, not a weighted-sum with −β Z2.

Constraint: Σ r_v x_uv + s = ε_k, s ≥ 0.

Because ρ s/r ≤ 10^{-3} is below a practical MIPGap, the default solver is
lexicographic (ρ → 0+):
  Stage A: max (Z1 − p ΣΔ)  s.t. Z2 + s = ε_k
  Stage B: max s            s.t. primary ≥ Stage-A incumbent

--mode rho keeps the scalar objective primary + ρ s/r (ablation only).

依赖：gurobipy + 有效 Gurobi 许可证。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

USER_CSV_PATH = Path(__file__).resolve().parent / "users.csv"
VIDEO_CSV_PATH = Path(__file__).resolve().parent / "videos.csv"
USER_VIDEO_CSV_PATH = Path(__file__).resolve().parent / "user_video_value.csv"
RESULT_JSON_PATH = Path(__file__).resolve().parent / "result/common_model_result5.json"

MAX_CATEGORY_PUSH_COUNT = 9275
TOTAL_COST = 43
# 全局风险预算 ε_k（ε-约束 RHS）；可用 --epsilon-k 覆盖
R_MIN = 13968.5
R_MAX = 99773.75
q=20
step = (R_MAX - R_MIN) / q
k =20
EPSILON_K = R_MIN + step * k
R_RANGE = R_MAX - R_MIN
RHO = 1e-3

PENALTY_COEFF = 0.01
ALPHA = 1
BETA = 0
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


def actual_gap_pct(incumbent: float, proven_bound: float) -> float | None:
    """Actual Gap = ((Best Bound - Incumbent) / |Incumbent|) × 100%."""
    if abs(incumbent) < 1e-12:
        return None
    return ((proven_bound - incumbent) / abs(incumbent)) * 100.0


def bypass_count(slack: float, step: float) -> int:
    if step <= 1e-12:
        return 0
    return int(float(slack) // float(step))


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
    risk_cap: float,
) -> dict[tuple[int, int], int]:
    """
    按用户分组，弧按 value/duration 降序贪心。
    先在时长 T 内尽量填满（上限 K）；若推送数仍低于 0.4·K，则允许超出 T（δ_T 补齐）继续补足下限。
    满足每用户推送数上限 K、每视频次数 N、全局成本与全局风险预算；尽量满足下限 0.4·K。
    不保证满足 R/A/E/类别上限；用于 MIP 初值，δ 由 compute_delta_starts 补齐。
    """
    by_user: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for u, v, val in arcs:
        by_user[u].append((u, v, val))

    remaining_video = {v: videos_data[v]["N"] for v in videos_data}
    user_time: dict[int, float] = defaultdict(float)
    user_count: dict[int, int] = defaultdict(int)
    spent_cost = 0.0
    spent_risk = 0.0
    start: dict[tuple[int, int], int] = {}

    def try_add(u: int, v: int, t_cap: float | None) -> bool:
        nonlocal spent_cost, spent_risk
        c_v = float(videos_data[v]["c"])
        t_v = float(videos_data[v]["t"])
        r_v = float(videos_data[v]["r"])
        if spent_cost + c_v > total_cost_cap + 1e-12:
            return False
        if spent_risk + r_v > risk_cap + 1e-12:
            return False
        if t_cap is not None and user_time[u] + t_v > t_cap + 1e-12:
            return False
        if remaining_video[v] <= 0:
            return False
        start[(u, v)] = 1
        user_time[u] += t_v
        user_count[u] += 1
        spent_cost += c_v
        spent_risk += r_v
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


LEX_PRIMARY_TOL = 1e-4


@dataclass
class AugmeconPack:
    model: gp.Model
    x: gp.tupledict
    delta_T: gp.tupledict
    delta_R: gp.tupledict
    delta_A: gp.tupledict
    delta_E: gp.tupledict
    s: gp.Var
    obj_primary: gp.LinExpr
    risk_expr: gp.LinExpr
    eps_constr: gp.Constr | None
    r_range: float
    u_v_pairs: list[tuple[int, int]]
    all_users: list[int]
    arcs: list[tuple[int, int, float]]
    users_data: dict[int, dict]
    videos_data: dict[int, dict]
    user_video_data: dict[int, dict[int, dict]]
    total_cost_f: float

    def set_epsilon(self, epsilon_k: float) -> None:
        if self.eps_constr is None:
            self.add_epsilon(epsilon_k)
            return
        self.eps_constr.RHS = float(epsilon_k)

    def drop_epsilon(self) -> None:
        if self.eps_constr is None:
            return
        self.model.remove(self.eps_constr)
        self.model.update()
        self.eps_constr = None

    def add_epsilon(self, epsilon_k: float) -> None:
        if self.eps_constr is not None:
            self.set_epsilon(epsilon_k)
            return
        self.eps_constr = self.model.addConstr(
            self.risk_expr + self.s == float(epsilon_k),
            name="total_risk_eps",
        )


def load_instance(
    user_csv: Path | str = USER_CSV_PATH,
    video_csv: Path | str = VIDEO_CSV_PATH,
    user_video_csv: Path | str = USER_VIDEO_CSV_PATH,
) -> tuple[dict[int, dict], dict[int, dict], dict[int, dict[int, dict]]]:
    return load_users(user_csv), load_videos(video_csv), load_user_video_data(user_video_csv)


def build_model(
    users_data: dict[int, dict],
    videos_data: dict[int, dict],
    user_video_data: dict[int, dict[int, dict]],
    epsilon_k: float,
    r_range: float,
    value_eps: float,
    time_limit: float | None,
    mip_gap: float | None,
    threads: int | None,
    mip_focus: int | None,
    output_flag: int,
    log_file: str | None = None,
) -> AugmeconPack:
    total_cost_f = float(TOTAL_COST)
    r_range_f = float(r_range) if abs(r_range) > 1e-12 else 1.0
    arcs = build_arc_list(videos_data, user_video_data, users_data, value_eps, total_cost_f)
    u_v_pairs = [(u, v) for u, v, _ in arcs]
    all_users = sorted(users_data.keys())
    all_videos = sorted(videos_data.keys())

    model = gp.Model("ProposedAugmecon")
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
    if log_file:
        model.setParam("LogFile", log_file)

    obj_coeff: dict[tuple[int, int], float] = {}
    for u, v, val in arcs:
        r_v = float(videos_data[v]["r"])
        obj_coeff[u, v] = ALPHA * val - BETA * r_v

    x = model.addVars(u_v_pairs, vtype=GRB.BINARY, name="x")
    delta_T = model.addVars(all_users, lb=0.0, vtype=GRB.CONTINUOUS, name="delta_T")
    delta_R = model.addVars(all_users, lb=0.0, vtype=GRB.CONTINUOUS, name="delta_R")
    delta_A = model.addVars(all_users, lb=0, vtype=GRB.INTEGER, name="delta_A")
    delta_E = model.addVars(all_users, lb=0, vtype=GRB.INTEGER, name="delta_E")
    s = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="s_aug")

    obj_x = gp.quicksum(obj_coeff[uv] * x[uv] for uv in u_v_pairs)
    obj_pen = PENALTY_COEFF * gp.quicksum(
        delta_T[u] + delta_R[u] + delta_A[u] + delta_E[u] for u in all_users
    )
    obj_primary = obj_x - obj_pen

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

    for cat in range(1, 5):
        pairs_k = [(u, v) for u, v in u_v_pairs if int(videos_data[v]["K"]) == cat]
        if pairs_k:
            model.addConstr(
                gp.quicksum(x[u, v] for u, v in pairs_k) <= MAX_CATEGORY_PUSH_COUNT,
                name=f"cat_{cat}",
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
            model.addConstr(delta_E[u] >= int(users_data[u]["E"]), name=f"edu_{u}")

    if u_v_pairs:
        model.addConstr(
            gp.quicksum(videos_data[v]["c"] * x[u, v] for u, v in u_v_pairs) <= total_cost_f,
            name="total_cost",
        )
        risk_expr = gp.quicksum(videos_data[v]["r"] * x[u, v] for u, v in u_v_pairs)
    else:
        risk_expr = gp.LinExpr(0.0)

    eps_constr = model.addConstr(risk_expr + s == float(epsilon_k), name="total_risk_eps")

    return AugmeconPack(
        model=model,
        x=x,
        delta_T=delta_T,
        delta_R=delta_R,
        delta_A=delta_A,
        delta_E=delta_E,
        s=s,
        obj_primary=obj_primary,
        risk_expr=risk_expr,
        eps_constr=eps_constr,
        r_range=r_range_f,
        u_v_pairs=u_v_pairs,
        all_users=all_users,
        arcs=arcs,
        users_data=users_data,
        videos_data=videos_data,
        user_video_data=user_video_data,
        total_cost_f=total_cost_f,
    )


def apply_mip_start(
    pack: AugmeconPack,
    x_on: dict[tuple[int, int], int],
    epsilon_k: float,
) -> None:
    dT, dR, dA, dE = compute_delta_starts(
        pack.users_data, pack.videos_data, x_on, pack.all_users
    )
    for uv, var in pack.x.items():
        var.Start = float(x_on.get(uv, 0))
    for u in pack.all_users:
        pack.delta_T[u].Start = dT[u]
        pack.delta_R[u].Start = dR[u]
        pack.delta_A[u].Start = float(dA[u])
        pack.delta_E[u].Start = float(dE[u])
    spent_risk = sum(float(pack.videos_data[v]["r"]) for (_u, v), z in x_on.items() if z)
    pack.s.Start = max(0.0, float(epsilon_k) - spent_risk)


def apply_greedy_start(pack: AugmeconPack, epsilon_k: float) -> None:
    h = heuristic_x_greedy(
        pack.arcs, pack.users_data, pack.videos_data, pack.total_cost_f, float(epsilon_k)
    )
    apply_mip_start(pack, h, epsilon_k)


def capture_assignment(pack: AugmeconPack) -> dict[tuple[int, int], int]:
    return {uv: 1 for uv in pack.u_v_pairs if pack.x[uv].X > 0.5}


def extract_metrics(
    pack: AugmeconPack,
    epsilon_k: float,
    step: float,
    primary_obj: float | None = None,
    primary_bound: float | None = None,
    stage_a_time: float = 0.0,
    stage_b_time: float = 0.0,
    include_push_info: bool = True,
) -> dict:
    status = pack.model.Status
    if pack.model.SolCount <= 0:
        return {
            "status": grb_status_to_label(status),
            "status_code": int(status),
            "infeasible": True,
            "epsilon_k": float(epsilon_k),
            "time_s": stage_a_time + stage_b_time,
            "stage_a_time_s": stage_a_time,
            "stage_b_time_s": stage_b_time,
        }

    chosen = [(u, v) for (u, v) in pack.u_v_pairs if pack.x[u, v].X > 0.5]
    total_value = sum(float(pack.user_video_data[u][v]["value"]) for u, v in chosen)
    total_cost = sum(float(pack.videos_data[v]["c"]) for _, v in chosen)
    total_risk = sum(float(pack.videos_data[v]["r"]) for _, v in chosen)
    s_k = float(pack.s.X)
    primary_now = float(pack.obj_primary.getValue())
    rho_term = RHO * s_k / pack.r_range
    inc = primary_obj if primary_obj is not None else primary_now
    bound = primary_bound if primary_bound is not None else (
        float(pack.model.ObjBound) if getattr(pack.model, "IsMIP", 0) else inc
    )
    gap = actual_gap_pct(inc, bound) if bound is not None else None

    by_user: dict[int, list[int]] = defaultdict(list)
    for u, v in chosen:
        by_user[u].append(v)

    payload: dict = {
        "status": grb_status_to_label(status),
        "status_code": int(status),
        "infeasible": False,
        "epsilon_k": float(epsilon_k),
        "value": total_value,
        "risk": total_risk,
        "cost": total_cost,
        "slack": s_k,
        "rho": RHO,
        "r_range": pack.r_range,
        "rho_term": rho_term,
        "primary_obj": inc,
        "primary_bound": bound,
        "primary_gap_pct": gap,
        "augmented_obj": inc + rho_term,
        "bypass": bypass_count(s_k, step),
        "time_s": stage_a_time + stage_b_time,
        "stage_a_time_s": stage_a_time,
        "stage_b_time_s": stage_b_time,
        "n_arcs": len(pack.u_v_pairs),
        "n_chosen": len(chosen),
    }
    if include_push_info:
        full = build_result_payload(
            status=status,
            objective=inc + rho_term,
            total_value=total_value,
            total_cost=total_cost,
            total_risk=total_risk,
            pushes_by_user=dict(by_user),
            users_data=pack.users_data,
            user_video_data=pack.user_video_data,
            videos_data=pack.videos_data,
        )
        payload["push_info"] = full["push_info"]
        payload["objective"] = full["objective"]
    return payload


def metrics_from_assignment(
    pack: AugmeconPack,
    x_on: dict[tuple[int, int], int],
    epsilon_k: float,
    step: float,
    *,
    primary_obj: float | None = None,
    primary_bound: float | None = None,
    stage_a_time: float = 0.0,
    stage_b_time: float = 0.0,
    status: str = "Copied",
    include_push_info: bool = False,
) -> dict:
    """Criteria and slack from a 0/1 assignment (bypass copies, Stage-B fallback)."""
    chosen = [(u, v) for (u, v), z in x_on.items() if z]
    total_value = sum(float(pack.user_video_data[u][v]["value"]) for u, v in chosen)
    total_cost = sum(float(pack.videos_data[v]["c"]) for _, v in chosen)
    total_risk = sum(float(pack.videos_data[v]["r"]) for _, v in chosen)
    s_k = max(0.0, float(epsilon_k) - total_risk)
    dT, dR, dA, dE = compute_delta_starts(
        pack.users_data, pack.videos_data, x_on, pack.all_users
    )
    pen = PENALTY_COEFF * sum(
        dT[u] + dR[u] + dA[u] + dE[u] for u in pack.all_users
    )
    primary_now = ALPHA * total_value - BETA * total_risk - pen
    inc = primary_obj if primary_obj is not None else primary_now
    bound = primary_bound if primary_bound is not None else inc
    gap = actual_gap_pct(inc, bound) if bound is not None else None
    rho_term = RHO * s_k / pack.r_range
    payload: dict = {
        "status": status,
        "status_code": None,
        "infeasible": False,
        "epsilon_k": float(epsilon_k),
        "value": total_value,
        "risk": total_risk,
        "cost": total_cost,
        "slack": s_k,
        "rho": RHO,
        "r_range": pack.r_range,
        "rho_term": rho_term,
        "primary_obj": inc,
        "primary_bound": bound,
        "primary_gap_pct": gap,
        "augmented_obj": inc + rho_term,
        "bypass": bypass_count(s_k, step),
        "time_s": stage_a_time + stage_b_time,
        "stage_a_time_s": stage_a_time,
        "stage_b_time_s": stage_b_time,
        "n_arcs": len(pack.u_v_pairs),
        "n_chosen": len(chosen),
    }
    if include_push_info:
        by_user: dict[int, list[int]] = defaultdict(list)
        for u, v in chosen:
            by_user[u].append(v)
        full = build_result_payload(
            status=0,
            objective=inc + rho_term,
            total_value=total_value,
            total_cost=total_cost,
            total_risk=total_risk,
            pushes_by_user=dict(by_user),
            users_data=pack.users_data,
            user_video_data=pack.user_video_data,
            videos_data=pack.videos_data,
        )
        payload["status"] = status
        payload["push_info"] = full["push_info"]
        payload["objective"] = full["objective"]
    return payload


def print_subproblem_metrics(m: dict) -> None:
    if m.get("infeasible"):
        print("No incumbent solution.")
        return
    print(f"Primary (Z1-pΣΔ): {m['primary_obj']:.6g}")
    if m.get("primary_bound") is not None:
        print(f"Primary bound: {m['primary_bound']:.6g}")
    gap = m.get("primary_gap_pct")
    if gap is None:
        print("Primary gap: N/A")
    else:
        print(f"Primary gap: {gap:.6g}%")
    print(f"Total value (Z1): {m['value']:.10g}")
    print(f"Total push cost: {m['cost']:.10g}")
    print(
        f"Total push risk (Z2): {m['risk']:.10g} "
        f"(cap ε_k={m['epsilon_k']:g})"
    )
    print(
        f"Slack s: {m['slack']:.10g} "
        f"(ρ·s/r={m.get('rho_term', 0.0):.10g})"
    )
    print(f"bypass: {m['bypass']}")
    print(
        f"Running time: {m['time_s']:.10g}s "
        f"(A={m.get('stage_a_time_s', 0.0):.10g}s, "
        f"B={m.get('stage_b_time_s', 0.0):.10g}s)"
    )


def solve_lex_subproblem(
    pack: AugmeconPack,
    epsilon_k: float,
    step: float,
    mode: str,
    include_push_info: bool,
) -> dict:
    """Solve one AUGMECON grid point. mode is 'lex' (default) or 'rho'."""
    pack.set_epsilon(epsilon_k)
    pack.model.ModelSense = GRB.MAXIMIZE

    if mode == "rho":
        pack.model.setObjective(
            pack.obj_primary + RHO * pack.s / pack.r_range, GRB.MAXIMIZE
        )
        print(f"EPSILON_K (ε_k risk cap): {epsilon_k:g}")
        print(f"AUGMECON: mode=rho, ρ={RHO:g}, r={pack.r_range:g} (term ρ·s/r)")
        pack.model.optimize()
        print(f"Status: {grb_status_to_label(pack.model.Status)} (code {pack.model.Status})")
        primary = None
        bound = None
        if pack.model.SolCount > 0:
            s_k = float(pack.s.X)
            primary = float(pack.obj_primary.getValue())
            bound = float(pack.model.ObjBound) - RHO * s_k / pack.r_range
            pack.last_assignment = capture_assignment(pack)
        return extract_metrics(
            pack,
            epsilon_k,
            step,
            primary_obj=primary,
            primary_bound=bound,
            stage_a_time=float(pack.model.Runtime),
            include_push_info=include_push_info,
        )

    pack.model.setObjective(pack.obj_primary, GRB.MAXIMIZE)
    print(f"EPSILON_K (ε_k risk cap): {epsilon_k:g}")
    print(f"AUGMECON: mode=lex (ρ→0+), r={pack.r_range:g}, ρ={RHO:g} (reported, not in Stage A)")
    pack.model.optimize()
    stage_a_time = float(pack.model.Runtime)
    status_a = pack.model.Status
    print(f"Stage A status: {grb_status_to_label(status_a)} (code {status_a})")

    if pack.model.SolCount <= 0:
        return extract_metrics(
            pack,
            epsilon_k,
            step,
            stage_a_time=stage_a_time,
            include_push_info=False,
        )

    primary_obj = float(pack.obj_primary.getValue())
    primary_bound = float(pack.model.ObjBound)
    print(f"Stage A primary incumbent: {primary_obj:.6g}  bound: {primary_bound:.6g}")
    assignment_a = capture_assignment(pack)

    lex_c = pack.model.addConstr(
        pack.obj_primary >= primary_obj - LEX_PRIMARY_TOL,
        name="lex_keep_primary",
    )
    pack.model.setObjective(pack.s, GRB.MAXIMIZE)
    pack.model.optimize()
    stage_b_time = float(pack.model.Runtime)
    status_b = pack.model.Status
    solcount_b = int(pack.model.SolCount)
    print(f"Stage B status: {grb_status_to_label(status_b)} (code {status_b})")
    # Read X before removing lex_c: update() drops the incumbent.
    if solcount_b > 0:
        assignment_b = capture_assignment(pack)
        metrics_b = extract_metrics(
            pack,
            epsilon_k,
            step,
            primary_obj=primary_obj,
            primary_bound=primary_bound,
            stage_a_time=stage_a_time,
            stage_b_time=stage_b_time,
            include_push_info=include_push_info,
        )
        pack.last_assignment = assignment_b
    else:
        assignment_b = None
        metrics_b = None
        pack.last_assignment = assignment_a

    pack.model.remove(lex_c)
    pack.model.update()
    pack.model.setObjective(pack.obj_primary, GRB.MAXIMIZE)

    if metrics_b is None:
        print("Stage B produced no incumbent; keeping Stage A solution.")
        return metrics_from_assignment(
            pack,
            assignment_a,
            epsilon_k,
            step,
            primary_obj=primary_obj,
            primary_bound=primary_bound,
            stage_a_time=stage_a_time,
            stage_b_time=stage_b_time,
            status="Stage-A fallback",
            include_push_info=include_push_info,
        )
    return metrics_b


def solve(
    time_limit: float | None,
    mip_gap: float | None,
    threads: int | None,
    mip_focus: int | None,
    value_eps: float,
    warm_start: bool,
    dump_model: str | None,
    output_flag: int,
    epsilon_k: float,
    mode: str,
    r_range: float,
    output_path: str | None,
    include_push_info: bool,
    log_file: str | None = None,
) -> dict:
    users_data, videos_data, user_video_data = load_instance()
    r_range_f = float(r_range) if abs(float(r_range)) > 1e-12 else 1.0
    step_f = r_range_f / float(q)
    pack = build_model(
        users_data,
        videos_data,
        user_video_data,
        epsilon_k=float(epsilon_k),
        r_range=r_range_f,
        value_eps=value_eps,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        mip_focus=mip_focus,
        output_flag=output_flag,
        log_file=log_file,
    )
    if dump_model:
        n = len(pack.u_v_pairs) + 4 * len(pack.all_users) + 1
        if n > 500_000:
            print(f"Skipping --dump-model: ~{n} vars (threshold 500000).")
        else:
            pack.model.write(dump_model)
            print(f"Wrote model to {dump_model}")
    if warm_start and pack.u_v_pairs:
        apply_greedy_start(pack, float(epsilon_k))
    metrics = solve_lex_subproblem(
        pack,
        float(epsilon_k),
        step_f,
        mode,
        include_push_info,
    )
    print_subproblem_metrics(metrics)
    out = Path(output_path) if output_path else RESULT_JSON_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as jf:
        json.dump(metrics, jf, ensure_ascii=False, indent=2)
    print(f"Wrote results to {out}")
    pack.model.dispose()
    return metrics



def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "AUGMECON subproblem: lex (default) max Z1-pΣΔ then max s, "
            "or --mode rho for scalar primary + ρ s/r. Constraint Z2 + s = ε_k."
        )
    )
    p.add_argument("--time-limit", type=float, default=None, help="Gurobi TimeLimit (seconds).")
    p.add_argument("--mip-gap", type=float, default=None, help="MIPGap (e.g. 0.001 for 0.1%%).")
    p.add_argument("--threads", type=int, default=None, help="Gurobi Threads.")
    p.add_argument("--mip-focus", type=int, default=None, help="MIPFocus (0 default, 1 feas, 2 optimality bound, 3 bound).")
    p.add_argument(
        "--value-eps",
        type=float,
        default=0.0,
        help="Drop arcs with value <= eps (dense CSV 上常为 0 与 PuLP 一致).",
    )
    p.add_argument(
        "--epsilon-k",
        type=float,
        default=EPSILON_K,
        help=f"Global risk cap ε_k: Σ r_v x_uv + s = ε_k (default {EPSILON_K:g}).",
    )
    p.add_argument(
        "--mode",
        choices=("lex", "rho"),
        default="lex",
        help="lex = two-stage AUGMECON (default); rho = scalar ρ s/r (ablation).",
    )
    p.add_argument("--r-min", type=float, default=None, help="Override R_MIN for r = R_max-R_min.")
    p.add_argument("--r-max", type=float, default=None, help="Override R_MAX for r = R_max-R_min.")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"JSON path (default {RESULT_JSON_PATH}).",
    )
    p.add_argument("--no-push-info", action="store_true", help="Omit per-user assignment lists in JSON.")
    p.add_argument("--no-warm-start", action="store_true", help="Disable heuristic MIP start.")
    p.add_argument(
        "--dump-model",
        type=str,
        default=None,
        metavar="PATH",
        help="Write .lp / .mps (skipped if estimated vars > 500k).",
    )
    p.add_argument("--quiet", action="store_true", help="Gurobi OutputFlag=0.")
    p.add_argument("--log-file", type=str, default=None, help="Gurobi LogFile path.")
    args = p.parse_args()

    r_min = R_MIN if args.r_min is None else args.r_min
    r_max = R_MAX if args.r_max is None else args.r_max
    solve(
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        mip_focus=args.mip_focus,
        value_eps=args.value_eps,
        warm_start=not args.no_warm_start,
        dump_model=args.dump_model,
        output_flag=0 if args.quiet else 1,
        epsilon_k=args.epsilon_k,
        mode=args.mode,
        r_range=float(r_max) - float(r_min),
        output_path=args.output,
        include_push_info=not args.no_push_info,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()
