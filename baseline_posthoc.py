"""
Baseline (i): greedy value maximisation + post-hoc safeguard.

Stage 1 — load an existing pure-value greedy solution (no R/A/E in the MIP).
Stage 2a — per-user hard repair to nominal limits (δ=0):
    drop by r/v until risk ≤ R[u] and entertainment ≤ A[u];
    add education videos by v until education ≥ E[u] (if feasible).
Stage 2b — budget alignment (confirmed policy):
    refill by v to match target cost first, then target volume
    (still enforcing R/A/E/T/K/N and TOTAL_COST).

Does not re-optimise αv−βr; allocation logic remains value-greedy.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
USER_CSV_PATH = ROOT / "users.csv"
VIDEO_CSV_PATH = ROOT / "videos.csv"
USER_VIDEO_CSV_PATH = ROOT / "user_video_value.csv"
DEFAULT_GREED_JSON = ROOT / "result/greed_model_result2.json"
DEFAULT_PROPOSED_JSON = ROOT / "result/common_model_result2.json"
DEFAULT_OUT_JSON = ROOT / "result/greed_posthoc_result.json"

TOTAL_COST = 43.0
EPS = 1e-12
VALUE_EPS = 1e-12


def load_users(path: Path) -> dict[int, dict]:
    users: dict[int, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = int(row["user_id"])
            users[uid] = {
                "T": float(row["T"]),
                "R": float(row["R"]),
                "K": int(row["K"]),
                "A": int(row["A"]),
                "E": int(row["E"]),
            }
    return users


def load_videos(path: Path) -> dict[int, dict]:
    videos: dict[int, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = int(row["video_id"])
            videos[vid] = {
                "K": int(row["K"]),
                "t": float(row["t"]),
                "entertainment_flag": int(row["entertainment_flag"]),
                "education_flag": int(row["education_flag"]),
                "r": float(row["r"]),
                "c": float(row["c"]),
                "N": int(row["N"]),
            }
    return videos


def load_user_video_values(path: Path) -> dict[int, dict[int, float]]:
    out: dict[int, dict[int, float]] = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["user_id"])][int(row["video_id"])] = float(row["v"])
    return dict(out)


def load_pushes_from_greed_json(path: Path) -> dict[int, list[int]]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    pushes: dict[int, list[int]] = {}
    for row in payload["push_info"]:
        pushes[int(row["user_id"])] = [int(v) for v in row["video_order"]]
    return pushes


def summarize_json(path: Path) -> tuple[float, int]:
    """Return (cost, total_push_count) from a result JSON."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    cost = float(payload["cost"])
    n = sum(int(u["video_num"]) for u in payload["push_info"])
    return cost, n


def user_stats(
    vids: list[int],
    videos: dict[int, dict],
    values: dict[int, float],
) -> dict[str, float]:
    risk = ent = edu = time = cost = value = 0.0
    for v in vids:
        vd = videos[v]
        risk += vd["r"]
        ent += vd["entertainment_flag"]
        edu += vd["education_flag"]
        time += vd["t"]
        cost += vd["c"]
        value += values.get(v, 0.0)
    return {
        "risk": risk,
        "ent": ent,
        "edu": edu,
        "time": time,
        "cost": cost,
        "value": value,
        "n": float(len(vids)),
    }


def can_add(
    u: int,
    v: int,
    selected: set[int],
    users: dict[int, dict],
    videos: dict[int, dict],
    values_u: dict[int, float],
    remain_n: dict[int, int],
    cost_spent: float,
    cost_cap: float,
) -> bool:
    if v in selected:
        return False
    if remain_n.get(v, 0) <= 0:
        return False
    if float(values_u.get(v, 0.0)) <= VALUE_EPS:
        return False
    vd = videos[v]
    if cost_spent + vd["c"] > cost_cap + EPS:
        return False
    lim = users[u]
    if len(selected) + 1 > lim["K"]:
        return False
    # provisional aggregates
    risk = ent = edu = time = 0.0
    for x in selected:
        xd = videos[x]
        risk += xd["r"]
        ent += xd["entertainment_flag"]
        edu += xd["education_flag"]
        time += xd["t"]
    risk += vd["r"]
    ent += vd["entertainment_flag"]
    edu += vd["education_flag"]
    time += vd["t"]
    if time > lim["T"] + EPS:
        return False
    if risk > lim["R"] + EPS:
        return False
    if ent > lim["A"] + EPS:
        return False
    # education is a lower bound — adding never hurts it
    _ = edu
    return True


def pick_drop_rv(
    selected: list[int],
    videos: dict[int, dict],
    values_u: dict[int, float],
) -> int | None:
    """Drop video with largest r / (v+eps); tie-break lower v, then higher video_id."""
    if not selected:
        return None
    best_v = None
    best_key = None
    for v in selected:
        val = float(values_u.get(v, 0.0))
        r = float(videos[v]["r"])
        ratio = r / (val + EPS)
        key = (ratio, -val, v)
        if best_key is None or key > best_key:
            best_key = key
            best_v = v
    return best_v


def repair_user(
    u: int,
    initial: list[int],
    users: dict[int, dict],
    videos: dict[int, dict],
    uv_values: dict[int, dict[int, float]],
    remain_n: dict[int, int],
    cost_spent: float,
    cost_cap: float,
) -> tuple[list[int], float, dict[str, int | float | bool]]:
    """
    Hard filter to R/A and try to meet E. Returns (new_list, new_cost_spent, stats).
    """
    lim = users[u]
    values_u = uv_values.get(u, {})
    selected = list(initial)
    # occupy N already counted globally from initial assignment; we only adjust deltas
    drops = 0
    edu_adds = 0

    def aggregates(sel: list[int]) -> tuple[float, float, float, float]:
        risk = ent = edu = time = 0.0
        for v in sel:
            vd = videos[v]
            risk += vd["r"]
            ent += vd["entertainment_flag"]
            edu += vd["education_flag"]
            time += vd["t"]
        return risk, ent, edu, time

    # --- A: drop by r/v until risk & entertainment satisfied ---
    while True:
        risk, ent, edu, time = aggregates(selected)
        if risk <= lim["R"] + EPS and ent <= lim["A"] + EPS:
            break
        victim = pick_drop_rv(selected, videos, values_u)
        if victim is None:
            break
        selected.remove(victim)
        remain_n[victim] = remain_n.get(victim, 0) + 1
        cost_spent -= float(videos[victim]["c"])
        drops += 1

    # --- B: add education by v (descending) ---
    edu_candidates = sorted(
        (
            v
            for v, val in values_u.items()
            if int(videos[v]["education_flag"]) == 1 and val > VALUE_EPS
        ),
        key=lambda v: (-float(values_u[v]), v),
    )
    while True:
        risk, ent, edu, time = aggregates(selected)
        if edu >= lim["E"] - EPS:
            break
        added = False
        for v in edu_candidates:
            if can_add(
                u, v, set(selected), users, videos, values_u, remain_n, cost_spent, cost_cap
            ):
                selected.append(v)
                remain_n[v] -= 1
                cost_spent += float(videos[v]["c"])
                edu_adds += 1
                added = True
                break
        if not added:
            break

    risk, ent, edu, time = aggregates(selected)
    stats: dict[str, int | float | bool] = {
        "drops": drops,
        "edu_adds": edu_adds,
        "edu_shortfall": max(0.0, float(lim["E"]) - edu),
        "risk_ok": risk <= lim["R"] + EPS,
        "ent_ok": ent <= lim["A"] + EPS,
        "edu_ok": edu + EPS >= lim["E"],
    }
    # stable order: value desc, video_id asc
    selected.sort(key=lambda v: (-float(values_u.get(v, 0.0)), v))
    return selected, cost_spent, stats


def iter_refill_candidates(
    pushes: dict[int, list[int]],
    uv_values: dict[int, dict[int, float]],
    videos: dict[int, dict],
) -> list[tuple[float, int, int]]:
    """Global (value, user, video) candidates not yet selected, value-desc."""
    cand: list[tuple[float, int, int]] = []
    for u, vals in uv_values.items():
        selected = set(pushes.get(u, []))
        for v, val in vals.items():
            if v in selected or val <= VALUE_EPS:
                continue
            if v not in videos:
                continue
            cand.append((float(val), u, v))
    cand.sort(key=lambda z: (-z[0], z[1], z[2]))
    return cand


def refill(
    pushes: dict[int, list[int]],
    users: dict[int, dict],
    videos: dict[int, dict],
    uv_values: dict[int, dict[int, float]],
    remain_n: dict[int, int],
    cost_spent: float,
    cost_cap: float,
    target_cost: float | None,
    target_volume: int | None,
) -> tuple[float, dict[str, int]]:
    """
    Align: first raise cost toward target_cost, then volume toward target_volume.
    Adds by pure value; never violates R/A/E/T/K/N or cost_cap.
    """
    adds_cost_phase = 0
    adds_vol_phase = 0
    selected_sets = {u: set(vs) for u, vs in pushes.items()}
    for u in users:
        selected_sets.setdefault(u, set())

    def current_volume() -> int:
        return sum(len(s) for s in selected_sets.values())

    def try_add_from_list(candidates: list[tuple[float, int, int]], phase: str) -> int:
        nonlocal cost_spent
        added = 0
        for val, u, v in candidates:
            if phase == "cost" and target_cost is not None and cost_spent + EPS >= target_cost:
                break
            if phase == "volume" and target_volume is not None and current_volume() >= target_volume:
                break
            sel = selected_sets[u]
            if not can_add(
                u, v, sel, users, videos, uv_values.get(u, {}), remain_n, cost_spent, cost_cap
            ):
                continue
            # education lower bound already satisfied or not — adding is fine;
            # can_add already enforces R/A/T/K/N/cost.
            sel.add(v)
            remain_n[v] -= 1
            cost_spent += float(videos[v]["c"])
            added += 1
        return added

    candidates = iter_refill_candidates(
        {u: list(s) for u, s in selected_sets.items()}, uv_values, videos
    )

    if target_cost is not None and cost_spent + EPS < target_cost:
        adds_cost_phase = try_add_from_list(candidates, "cost")
        # rebuild candidate list after cost-phase adds
        candidates = iter_refill_candidates(
            {u: list(s) for u, s in selected_sets.items()}, uv_values, videos
        )

    if target_volume is not None and current_volume() < target_volume:
        adds_vol_phase = try_add_from_list(candidates, "volume")

    for u, sel in selected_sets.items():
        vals = uv_values.get(u, {})
        pushes[u] = sorted(sel, key=lambda v: (-float(vals.get(v, 0.0)), v))

    return cost_spent, {"refill_cost_phase": adds_cost_phase, "refill_volume_phase": adds_vol_phase}


def build_result_payload(
    pushes: dict[int, list[int]],
    users: dict[int, dict],
    videos: dict[int, dict],
    uv_values: dict[int, dict[int, float]],
    diagnostics: dict,
) -> dict:
    push_info: list[dict] = []
    total_value = total_risk = total_cost = 0.0
    for u in sorted(pushes.keys()):
        vids = pushes[u]
        if not vids:
            continue
        values_u = uv_values.get(u, {})
        video_order = sorted(vids, key=lambda vid: (-float(values_u.get(vid, 0.0)), vid))
        video_time = sum(videos[vid]["t"] for vid in video_order)
        videos_detail = []
        for vid in video_order:
            val = float(values_u.get(vid, 0.0))
            r = float(videos[vid]["r"])
            total_value += val
            total_risk += r
            total_cost += float(videos[vid]["c"])
            videos_detail.append(
                {
                    "video_id": vid,
                    "value": val,
                    "risk": r,
                    "category": int(videos[vid]["K"]),
                }
            )
        push_info.append(
            {
                "user_id": u,
                "video_time_limit": int(users[u]["T"]),
                "video_time": video_time,
                "video_num": len(video_order),
                "video_order": video_order,
                "videos": videos_detail,
            }
        )
    return {
        "status": "PostHoc",
        "objective": total_value,
        "value": total_value,
        "risk": total_risk,
        "cost": total_cost,
        "diagnostics": diagnostics,
        "push_info": push_info,
    }


def run(
    greed_json: Path,
    proposed_json: Path | None,
    out_json: Path,
    target_cost: float | None,
    target_volume: int | None,
    cost_cap: float,
) -> dict:
    users = load_users(USER_CSV_PATH)
    videos = load_videos(VIDEO_CSV_PATH)
    uv_values = load_user_video_values(USER_VIDEO_CSV_PATH)
    pushes0 = load_pushes_from_greed_json(greed_json)

    if proposed_json is not None and proposed_json.exists():
        prop_cost, prop_vol = summarize_json(proposed_json)
        if target_cost is None:
            target_cost = prop_cost
        if target_volume is None:
            target_volume = prop_vol

    # Occupy the full Stage-1 assignment first so later users' N[v] slots
    # cannot be stolen during earlier users' education top-ups.
    remain_n = {v: int(videos[v]["N"]) for v in videos}
    cost_spent = 0.0
    for u, vids in pushes0.items():
        for v in vids:
            remain_n[v] -= 1
            cost_spent += float(videos[v]["c"])

    # Stage 2a — repair in place (drops free capacity; adds use residual N/cost)
    pushes: dict[int, list[int]] = {}
    users_dropped = 0
    users_edu_added = 0
    users_edu_short = 0
    total_drops = 0
    total_edu_adds = 0

    for u in sorted(users.keys()):
        initial = list(pushes0.get(u, []))
        repaired, cost_spent, st = repair_user(
            u,
            initial,
            users,
            videos,
            uv_values,
            remain_n,
            cost_spent,
            cost_cap,
        )
        pushes[u] = repaired
        total_drops += int(st["drops"])
        total_edu_adds += int(st["edu_adds"])
        if int(st["drops"]) > 0:
            users_dropped += 1
        if int(st["edu_adds"]) > 0:
            users_edu_added += 1
        if not bool(st["edu_ok"]):
            users_edu_short += 1

    vol_after_repair = sum(len(vs) for vs in pushes.values())
    cost_after_repair = cost_spent

    # Stage 2b
    cost_spent, refill_stats = refill(
        pushes,
        users,
        videos,
        uv_values,
        remain_n,
        cost_spent,
        cost_cap,
        target_cost=target_cost,
        target_volume=target_volume,
    )

    vol_final = sum(len(vs) for vs in pushes.values())
    diagnostics = {
        "policy": {
            "drop": "r/(v+eps)",
            "edu_add": "v descending",
            "align": "cost then volume",
        },
        "targets": {
            "cost": target_cost,
            "volume": target_volume,
            "cost_cap": cost_cap,
        },
        "stage1_source": str(greed_json),
        "after_repair": {
            "cost": cost_after_repair,
            "volume": vol_after_repair,
            "users_with_drops": users_dropped,
            "users_with_edu_adds": users_edu_added,
            "users_edu_shortfall": users_edu_short,
            "total_drops": total_drops,
            "total_edu_adds": total_edu_adds,
        },
        "refill": refill_stats,
        "final": {
            "cost": cost_spent,
            "volume": vol_final,
            "cost_gap_vs_target": None
            if target_cost is None
            else cost_spent - float(target_cost),
            "volume_gap_vs_target": None
            if target_volume is None
            else vol_final - int(target_volume),
        },
    }

    payload = build_result_payload(pushes, users, videos, uv_values, diagnostics)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Greedy + post-hoc R/A/E safeguard baseline.")
    p.add_argument("--target-cost", type=float, default=None)
    p.add_argument("--target-volume", type=int, default=None)
    p.add_argument("--cost-cap", type=float, default=TOTAL_COST)
    p.add_argument(
        "--no-proposed-targets",
        action="store_true",
        help="Do not read proposed JSON for targets; refill only up to --cost-cap.",
    )
    args = p.parse_args()

    proposed = None if args.no_proposed_targets else DEFAULT_PROPOSED_JSON
    payload = run(
        greed_json=DEFAULT_GREED_JSON,
        proposed_json=proposed,
        out_json=DEFAULT_OUT_JSON,
        target_cost=args.target_cost,
        target_volume=args.target_volume,
        cost_cap=float(args.cost_cap),
    )
    d = payload["diagnostics"]
    print(f"value={payload['value']:.6g}  risk={payload['risk']:.6g}  cost={payload['cost']:.6g}")
    print(
        f"volume={d['final']['volume']}  "
        f"cost_gap={d['final']['cost_gap_vs_target']}  "
        f"vol_gap={d['final']['volume_gap_vs_target']}"
    )
    print(
        f"repair: drops={d['after_repair']['total_drops']}  "
        f"edu_adds={d['after_repair']['total_edu_adds']}  "
        f"edu_short_users={d['after_repair']['users_edu_shortfall']}"
    )
    print(
        f"refill: cost_phase={d['refill']['refill_cost_phase']}  "
        f"vol_phase={d['refill']['refill_volume_phase']}"
    )
    print(f"Wrote {DEFAULT_OUT_JSON}")


if __name__ == "__main__":
    main()
