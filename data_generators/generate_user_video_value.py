"""
针对每个用户 × 每个视频，生成截断在 [0, 1] 上的商用价值 v（异构截断正态）。
用户来自 users.csv，视频来自 videos.csv；输出列：user_id, video_id, v。

年龄分段（与 users 中浮点年龄一致）：
- 12–14 岁段：12 <= age < 15
- 15–18 岁段：15 <= age <= 18

时长与 videos.csv 列 D 一致：D=1 短视频，D=2 中视频，D=3 长视频。

类别口径（基于 videos.csv）：
- 教育：education_flag == 1 且 entertainment_flag == 0（仅教育）
- 娱乐：entertainment_flag == 1（与 education_flag 同时为 1 时，仍按娱乐处理：娱乐优先）
  - 细分：按 K（与 generators/generate_video_data.py 一致）
    - K=2：游戏
    - K=3：搞笑
- 生活与纪实：entertainment_flag == 0 且 education_flag == 0（两类都不是；通常 K=4）

截断正态参数（方差为 σ²，标准差 σ = sqrt(方差)），按代码中规则列表顺序依次匹配（先匹配者不重复）。
μ、σ² 表示 **截断前** 底层正态 X ~ N(μ, σ²) 的均值与方差；
v 为 X 在 [0,1] 上截断后的样本，故 v 的边际 E[v]、Var(v) 一般 **不等于** μ、σ²。

规则表（年龄 × 视频类型 × 视频长度 → 截断前均值 μ、方差 σ²）：

| 年龄   | 视频类型   | 视频长度 | 均值 μ | 方差 σ² |
|--------|------------|----------|--------|---------|
| 12–14  | 游戏       | 短       | 0.90   | 0.10    |
| 12–14  | 游戏       | 中       | 0.85   | 0.05    |
| 12–14  | 游戏       | 长       | 0.80   | 0.05    |
| 12–14  | 搞笑       | 短       | 0.60   | 0.10    |
| 12–14  | 搞笑       | 中       | 0.50   | 0.05    |
| 12–14  | 搞笑       | 长       | 0.40   | 0.05    |
| 12–14  | 教育       | 短       | 0.15   | 0.03    |
| 12–14  | 教育       | 中       | 0.30   | 0.05    |
| 12–14  | 教育       | 长       | 0.20   | 0.05    |
| 12–14  | 生活与纪实 | 短       | 0.15   | 0.03    |
| 12–14  | 生活与纪实 | 中       | 0.20   | 0.05    |
| 12–14  | 生活与纪实 | 长       | 0.10   | 0.03    |
| 15–18  | 游戏       | 短       | 0.95   | 0.10    |
| 15–18  | 游戏       | 中       | 0.90   | 0.05    |
| 15–18  | 游戏       | 长       | 0.85   | 0.05    |
| 15–18  | 搞笑       | 短       | 0.65   | 0.10    |
| 15–18  | 搞笑       | 中       | 0.55   | 0.05    |
| 15–18  | 搞笑       | 长       | 0.45   | 0.05    |
| 15–18  | 教育       | 短       | 0.20   | 0.05    |
| 15–18  | 教育       | 中       | 0.40   | 0.05    |
| 15–18  | 教育       | 长       | 0.35   | 0.05    |
| 15–18  | 生活与纪实 | 短       | 0.15   | 0.03    |
| 15–18  | 生活与纪实 | 中       | 0.25   | 0.05    |
| 15–18  | 生活与纪实 | 长       | 0.15   | 0.05    |

兜底（未命中上表：非 12–18 岁、D 非 1/2/3 等）：μ=0.2，σ²=0.05，截断到 [0,1]。

在 [0,1] 上截断后，由 μ、σ² 得到的理论边际矩可用 `marginal_moments_truncated_gaussian` 计算。
下列为若干代表性规则及兜底的截断后理论矩（其余组合同理可算）：

| 规则            | 截断前 μ | 截断前 σ² | 截断后 E[v] | 截断后 Var(v) |
|-----------------|----------|-----------|-------------|---------------|
| 12–14 游戏+短   | 0.90     | 0.10      | ≈0.711      | ≈0.0416       |
| 12–14 搞笑+短   | 0.60     | 0.10      | ≈0.559      | ≈0.0578       |
| 12–14 教育+长   | 0.15     | 0.05      | ≈0.245      | ≈0.0266       |
| 15–18 教育+长   | 0.30     | 0.05      | ≈0.339      | ≈0.0360       |
| 兜底            | 0.20     | 0.05      | ≈0.273      | ≈0.0298       |

若业务上希望「样本 v 的均值 / 方差」直接等于某目标值，需改用 Beta 等可在 [0,1] 上独立设定前两矩的分布，
或对「截断后矩」做数值反标定（可能无解或仅能逼近，见 `marginal_moments_truncated_gaussian`）。
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import pandas as pd

USERS_PATH = "users.csv"
VIDEOS_PATH = "videos.csv"
OUTPUT_PATH = "user_video_value.csv"
RNG_SEED = 42
# 每次写入的用户块大小，控制内存（全量笛卡尔积约 用户数×视频数 行）
USER_CHUNK_SIZE = 100


def marginal_moments_truncated_gaussian(
    mean: float,
    std: float,
    low: float = 0.0,
    high: float = 1.0,
) -> tuple[float, float]:
    """
    若 Y ~ N(mean, std²)，v = Y | low <= Y <= high（双侧截断），返回 (E[v], Var[v]) 的闭式解。
    std 须 > 0；low < high。
    """
    if std <= 0 or low >= high:
        raise ValueError("std>0 且 low<high 才定义截断矩")

    def _phi(z: float) -> float:
        return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

    def _Phi(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    a = (low - mean) / std
    b = (high - mean) / std
    zmass = _Phi(b) - _Phi(a)
    if zmass <= 1e-18:
        raise ValueError("截断区间上质量过小，请检查 mean/std 与 [low,high]")

    ez = (_phi(a) - _phi(b)) / zmass
    ez2 = 1.0 + (a * _phi(a) - b * _phi(b)) / zmass
    vz = max(ez2 - ez * ez, 0.0)
    ex = mean + std * ez
    vx = std * std * vz
    return ex, vx


def _truncated_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    low: float,
    high: float,
    size: int,
) -> np.ndarray:
    """N(mean, std²) 截断到 [low, high]，仅用 NumPy（拒绝采样）。"""
    if size == 0:
        return np.empty(0, dtype=np.float64)
    out = np.empty(size, dtype=np.float64)
    pos = 0
    # 略多抽样以提高填满概率
    batch = min(max(size * 2, 4096), 5_000_000)
    while pos < size:
        raw = rng.normal(mean, std, size=batch)
        good = raw[(raw >= low) & (raw <= high)]
        take = min(len(good), size - pos)
        if take > 0:
            out[pos : pos + take] = good[:take]
            pos += take
        if take == 0 and batch < 50_000_000:
            batch = min(batch * 2, 50_000_000)
    return out


def _assign_v_for_chunk(
    ages: np.ndarray,
    D: np.ndarray,
    K: np.ndarray,
    entertainment: np.ndarray,
    education: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """ages / D / flags 均为与用户×视频对齐后的等长向量（float / int）。"""
    n = len(ages)
    v = np.empty(n, dtype=np.float64)
    young = (ages >= 12.0) & (ages < 15.0)
    older = (ages >= 15.0) & (ages <= 18.0)
    short = D.astype(np.int64) == 1
    mid = D.astype(np.int64) == 2
    long_ = D.astype(np.int64) == 3
    k = K.astype(np.int64)
    ent = entertainment.astype(np.int64) == 1
    edu = education.astype(np.int64) == 1
    # 类别口径：娱乐优先，其次教育，其余视为生活与纪实
    edu_only = edu & ~ent
    life = ~ent & ~edu
    game = ent & (k == 2)
    funny = ent & (k == 3)
    ent_other = ent & ~(game | funny)

    # 优先级：按规则列表顺序依次匹配（先匹配者不重复）
    rules: list[tuple[np.ndarray, float, float]] = [
        # --- 娱乐：细分 游戏 / 搞笑 ---
        # 12–14 游戏
        (young & game & short, 0.90, np.sqrt(0.10)),
        (young & game & mid, 0.85, np.sqrt(0.05)),
        (young & game & long_, 0.80, np.sqrt(0.05)),
        # 12–14 搞笑
        (young & funny & short, 0.60, np.sqrt(0.10)),
        (young & funny & mid, 0.50, np.sqrt(0.05)),
        (young & funny & long_, 0.40, np.sqrt(0.05)),
        # 15–18 游戏
        (older & game & short, 0.95, np.sqrt(0.10)),
        (older & game & mid, 0.90, np.sqrt(0.05)),
        (older & game & long_, 0.85, np.sqrt(0.05)),
        # 15–18 搞笑
        (older & funny & short, 0.65, np.sqrt(0.10)),
        (older & funny & mid, 0.55, np.sqrt(0.05)),
        (older & funny & long_, 0.45, np.sqrt(0.05)),
        # 若存在 entertainment_flag==1 但 K 非 2/3 的异常数据，按“搞笑”参数处理
        (young & ent_other & short, 0.60, np.sqrt(0.10)),
        (young & ent_other & mid, 0.50, np.sqrt(0.05)),
        (young & ent_other & long_, 0.40, np.sqrt(0.05)),
        (older & ent_other & short, 0.65, np.sqrt(0.10)),
        (older & ent_other & mid, 0.55, np.sqrt(0.05)),
        (older & ent_other & long_, 0.45, np.sqrt(0.05)),
        # --- 教育 ---
        (young & edu_only & long_, 0.20, np.sqrt(0.05)),
        (young & edu_only & short, 0.15, np.sqrt(0.03)),
        (young & edu_only & mid, 0.30, np.sqrt(0.05)),
        (older & edu_only & long_, 0.35, np.sqrt(0.05)),
        (older & edu_only & short, 0.20, np.sqrt(0.05)),
        (older & edu_only & mid, 0.40, np.sqrt(0.05)),
        # 生活与纪实（两年龄段分别设定）
        (young & life & short, 0.15, np.sqrt(0.03)),
        (young & life & mid, 0.20, np.sqrt(0.05)),
        (young & life & long_, 0.10, np.sqrt(0.03)),
        (older & life & short, 0.15, np.sqrt(0.03)),
        (older & life & mid, 0.25, np.sqrt(0.05)),
        (older & life & long_, 0.15, np.sqrt(0.05)),
    ]

    assigned = np.zeros(n, dtype=bool)
    for mask, mu, sigma in rules:
        m = mask & ~assigned
        cnt = int(m.sum())
        if cnt == 0:
            continue
        v[m] = _truncated_normal(rng, mu, sigma, 0.0, 1.0, cnt)
        assigned |= m

    rest = ~assigned
    cnt = int(rest.sum())
    if cnt > 0:
        # 兜底：非 12–18 岁用户或异常数据（如 D 非 1/2/3）
        v[rest] = _truncated_normal(rng, 0.2, np.sqrt(0.05), 0.0, 1.0, cnt)
    return v


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    users = pd.read_csv(USERS_PATH)
    videos = pd.read_csv(VIDEOS_PATH)

    required_u = {"user_id", "age"}
    required_v = {"video_id", "K", "D", "entertainment_flag", "education_flag"}
    if not required_u.issubset(users.columns):
        raise ValueError(f"users.csv 需包含列: {required_u}")
    if not required_v.issubset(videos.columns):
        raise ValueError(f"videos.csv 需包含列: {required_v}")

    n_v = len(videos)
    video_ids = videos["video_id"].to_numpy()
    K_col = videos["K"].to_numpy(dtype=np.int64)
    D_col = videos["D"].to_numpy(dtype=np.int64)
    ent_col = videos["entertainment_flag"].to_numpy(dtype=np.int64)
    edu_col = videos["education_flag"].to_numpy(dtype=np.int64)

    first = True
    for start in range(0, len(users), USER_CHUNK_SIZE):
        chunk = users.iloc[start : start + USER_CHUNK_SIZE]
        n_u = len(chunk)
        # 每行重复 n_v 次，与视频对齐
        ages = np.repeat(chunk["age"].to_numpy(dtype=np.float64), n_v)
        K_rep = np.tile(K_col, n_u)
        D_rep = np.tile(D_col, n_u)
        ent_rep = np.tile(ent_col, n_u)
        edu_rep = np.tile(edu_col, n_u)
        user_ids = np.repeat(chunk["user_id"].to_numpy(), n_v)
        vids = np.tile(video_ids, n_u)

        v = _assign_v_for_chunk(ages, D_rep, K_rep, ent_rep, edu_rep, rng)
        out = pd.DataFrame({"user_id": user_ids, "video_id": vids, "v": v})
        out.to_csv(
            OUTPUT_PATH,
            mode="w" if first else "a",
            header=first,
            index=False,
            float_format="%.8f",
        )
        first = False

    print(f"已写入 {OUTPUT_PATH}，共 {len(users) * n_v} 行。")


def _verify_stats_against_theory(value_csv: str = OUTPUT_PATH) -> None:
    """对照 user×video 结果与截断后理论矩（分块读 CSV，控制内存）。"""
    users = pd.read_csv(USERS_PATH)
    # users.csv 与 videos.csv 均含列名 "K"（含义不同），为避免 merge 后列名冲突，先重命名视频侧列
    videos = pd.read_csv(VIDEOS_PATH).rename(columns={"K": "video_K"})

    # 每类：n, sum, sumsq（总体方差 = sumsq/n - mean^2）
    keys = ["y_game_s", "y_funny_s", "y_edu_l", "o_game_s", "o_edu_l"]
    n_acc = {k: 0 for k in keys}
    sum_acc = {k: 0.0 for k in keys}
    sumsq_acc = {k: 0.0 for k in keys}

    chunk_rows = 400_000
    reader = pd.read_csv(value_csv, usecols=["user_id", "video_id", "v"], chunksize=chunk_rows)
    for uv in reader:
        m = uv.merge(users, on="user_id", how="left").merge(videos, on="video_id", how="left")
        age = m["age"].to_numpy(dtype=np.float64)
        D = m["D"].to_numpy(dtype=np.int64)
        ent = m["entertainment_flag"].to_numpy(dtype=np.int64) == 1
        edu = m["education_flag"].to_numpy(dtype=np.int64) == 1
        k = m["video_K"].to_numpy(dtype=np.int64)
        young = (age >= 12.0) & (age < 15.0)
        older = (age >= 15.0) & (age <= 18.0)
        short = D == 1
        long_ = D == 3
        edu_only = edu & ~ent
        game = ent & (k == 2)
        funny = ent & (k == 3)
        masks = {
            "y_game_s": young & game & short,
            "y_funny_s": young & funny & short,
            "y_edu_l": young & edu_only & long_,
            "o_game_s": older & game & short,
            "o_edu_l": older & edu_only & long_,
        }
        vv = m["v"].to_numpy(dtype=np.float64)
        for k, mask in masks.items():
            if not mask.any():
                continue
            sub = vv[mask]
            c = int(sub.size)
            n_acc[k] += c
            sum_acc[k] += float(sub.sum())
            sumsq_acc[k] += float(np.dot(sub, sub))

    spec_meta: list[tuple[str, str, float, float]] = [
        ("12-14 游戏+短", "y_game_s", 0.90, np.sqrt(0.10)),
        ("12-14 搞笑+短", "y_funny_s", 0.60, np.sqrt(0.10)),
        ("12-14 教育+长", "y_edu_l", 0.20, np.sqrt(0.05)),
        ("15-18 游戏+短", "o_game_s", 0.95, np.sqrt(0.10)),
        ("15-18 教育+长", "o_edu_l", 0.35, np.sqrt(0.05)),
    ]

    print("--- 与理论截断矩对照（μ、σ 为截断前参数）---")
    for label, key, mu, sig in spec_meta:
        n = n_acc[key]
        if n == 0:
            print(f"{label}: 无样本")
            continue
        mean_s = sum_acc[key] / n
        var_s = max(sumsq_acc[key] / n - mean_s * mean_s, 0.0)
        ex_t, vx_t = marginal_moments_truncated_gaussian(mu, sig, 0.0, 1.0)
        print(
            f"{label}: n={n} 样本 mean={mean_s:.6f} var={var_s:.6f} | "
            f"理论 E={ex_t:.6f} Var={vx_t:.6f} (parent mu={mu}, sig2={sig**2})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 user×video 截断正态价值 v")
    parser.add_argument(
        "--verify-stats",
        nargs="?",
        const=OUTPUT_PATH,
        metavar="CSV",
        help=f"不生成数据；读取 CSV（默认 {OUTPUT_PATH}）与 users/videos 对照理论截断矩",
    )
    args = parser.parse_args()
    if args.verify_stats is not None:
        _verify_stats_against_theory(args.verify_stats)
        sys.exit(0)
    main()
