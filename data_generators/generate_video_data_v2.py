"""
使用 NumPy、Pandas 生成实验用视频数据，并保存为 CSV。

在 generate_video_data.py 基础上，采用 update_video_risk_only.py 中的沉迷风险参数：

- 类别 K：1 教育 15%，2 游戏 30%，3 搞笑 40%，4 生活 15%
- 播放时长 t（分钟）：对数正态分布，均值约 2，截断到 [0.5, 60]
- 时长区间 D：≤3→1 短，(3,15]→2 中，>15→3 长
- 娱乐标志：K∈{2,3} 为 1
- 教育标志：K=1 为 1
- 沉迷风险 r = 基础内容风险 × 时长诱导乘数
    基础内容风险（K）：教育 1，游戏 3.5，搞笑 3，生活 2
    时长诱导乘数（D）：短 1.5，中 1.0，长 0.8
- 分发成本 c = 0.001 × t
- 最大曝光次数 N：Type I Pareto（xmin=18, α=1.28），floor 后截断到 [20, 1000]
  numpy Generator.pareto 为 Lomax，故 N 的连续原像为 xmin * (1 + Lomax(α))
  使用独立种子 N_RNG_SEED，不占用主 RNG（类别/时长）的随机流
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_VIDEOS = 5_000
RNG_SEED = 42
N_RNG_SEED = 20260411  # N 单独抽样，与主种子 42 解耦
OUTPUT_PATH = "videos.csv"

# 对数正态：ln(t) ~ N(mu_ln, sigma_ln)，使 E[t] ≈ exp(mu_ln + sigma_ln^2/2) ≈ 2（截断前）
SIGMA_LN = 1.0
MU_LN = float(np.log(2.0) - 0.5 * SIGMA_LN**2)

T_MIN, T_MAX = 0.5, 60.0

# 类别与占比：1 教育 15%，2 游戏 30%，3 搞笑 40%，4 生活 15%
CATEGORIES = np.array([1, 2, 3, 4], dtype=np.int8)
CAT_PROBS = np.array([0.15, 0.30, 0.40, 0.15], dtype=np.float64)

# 基础内容风险：教育 1，游戏 3.5，搞笑 3，生活 2（索引 0 占位，1..4 有效）
BASE_RISK = np.array([0.0, 1.0, 3.5, 3.0, 2.0], dtype=np.float64)

# D→时长诱导乘数：1→1.5，2→1.0，3→0.8
MULT_BY_D = np.array([0.0, 1.5, 1.0, 0.8], dtype=np.float64)

# Type I Pareto：xmin * (1 + Lomax(α))；floor 后 clip 到 [N_MIN, N_MAX]
N_XMIN = 18.0
N_ALPHA = 1.28
N_MIN, N_MAX = 20, 1000


def sample_duration(rng: np.random.Generator, n: int) -> np.ndarray:
    """对数正态采样并截断到 [T_MIN, T_MAX]。"""
    t = rng.lognormal(mean=MU_LN, sigma=SIGMA_LN, size=n)
    return np.clip(t, T_MIN, T_MAX)


def duration_bin(t: np.ndarray) -> np.ndarray:
    """t → D：1 短 ≤3，2 中 (3,15]，3 长 >15。"""
    d = np.empty_like(t, dtype=np.int8)
    d[t <= 3.0] = 1
    d[(t > 3.0) & (t <= 15.0)] = 2
    d[t > 15.0] = 3
    return d


def compute_r(K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """r = 基础内容风险(K) × 时长诱导乘数(D)。"""
    base = BASE_RISK[K.astype(np.int64)]
    mult = MULT_BY_D[D.astype(np.int64)]
    return np.round(base * mult, 6)


def sample_exposure_cap(rng: np.random.Generator, n: int) -> np.ndarray:
    """Type I Pareto 抽样最大曝光次数 N，floor 后截断到 [N_MIN, N_MAX]。"""
    raw = N_XMIN * (1.0 + rng.pareto(N_ALPHA, size=n))
    return np.clip(np.floor(raw).astype(np.int64), N_MIN, N_MAX)


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    K = rng.choice(CATEGORIES, size=N_VIDEOS, replace=True, p=CAT_PROBS).astype(np.int8)
    t = sample_duration(rng, N_VIDEOS)
    D = duration_bin(t)

    entertainment_flag = np.isin(K, [2, 3]).astype(np.int8)
    education_flag = (K == 1).astype(np.int8)

    r = compute_r(K, D)

    cost_coef = 0.001
    c = cost_coef * t

    n_rng = np.random.default_rng(N_RNG_SEED)
    N = sample_exposure_cap(n_rng, N_VIDEOS)

    df = pd.DataFrame(
        {
            "video_id": np.arange(1, N_VIDEOS + 1, dtype=np.int64),
            "K": K,
            "t": np.round(t, 4),
            "D": D,
            "entertainment_flag": entertainment_flag,
            "education_flag": education_flag,
            "r": r,
            "c": np.round(c, 6),
            "N": N,
        }
    )

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"已生成 {N_VIDEOS} 条视频数据，保存至 {OUTPUT_PATH}")
    print(
        "类别占比（%）:",
        df["K"].value_counts(normalize=True).sort_index().mul(100).round(2).to_dict(),
    )
    print(f"时长 t：均值 {df['t'].mean():.3f} 分钟，min={df['t'].min():.3f}, max={df['t'].max():.3f}")
    print(
        "时长区间 D 占比（%）:",
        df["D"].value_counts(normalize=True).sort_index().mul(100).round(2).to_dict(),
    )
    print(f"沉迷风险 r：min={df['r'].min():.4f}, max={df['r'].max():.4f}")
    print(f"分发成本 c：min={df['c'].min():.6f}, max={df['c'].max():.6f}（c = {cost_coef} × t）")
    print(
        f"最大曝光 N：均值 {df['N'].mean():.2f}，"
        f"min={int(df['N'].min())}, max={int(df['N'].max())}，"
        f"P50={int(df['N'].median())}"
        f"（Pareto xmin={N_XMIN:g}, α={N_ALPHA}，clip [{N_MIN}, {N_MAX}]）"
    )


if __name__ == "__main__":
    main()
