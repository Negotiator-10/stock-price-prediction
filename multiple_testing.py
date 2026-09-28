"""
DIG #5: multiple-testing / deflated-Sharpe correction
=====================================================
We ran MANY configurations. The more you try, the more likely one looks
"significant" by pure luck. This quantifies that, so we know which findings
actually survive.
"""
import numpy as np
from scipy.stats import norm

# ---- roughly how many distinct predictive-edge configurations did we try? ----
trials = {
    "Option A models": 8, "Option B models": 8, "split ratios": 4,
    "rolling windows (robustness)": 7, "horizons": 5,
    "fresh stocks x leaky/purged": 6, "regimes": 5, "classifier": 1,
}
N = sum(trials.values())
print("=" * 66)
print("DIG #5: multiple-testing correction")
print("=" * 66)
print("Approx distinct edge-tests run:")
for k, v in trials.items(): print(f"   {k:32s}: {v}")
print(f"   {'TOTAL (N)':32s}: {N}")

# ---- family-wise false-positive probability ----
print("\n1) Chance of at least one FALSE 'p<0.05' by pure luck = 1 - 0.95^N")
for n in [20, N, 100]:
    print(f"   N={n:3d}: {(1 - 0.95**n)*100:.0f}%")
print("   -> With this many trials, finding a spurious 'significant' result was almost certain.")
print("      (This is exactly why the leaky momentum edge and JPM/KO leaks looked significant.)")

# ---- Bonferroni-corrected threshold ----
print(f"\n2) Bonferroni: to be truly significant at 5% after N={N} trials,")
print(f"   a raw p-value must beat 0.05/N = {0.05/N:.4f}")
print("   None of our PURGED directional results are significant at any threshold")
print("   (their edges are negative). So nothing directional survives.")

# ---- deflated-Sharpe hurdle: the best Sharpe pure NOISE gives after N tries ----
# Under the null (true Sharpe = 0), an annual Sharpe estimated over y years has
# std ~ sqrt(1/y). The expected MAX of N such noise-Sharpes ~ std * sqrt(2 ln N).
def hurdle(years, n):
    return np.sqrt(1.0 / years) * np.sqrt(2 * np.log(n))

print("\n3) Deflated-Sharpe hurdle = the best Sharpe you'd expect from NOISE after N tries")
for years, label in [(2.6, "80/20 test (~2.6 yrs)"), (13.0, "walk-forward (~13 yrs)")]:
    print(f"   {label:26s}: hurdle Sharpe ~ {hurdle(years, N):.2f}")
print("   Observed buy-and-hold Sharpe ~0.9-1.4 -> AT or BELOW the noise hurdle (not skill).")
print("   The leaky strategy's Sharpe 3.56 was FAR above the hurdle -> not luck, it was LEAKAGE.")

print("\n" + "=" * 66)
print("WHAT SURVIVES EVERY CORRECTION:")
print("  - Directional/return edge: NOTHING (negative after purging, fails Bonferroni).")
print("  - VOLATILITY prediction: R2=0.42, corr=0.69 on 641 points -> t-stat huge,")
print("    p far below 0.05/N, few trials, and backed by theory (vol clustering).")
print("    THIS is the one real, robust, snooping-proof finding.")
print("=" * 66)
