"""
main.py
=======
Entry point for the Climate Temperature Reconstruction project.

Run:
    python main.py

Outputs:
    results/climate_reconstruction_results.png
    results/heatmap.png
    results/curve_fitting.png
    results/per_year_reconstruction.png
    results/error_report.txt
"""

import sys
import os
import numpy as np
from pathlib import Path

# allow imports from src/
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader   import load_data, get_monthly_means, dataset_info, MONTHS, MONTH_NUMS
from numerical_methods import (
    forward_difference_table, print_forward_diff_table,
    newton_forward, lagrange,
    cubic_spline_coeffs, cubic_spline_eval,
    linear_spline, interpolate_array, interpolate_cubic_array,
    error_summary, rmse,
    least_squares_linear, least_squares_polynomial,
    least_squares_exponential, least_squares_power,
    thomas_algorithm
)
from plots import (
    plot_main_results, plot_heatmap,
    plot_curve_fitting, plot_per_year
)

# ── paths ────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent
DATA    = ROOT / "data"  / "kathmandu_temperature_station1030.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

os.chdir(ROOT)   # make relative paths work for plots.py


def section(title):
    w = 64
    print()
    print("═" * w)
    print(f"  {title}")
    print("═" * w)


def run():
    # ──────────────────────────────────────────────────────
    #  LOAD & INSPECT
    # ──────────────────────────────────────────────────────
    section("0. Dataset Overview")
    df          = load_data(DATA)
    mean_temps  = get_monthly_means(df)
    annual_means = df['Annual_Mean'].values
    years        = df['Year'].values.astype(float)
    dataset_info(df)

    # ──────────────────────────────────────────────────────
    #  FORWARD DIFFERENCE TABLE  (Unit 4)
    # ──────────────────────────────────────────────────────
    section("1. Forward Difference Table  (Unit 4 — Newton Forward)")
    print("  Using multi-year monthly mean temperatures as input\n")
    print_forward_diff_table(MONTH_NUMS, mean_temps, max_order=5)

    # ──────────────────────────────────────────────────────
    #  THOMAS ALGORITHM DEMO  (Unit 3)
    # ──────────────────────────────────────────────────────
    section("2. Thomas Algorithm Demo  (Unit 3)")
    print("  Solving tridiagonal system for cubic spline M-values\n")
    M = cubic_spline_coeffs(MONTH_NUMS, mean_temps)
    print("  Cubic spline second derivatives M[i]:")
    for i, (m, mn) in enumerate(zip(MONTH_NUMS, M)):
        print(f"    Month {int(m):>2} ({MONTHS[i]}):  M = {m:>6.4f}")

    # ──────────────────────────────────────────────────────
    #  INTERPOLATION + ERROR ANALYSIS  (Units 1 & 4)
    # ──────────────────────────────────────────────────────
    section("3. Interpolation & Error Analysis  (Units 1 & 4)")

    for n_known in [4, 6, 8, 10]:
        idx   = np.linspace(0, 11, n_known, dtype=int)
        x_k   = MONTH_NUMS[idx]
        y_k   = mean_temps[idx]
        M_cs  = cubic_spline_coeffs(x_k, y_k)

        y_nf  = interpolate_array(newton_forward, MONTH_NUMS, x_k, y_k)
        y_lag = interpolate_array(lagrange,        MONTH_NUMS, x_k, y_k)
        y_cs  = np.array([cubic_spline_eval(x, x_k, y_k, M_cs) for x in MONTH_NUMS])
        y_ls  = interpolate_array(linear_spline,   MONTH_NUMS, x_k, y_k)

        print(f"\n  ── n_known = {n_known} ──────────────────────────────────")
        print(f"  {'Method':<20} {'RMSE':>8} {'MAE':>8} {'Max|e|':>8} {'AvgRel%':>9}")
        print(f"  {'-'*57}")
        for name, yp in [('Newton Forward', y_nf), ('Lagrange', y_lag),
                          ('Cubic Spline',  y_cs), ('Linear Spline', y_ls)]:
            es = error_summary(mean_temps, yp, name)
            print(f"  {name:<20} {es['RMSE']:>8.4f} {es['MAE']:>8.4f} "
                  f"{es['Max |e|']:>8.4f} {es['Avg Rel%']:>9.4f}")

    # Month-by-month detail for cubic spline (n=6)
    print("\n  ── Cubic Spline month-by-month detail  (n_known = 6) ──")
    idx  = np.linspace(0, 11, 6, dtype=int)
    x_k  = MONTH_NUMS[idx];  y_k = mean_temps[idx]
    M_cs = cubic_spline_coeffs(x_k, y_k)
    y_cs = np.array([cubic_spline_eval(x, x_k, y_k, M_cs) for x in MONTH_NUMS])
    print(f"  {'Month':<6} {'True (°C)':>10} {'Pred (°C)':>10} "
          f"{'|Error|':>10} {'Rel %':>8} {'Known?':>7}")
    print(f"  {'-'*55}")
    known_set = set(idx)
    for i, (m, yt, yp) in enumerate(zip(MONTHS, mean_temps, y_cs)):
        ae = abs(yp - yt)
        re = ae / abs(yt) * 100
        kn = '  ✓' if i in known_set else ''
        print(f"  {m:<6} {yt:>10.2f} {yp:>10.4f} {ae:>10.4f} {re:>8.4f}{kn}")

    # ──────────────────────────────────────────────────────
    #  RUNGE'S PHENOMENON  (Unit 1 — truncation/round-off)
    # ──────────────────────────────────────────────────────
    section("4. Runge's Phenomenon  (Unit 1 — Error Growth)")
    x_fine = np.linspace(1, 12, 500)
    y_ref  = np.interp(x_fine, MONTH_NUMS, mean_temps)
    print(f"\n  {'n pts':<8} {'Lagrange RMSE':>14} {'Cubic RMSE':>12} {'Newton RMSE':>13}")
    print(f"  {'-'*52}")
    for nk in range(3, 12):
        idx_r = np.linspace(0, 11, nk, dtype=int)
        xk_r  = MONTH_NUMS[idx_r];  yk_r = mean_temps[idx_r]
        M_r   = cubic_spline_coeffs(xk_r, yk_r)
        y_lag_r = np.array([lagrange(x, xk_r, yk_r)  for x in x_fine])
        y_cs_r  = np.array([cubic_spline_eval(x, xk_r, yk_r, M_r) for x in x_fine])
        y_nf_r  = np.array([newton_forward(x, xk_r, yk_r) for x in x_fine])
        print(f"  {nk:<8} {rmse(y_ref,y_lag_r):>14.6f} "
              f"{rmse(y_ref,y_cs_r):>12.6f} {rmse(y_ref,y_nf_r):>13.6f}")

    # ──────────────────────────────────────────────────────
    #  CURVE FITTING  (Unit 4.3)
    # ──────────────────────────────────────────────────────
    section("5. Least-Squares Curve Fitting  (Unit 4.3)")
    x_off = years - years[0]   # offset for numerical stability

    a, b, r2 = least_squares_linear(years, annual_means)
    print(f"\n  Linear fit:      T = {a:.5f}·year + ({b:.3f})")
    print(f"  Warming rate:    {a*10:+.4f} °C per decade")
    print(f"  R²:              {r2:.4f}")

    coeffs = least_squares_polynomial(x_off, annual_means, degree=2)
    print(f"\n  Quadratic fit:   T = {coeffs[0]:.4f} + {coeffs[1]:.5f}·x + {coeffs[2]:.6f}·x²")

    a_e, b_e = least_squares_exponential(x_off + 1, annual_means)
    print(f"\n  Exponential fit: T = {a_e:.4f} · e^({b_e:.5f}·x)")

    a_p, b_p = least_squares_power(x_off + 1, annual_means)
    print(f"\n  Power fit:       T = {a_p:.4f} · x^({b_p:.5f})")

    # ──────────────────────────────────────────────────────
    #  GENERATE FIGURES
    # ──────────────────────────────────────────────────────
    section("6. Generating Figures")
    plot_main_results(
        mean_temps, annual_means, years, n_known=6,
        save_path=str(RESULTS / "climate_reconstruction_results.png")
    )
    plot_heatmap(df,
        save_path=str(RESULTS / "heatmap.png")
    )
    plot_curve_fitting(years, annual_means,
        save_path=str(RESULTS / "curve_fitting.png")
    )
    plot_per_year(df, n_known=6,
        save_path=str(RESULTS / "per_year_reconstruction.png")
    )

    # ──────────────────────────────────────────────────────
    #  SAVE TEXT REPORT
    # ──────────────────────────────────────────────────────
    section("7. Saving Error Report")
    report_lines = []
    report_lines.append("Climate Temperature Reconstruction — Error Report")
    report_lines.append("Kathmandu Station 1030, 2004–2014")
    report_lines.append("=" * 60)
    for n_known in [4, 6, 8, 10]:
        idx   = np.linspace(0, 11, n_known, dtype=int)
        x_k   = MONTH_NUMS[idx];  y_k = mean_temps[idx]
        M_cs  = cubic_spline_coeffs(x_k, y_k)
        y_nf  = interpolate_array(newton_forward, MONTH_NUMS, x_k, y_k)
        y_lag = interpolate_array(lagrange, MONTH_NUMS, x_k, y_k)
        y_cs  = np.array([cubic_spline_eval(x, x_k, y_k, M_cs) for x in MONTH_NUMS])
        y_ls  = interpolate_array(linear_spline, MONTH_NUMS, x_k, y_k)
        report_lines.append(f"\nn_known = {n_known}")
        report_lines.append(f"{'Method':<20} {'RMSE':>8} {'MAE':>8} {'Max|e|':>8}")
        report_lines.append("-" * 48)
        for name, yp in [('Newton Forward', y_nf), ('Lagrange', y_lag),
                          ('Cubic Spline', y_cs),  ('Linear Spline', y_ls)]:
            es = error_summary(mean_temps, yp)
            report_lines.append(f"{name:<20} {es['RMSE']:>8.4f} "
                                 f"{es['MAE']:>8.4f} {es['Max |e|']:>8.4f}")

    report_path = RESULTS / "error_report.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"  [Saved] {report_path}")

    section("✓ Project Complete")
    print("  All outputs saved to results/\n")


if __name__ == "__main__":
    run()
