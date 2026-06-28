"""
plots.py
========
All matplotlib visualisations for the Climate Reconstruction project.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from data_loader import MONTHS, MONTH_NUMS
from numerical_methods import (
    newton_forward, lagrange,
    cubic_spline_coeffs, cubic_spline_eval,
    linear_spline, interpolate_array, interpolate_cubic_array,
    error_summary, rmse, forward_difference_table,
    least_squares_linear, least_squares_polynomial
)

# ── colour palette ──────────────────────────────────────────────
C = {
    'newton':  '#378ADD',   # blue
    'lagrange':'#1D9E75',   # teal
    'cubic':   '#D85A30',   # coral
    'linear':  '#888780',   # grey
    'true':    '#2C2C2A',   # near-black
    'known':   '#E24B4A',   # red dots
    'trend':   '#D85A30',
    'bar':     '#B5D4F4',
    'bar_edge':'#378ADD',
}
LSTYLE = {'newton':'-','lagrange':'-','cubic':'-','linear':'--'}
LABELS = {
    'newton': 'Newton Forward',
    'lagrange':'Lagrange',
    'cubic':  'Cubic Spline',
    'linear': 'Linear Spline',
}


# ── helpers ─────────────────────────────────────────────────────
def _eval_all(x_arr, x_k, y_k):
    M = cubic_spline_coeffs(x_k, y_k)
    return {
        'newton':  interpolate_array(newton_forward, x_arr, x_k, y_k),
        'lagrange':interpolate_array(lagrange,       x_arr, x_k, y_k),
        'cubic':   np.array([cubic_spline_eval(x, x_k, y_k, M) for x in x_arr]),
        'linear':  interpolate_array(linear_spline,  x_arr, x_k, y_k),
    }

def _style(ax):
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[['top','right']].set_visible(False)


# ═══════════════════════════════════════════════════════════════
#  FIGURE 1 — Main 5-panel results figure
# ═══════════════════════════════════════════════════════════════
def plot_main_results(mean_temps, annual_means, years,
                      n_known=6, save_path=None):
    idx   = np.linspace(0, 11, n_known, dtype=int)
    x_k   = MONTH_NUMS[idx]
    y_k   = mean_temps[idx]
    preds = _eval_all(MONTH_NUMS, x_k, y_k)

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        "Climate Temperature Reconstruction — Kathmandu Station 1030\n"
        "Open Data Nepal · CBS Statistical Year Book 2015 · 2004–2014",
        fontsize=13, fontweight='bold', y=0.985
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.46, wspace=0.32)

    # A — Interpolation comparison
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(MONTH_NUMS, mean_temps, 'o--', color=C['true'],
             lw=1.8, ms=5, label='Multi-year mean (reference)', zorder=5)
    for key, y_pred in preds.items():
        ax1.plot(MONTH_NUMS, y_pred, LSTYLE[key],
                 color=C[key], lw=2.2, label=LABELS[key])
    ax1.scatter(x_k, y_k, color=C['known'], zorder=10,
                s=70, label=f'Known points ({n_known})', edgecolors='white', lw=1)
    ax1.set_xlabel('Month'); ax1.set_ylabel('Temperature (°C)')
    ax1.set_title(f'A — All Four Interpolation Methods  ({n_known} known points)')
    ax1.set_xticks(MONTH_NUMS); ax1.set_xticklabels(MONTHS)
    ax1.legend(fontsize=8.5, ncol=3, framealpha=0.9); _style(ax1)

    # B — Absolute error
    ax2 = fig.add_subplot(gs[1, 0])
    for key, y_pred in preds.items():
        ax2.plot(MONTH_NUMS, np.abs(y_pred - mean_temps),
                 '-o', color=C[key], lw=1.8, ms=4, label=LABELS[key])
    ax2.set_xlabel('Month'); ax2.set_ylabel('|Error| (°C)')
    ax2.set_title('B — Absolute Error at Each Month')
    ax2.set_xticks(MONTH_NUMS); ax2.set_xticklabels(MONTHS, fontsize=8)
    ax2.legend(fontsize=8); _style(ax2)

    # C — RMSE vs n_known
    ax3 = fig.add_subplot(gs[1, 1])
    n_vals = list(range(4, 12))
    rmse_by_n = {k: [] for k in preds}
    for nk in n_vals:
        idx_n = np.linspace(0, 11, nk, dtype=int)
        xk_n  = MONTH_NUMS[idx_n]
        yk_n  = mean_temps[idx_n]
        p_n   = _eval_all(MONTH_NUMS, xk_n, yk_n)
        for key, yp in p_n.items():
            rmse_by_n[key].append(rmse(mean_temps, yp))
    for key, vals in rmse_by_n.items():
        ax3.plot(n_vals, vals, '-o', color=C[key],
                 lw=2, ms=5, label=LABELS[key])
    ax3.axvline(n_known, color='grey', lw=1, ls=':', alpha=0.7,
                label=f'Current n={n_known}')
    ax3.set_xlabel('Number of known data points')
    ax3.set_ylabel('RMSE (°C)')
    ax3.set_title("C — RMSE vs. Number of Known Points")
    ax3.legend(fontsize=8); _style(ax3)

    # D — Runge's phenomenon
    ax4 = fig.add_subplot(gs[2, 0])
    x_fine = np.linspace(1, 12, 300)
    y_ref  = np.interp(x_fine, MONTH_NUMS, mean_temps)
    n_range = list(range(3, 12))
    runge_rmse = {k: [] for k in preds}
    for nk in n_range:
        idx_r = np.linspace(0, 11, nk, dtype=int)
        xk_r  = MONTH_NUMS[idx_r]
        yk_r  = mean_temps[idx_r]
        p_r   = _eval_all(x_fine, xk_r, yk_r)
        for key, yp in p_r.items():
            runge_rmse[key].append(rmse(y_ref, yp))
    for key, vals in runge_rmse.items():
        ax4.plot(n_range, vals, '-o', color=C[key],
                 lw=2, ms=5, label=LABELS[key])
    ax4.set_xlabel('Number of interpolation points')
    ax4.set_ylabel('RMSE vs. reference curve (°C)')
    ax4.set_title("D — Runge's Phenomenon: Stability vs. Polynomial Degree")
    ax4.legend(fontsize=8); _style(ax4)

    # E — Warming trend
    ax5 = fig.add_subplot(gs[2, 1])
    a, b, r2 = least_squares_linear(years, annual_means)
    y_fit = a * years + b
    ax5.bar(years, annual_means, color=C['bar'],
            edgecolor=C['bar_edge'], linewidth=0.8, label='Annual mean temp')
    ax5.plot(years, y_fit, '-', color=C['trend'], lw=2.5,
             label=f'Trend: {a*10:+.3f} °C/decade  (R²={r2:.3f})')
    ax5.set_xlabel('Year'); ax5.set_ylabel('Mean Temperature (°C)')
    ax5.set_title('E — Annual Mean Temperature + Warming Trend (Unit 4.3)')
    ax5.legend(fontsize=8.5)
    ax5.set_xticks(years); ax5.set_xticklabels(years.astype(int), rotation=45)
    _style(ax5)

    plt.savefig(save_path or 'results/climate_reconstruction_results.png',
                dpi=150, bbox_inches='tight')
    print(f"[Saved] {save_path or 'results/climate_reconstruction_results.png'}")
    plt.close()


# ═══════════════════════════════════════════════════════════════
#  FIGURE 2 — Year-by-year heatmap of monthly temperatures
# ═══════════════════════════════════════════════════════════════
def plot_heatmap(df, save_path=None):
    data = df[MONTHS].values
    years = df['Year'].values

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(data, cmap='RdYlBu_r', aspect='auto',
                   vmin=data.min(), vmax=data.max())
    plt.colorbar(im, ax=ax, label='Temperature (°C)', shrink=0.8)
    ax.set_xticks(range(12)); ax.set_xticklabels(MONTHS)
    ax.set_yticks(range(len(years))); ax.set_yticklabels(years.astype(int))
    ax.set_xlabel('Month'); ax.set_ylabel('Year')
    ax.set_title('Monthly Mean Temperature Heatmap — Kathmandu Station 1030\n'
                 'Open Data Nepal · 2004–2014')
    for i in range(len(years)):
        for j in range(12):
            ax.text(j, i, f'{data[i,j]:.1f}', ha='center', va='center',
                    fontsize=7.5, color='white' if data[i,j] > 20 else 'black')
    plt.tight_layout()
    plt.savefig(save_path or 'results/heatmap.png', dpi=150, bbox_inches='tight')
    print(f"[Saved] {save_path or 'results/heatmap.png'}")
    plt.close()


# ═══════════════════════════════════════════════════════════════
#  FIGURE 3 — Curve fitting showcase (linear, poly, exp, power)
# ═══════════════════════════════════════════════════════════════
def plot_curve_fitting(years, annual_means, save_path=None):
    from numerical_methods import least_squares_exponential, least_squares_power

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Curve Fitting on Annual Mean Temperature — Kathmandu 1030\n'
                 'Unit 4.3: Least-Squares Methods', fontsize=12, fontweight='bold')

    x = years - years[0]   # shift so x starts at 0 for poly/exp stability
    y = annual_means

    panels = []

    # Linear fit
    a, b, r2 = least_squares_linear(years, y)
    y_lin = a * years + b
    panels.append(('Linear  y = a·x + b',
                   years, y_lin,
                   f'a={a:.4f}, b={b:.2f}\nR²={r2:.4f}'))

    # Quadratic fit
    c = least_squares_polynomial(x, y, degree=2)
    y_quad = sum(c[k] * x**k for k in range(3))
    ss_res = np.sum((y - y_quad)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2_q   = 1 - ss_res/ss_tot
    panels.append(('Quadratic  y = a₀+a₁x+a₂x²',
                   x, y_quad,
                   f'a₀={c[0]:.3f}, a₁={c[1]:.4f}, a₂={c[2]:.4f}\nR²={r2_q:.4f}'))

    # Exponential fit
    a_e, b_e = least_squares_exponential(x + 1, y)
    y_exp = a_e * np.exp(b_e * (x + 1))
    ss_res = np.sum((y - y_exp)**2)
    r2_e   = 1 - ss_res/np.sum((y - y.mean())**2)
    panels.append(('Exponential  y = a·eˢˣ',
                   x, y_exp,
                   f'a={a_e:.4f}, b={b_e:.4f}\nR²={r2_e:.4f}'))

    # Power fit
    a_p, b_p = least_squares_power(x + 1, y)
    y_pow = a_p * (x + 1)**b_p
    ss_res = np.sum((y - y_pow)**2)
    r2_p   = 1 - ss_res/np.sum((y - y.mean())**2)
    panels.append(('Power  y = a·xᵇ',
                   x, y_pow,
                   f'a={a_p:.4f}, b={b_p:.4f}\nR²={r2_p:.4f}'))

    x_labels = [years, x, x, x]
    for ax, (title, xp, yp, info), xl in zip(axes.flat, panels, x_labels):
        ax.scatter(xl, y, color=C['known'], s=60, zorder=5, label='Data')
        ax.plot(xl, yp, '-', color=C['trend'], lw=2.5, label='Fit')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Year' if xl is years else 'Year offset')
        ax.set_ylabel('Annual mean temp (°C)')
        ax.text(0.04, 0.93, info, transform=ax.transAxes,
                fontsize=8.5, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
        ax.legend(fontsize=8); _style(ax)

    plt.tight_layout()
    plt.savefig(save_path or 'results/curve_fitting.png',
                dpi=150, bbox_inches='tight')
    print(f"[Saved] {save_path or 'results/curve_fitting.png'}")
    plt.close()


# ═══════════════════════════════════════════════════════════════
#  FIGURE 4 — Per-year interpolation quality grid
# ═══════════════════════════════════════════════════════════════
def plot_per_year(df, n_known=6, save_path=None):
    years = df['Year'].values
    ny    = len(years)
    ncols = 4
    nrows = (ny + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2),
                             sharey=False)
    fig.suptitle(f'Per-Year Cubic Spline Reconstruction  (n={n_known} known points)\n'
                 'Kathmandu Station 1030', fontsize=12, fontweight='bold')
    axes = axes.flat

    for ax, yr in zip(axes, years):
        y_full = df[df['Year'] == yr][MONTHS].values[0]
        idx    = np.linspace(0, 11, n_known, dtype=int)
        x_k    = MONTH_NUMS[idx]
        y_k    = y_full[idx]
        y_cs   = interpolate_cubic_array(MONTH_NUMS, x_k, y_k)

        ax.plot(MONTH_NUMS, y_full, 'o--', color=C['true'],
                lw=1.4, ms=4, label='True')
        ax.plot(MONTH_NUMS, y_cs, '-', color=C['cubic'],
                lw=2, label='Cubic Spline')
        ax.scatter(x_k, y_k, color=C['known'], zorder=5, s=40)
        rmse_val = rmse(y_full, y_cs)
        ax.set_title(f'{int(yr)}  (RMSE={rmse_val:.2f}°C)', fontsize=9)
        ax.set_xticks(MONTH_NUMS[::2])
        ax.set_xticklabels(MONTHS[::2], fontsize=7)
        _style(ax)

    # hide unused axes
    for ax in list(axes)[ny:]:
        ax.set_visible(False)

    handles = [Line2D([0],[0], color=C['true'],   lw=1.5, ls='--', label='True'),
               Line2D([0],[0], color=C['cubic'],  lw=2,             label='Cubic Spline'),
               Line2D([0],[0], marker='o', color=C['known'], ls='', ms=6, label='Known pts')]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path or 'results/per_year_reconstruction.png',
                dpi=150, bbox_inches='tight')
    print(f"[Saved] {save_path or 'results/per_year_reconstruction.png'}")
    plt.close()
