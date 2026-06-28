"""
numerical_methods.py
====================
All numerical methods implemented from scratch for the
Climate Temperature Reconstruction project.

Covers:
  Unit 1  — Error metrics (absolute, relative, RMSE)
  Unit 3  — Thomas Algorithm (tridiagonal solver)
  Unit 4  — Newton Forward, Lagrange, Cubic Spline, Linear Spline
  Unit 4.3— Least-squares linear & polynomial curve fitting
"""

import numpy as np


# ══════════════════════════════════════════════════════════════
#  UNIT 1 — ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════

def absolute_error(y_true, y_pred):
    """Element-wise absolute error."""
    return np.abs(np.array(y_true) - np.array(y_pred))

def relative_error_percent(y_true, y_pred):
    """Element-wise relative error in percent."""
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    return np.abs(yt - yp) / (np.abs(yt) + 1e-12) * 100.0

def rmse(y_true, y_pred):
    """Root Mean Square Error."""
    diff = np.array(y_true) - np.array(y_pred)
    return float(np.sqrt(np.mean(diff ** 2)))

def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

def error_summary(y_true, y_pred, method_name=""):
    """Return a dict of all error metrics."""
    ae = absolute_error(y_true, y_pred)
    re = relative_error_percent(y_true, y_pred)
    return {
        "method":   method_name,
        "RMSE":     rmse(y_true, y_pred),
        "MAE":      mae(y_true, y_pred),
        "Max |e|":  float(ae.max()),
        "Avg Rel%": float(re.mean()),
        "Max Rel%": float(re.max()),
    }


# ══════════════════════════════════════════════════════════════
#  UNIT 3 — THOMAS ALGORITHM (tridiagonal solver)
# ══════════════════════════════════════════════════════════════

def thomas_algorithm(lower, diag, upper, rhs):
    """
    Solve a tridiagonal system  A·x = rhs  using the Thomas Algorithm.

    Parameters
    ----------
    lower : array-like, length n   (lower[0] unused)
    diag  : array-like, length n
    upper : array-like, length n   (upper[-1] unused)
    rhs   : array-like, length n

    Returns
    -------
    x : ndarray, length n
    """
    n = len(rhs)
    a = np.array(lower, dtype=float)
    b = np.array(diag,  dtype=float)
    c = np.array(upper, dtype=float)
    d = np.array(rhs,   dtype=float)

    # Forward sweep
    c_ = np.zeros(n)
    d_ = np.zeros(n)
    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * c_[i - 1]
        c_[i] = c[i] / denom if i < n - 1 else 0.0
        d_[i] = (d[i] - a[i] * d_[i - 1]) / denom

    # Back substitution
    x = np.zeros(n)
    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]
    return x


# ══════════════════════════════════════════════════════════════
#  UNIT 4 — FINITE DIFFERENCES
# ══════════════════════════════════════════════════════════════

def forward_difference_table(y):
    """
    Build the full forward difference table.

    Returns
    -------
    table : ndarray shape (n, n)
        table[:, 0] = y
        table[:, k] = k-th forward difference
    """
    y = np.array(y, dtype=float)
    n = len(y)
    table = np.zeros((n, n))
    table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = table[i + 1, j - 1] - table[i, j - 1]
    return table

def print_forward_diff_table(x_pts, y_pts, max_order=5):
    """Pretty-print the forward difference table."""
    tbl = forward_difference_table(y_pts)
    n   = len(x_pts)
    orders = min(max_order + 1, n)
    header = f"{'x':>8} | {'y':>7} |" + "".join(f" {'Δ'+str(k)+'y':>8} |" for k in range(1, orders))
    print(header)
    print("-" * len(header))
    for i in range(n):
        row = f"{x_pts[i]:>8.2f} | {tbl[i,0]:>7.4f} |"
        for k in range(1, orders):
            if i + k < n or True:   # show cell if computed
                try:
                    row += f" {tbl[i,k]:>8.4f} |" if i + k <= n - 1 else f" {'':>8} |"
                except IndexError:
                    row += f" {'':>8} |"
        print(row)


# ══════════════════════════════════════════════════════════════
#  UNIT 4 — NEWTON FORWARD INTERPOLATION
#  (equally spaced data, uses forward difference table)
# ══════════════════════════════════════════════════════════════

def newton_forward(x, x_pts, y_pts):
    """
    Newton's Forward Difference Interpolation formula.

    Valid for equally-spaced x_pts.
    s = (x - x0) / h,  then uses Gregory-Newton forward formula.
    """
    x_pts = np.array(x_pts, dtype=float)
    y_pts = np.array(y_pts, dtype=float)
    n     = len(x_pts)
    h     = x_pts[1] - x_pts[0]          # assumes uniform spacing
    s     = (x - x_pts[0]) / h
    table = forward_difference_table(y_pts)

    result = table[0, 0]
    s_term = 1.0
    for k in range(1, n):
        s_term *= (s - (k - 1)) / k
        result += s_term * table[0, k]
    return float(result)


# ══════════════════════════════════════════════════════════════
#  UNIT 4 — LAGRANGE INTERPOLATION
#  (unequally spaced data)
# ══════════════════════════════════════════════════════════════

def lagrange(x, x_pts, y_pts):
    """
    Lagrange Interpolation polynomial.

    L(x) = Σ yᵢ · ∏_{j≠i} (x - xⱼ)/(xᵢ - xⱼ)
    """
    x_pts  = np.array(x_pts, dtype=float)
    y_pts  = np.array(y_pts, dtype=float)
    n      = len(x_pts)
    result = 0.0
    for i in range(n):
        term = y_pts[i]
        for j in range(n):
            if j != i:
                term *= (x - x_pts[j]) / (x_pts[i] - x_pts[j])
        result += term
    return float(result)


# ══════════════════════════════════════════════════════════════
#  UNIT 4 — CUBIC SPLINE (natural boundary conditions)
#  Uses Thomas Algorithm internally
# ══════════════════════════════════════════════════════════════

def cubic_spline_coeffs(x_pts, y_pts):
    """
    Compute natural cubic spline second derivatives M using Thomas Algorithm.

    The spline on segment [xᵢ, xᵢ₊₁]:
      S(x) = A·yᵢ + B·yᵢ₊₁ + (A³-A)·Mᵢ·h²/6 + (B³-B)·Mᵢ₊₁·h²/6
      where A=(xᵢ₊₁-x)/h, B=(x-xᵢ)/h

    Returns M : ndarray of length n (second derivatives)
    """
    x = np.array(x_pts, dtype=float)
    y = np.array(y_pts, dtype=float)
    n = len(x)
    h = np.diff(x)

    # Build tridiagonal system
    rhs  = np.zeros(n)
    diag = np.ones(n)
    lo   = np.zeros(n)
    up   = np.zeros(n)

    for i in range(1, n - 1):
        diag[i] = 2.0 * (h[i - 1] + h[i])
        lo[i]   = h[i - 1]
        up[i]   = h[i]
        rhs[i]  = 6.0 * ((y[i + 1] - y[i]) / h[i]
                        - (y[i] - y[i - 1]) / h[i - 1])

    # Natural boundary: M[0] = M[n-1] = 0
    diag[0]  = 1.0;  diag[-1] = 1.0
    rhs[0]   = 0.0;  rhs[-1]  = 0.0

    M = thomas_algorithm(lo, diag, up, rhs)
    return M

def cubic_spline_eval(x, x_pts, y_pts, M):
    """Evaluate cubic spline at scalar x, given precomputed M."""
    x_pts = np.array(x_pts, dtype=float)
    y_pts = np.array(y_pts, dtype=float)
    n = len(x_pts)
    i = int(np.searchsorted(x_pts, x, side='right')) - 1
    i = int(np.clip(i, 0, n - 2))
    h = x_pts[i + 1] - x_pts[i]
    A = (x_pts[i + 1] - x) / h
    B = (x - x_pts[i])     / h
    return float(
        A * y_pts[i] + B * y_pts[i + 1]
        + ((A**3 - A) * M[i] + (B**3 - B) * M[i + 1]) * h**2 / 6.0
    )

def cubic_spline(x, x_pts, y_pts):
    """Convenience wrapper: compute M then evaluate."""
    M = cubic_spline_coeffs(x_pts, y_pts)
    return cubic_spline_eval(x, x_pts, y_pts, M)


# ══════════════════════════════════════════════════════════════
#  UNIT 4 — LINEAR SPLINE (piecewise linear, baseline)
# ══════════════════════════════════════════════════════════════

def linear_spline(x, x_pts, y_pts):
    """Piecewise linear interpolation (baseline)."""
    x_pts = np.array(x_pts, dtype=float)
    y_pts = np.array(y_pts, dtype=float)
    i = int(np.searchsorted(x_pts, x, side='right')) - 1
    i = int(np.clip(i, 0, len(x_pts) - 2))
    t = (x - x_pts[i]) / (x_pts[i + 1] - x_pts[i])
    return float(y_pts[i] * (1 - t) + y_pts[i + 1] * t)


# ══════════════════════════════════════════════════════════════
#  UNIT 4.3 — LEAST-SQUARES CURVE FITTING
# ══════════════════════════════════════════════════════════════

def least_squares_linear(x, y):
    """
    Fit y = a*x + b via normal equations (closed form).

    Returns (a, b, r_squared)
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n   = len(x)
    sx  = x.sum();    sy  = y.sum()
    sx2 = (x**2).sum(); sxy = (x * y).sum()
    a   = (n * sxy - sx * sy) / (n * sx2 - sx**2)
    b   = (sy - a * sx) / n
    y_fit  = a * x + b
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(a), float(b), float(r2)

def least_squares_polynomial(x, y, degree=2):
    """
    Fit polynomial of given degree using normal equations (manual Vandermonde).

    Returns coefficients [a0, a1, ..., a_degree] for
      y = a0 + a1*x + a2*x² + ...
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = degree + 1
    # Build Vandermonde matrix A (n_pts × n_coeffs)
    A = np.column_stack([x**k for k in range(n)])
    # Normal equations: AᵀA · c = Aᵀy
    ATA = A.T @ A
    ATy = A.T @ y
    # Solve using numpy (we already have Thomas for tridiagonal;
    # general system: use Gaussian elimination or numpy)
    coeffs = np.linalg.solve(ATA, ATy)
    return coeffs

def least_squares_exponential(x, y):
    """
    Fit y = a * exp(b*x) via linearisation: ln(y) = ln(a) + b*x
    Returns (a, b)
    """
    x  = np.array(x, dtype=float)
    y  = np.array(y, dtype=float)
    ln_y = np.log(np.clip(y, 1e-10, None))
    b, ln_a, _ = least_squares_linear(x, ln_y)
    return float(np.exp(ln_a)), float(b)

def least_squares_power(x, y):
    """
    Fit y = a * x^b via linearisation: ln(y) = ln(a) + b*ln(x)
    Returns (a, b)
    """
    x    = np.array(x, dtype=float)
    y    = np.array(y, dtype=float)
    ln_x = np.log(np.clip(x, 1e-10, None))
    ln_y = np.log(np.clip(y, 1e-10, None))
    b, ln_a, _ = least_squares_linear(ln_x, ln_y)
    return float(np.exp(ln_a)), float(b)


# ══════════════════════════════════════════════════════════════
#  VECTORISED HELPERS
# ══════════════════════════════════════════════════════════════

def interpolate_array(fn, x_arr, x_pts, y_pts, **kwargs):
    """Evaluate an interpolation function over an array of x values."""
    return np.array([fn(xi, x_pts, y_pts, **kwargs) for xi in x_arr])

def interpolate_cubic_array(x_arr, x_pts, y_pts):
    """Evaluate cubic spline (precomputes M once) over an array."""
    M = cubic_spline_coeffs(x_pts, y_pts)
    return np.array([cubic_spline_eval(xi, x_pts, y_pts, M) for xi in x_arr])
