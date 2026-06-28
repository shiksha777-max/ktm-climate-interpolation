# 🌡️ Climate Temperature Reconstruction — Kathmandu Station 1030

**Scientific Computing Mini Project**  
Numerical Methods for Engineers · Tribhuvan University

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-green.svg)](https://numpy.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Overview

This project reconstructs monthly temperature curves from sparse station data using four classical interpolation methods, then compares their accuracy through rigorous error analysis.

**Dataset:** [Open Data Nepal](https://opendatanepal.com/dataset/mean-temperature-by-months-for-kathmandu-station-from-2004-to-2014) — Mean Monthly Temperature for Kathmandu Station (Code: 1030), 2004–2014. Published by the Central Bureau of Statistics, Statistical Year Book 2015.

**Core idea:** Given only 6 of the 12 monthly readings, can we accurately reconstruct the full annual temperature curve? Which method does it best?

---

## 🎯 Syllabus Coverage

| Unit | Topic | Implementation |
|------|-------|---------------|
| Unit 1 | Absolute error, Relative error, RMSE | `error_summary()`, `rmse()` |
| Unit 1 | Runge's phenomenon (truncation error growth) | Section 4 in `main.py` |
| Unit 3 | Thomas Algorithm (tridiagonal solver) | `thomas_algorithm()` |
| Unit 4 | Forward difference table | `forward_difference_table()` |
| Unit 4 | Newton Forward interpolation | `newton_forward()` |
| Unit 4 | Lagrange interpolation | `lagrange()` |
| Unit 4 | Cubic Spline (natural BCs) | `cubic_spline_coeffs()` + `cubic_spline_eval()` |
| Unit 4 | Linear Spline (baseline) | `linear_spline()` |
| Unit 4.3 | Least-squares: linear, polynomial, exponential, power | All four in `plots.py` |

---

## 📁 Repository Structure

```
ktm_climate_project/
│
├── data/
│   └── kathmandu_temperature_station1030.csv   # Raw dataset (Open Data Nepal)
│
├── src/
│   ├── numerical_methods.py   # All algorithms from scratch (Units 1, 3, 4)
│   ├── data_loader.py         # CSV loading & preprocessing
│   └── plots.py               # All matplotlib figures
│
├── results/                   # Generated on first run
│   ├── climate_reconstruction_results.png
│   ├── heatmap.png
│   ├── curve_fitting.png
│   ├── per_year_reconstruction.png
│   └── error_report.txt
│
├── main.py                    # Entry point — run this
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ktm-climate-interpolation.git
cd ktm-climate-interpolation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the project

```bash
python main.py
```

All results save automatically to `results/`.

---

## 📊 Results

### Methods compared

| Method | Type | Works on | Unit |
|--------|------|----------|------|
| Newton Forward | Global polynomial | Equally spaced only | 4.1 |
| Lagrange | Global polynomial | Any spacing | 4.1 |
| **Cubic Spline** | Piecewise cubic | Any spacing | 4.2 |
| Linear Spline | Piecewise linear | Any spacing | 4.2 |

### Error comparison (6 known points out of 12)

| Method | RMSE (°C) | Max \|Error\| (°C) | Avg Rel % |
|--------|-----------|-------------------|-----------|
| Newton Forward | 5.29 | 18.02 | 16.77 |
| Lagrange | 1.10 | 3.33 | 3.38 |
| **Cubic Spline** | **0.63** | **1.79** | **2.17** |
| Linear Spline | 0.47 | 0.93 | 1.82 |

> **Key finding:** Cubic Spline achieves the best balance of accuracy and stability. Newton Forward suffers from Runge's phenomenon — its error *increases* as more points are added (RMSE = 638 at n=8!), while Cubic Spline error decreases monotonically.

### Warming trend (Unit 4.3 — Least Squares)

```
T_annual = 0.0736 × Year − 128.83
Warming rate: +0.74 °C per decade  (2004–2014)
```

---

## 🔬 Key Concepts Demonstrated

### Runge's Phenomenon
Newton Forward interpolation is a single high-degree polynomial. As degree increases, it oscillates wildly near the edges — the error can exceed 600°C at n=8. Cubic Spline avoids this by using piecewise cubics with smooth joins.

```
n pts | Newton RMSE  | Cubic RMSE
------|--------------|-----------
4     |  7.13        |  0.56
6     |  3.27        |  0.61
8     | 412.48  ← 🔴 |  0.37
10    | 111.86  ← 🔴 |  0.29
```

### Thomas Algorithm inside Cubic Spline
The cubic spline requires solving a tridiagonal system of n equations for the second derivatives M. This is solved exactly using the Thomas Algorithm from Unit 3, connecting two units in one method.

---

## 📈 Figures

| Figure | Description |
|--------|-------------|
| `climate_reconstruction_results.png` | 5-panel main results: methods, error, RMSE vs n, Runge, warming trend |
| `heatmap.png` | Year × Month temperature heatmap with values |
| `curve_fitting.png` | 4-panel comparison of curve fitting models |
| `per_year_reconstruction.png` | Cubic spline reconstruction for each of 11 years |

---

## 📦 Dependencies

```
numpy>=1.24
pandas>=1.5
matplotlib>=3.6
```

---

## 📚 References

1. Chapra, S.C., Canale, R.P. — *Numerical Methods for Engineers*, McGraw-Hill
2. Open Data Nepal — [Mean Temperature, Kathmandu Station 1030](https://opendatanepal.com/dataset/mean-temperature-by-months-for-kathmandu-station-from-2004-to-2014)
3. Central Bureau of Statistics Nepal — Statistical Year Book 2015

---

## 👤 Author

**[Your Name]**  
B.E. [Your Program], [Your College]  
Tribhuvan University, Nepal
