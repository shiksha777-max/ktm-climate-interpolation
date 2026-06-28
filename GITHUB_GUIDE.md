# GitHub Upload Guide
## Climate Temperature Reconstruction Project

Complete step-by-step instructions to put this project on GitHub.

---

## Prerequisites

- A GitHub account → [github.com](https://github.com)
- Git installed on your PC
  - Windows: download from [git-scm.com](https://git-scm.com)
  - Check: open terminal and type `git --version`

---

## Step 1 — Create the GitHub repository

1. Go to **github.com** and sign in
2. Click the **+** button (top right) → **New repository**
3. Fill in:
   - **Repository name:** `ktm-climate-interpolation`
   - **Description:** `Climate Temperature Reconstruction using Numerical Interpolation Methods — Kathmandu Station 1030`
   - **Visibility:** Public *(so your professor can see it)*
   - ✅ **Do NOT** tick "Add a README" (we have our own)
4. Click **Create repository**
5. Copy the repo URL shown — it looks like:
   ```
   https://github.com/YOUR_USERNAME/ktm-climate-interpolation.git
   ```

---

## Step 2 — Set up your local project folder

Open terminal (Command Prompt or PowerShell on Windows, Terminal on Mac/Linux).

Navigate to where you saved the project files:

```bash
cd path/to/ktm_climate_project
```

For example:
```bash
cd C:\Users\YourName\Downloads\ktm_climate_project   # Windows
cd ~/Downloads/ktm_climate_project                   # Mac/Linux
```

---

## Step 3 — Initialise Git and push

Run these commands one by one:

```bash
# 1. Initialise a git repository
git init

# 2. Tell git who you are (first time only)
git config --global user.name  "Your Name"
git config --global user.email "your@email.com"

# 3. Stage all project files
git add .

# 4. Make your first commit
git commit -m "Initial commit: Climate Temperature Reconstruction project"

# 5. Rename branch to main (GitHub default)
git branch -M main

# 6. Connect to your GitHub repo (paste YOUR repo URL here)
git remote add origin https://github.com/YOUR_USERNAME/ktm-climate-interpolation.git

# 7. Push to GitHub
git push -u origin main
```

After step 7, refresh your GitHub page — all files should appear!

---

## Step 4 — Verify your repo structure

Your GitHub repo should look like this:

```
ktm-climate-interpolation/
├── 📁 data/
│   └── kathmandu_temperature_station1030.csv
├── 📁 src/
│   ├── numerical_methods.py
│   ├── data_loader.py
│   └── plots.py
├── 📁 results/
│   ├── climate_reconstruction_results.png
│   ├── heatmap.png
│   ├── curve_fitting.png
│   ├── per_year_reconstruction.png
│   └── error_report.txt
├── main.py
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

---

## Step 5 — Add result images to README (optional but impressive)

GitHub renders images in README if you push them to `results/`. The README already references them. Make sure the results folder is committed:

```bash
# Run the project first to generate results
python main.py

# Then add and push the results
git add results/
git commit -m "Add generated result figures and error report"
git push
```

---

## Step 6 — Every time you make changes

```bash
git add .
git commit -m "Describe what you changed"
git push
```

---

## Common errors and fixes

| Error | Fix |
|-------|-----|
| `git: command not found` | Install Git from git-scm.com |
| `remote origin already exists` | Run `git remote remove origin` then re-add |
| `Authentication failed` | Use a Personal Access Token instead of password — see below |
| `src refspec main does not match` | Run `git add .` and `git commit` before pushing |

### Personal Access Token (if password doesn't work)

GitHub no longer accepts passwords for push. Use a token:

1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click **Generate new token**
3. Select scope: ✅ **repo**
4. Copy the token
5. When git asks for password, paste the token

---

## Step 7 — Share with your professor

Your project URL will be:
```
https://github.com/YOUR_USERNAME/ktm-climate-interpolation
```

The README renders automatically on the homepage, showing your results table, figures, and full explanation.

---

## Quick reference card

```bash
git init                          # initialise repo (once)
git add .                         # stage all changes
git add filename.py               # stage one file
git commit -m "your message"      # save snapshot
git push                          # upload to GitHub
git status                        # see what's changed
git log --oneline                 # see commit history
```
