# 🚀 Deployment Guide: Uttar Pradesh Drought Prediction Dashboard

This guide provides step-by-step instructions to deploy the interactive Streamlit dashboard (`03_streamlit_app.py`) online so that anyone can access, interact with, and view your drought prediction models and forecasts.

We have already configured your project files with:
1. A streamlined production **`requirements.txt`** containing only the essential libraries (`streamlit`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `pillow`) so that the deployment builds extremely fast (under 2 minutes) and stays well within the free-tier RAM limits.
2. A separate **`requirements_dev.txt`** containing all heavy local libraries (`tensorflow`, `keras`, `xarray`, `geopandas`, etc.) if you need to rerun preprocessing or training locally.
3. A customized **`.gitignore`** to ensure you do not upload large raw NetCDF datasets (`*.nc`) or local virtual environments (`.venv/`) to your public repository.

---

## Method 1: Deploy on Streamlit Community Cloud (Recommended & Fastest)

Streamlit Community Cloud is a free, secure, and officially supported platform by Streamlit to host your applications directly from GitHub.

### Step 1: Create a GitHub Repository
1. Open your browser and go to [github.com](https://github.com).
2. Log in and click the **New** button to create a new repository.
3. Name your repository (e.g., `drought-prediction-up`).
4. Set the visibility to **Public** or **Private** (Public is recommended for easy sharing), and **do not** initialize it with a README, `.gitignore`, or license (we already have these).
5. Click **Create repository**.

### Step 2: Push your Code to GitHub
Open your terminal (PowerShell, CMD, or Git Bash) in your project directory `c:\Users\Lenovo\OneDrive\Desktop\drought` and run:

```bash
# Initialize a local Git repository
git init

# Add all project files (large raw files are automatically ignored by our .gitignore)
git add .

# Commit the changes
git commit -m "Configure project for deployment"

# Rename the default branch to main
git branch -M main

# Add the remote GitHub repository URL (replace with your repository's URL)
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/drought-prediction-up.git

# Push your code to GitHub
git push -u origin main
```

### Step 3: Launch on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Click **Connect GitHub account** and sign in.
3. Once logged in, click **Create app** (or **Deploy an app**).
4. Fill in the deployment details:
   - **Repository:** `YOUR_GITHUB_USERNAME/drought-prediction-up`
   - **Branch:** `main`
   - **Main file path:** `03_streamlit_app.py`
5. Click the **Deploy!** button.

Your app will start building immediately! Within 1-2 minutes, you will have a live, public URL (e.g., `https://drought-prediction-up.streamlit.app`) to share.

---

## Method 2: Deploy on Hugging Face Spaces (Alternative Free Platform)

Hugging Face Spaces is a popular free hosting platform for machine learning demos.

### Step 1: Create a Space on Hugging Face
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces).
2. Log in or create a free account, then click **Create new Space**.
3. Fill in the details:
   - **Space name:** e.g., `drought-prediction-up`
   - **License:** Open source (e.g., mit)
   - **SDK:** Select **Streamlit**.
   - **Space Hardware:** Select **CPU basic (Free)**.
   - **Visibility:** Public.
4. Click **Create Space**.

### Step 2: Upload or Push the Code
You can upload files directly through the Hugging Face web UI, or clone the Space using Git:

```bash
# Clone your Hugging Face Space repository
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/drought-prediction-up

# Copy your local project files into the cloned Space directory
# (Exclude large raw data folders and the .git/ folder from your local workspace)

# Move into the cloned Space directory
cd drought-prediction-up

# Commit and push
git add .
git commit -m "Deploy drought prediction dashboard"
git push
```

Hugging Face will automatically install the libraries listed in `requirements.txt` and serve the dashboard.

---

## 🛠️ Verification Checklist Before Deploying

To ensure the dashboard works flawlessly upon launch:
- [x] **Lightweight Dependencies:** Root `requirements.txt` has only deployment dependencies.
- [x] **Relative Paths:** `03_streamlit_app.py` is configured to load processed data using relative paths (e.g., `data/processed_features.csv` and `outputs/figures/`).
- [x] **Pre-generated Artifacts:** All metrics CSVs and figures are present in the `outputs/` and `data/` directories so the dashboard has everything it needs to display immediately without running time-consuming ML calculations on start-up.
