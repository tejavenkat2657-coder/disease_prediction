# 🩺 Disease Prediction App — Malaria & Dengue
### Symptom-based ML diagnosis · Decision Tree · Random Forest · Naïve Bayes · Streamlit

---

## 📁 Project Structure

```
disease_prediction/
├── generate_dataset.py        # Step 1 – Create / download dataset
├── train_model.py             # Step 2 – Preprocess + train + save model
├── predictor.py               # Step 3 – Prediction helper (used by app)
├── app.py                     # Step 4 – Streamlit web application
├── requirements.txt           # Python dependencies
├── .streamlit/
│   └── config.toml            # Streamlit theme & server config
├── data/
│   └── disease_dataset.csv    # Generated dataset (1 200 rows)
└── model/
    ├── best_model.pkl         # Saved model + encoders
    └── model_comparison.csv   # Accuracy table
```

---

## 🚀 Quick Start (Local)

### 1. Clone / copy the project
```bash
git clone <your-repo-url>
cd disease_prediction
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate the dataset
```bash
python generate_dataset.py
# → data/disease_dataset.csv  (1 200 rows, 26 symptom + meta columns)
```

### 5. Train the models & save the best one
```bash
python train_model.py
# → model/best_model.pkl
# → model/model_comparison.csv
```

### 6. Run the Streamlit app
```bash
streamlit run app.py
# Opens http://localhost:8501
```

---

## 🧪 Using a Real Kaggle Dataset (Optional)

Instead of the synthetic dataset, you can use a real Kaggle dataset:

1. Install the Kaggle CLI:
   ```bash
   pip install kaggle
   ```

2. Download a symptom-disease dataset:
   ```bash
   kaggle datasets download -d itachi9604/disease-symptom-description-dataset
   unzip disease-symptom-description-dataset.zip -d data/
   ```

3. Adapt `generate_dataset.py` to load and reshape the CSV into the same column format (one binary column per symptom + `age_group`, `gender`, `disease`).

---

## 📊 Code Walkthrough

### `generate_dataset.py`
| What it does | Key concept |
|---|---|
| Defines Malaria & Dengue symptom lists | Domain knowledge encoding |
| Randomly generates 1 200 patient records | Controlled synthetic data |
| Adds age group + gender metadata | Demographic features |
| Saves to `data/disease_dataset.csv` | Reproducible with `random_state=42` |

### `train_model.py`
| Step | What it does | scikit-learn API |
|---|---|---|
| Load | `pd.read_csv()` | — |
| Handle NaN | `fillna(0)` for symptoms, mode for categoricals | — |
| Encode | `LabelEncoder` for age, gender, disease | `fit_transform()` |
| Split | 80 % train / 20 % test, stratified | `train_test_split()` |
| Train | Decision Tree, Random Forest, Naïve Bayes | `.fit()` |
| Evaluate | Accuracy, Classification Report, 5-fold CV | `cross_val_score()` |
| Save | Bundle model + all encoders | `pickle.dump()` |

### `predictor.py`
| Function | Purpose |
|---|---|
| `_load()` | Loads Pickle once and caches it in memory |
| `get_symptom_list()` | Returns symptom column names for the UI |
| `get_model_info()` | Returns model name + accuracy for the badge |
| `predict(symptoms, age, gender)` | Encodes input → runs model → returns disease + confidence |

### `app.py`
| Section | Purpose |
|---|---|
| `st.set_page_config()` | Page title, icon, wide layout |
| Custom CSS | Hero banner, result cards, symptom chips |
| Sidebar | Model comparison table, disclaimer |
| Symptom grid | 4-column checkbox grid for all 24 symptoms |
| Predict button | Calls `predictor.predict()`, shows result card + bar chart |

---

## ☁️ Deployment

### Option A — Streamlit Community Cloud (Free, Easiest)

1. Push your project to a **public GitHub repo**
2. Visit [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **"New app"** → select your repo → set **Main file path** to `app.py`
4. Add a `setup.sh` if you need system packages, otherwise just requirements.txt is enough
5. Click **Deploy** — your app gets a public URL like `https://your-app.streamlit.app`

> **Important:** Make sure `model/best_model.pkl` and `data/` are committed to the repo, OR add a `@st.cache_resource` block in `app.py` that calls `generate_dataset.py` + `train_model.py` on first boot.

#### `packages.txt` (if needed for system libs)
```
libgomp1
```

---

### Option B — Render (Free Tier)

1. Push to GitHub
2. Go to [render.com](https://render.com) → New → **Web Service**
3. Connect your GitHub repo
4. Set:
   - **Build Command:** `pip install -r requirements.txt && python generate_dataset.py && python train_model.py`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Set **Environment** to Python 3
6. Click **Create Web Service**

---

### Option C — Docker (Self-hosted / any cloud)

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN python generate_dataset.py && python train_model.py
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build & run:
```bash
docker build -t disease-predictor .
docker run -p 8501:8501 disease-predictor
```

---

## 📌 Notes & Disclaimers

- This app uses **synthetic data** designed to closely mimic real symptom patterns.
- For production medical software, use validated clinical datasets and obtain proper regulatory approval.
- The 100 % accuracy on synthetic data is expected — real-world data will show ~85–95 % depending on data quality.
- Always consult a licensed physician for actual diagnosis.

---

## 🛠 Extending the Project

| Idea | How |
|---|---|
| Add more diseases (Typhoid, COVID-19) | Add symptom lists in `generate_dataset.py`, re-train |
| Use real Kaggle data | Adapt the data loading step in `train_model.py` |
| Add SHAP explainability | `pip install shap` + `shap.TreeExplainer` |
| REST API version | Use **FastAPI** + `predictor.predict()` |
| Database logging | SQLite + `sqlite3` to log predictions |
