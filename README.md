# ML Prediction Pipeline — Time‑Series Regression (CNN + GRU)

Professional, end‑to‑end pipeline for **time‑series / tabular regression**:
- **Model**: CNN → GRU → Dense (Keras/TensorFlow)
- **Data processing**: `StandardScaler` for inputs/target, optional **One‑Hot** for categoricals
- **Training artifacts**: model (`.keras`), scalers/encoders (`.gz`), `feature_config.json`
- **Evaluation**: MAE/RMSE/R², learning curves, interactive HTML plots
- **UX**: clean notebooks, repo structure ready for CI/packaging

> Fits portfolio use‑cases and predictive maintenance demos. Works on CPU/GPU.

---

## 📦 Repository structure

```
.
├─ notebooks/
│  └─ 01_train.ipynb           # Training notebook (end‑to‑end)
├─ src/mlpp/                    # (slot for shared code if needed later)
├─ TRAIN/                       # Put your training CSVs here (examples included)
├─ TEST/                        # Put your test CSVs here (examples included)
├─ OUTPUTS/                     # Training artifacts & plots (git‑ignored)
├─ scripts/
│  └─ quickstart.sh             # (optional) one‑command setup & run
├─ requirements.txt
├─ .gitignore
├─ LICENSE
└─ README.md
```

> If you keep a separate prediction workflow (GUI/CLI), add `notebooks/02_predict.ipynb` or `scripts/predict.py` and place shared helpers in `src/mlpp/`.

---

## 🧠 Task & data schema

**Task:** univariate regression on time‑ordered data (predict a continuous `target`).

**Default input columns:**
```
feature_01 .. feature_12, comp_active
```
**Target column:**
```
target
```
You can change column names inside the notebook (section **Konfiguracja**).

> CSV delimiter is auto‑detected; a robust fallback to `;` and to default `,` is included.

---

## 🚀 Quickstart

### 1) Environment
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Data
- Put your CSVs into **TRAIN/** and **TEST/**.  
- See examples in `TRAIN/example_train.csv` and `TEST/example_test.csv` for the expected headers.

### 3) Run
Open the notebook and run all cells:
```bash
jupyter lab   # or: jupyter notebook / VS Code
```
Artifacts (models, scalers, encoders, plots) land under **OUTPUTS/** in a timestamped subfolder.

---

## 📊 Outputs
- `OUTPUTS/<timestamp>/best_model.keras` — best checkpoint by `val_loss`
- `OUTPUTS/<timestamp>/scaler.gz`, `output_scaler.gz`, `encoders.gz` — preprocessing
- `OUTPUTS/<timestamp>/feature_config.json` — persisted input/output columns
- `OUTPUTS/<timestamp>/*.png` — loss/MAE curves
- `OUTPUTS/<timestamp>/*.html` — interactive prediction diff plots
- `OUTPUTS/<timestamp>/test_metrics.csv` — per‑file MSE/RMSE/MAE/R²

---

## 🛠️ Tech stack
- TensorFlow / Keras (2.x), NumPy, Pandas
- scikit‑learn (StandardScaler, OneHotEncoder)
- Matplotlib & Plotly (interactive HTML)
- Joblib for artifact persistence
- Jupyter environment

---

## 🔄 Reproducibility & quality
- Deterministic seeds for NumPy/TensorFlow
- EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
- Schema validation (`STRICT_SCHEMA`) with a tolerant fallback
- Robust CSV delimiter detection

---

## 🗺️ Roadmap (suggested)
- [ ] Notebook `02_predict.ipynb` for inference from saved artifacts
- [ ] `scripts/predict.py` (CLI): `--artifacts`, `--csv`, `--out`
- [ ] `src/mlpp/` helpers for shared preprocessing & plotting
- [ ] GitHub Actions: smoke‑test `pip install` + notebook import
- [ ] MLflow experiment tracking (optional)

---

## 📄 License
Released under the **MIT License** (see [LICENSE](LICENSE)).

---

## 🤝 Acknowledgements
Built as a clean, portfolio‑friendly baseline for regression on time‑ordered industrial data.
