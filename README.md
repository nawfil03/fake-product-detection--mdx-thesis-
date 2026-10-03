# 🛡️ Fake Product Detection — MSc Data Science Thesis

A machine-learning web app that checks an online product listing and estimates how likely it is to be **counterfeit or fake**. Paste a product URL from a UAE marketplace (Amazon.ae, Noon, Namshi, Ounass, Sharaf DG …) and the app reads the listing, scores it with a trained model, and explains *why* it reached that verdict.

Built as the final-year thesis project for my MSc in Data Science.

---

## ✨ Features

- **URL-based detection** – paste a full or short product link; Gemini reads the listing page and extracts price, rating, reviews, seller and wording.
- **Ensemble ML model** – Logistic Regression, Random Forest, Gradient Boosting and XGBoost combined into a `FusionModel`.
- **Transparent evidence layer** (`evidence.py`) – rule-based checks a shopper would make (price vs. UAE market price, review count, rating, seller, spelling, suspicious wording). The final risk is the higher of the model probability and the evidence risk.
- **Explainability** – SHAP feature importance shown in the app.
- **Three tabs** – 🔍 Detect Product · 📊 Models & SHAP · 🎓 How the verdict is made.

## 📊 Dataset & results

| | |
|---|---|
| Listings | 3,678 (2,557 fake · 1,121 genuine) |
| Features | 52 (incl. adversarial-wording features) |
| Sources | Scraped Amazon.ae listings + public fake-pattern datasets |
| Brand price DB | ~100 UAE retail reference prices |

| Model | Accuracy | F1 |
|---|---|---|
| Gradient Boosting 🏆 | 99.73% | 99.80% |
| Random Forest | 99.73% | 99.80% |
| XGBoost | 99.73% | 99.80% |
| FusionModel | 99.73% | 99.80% |
| Logistic Regression | 97.96% | 98.53% |

## 🗂️ Project structure

```
app_v2.py            # Streamlit app (UI + Gemini extraction + prediction)
evidence.py          # Rule-based evidence layer and UAE brand price DB
model_*.pkl          # Trained models (lr, rf, gb, xgb, fusion)
scaler.pkl           # Feature scaler
features_final.pkl   # Ordered feature list
shap_importance.pkl  # SHAP importances for the dashboard
model_stats.json     # Evaluation metrics shown in the app
dataset_info.json    # Dataset summary
requirements.txt
```

## 🚀 How to run

**1. Clone and install**

```bash
git clone https://github.com/nawfil03/fake-product-detection--mdx-thesis-.git
cd fake-product-detection--mdx-thesis-
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Add your Gemini API key** (get one from [Google AI Studio](https://aistudio.google.com/))

```bash
export GEMINI_API_KEY="your-key-here"     # Windows: set GEMINI_API_KEY=your-key-here
```

Or, when hosting on Streamlit Cloud, add `GEMINI_API_KEY` to the app's **Secrets**.

**3. Start the app**

```bash
streamlit run app_v2.py
```

Open http://localhost:8501, paste a product link and click **Analyze Product**.

> 💡 If you see a pickle/version error when loading the models, pin `scikit-learn`, `xgboost`, `numpy` and `joblib` to the versions used for training (e.g. `scikit-learn==1.6.1`).

### Optional environment variables

| Variable | Purpose |
|---|---|
| `APP_DIR` | Folder containing the model files (defaults to the app folder) |
| `CLEAN_CSV` | Path to the cleaned dataset CSV (used for extra charts) |
| `SURVEY_N` | Number of survey respondents shown in the explanation tab |

## 🛠️ Tech stack

Python · Streamlit · scikit-learn · XGBoost · SHAP · Plotly · pandas · Google Gemini (`gemini-2.5-flash`)

## ⚠️ Disclaimer

This is an academic project. Predictions are estimates and should not be treated as proof that a product is fake or genuine.

---

👤 **Nawfil Faraaz** · [GitHub](https://github.com/nawfil03)
