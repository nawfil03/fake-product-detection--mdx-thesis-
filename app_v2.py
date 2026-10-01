import os, sys, re, json, html, urllib.parse, urllib.request

_COLAB_DIR = "/content/drive/MyDrive/FakeProductThesis/streamlit_app/"
APP_DIR = os.environ.get("APP_DIR") or (_COLAB_DIR if os.path.isdir(_COLAB_DIR) else os.path.dirname(os.path.abspath(__file__)))
if not APP_DIR.endswith("/"):
    APP_DIR += "/"
sys.path.insert(0, APP_DIR)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
SURVEY_N = os.environ.get("SURVEY_N", "150")
CLEAN_CSV = os.environ.get("CLEAN_CSV", APP_DIR.replace("streamlit_app/", "cleaned_data/") + "combined_full_dataset.csv")
GEMINI_MODEL = "gemini-2.5-flash"

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
from google import genai
from google.genai import types
import evidence as ev

if not GEMINI_API_KEY:          # hosted version: also accept the key from Streamlit's secrets store
    try:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

try:
    _V = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:
    _V = (1, 0)
WIDE = {"width": "stretch"} if _V >= (1, 50) else {"use_container_width": True}

st.set_page_config(page_title="UAE Fake Product Detector", page_icon="\U0001F50D", layout="wide")
st.markdown("""
<style>
.main-title{font-size:2.8rem;font-weight:900;text-align:center;color:#1a1a2e;}
.subtitle{font-size:1rem;color:#888;text-align:center;margin-bottom:1.5rem;}
.verdict-fake{background:linear-gradient(135deg,#ff4444,#cc0000);color:white;padding:2rem;border-radius:16px;text-align:center;font-size:2rem;font-weight:900;margin:1rem 0;}
.verdict-suspicious{background:linear-gradient(135deg,#ff9800,#e65100);color:white;padding:2rem;border-radius:16px;text-align:center;font-size:2rem;font-weight:900;margin:1rem 0;}
.verdict-genuine{background:linear-gradient(135deg,#00c853,#1b5e20);color:white;padding:2rem;border-radius:16px;text-align:center;font-size:2rem;font-weight:900;margin:1rem 0;}
.verdict-unknown{background:linear-gradient(135deg,#78909c,#37474f);color:white;padding:2rem;border-radius:16px;text-align:center;font-size:2rem;font-weight:900;margin:1rem 0;}
.model-box{border-radius:12px;padding:0.8rem;margin:0.4rem 0;text-align:center;border:2px solid #e0e0e0;background:white;}
.model-selected{border:3px solid #9C27B0!important;background:linear-gradient(135deg,#f3e5f5,#e1bee7)!important;}
.card{background:white;border-radius:12px;padding:0.8rem 1.4rem;margin:0.4rem 0;border-left:5px solid #2196F3;box-shadow:0 2px 8px rgba(0,0,0,0.05);}
.card-red{border-left-color:#f44336!important;background:#fff8f8;}
.card-green{border-left-color:#4CAF50!important;background:#f8fff8;}
.card-orange{border-left-color:#FF9800!important;background:#fff8f0;}
.card-grey{border-left-color:#90a4ae!important;background:#f7f9fa;}
.explanation-box{background:linear-gradient(135deg,#f3e5f5,#e8eaf6);border:1px solid #9C27B0;border-radius:12px;padding:1.5rem;margin:1rem 0;font-size:1rem;line-height:1.8;}
.combo-box{background:#f9f0ff;border:1px solid #9C27B0;border-radius:10px;padding:0.8rem 1.4rem;margin:0.5rem 0;}
</style>
""", unsafe_allow_html=True)

MODEL_INFO = {
    "FusionModel":         {"icon": "\U0001F52C", "color": "#9C27B0"},
    "Gradient Boosting":   {"icon": "\U0001F3C6", "color": "#4CAF50"},
    "Random Forest":       {"icon": "\U0001F332", "color": "#2196F3"},
    "XGBoost":             {"icon": "\u26A1",     "color": "#FF9800"},
    "Logistic Regression": {"icon": "\U0001F4C9", "color": "#E91E63"},
}
VERDICTS = {
    "fake":         ("\U0001F534 HIGH RISK \u2014 LIKELY FAKE", "verdict-fake", "#ff4444"),
    "suspicious":   ("\U0001F7E1 MEDIUM RISK \u2014 SUSPICIOUS", "verdict-suspicious", "#ff9800"),
    "genuine":      ("\U0001F7E2 LOW RISK \u2014 LIKELY GENUINE", "verdict-genuine", "#4CAF50"),
    "inconclusive": ("\u26AA INCONCLUSIVE \u2014 NOT ENOUGH DATA TO CLEAR THIS LISTING", "verdict-unknown", "#78909c"),
}
STATUS_ICON = {"bad": "\U0001F534", "warn": "\U0001F7E1", "ok": "\U0001F7E2", "unknown": "\u26AA"}


# ============================ loading ============================
def load_category_stats():
    default_codes = {"cosmetics": 0, "electronics": 1, "fashion": 2, "supplements": 3}
    try:
        df = pd.read_csv(CLEAN_CSV, usecols=["category", "price_clean"]).dropna()
        g = df.groupby("category")["price_clean"].agg(["median", "mean", "std"])
        stats = {k: {"median": float(r["median"]), "mean": float(r["mean"]), "std": float(r["std"] or 1.0)}
                 for k, r in g.iterrows()}
        glob = {"median": float(df["price_clean"].median()), "mean": float(df["price_clean"].mean()),
                "std": float(df["price_clean"].std() or 1.0)}
        codes = {c: i for i, c in enumerate(sorted(df["category"].unique()))}
        return {"stats": stats, "global": glob, "codes": codes, "from_training": True}
    except Exception:
        return {"stats": {}, "global": {"median": 150.0, "mean": 150.0, "std": 80.0},
                "codes": default_codes, "from_training": False}


@st.cache_resource(show_spinner="Loading the trained models from Google Drive...")
def load_all():
    ms = {
        "FusionModel":         joblib.load(APP_DIR + "model_fusion.pkl"),
        "Gradient Boosting":   joblib.load(APP_DIR + "model_gb.pkl"),
        "Random Forest":       joblib.load(APP_DIR + "model_rf.pkl"),
        "XGBoost":             joblib.load(APP_DIR + "model_xgb.pkl"),
        "Logistic Regression": joblib.load(APP_DIR + "model_lr.pkl"),
    }
    sc = joblib.load(APP_DIR + "scaler.pkl")
    ft = joblib.load(APP_DIR + "features_final.pkl")
    sd = joblib.load(APP_DIR + "shap_importance.pkl")
    with open(APP_DIR + "model_stats.json") as f:
        stats = json.load(f)
    with open(APP_DIR + "dataset_info.json") as f:
        ds_info = json.load(f)
    cl = genai.Client(api_key=GEMINI_API_KEY)
    return ms, sc, ft, cl, stats, sd, ds_info, load_category_stats()


if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is not set. Run the launch cell again with your key filled in.")
    st.stop()
try:
    models, scaler, features, client, MODEL_STATS, shap_df, DS_INFO, CAT = load_all()
except Exception as e:
    st.error("Could not load the trained models from %s\n\n%s\n\nMount Google Drive and check the .pkl files exist." % (APP_DIR, e))
    st.stop()
FUSION_F1 = MODEL_STATS.get("FusionModel", {}).get("f1", "?")
FUSION_AUC = MODEL_STATS.get("FusionModel", {}).get("auc", "?")


# ============================ extraction ============================
def expand_url(url):
    try:
        if any(x in url for x in ["amzn.eu", "amzn.to", "noon.com/s/", "bit.ly", "tinyurl"]):
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            return urllib.request.urlopen(req, timeout=8).url
    except Exception:
        pass
    return url


def detect_platform(url):
    d = urllib.parse.urlparse(url).netloc.lower()
    for key, name in [("amazon", "Amazon.ae"), ("noon", "Noon.com"), ("namshi", "Namshi"), ("dubizzle", "Dubizzle"),
                      ("ounass", "Ounass"), ("sharaf", "Sharaf DG"), ("carrefour", "Carrefour UAE"),
                      ("shein", "Shein"), ("aliexpress", "AliExpress")]:
        if key in d:
            return name
    return d.replace("www.", "").split(".")[0].title() or "Unknown"


PROMPT = """You are the data-extraction step of a UAE fake-product checker. Read this product listing.
URL: __URL__
Platform: __PLATFORM__

HOW TO WORK
- Open the listing page itself and read it. If the page cannot be opened, use Google Search results for the SAME product
  (match the product id / ASIN / title in the URL) and set page_read to false.
- Report ONLY what you actually see. If something is not visible, use null. NEVER guess or estimate review_count, rating or seller.
- rating_text_seen and review_text_seen: copy the exact short text you saw next to the stars (for example "4.6 (443)" or
  "27.2K ratings" or "No customer reviews yet"). If you did not see any, use null.
- price: the current selling price as a number, in the currency shown on the page. currency: the ISO code shown (AED, USD, INR ...).
  If there are several options, use the lowest current price of the selected option.
- review_count: integer (27.2K = 27200, 1.5M = 1500000). Use 0 ONLY if the page explicitly says there are no reviews yet.
- seller_type must be one of: official_brand_store, platform_fulfilled (sold or fulfilled by Amazon/Noon), third_party_marketplace,
  individual_classified, unknown. seller_rating is out of 5 if shown, else null.
- min_market_price and max_market_price: typical genuine retail price range in AED for this exact product in the UAE, found with
  Google Search. null if you cannot find it. market_sources: up to 3 site names you used.
- category: one of electronics, fashion, cosmetics, supplements, other.

Return ONLY this JSON and nothing else:
{"title": null, "brand": null, "category": "other", "price": null, "currency": "AED", "rating": null, "rating_text_seen": null,
 "review_count": null, "review_text_seen": null, "seller_name": null, "seller_type": "unknown", "seller_rating": null,
 "seller_rating_count": null, "min_market_price": null, "max_market_price": null, "market_sources": [],
 "page_read": false, "red_flags": []}"""


def tool_sets():
    sets = []
    try:
        sets.append(("page access + Google Search", [types.Tool(url_context=types.UrlContext()), types.Tool(google_search=types.GoogleSearch())]))
        sets.append(("page access", [types.Tool(url_context=types.UrlContext())]))
    except Exception:
        pass
    sets.append(("Google Search only", [types.Tool(google_search=types.GoogleSearch())]))
    return sets


def extract_listing(url):
    url = expand_url(url)
    platform = detect_platform(url)
    prompt = PROMPT.replace("__URL__", url).replace("__PLATFORM__", platform)
    best, best_score, errors = None, -1, []
    for label, tools in tool_sets():
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt,
                config=types.GenerateContentConfig(tools=tools, temperature=0))
            raw = ev.extract_json(getattr(resp, "text", "") or "")
            if not raw:
                errors.append("%s: no JSON returned" % label)
                continue
            d = ev.normalize(raw, platform)
            d["read_with"] = label
            score = sum(d.get(k) is not None for k in ("title", "price", "rating", "review_count", "min_market_price")) \
                + (1 if d.get("seller_name") else 0)
            if score > best_score:
                best, best_score = d, score
            if d.get("price") and d.get("title"):
                break
        except Exception as e:
            errors.append("%s: %s" % (label, str(e)[:100]))
    if best is None:
        return {"success": False, "error": "; ".join(errors) or "no data returned", "platform": platform, "url": url}
    best["success"] = True
    best["url"] = url
    return best


# ============================ model input ============================
def engineer_features(data):
    """Same 52 features as training. Unknown rating/reviews get a neutral value (verdict is gated separately)."""
    price = float(data.get("price") or 0)
    rating = float(data["rating"]) if data.get("rating") is not None else 4.3
    reviews = int(data["review_count"]) if data.get("review_count") is not None else 100
    title = str(data.get("title") or "")
    min_p = float(data.get("min_market_price") or 0)
    max_p = float(data.get("max_market_price") or 0)

    cs = CAT["stats"].get(data.get("category")) or CAT["global"]
    cat_med, cat_mean, cat_std = cs["median"], cs["mean"], (cs["std"] or 1.0)
    if min_p > 0 and max_p > 0 and price > 0:
        price_deviation = ((min_p + max_p) / 2 - price) / ((min_p + max_p) / 2)
    elif min_p > 0 and price > 0:
        price_deviation = (min_p - price) / min_p
    elif price > 0 and cat_med > 0:
        price_deviation = (cat_med - price) / cat_med
    else:
        price_deviation = 0.0
    high_pd = int(price_deviation > 0.5); extreme_pd = int(price_deviation > 0.7); critical_pd = int(price_deviation > 0.85)
    p2mr = price / cat_med if cat_med > 0 else 1.0
    pz = (price - cat_mean) / cat_std if price > 0 else 0.0
    p_sig = high_pd * 0.72; ep_sig = extreme_pd * 0.85
    log_p = np.log1p(price); pb10 = int(price < 10); pb50 = int(price < 50)
    prn = int(price > 0 and price % 10 == 0)

    violation, brand_min, _ = ev.get_brand_info(title, price)
    bpr = price / brand_min if brand_min > 0 else 1.0
    bps = violation / 5.0; hlb = int(violation > 0); ibp = int(violation >= 4)

    z_rev = int(reviews == 0); f_rev = int(reviews < 10); m_rev = int(reviews > 100); vm_rev = int(reviews > 1000)
    log_rev = np.log1p(reviews); rev_sig = z_rev * 0.39; sel_sig = z_rev * 0.39
    rev_cred = rating * np.log1p(reviews) / 5.0 if reviews > 0 else 0

    perf_rat = int(rating == 5.0); low_rat = int(rating < 3.5)
    sus_comb = int(rating >= 4.8 and reviews < 10); pnr = int(rating == 5.0 and reviews == 0)
    rat_sig = pnr * 0.28; rrr = rating / (np.log1p(reviews) + 1)
    rlr = rating * np.log1p(1 / (reviews + 1)); epr = int(rating == 5.0 and reviews > 100)

    frs = 0
    if rating == 5.0 and reviews == 0:     frs += 90
    elif rating == 5.0 and reviews < 3:    frs += 75
    elif rating == 5.0 and reviews < 10:   frs += 55
    elif rating >= 4.9 and reviews < 5:    frs += 65
    elif rating >= 4.8 and reviews < 10:   frs += 45
    elif rating == 5.0 and reviews < 50:   frs += 25
    elif rating >= 4.7 and reviews < 20:   frs += 20
    if reviews == 0:   frs += 20
    elif reviews < 3:  frs += 15
    elif reviews < 5:  frs += 8
    if reviews > 0:
        rr = rating / np.log1p(reviews)
        if rr > 5.5:   frs += 30
        elif rr > 5.0: frs += 20
        elif rr > 4.5: frs += 12
        elif rr > 4.0: frs += 6
    if reviews > 200 and rating == 5.0: frs += 25
    fake_review_score = min(frs, 100); frn = fake_review_score / 100.0

    # These title features are computed exactly as in training (substring matching), so the trained models see the same kind of input.
    tl = title.lower()
    sp_list = ["niike", "nkie", "samsang", "addidas", "abidas", "guci", "chanell", "rollex", "i-phone", "aple", "iphon",
               "galxy", "samsun", "nikee", "addidass", "louiss", "vuittton"]
    kw_list = ["replica", "copy", "clone", "inspired", "aaa", "1:1", "mirror quality", "grade a", "factory sealed", "clearance",
               "wholesale", "bulk stock", "direct factory", "best copy", "import", "china stock"]
    susp = ["original", "authentic", "genuine", "100%", "real", "official", "brand new sealed", "factory"]
    twc = len(title.split())
    hse = int(any(s in tl for s in sp_list)); ss = hse * 0.35
    hfk = int(any(k in tl for k in kw_list)); hsw = int(any(w in tl for w in susp))
    srs = p_sig + rev_sig + sel_sig + ss + rat_sig + bps
    adv = hfk * 0.5 + ibp * 0.9 + critical_pd * 0.8 + epr * 0.4
    code = CAT["codes"].get(data.get("category"), CAT["codes"].get("electronics", 0))

    row = {
        "price_clean": price, "price_deviation": price_deviation, "high_price_deviation": high_pd,
        "extreme_price_dev": extreme_pd, "critical_price_dev": critical_pd, "price_to_median_ratio": p2mr,
        "price_zscore": pz, "price_signal": p_sig, "extreme_price_signal": ep_sig, "log_price": log_p,
        "price_below_10": pb10, "price_round_number": prn, "brand_violation_level": violation,
        "brand_price_ratio": bpr, "brand_price_signal": bps, "has_luxury_brand": hlb,
        "impossible_brand_price": ibp, "price_below_50": pb50, "num_reviews": reviews, "zero_reviews": z_rev,
        "few_reviews": f_rev, "many_reviews": m_rev, "very_many_reviews": vm_rev, "log_reviews": log_rev,
        "review_signal": rev_sig, "seller_signal": sel_sig, "review_credibility": rev_cred, "rating": rating,
        "perfect_rating": perf_rat, "low_rating": low_rat, "suspicious_combo": sus_comb, "perfect_no_reviews": pnr,
        "rating_signal": rat_sig, "rating_review_ratio": rrr, "rating_log_ratio": rlr, "extreme_perfect_ratio": epr,
        "fake_review_score": fake_review_score, "fake_review_score_norm": frn, "title_length": len(title),
        "title_word_count": twc, "title_caps_ratio": sum(1 for c in title if c.isupper()) / max(len(title), 1),
        "title_digit_ratio": sum(1 for c in title if c.isdigit()) / max(len(title), 1), "has_spelling_error": hse,
        "spelling_signal": ss, "has_fake_keyword": hfk, "has_suspicion_word": hsw, "title_too_short": int(twc < 3),
        "has_exclamation": int("!" in title), "has_all_caps_word": int(any(w.isupper() and len(w) > 2 for w in title.split())),
        "survey_risk_score": srs, "adversarial_risk_score": adv, "category_code": code,
    }
    df_row = pd.DataFrame([row])
    for f in features:
        if f not in df_row.columns:
            df_row[f] = 0
    return df_row[features], fake_review_score


def get_all_predictions(X):
    preds, X_sc = {}, scaler.transform(X)
    for name, model in models.items():
        try:
            preds[name] = round(float(model.predict_proba(X_sc if name == "Logistic Regression" else X)[0][1]), 4)
        except Exception:
            preds[name] = 0.0
    return preds


def agreement(all_probs):
    sd = float(np.std(list(all_probs.values())))
    return ("HIGH" if sd < 0.10 else "MEDIUM" if sd < 0.20 else "LOW"), sd


def local_explanation(A):
    lines = ["Final risk is %.0f%%: the higher of the FusionModel (%.1f%%) and the evidence checks (%.0f%%)." %
             (A["final"] * 100, A["p_model"] * 100, A["evidence"] * 100)]
    if A["drivers"]:
        lines.append("Main reasons: " + "; ".join("%s (%s)" % (c["check"].lower(), c["finding"]) for c in A["drivers"][:3]) + ".")
    if A["missing"]:
        lines.append("Could not read: " + ", ".join(A["missing"]) + ".")
    return " ".join(lines)


def gemini_explain(data, A):
    facts = "\n".join("- %s: %s -> %s" % (c["check"], c["read"], c["finding"]) for c in A["checks"])
    prompt = ("Explain this fake-product check to a UAE shopper in 4 or 5 plain sentences. Only explain; do NOT change the verdict "
              "and do NOT invent any number that is not below.\n\n"
              "Product: %s (%s)\nVerdict: %s\nFinal risk: %.0f%% (FusionModel %.1f%%, evidence checks %.0f%%)\n"
              "Checks performed:\n%s\nNot readable from the page: %s\n\n"
              "Finish with one clear action for the shopper." %
              (data.get("title"), data.get("platform"), A["verdict"], A["final"] * 100, A["p_model"] * 100,
               A["evidence"] * 100, facts, ", ".join(A["missing"]) or "nothing"))
    try:
        r = client.models.generate_content(model=GEMINI_MODEL, contents=prompt,
                                           config=types.GenerateContentConfig(temperature=0.2))
        return (r.text or "").strip() or local_explanation(A)
    except Exception:
        return local_explanation(A)


def esc(x):
    return html.escape(str(x)) if x is not None else ""


# ============================ UI ============================
st.markdown('<div class="main-title">\U0001F50D UAE Fake Product Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">FusionModel (F1: %s%% AUC: %s) \u2022 52 features \u2022 107 UAE brand prices \u2022 live evidence checks \u2022 '
            'Gemini AI \u2022 MSc Data Science \u2022 Middlesex University Dubai</div>' % (FUSION_F1, FUSION_AUC), unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["\U0001F50D Detect Product", "\U0001F4CA Models & SHAP", "\U0001F393 How the verdict is made"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("FusionModel F1", "%s%%" % FUSION_F1, "AUC: %s" % FUSION_AUC)
    c2.metric("Training listings", "{:,}".format(DS_INFO.get("total_products", 0)), "scraped + patterns")
    c3.metric("Features", "52", "incl. adversarial")
    c4.metric("Brand DB", "107", "UAE prices")
    st.markdown("---")
    url = st.text_input("\U0001F517 Paste a product URL (full or short link):",
                        placeholder="Noon.com, Amazon.ae, Namshi, Ounass, Sharaf DG ...")
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        btn = st.button("\U0001F50D Analyze Product", type="primary", **WIDE)

    if btn and not url:
        st.error("Please paste a product URL first.")
    if btn and url:
        s1, s2, s3 = st.empty(), st.empty(), st.empty()
        s1.info("Step 1/3: Gemini is reading the listing page (this can take 10-30 seconds)...")
        data = extract_listing(url.strip())
        if not data.get("success"):
            s1.error("Could not read this listing: %s" % data.get("error"))
            st.stop()
        s1.success("Step 1/3: listing read (%s)" % data.get("read_with"))
        price = data.get("price")

        if not price or not data.get("title"):
            s2.empty(); s3.empty()
            st.markdown('<div class="verdict-unknown">\u26AA CANNOT SCORE \u2014 THE PRICE OR TITLE COULD NOT BE READ'
                        '<br><span style="font-size:1.05rem;opacity:0.9">No verdict is given when the basics are missing. '
                        'Try the full product link, or a Noon.com link.</span></div>', unsafe_allow_html=True)
            st.json({k: v for k, v in data.items() if k in ("title", "platform", "price", "rating", "review_count", "seller_name", "read_with")})
            st.stop()

        s2.info("Step 2/3: calculating 52 features and running the 5 models...")
        X, frs_val = engineer_features(data)
        all_probs = get_all_predictions(X)
        p_fusion = all_probs["FusionModel"]
        agree, agree_sd = agreement(all_probs)
        s2.success("Step 2/3: FusionModel probability %.1f%% (model agreement: %s)" % (p_fusion * 100, agree))
        A = ev.assess(data, p_fusion)
        s3.info("Step 3/3: writing the explanation...")
        explanation = gemini_explain(data, A)
        s3.success("Step 3/3: done")

        label, css, color = VERDICTS[A["verdict"]]
        st.markdown("---")
        st.markdown('<div class="%s">%s<br><span style="font-size:1.1rem;opacity:0.92">Final risk %.0f%% = the higher of the FusionModel '
                    '(%.1f%%) and the evidence checks (%.0f%%)</span></div>' %
                    (css, label, A["final"] * 100, p_fusion * 100, A["evidence"] * 100), unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("FusionModel (ML)", "%.1f%%" % (p_fusion * 100), "model agreement: %s" % agree)
        m2.metric("Evidence checks", "%.0f%%" % (A["evidence"] * 100), "price, reviews, rating, seller, wording")
        m3.metric("Final risk", "%.0f%%" % (A["final"] * 100), A["verdict"])

        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=A["final"] * 100,
            title={"text": "Final fake risk %", "font": {"size": 14}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                   "steps": [{"range": [0, 40], "color": "#e8f5e9"}, {"range": [40, 70], "color": "#fff8e1"},
                             {"range": [70, 100], "color": "#ffebee"}],
                   "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 70}}))
        fig_g.update_layout(height=240, margin=dict(t=40, b=0, l=20, r=20))
        st.plotly_chart(fig_g, **WIDE, key="gauge_k")

        if A["missing"]:
            st.warning("Could not read from the page: **%s**. Missing data is never counted as zero, and a listing cannot be cleared "
                       "as genuine without a genuine-price reference and some review/rating history." % ", ".join(A["missing"]))
        for n in data.get("notes", []):
            st.info(n)

        st.markdown("### \u2705 Checks performed")
        rows = [{"": STATUS_ICON[c["status"]], "Check": c["check"], "What was read": c["read"], "Finding": c["finding"],
                 "Risk added": ("+%.0f%%" % (c["points"] * 100)) if c["points"] > 0 else "\u2014"} for c in A["checks"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, **WIDE)
        st.caption("Read by Gemini using: %s. Page actually read: %s. Weights come from the shopper survey (n = %s): price 72%%, "
                   "reviews 39%%, seller 39%%, spelling 35%%, rating 28%%." %
                   (data.get("read_with"), "yes" if data.get("page_read") else "not confirmed (search results only)", SURVEY_N))

        left, right = st.columns(2)
        seller_txt = data.get("seller_name") or "not identified"
        mk = ("AED %s \u2013 %s" % (format(data["min_market_price"], ",.0f"), format(data["max_market_price"], ",.0f"))) \
            if data.get("min_market_price") else "not found"
        with left:
            st.markdown("### \U0001F4E6 Product data read")
            for title_, val, cls in [
                ("\U0001F4CC Title", esc(data.get("title"))[:110], "card"),
                ("\U0001F3EA Platform", esc(data.get("platform")), "card"),
                ("\U0001F4B0 Listed price", "AED %s" % format(price, ",.2f"), "card"),
                ("\U0001F4C8 Genuine market price (Gemini + Google)", esc(mk), "card card-green" if data.get("min_market_price") else "card card-grey"),
                ("\u2B50 Rating", ("%.1f / 5" % data["rating"]) if data.get("rating") is not None else "could not be read",
                 "card" if data.get("rating") is not None else "card card-grey"),
                ("\U0001F4AC Reviews", ("{:,}".format(data["review_count"])) if data.get("review_count") is not None else "could not be read",
                 "card" if data.get("review_count") is not None else "card card-grey"),
                ("\U0001F3F7\uFE0F Seller", "%s (%s)" % (esc(seller_txt), esc(data.get("seller_type", "unknown")).replace("_", " ")),
                 "card" if data.get("seller_name") else "card card-grey"),
            ]:
                st.markdown('<div class="%s"><b>%s</b><br>%s</div>' % (cls, title_, val), unsafe_allow_html=True)
            if data.get("gemini_flags"):
                st.markdown("**Red flags Gemini noticed:** " + "; ".join(esc(f) for f in data["gemini_flags"]))
        with right:
            st.markdown("### \U0001F9E0 All 5 models")
            st.caption("The FusionModel is the machine-learning verdict. The other four are shown so you can see whether they agree.")
            for mname, prob in all_probs.items():
                info = MODEL_INFO[mname]; sel = mname == "FusionModel"
                mcol = "#e53935" if prob > 0.7 else "#fb8c00" if prob > 0.4 else "#43a047"
                mst = MODEL_STATS.get(mname, {})
                st.markdown('<div class="model-box%s"><div style="font-weight:700">%s %s%s</div>'
                            '<div style="font-size:1.6rem;font-weight:900;color:%s">%.1f%% fake</div>'
                            '<div style="font-size:0.75rem;color:#555">F1 %s%% | AUC %s</div></div>' %
                            (" model-selected" if sel else "", info["icon"], mname, " \u2190 ML verdict" if sel else "", mcol, prob * 100,
                             mst.get("f1", "?"), mst.get("auc", "?")), unsafe_allow_html=True)
            fig_m = go.Figure(go.Bar(x=list(all_probs.keys()), y=[v * 100 for v in all_probs.values()],
                                     marker_color=[MODEL_INFO[m]["color"] for m in all_probs],
                                     text=["%.1f%%" % (v * 100) for v in all_probs.values()], textposition="outside"))
            fig_m.add_hline(y=50, line_dash="dash", line_color="red")
            fig_m.update_layout(title="Model agreement: %s" % agree, height=270, showlegend=False,
                                yaxis=dict(title="Fake %", range=[0, 120]), margin=dict(t=40, b=20, l=10, r=10), plot_bgcolor="white")
            st.plotly_chart(fig_m, **WIDE, key="agree_k")

        st.markdown("### \U0001F4AC Explanation")
        st.caption("Gemini explains the result in plain English. It does not change the score or the verdict.")
        st.markdown('<div class="explanation-box">%s</div>' % esc(explanation).replace("\n", "<br>"), unsafe_allow_html=True)

        st.markdown("### \U0001F465 What UAE shoppers told us (survey, n = %s)" % SURVEY_N)
        for pct, lab, col in [(72, "Flag suspiciously low prices", "#ff4444"), (39, "Distrust zero / generic reviews", "#ff9800"),
                              (39, "Distrust unknown sellers", "#ff9800"), (35, "Flag spelling errors", "#ff9800"),
                              (28, "Suspicious of too many perfect ratings", "#ffc107")]:
            st.markdown('<div style="margin:0.35rem 0"><div style="display:flex;justify-content:space-between;font-size:0.85rem">'
                        '<span>%s</span><b style="color:%s">%d%%</b></div><div style="background:#f0f0f0;border-radius:10px;height:8px">'
                        '<div style="background:%s;width:%d%%;height:8px;border-radius:10px"></div></div></div>' % (lab, col, pct, col, pct),
                        unsafe_allow_html=True)
        st.warning("For academic research. The result is a probability, not proof. Always verify with the brand's official store.")

with tab2:
    st.markdown("## \U0001F4CA Model performance and SHAP")
    st.info("Test-set results on %s listings (training run saved with the models). High scores partly reflect rule-based labels and "
            "clean synthetic fakes; real-world accuracy is expected to be lower (see dissertation, Chapter 7)." % "{:,}".format(DS_INFO.get("total_products", 0)))
    df_res = pd.DataFrame({
        "Model": list(MODEL_STATS.keys()),
        "Accuracy": ["%s%%" % v["accuracy"] for v in MODEL_STATS.values()],
        "Precision": ["%s%%" % v["precision"] for v in MODEL_STATS.values()],
        "Recall": ["%s%%" % v["recall"] for v in MODEL_STATS.values()],
        "F1": ["%s%%" % v["f1"] for v in MODEL_STATS.values()],
        "AUC": [str(v.get("auc", "N/A")) for v in MODEL_STATS.values()],
        "CV F1": ["%s%%" % v["cv_mean"] for v in MODEL_STATS.values()]})
    st.dataframe(df_res, hide_index=True, **WIDE)
    top = shap_df.head(15)
    fig_s = go.Figure(go.Bar(x=top["shap_value"], y=top["feature"], orientation="h", marker_color="#9C27B0",
                             text=["%.3f" % v for v in top["shap_value"]], textposition="outside"))
    fig_s.update_layout(title="Top 15 features by mean |SHAP|", xaxis_title="Mean |SHAP value|", yaxis=dict(autorange="reversed"),
                        height=500, plot_bgcolor="white", margin=dict(l=250, r=50, t=40, b=40))
    st.plotly_chart(fig_s, **WIDE, key="shap_k")

with tab3:
    st.markdown("## \U0001F393 How the verdict is made")
    st.markdown("""
<div class="combo-box">
<b>1. Read the listing.</b> Gemini opens the product page (page access + Google Search) and returns title, price, rating, review count,
seller and the genuine UAE market price. It must quote the text it saw; anything it cannot see stays <i>unknown</i>, never zero.<br><br>
<b>2. FusionModel (machine learning).</b> 52 features go to three calibrated models (Gradient Boosting, Random Forest, XGBoost).
A Logistic Regression meta-learner combines them into one fake probability. Logistic Regression is also shown as a simple baseline.<br><br>
<b>3. Evidence checks (transparent rules).</b> Price margin against the genuine price, review count, rating pattern, seller, fake-listing wording.
The weights are the shopper-survey percentages (price 72%, reviews 39%, seller 39%, spelling 35%, rating 28%).<br><br>
<b>4. Final risk = the higher of the two.</b> Above 70% is red, 40-70% amber, below 40% green.<br><br>
<b>5. No green without evidence.</b> A listing is only called low risk if a genuine-price reference exists and some review or rating
history could be read. Otherwise the answer is <i>inconclusive</i>. If the price cannot be read, no verdict is given.
</div>""", unsafe_allow_html=True)
    st.markdown("### Limits to keep in mind")
    st.markdown("- The models were trained partly on rule-based labels and synthetic fake patterns, so scores are optimistic.\n"
                "- Gemini can misread a page; the checks table shows exactly what it read so a person can verify.\n"
                "- Seller information is self-reported by Gemini from the page and is not independently verified.\n"
                "- Noon.com links are the most reliable; Amazon.ae often blocks automated page reads.")

st.markdown("""<div style="text-align:center;color:#aaa;font-size:0.8rem;margin-top:2rem;padding-top:1rem;border-top:1px solid #eee">
MSc Data Science \u2022 Middlesex University Dubai \u2022 Nawfil Faraaz M01032971</div>""", unsafe_allow_html=True)
