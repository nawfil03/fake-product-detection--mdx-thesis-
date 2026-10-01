"""evidence.py - helpers and the transparent "evidence layer" used by app_v2.py

Plain Python, no Streamlit, so every rule can be unit-tested.

The FusionModel gives one probability. This module checks the things a shopper
would check (price margin, review count, rating, seller, wording), says exactly
what it read from the page, and never treats "could not read" as "zero".
Final risk = the higher of the FusionModel probability and the evidence risk.
"""
import re
import json

# ---- weights = share of surveyed shoppers who selected each red flag ----------
W_PRICE, W_REVIEWS, W_SELLER, W_SPELLING, W_RATING = 0.72, 0.39, 0.39, 0.35, 0.28
W_STRONG_KW, W_MEDIUM_KW, W_CLAIMS = 0.85, 0.25, 0.30   # adversarial wording
RED_T, AMBER_T = 0.70, 0.40

UAE_BRAND_DB = {
    "iphone 17 pro max":5500,"iphone 17 pro":4800,"iphone 17":4200,
    "iphone 16 pro max":5000,"iphone 16 pro":4300,"iphone 16":3800,
    "iphone 15 pro max":4500,"iphone 15 pro":3800,"iphone 15":3200,
    "iphone 14 pro max":4000,"iphone 14 pro":3500,"iphone 14":2800,
    "iphone 13":2200,"iphone 12":1800,"iphone":1800,
    "samsung galaxy s24 ultra":4500,"samsung galaxy s24":3200,
    "samsung galaxy s23":2500,"samsung galaxy":800,"samsung":400,
    "macbook pro 16":9000,"macbook pro 14":7000,"macbook pro":6000,
    "macbook air m3":5500,"macbook air m2":4500,"macbook air":4000,
    "macbook":3500,"ipad pro":4000,"ipad air":3000,"ipad":1800,
    "apple watch ultra":3800,"apple watch series 9":2200,"apple watch":1500,
    "airpods pro":900,"airpods max":2200,"airpods":500,
    "rolex submariner":45000,"rolex daytona":85000,"rolex":20000,
    "omega seamaster":8000,"omega":5000,"cartier":8000,
    "tag heuer":4000,"breitling":8000,
    "louis vuitton neverfull":7000,"louis vuitton":3000,
    "gucci marmont":4500,"gucci belt":1800,"gucci":2000,
    "prada":2000,"balenciaga triple s":2500,"balenciaga":1800,
    "burberry":1200,"versace":900,"fendi":2500,
    "chanel classic flap":30000,"chanel":3000,
    "hermes birkin":80000,"hermes":5000,"dior bag":8000,"dior":500,
    "nike air jordan 1":1200,"nike air jordan":800,
    "nike air max":500,"nike dunk":600,"nike":200,
    "adidas yeezy 350":1800,"adidas yeezy":1500,
    "adidas ultraboost":600,"adidas":150,
    "new balance 990":800,"new balance":400,
    "dyson v15":2500,"dyson airwrap":2800,"dyson":800,
    "sony wh-1000xm5":1400,"sony":300,"bose qc45":1200,"bose":600,
    "beats studio":900,"beats":500,
    "ps5":1900,"playstation 5":1900,"xbox series x":1800,
    "optimum nutrition":150,"myprotein":100,"gnc":80,
    "la mer":800,"sk-ii":400,"estee lauder":300,
    "tom ford":600,"jo malone":400,"creed aventus":1500,
    "dior sauvage":450,"chanel no 5":800,
    "ralph lauren":400,"tommy hilfiger":300,"lacoste":400,
    "stone island":2000,"moncler":4000,"canada goose":3000,
    "under armour":200,"asics":400,"hoka":600,
}

FAKE_SP = ["niike","nkie","samsang","addidas","abidas","guci","chanell","rollex",
           "i-phone","aple","iphon","galxy","samsun","nikee","addidass","louiss","vuittton"]

STRONG_RE = re.compile(
    r"\b(replica|replicas|clone|copy|knock-?off|counterfeit|fake|mirror quality|"
    r"mirror copy|first copy|master copy|super copy|best copy|inspired by|1:1)\b|\baaa\b\+?|\bgrade a\b")
MEDIUM_RE = re.compile(
    r"\b(wholesale|bulk stock|direct factory|china stock|factory price|clearance|"
    r"imported?|international version|factory sealed|grey market)\b")
CLAIM_WORDS = ["original", "authentic", "genuine", "100%", "real", "official",
               "brand new sealed", "factory"]
SPELL_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in FAKE_SP) + r")\b")
SELLER_BAD_RE = re.compile(r"\b(wholesale|factory|replica|clone|dropship\w*|china|shenzhen|guangzhou)\b")

RATES_TO_AED = {"AED": 1.0, "USD": 3.6725, "EUR": 4.0, "GBP": 4.7, "INR": 0.044,
                "SAR": 0.979, "QAR": 1.009, "CNY": 0.51, "PKR": 0.013}
CURRENCY_ALIASES = {"DH": "AED", "DHS": "AED", "DIRHAM": "AED", "DIRHAMS": "AED", "AED": "AED",
                    "$": "USD", "US$": "USD", "USD": "USD", "\u20b9": "INR", "INR": "INR",
                    "\u20ac": "EUR", "EUR": "EUR", "\u00a3": "GBP", "GBP": "GBP",
                    "SAR": "SAR", "QAR": "QAR", "CNY": "CNY", "RMB": "CNY", "PKR": "PKR"}
CATEGORIES = ("electronics", "fashion", "cosmetics", "supplements")
SELLER_TYPES = ("official_brand_store", "platform_fulfilled", "third_party_marketplace",
                "individual_classified", "unknown")


# ============================ parsing helpers ============================
def _nullish(s):
    return s is None or str(s).strip().lower() in ("", "null", "none", "n/a", "na", "unknown",
                                                    "not found", "not available", "-")

def clean_text(raw, limit=140):
    return None if _nullish(raw) else str(raw).strip()[:limit]

def parse_count(raw):
    """'27.2K' -> 27200, '1,234' -> 1234, 0 -> 0, unreadable/None -> None (NOT zero)."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return int(round(raw)) if raw >= 0 else None
    s = str(raw).strip().lower().replace(",", "").replace(" ", "")
    if _nullish(s):
        return None
    m = re.match(r"^(\d+(?:\.\d+)?)([km])?", s)
    if not m:
        return None
    num = float(m.group(1))
    if m.group(2) == "k":
        num *= 1000
    elif m.group(2) == "m":
        num *= 1000000
    return int(round(num))

def to_float(raw):
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = re.sub(r"[^0-9.,]", "", str(raw)).replace(",", "")
    try:
        return float(s) if s else None
    except ValueError:
        return None

def extract_json(text):
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text)
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        return json.loads(t[a:b + 1])
    except Exception:
        return None

def get_brand_info(title, price):
    """UAE Brand Price Database lookup (unchanged from the thesis): level 0-5."""
    try:
        price = float(price or 0)
    except (TypeError, ValueError):
        return 0, 0, ""
    if not title or price <= 0:
        return 0, 0, ""
    tl = str(title).lower()
    best_brand, best_min, best_len = "", 0, 0
    for brand, min_p in UAE_BRAND_DB.items():
        if brand in tl and len(brand) > best_len:
            best_brand, best_min, best_len = brand, min_p, len(brand)
    if best_min > 0:
        level = 0
        if price < best_min * 0.15:   level = 5
        elif price < best_min * 0.25: level = 4
        elif price < best_min * 0.40: level = 3
        elif price < best_min * 0.55: level = 2
        elif price < best_min * 0.70: level = 1
        return level, best_min, best_brand
    return 0, 0, ""

def normalize(raw, platform=""):
    """Turn Gemini's JSON into clean, validated fields. Unknown stays None."""
    notes = []
    d = {"platform": platform}
    d["title"] = clean_text(raw.get("title"), 200)
    d["brand"] = clean_text(raw.get("brand"), 60)
    cat = str(raw.get("category") or "").lower().strip()
    d["category"] = cat if cat in CATEGORIES else "other"

    price = to_float(raw.get("price"))
    if price is not None and price <= 0:
        price = None
    cur_raw = clean_text(raw.get("currency"), 12)
    cur = CURRENCY_ALIASES.get(str(cur_raw or "AED").upper().strip(), str(cur_raw or "").upper())
    if price is not None and cur != "AED":
        rate = RATES_TO_AED.get(cur)
        if rate:
            notes.append("Price was shown in %s; converted to AED at about %s." % (cur, rate))
            price = round(price * rate, 2)
        else:
            notes.append("Price currency '%s' not recognised, price ignored." % cur)
            price = None
    d["price"] = price

    rating = to_float(raw.get("rating"))
    d["rating"] = rating if (rating is not None and 0 < rating <= 5) else None
    d["rating_text_seen"] = clean_text(raw.get("rating_text_seen"))

    count = parse_count(raw.get("review_count"))
    rtxt = clean_text(raw.get("review_text_seen"))
    if count == 0 and not rtxt:
        count = None   # an unread page is not the same as "zero reviews"
    d["review_count"] = count
    d["review_text_seen"] = rtxt

    d["seller_name"] = clean_text(raw.get("seller_name"), 80)
    st_ = str(raw.get("seller_type") or "unknown").lower().strip()
    d["seller_type"] = st_ if st_ in SELLER_TYPES else "unknown"
    sr = to_float(raw.get("seller_rating"))
    d["seller_rating"] = sr if (sr is not None and 0 < sr <= 5) else None
    d["seller_rating_count"] = parse_count(raw.get("seller_rating_count"))

    mn, mx = to_float(raw.get("min_market_price")), to_float(raw.get("max_market_price"))
    mn = mn if (mn and mn > 0) else None
    mx = mx if (mx and mx > 0) else None
    if mn and mx and mn > mx:
        mn, mx = mx, mn
    mn, mx = (mn or mx), (mx or mn)
    d["min_market_price"], d["max_market_price"] = mn, mx
    d["market_sources"] = [str(x)[:80] for x in (raw.get("market_sources") or []) if x][:3]

    d["page_read"] = bool(raw.get("page_read"))
    d["gemini_flags"] = [str(x)[:100] for x in (raw.get("red_flags") or []) if x][:6]
    if price and d["max_market_price"] and price > 3 * d["max_market_price"]:
        notes.append("Listed price is more than 3x the market range: check currency / extraction.")
    d["notes"] = notes
    return d


# ============================ risk checks ============================
def _sev_price(r):
    """r = listed price / lowest genuine price. Below 40% of it is red on its own."""
    if r >= 0.90: return 0.0
    if r >= 0.80: return 0.15
    if r >= 0.70: return 0.35
    if r >= 0.55: return 0.60
    if r >= 0.40: return 0.85
    return 1.0

def _sev_reviews(n):
    if n == 0: return 1.0
    if n < 5:  return 0.7
    if n < 10: return 0.4
    if n < 50: return 0.15
    return 0.0

def _sev_rating(r, n):
    few = n is not None and n < 10
    if r >= 4.95 and few: return 1.0
    if r >= 4.8 and few:  return 0.7
    if r >= 4.9 and n is not None and n > 200: return 0.5
    if r < 3.0: return 0.4
    return 0.0

def strong_keywords(title):
    tl = (title or "").lower()
    found = [m.group(0).strip("+") for m in STRONG_RE.finditer(tl)]
    out = []
    for w in found:
        if w.startswith("replica") and ("margiela" in tl or "maison" in tl):   # real perfume line
            continue
        if w == "aaa" and "batter" in tl:                                      # AAA batteries
            continue
        if w == "copy" and any(x in tl for x in ("paper", "print", "scan")):   # printers
            continue
        if w == "fake" and any(x in tl for x in ("plant", "flower", "tan", "lash", "nail", "snow")):
            continue
        if w not in out:
            out.append(w)
    return out

def seller_severity(d):
    name, typ = d.get("seller_name"), d.get("seller_type", "unknown")
    rate, cnt = d.get("seller_rating"), d.get("seller_rating_count")
    shown = name or "not identified"
    if name and SELLER_BAD_RE.search(name.lower()):
        return 0.9, shown, "seller name contains a wholesale/factory-style word", "bad"
    if typ in ("official_brand_store", "platform_fulfilled"):
        label = "official brand store" if typ == "official_brand_store" else "fulfilled by the platform"
        return 0.0, "%s (%s)" % (shown, label), "verified-type seller", "ok"
    if typ == "individual_classified":
        return 0.6, "%s (individual seller)" % shown, "private classified seller, no buyer protection", "warn"
    if typ == "third_party_marketplace":
        if rate is not None:
            if rate < 3.5: return 0.85, "%s (rating %.1f)" % (shown, rate), "low seller rating", "bad"
            if rate < 4.0: return 0.45, "%s (rating %.1f)" % (shown, rate), "mediocre seller rating", "warn"
            if cnt is not None and cnt < 20:
                return 0.35, "%s (rating %.1f, %d ratings)" % (shown, rate, cnt), "very new seller", "warn"
            return 0.0 if (rate >= 4.5 and (cnt or 0) >= 100) else 0.1, \
                   "%s (rating %.1f)" % (shown, rate), "rated third-party seller", "ok"
        return 0.35, "%s (third party, no rating)" % shown, "unrated third-party seller", "warn"
    if name:
        return 0.3, "%s (type unknown)" % shown, "seller named but could not be verified", "warn"
    return 0.5, "not identified", "no seller information found", "unknown"

def assess(data, p_model):
    """Combine the FusionModel probability with transparent evidence checks."""
    price = data.get("price")
    title = data.get("title") or ""
    mmin, mmax = data.get("min_market_price"), data.get("max_market_price")
    reviews, rating = data.get("review_count"), data.get("rating")
    checks, comps = [], []

    def add(name, read, finding, status, w, s):
        pts = round(w * s, 3)
        comps.append(pts)
        checks.append({"check": name, "read": read, "finding": finding, "status": status,
                       "points": pts})

    # 1 ---- price margin (survey weight 0.72)
    level, bmin, bname = get_brand_info(title, price)
    ref_low = ref_txt = None
    if mmin:
        ref_low = mmin
        ref_txt = "market AED %s-%s" % (format(mmin, ",.0f"), format(mmax or mmin, ",.0f"))
    elif bmin:
        ref_low = bmin
        ref_txt = "UAE Brand DB minimum AED %s (%s)" % (format(bmin, ",.0f"), bname)
    price_known = bool(price)
    if price_known and ref_low:
        r = price / ref_low
        s = _sev_price(r)
        gap = (1 - r) * 100
        finding = ("%.0f%% below the genuine minimum" % gap) if r < 1 else "at or above the genuine minimum"
        status = "bad" if s >= 0.85 else "warn" if s >= 0.35 else "ok"
        add("Price margin", "AED %s vs %s" % (format(price, ",.0f"), ref_txt), finding, status, W_PRICE, s)
    elif price_known:
        checks.append({"check": "Price margin", "read": "AED %s, no genuine price found" % format(price, ",.0f"),
                       "finding": "cannot judge the price", "status": "unknown", "points": 0.0})
    else:
        checks.append({"check": "Price margin", "read": "price not read", "finding": "cannot judge",
                       "status": "unknown", "points": 0.0})
    if mmin and bmin and level >= 2 and price_known:
        checks.append({"check": "Brand price database", "read": "%s, level %d/5" % (bname, level),
                       "finding": "also below the brand minimum", "status": "warn", "points": 0.0})

    # 2 ---- review count (0.39)
    if reviews is None:
        checks.append({"check": "Review count", "read": "could not be read", "finding": "unknown, NOT treated as zero",
                       "status": "unknown", "points": 0.0})
    else:
        s = _sev_reviews(reviews)
        quote = (" (seen: '%s')" % data["review_text_seen"]) if data.get("review_text_seen") else " (not quoted from page)"
        add("Review count", "%s reviews%s" % (format(reviews, ","), quote),
            "no review history" if reviews == 0 else "very few reviews" if reviews < 10 else
            "thin history" if reviews < 50 else "established history",
            "bad" if s >= 0.7 else "warn" if s >= 0.15 else "ok", W_REVIEWS, s)

    # 3 ---- rating pattern (0.28)
    if rating is None:
        checks.append({"check": "Rating pattern", "read": "could not be read", "finding": "unknown",
                       "status": "unknown", "points": 0.0})
    else:
        s = _sev_rating(rating, reviews)
        finding = ("perfect rating with almost no reviews" if s >= 0.7 else
                   "near-perfect rating with a huge review count (review bombing?)" if s == 0.5 and rating >= 4.9 else
                   "very poor rating" if s == 0.4 else "normal")
        add("Rating pattern", "%.1f / 5" % rating, finding,
            "bad" if s >= 0.7 else "warn" if s > 0 else "ok", W_RATING, s)

    # 4 ---- seller (0.39)
    s, shown, why, status = seller_severity(data)
    add("Seller check", shown, why, status, W_SELLER, s)

    # 5 ---- wording (adversarial)
    strong = strong_keywords(title)
    if strong:
        add("Fake-listing words", ", ".join(strong), "title openly uses replica/copy-style wording", "bad", W_STRONG_KW, 1.0)
    else:
        checks.append({"check": "Fake-listing words", "read": "none", "finding": "clean", "status": "ok", "points": 0.0})
    tl = title.lower()
    med = sorted(set(m.group(0) for m in MEDIUM_RE.finditer(tl)))
    if med:
        add("Suspicious sales wording", ", ".join(med), "wholesale / clearance / import style wording", "warn", W_MEDIUM_KW, 1.0)
    claims = [w for w in CLAIM_WORDS if w in tl]
    if len(claims) >= 2:
        add("Authenticity claims", ", ".join(claims), "piled-up 'original/genuine' claims", "warn", W_CLAIMS, 0.6)
    sp = sorted(set(m.group(0) for m in SPELL_RE.finditer(tl)))
    if sp:
        add("Misspelt brand", ", ".join(sp), "misspelt brand name in title", "bad", W_SPELLING, 1.0)

    # ---- combine: noisy-OR over the checks, then max with the model
    keep = 1.0
    for c in comps:
        keep *= (1.0 - min(max(c, 0.0), 0.99))
    evidence = round(1.0 - keep, 4)
    final = round(max(float(p_model), evidence), 4)

    # ---- data quality and the "no green without evidence" rule
    has_ref = bool(mmin or bmin)
    has_hist = (reviews is not None) or (rating is not None)
    seller_known = bool(data.get("seller_name")) or data.get("seller_type", "unknown") != "unknown"
    fields = {"price": price_known, "genuine price reference": has_ref, "reviews": reviews is not None,
              "rating": rating is not None, "seller": seller_known}
    missing = [k for k, v in fields.items() if not v]
    completeness = sum(fields.values()) / len(fields)

    if final >= RED_T:
        verdict = "fake"
    elif final >= AMBER_T:
        verdict = "suspicious"
    elif has_ref and has_hist:
        verdict = "genuine"
    else:
        verdict = "inconclusive"
    drivers = [c for c in sorted(checks, key=lambda c: -c["points"]) if c["points"] >= 0.08]
    return {"evidence": evidence, "final": final, "p_model": float(p_model), "verdict": verdict,
            "checks": checks, "missing": missing, "completeness": completeness,
            "drivers": drivers, "brand_level": level, "brand_name": bname}
