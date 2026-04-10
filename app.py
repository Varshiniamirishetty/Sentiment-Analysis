# ================= IMPORTS =================
import streamlit as st
import pandas as pd
import re
import requests
import matplotlib.pyplot as plt
from collections import Counter
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk import download
from transformers import pipeline

download('stopwords')

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Smart Review Analyzer", layout="wide")
st.title("🎬 Sentiment Analysis of Product Reviews")

st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1524985069026-dd778a71c7b4");
    background-size: cover;
}
.stApp::before {
    content: "";
    position: fixed; width: 100%; height: 100%;
    background: rgba(0,0,0,0.6);
    z-index: -1;
}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.title("🔧 Controls")

input_type = st.sidebar.selectbox(
    "Select Input Type",
    ["IMDb URL", "Upload CSV", "Type Your Review"]
)

analysis_type = st.sidebar.radio(
    "Choose Analysis",
    ["Sentiment Analysis", "Emotion Analysis", "Word Cloud"]
)

if input_type == "IMDb URL":
    target_count = st.sidebar.slider(
        "Number of reviews to fetch",
        min_value=25, max_value=300, value=100, step=25
    )
    debug_mode = st.sidebar.checkbox(
        "🛠️ Debug mode (show raw HTML)", value=False
    )
else:
    target_count = 100
    debug_mode   = False


# ================= HELPERS =================
def extract_title_id(url):
    match = re.search(r'(tt\d+)', url)
    return match.group(1) if match else None


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.imdb.com/",
}

REVIEW_SELECTORS = [
    ("css",  ".ipc-html-content-inner-div"),
    ("css",  ".text.show-more__control"),
    ("css",  ".review-container .text"),
    ("css",  ".imdb-user-review .content .text"),
    ("attr", {"data-testid": "review-overflow"}),
    ("attr", {"data-testid": "review-text"}),
    ("css",  "[class*='ReviewContent'] [class*='text']"),
    ("css",  ".lister-item-content p"),
]


def parse_reviews_from_soup(soup, debug=False):
    for kind, selector in REVIEW_SELECTORS:
        divs  = soup.select(selector) if kind == "css" else soup.find_all("div", attrs=selector)
        texts = [d.get_text(separator=" ", strip=True) for d in divs if len(d.get_text(strip=True)) > 30]
        if texts:
            if debug:
                st.success(f"✅ Selector matched `{selector}` → {len(texts)} items")
            return texts
    if debug:
        st.warning("⚠️ No CSS selector matched. Using long <p> fallback.")
    return [p.get_text(separator=" ", strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 80]


# ================= FETCH STRATEGIES =================
def fetch_via_ajax(title_id, target, debug):
    ajax_url = f"https://www.imdb.com/title/{title_id}/reviews/_ajax"
    reviews, seen, page_key = [], set(), ""
    progress = st.progress(0, text="🚀 Strategy 1: AJAX endpoint…")
    for _ in range(50):
        try:
            resp = requests.get(ajax_url, headers=HEADERS,
                                params={"ref_": "undefined", "paginationKey": page_key}, timeout=15)
        except Exception as e:
            st.error(f"AJAX request error: {e}"); break
        if debug:
            st.code(f"AJAX status={resp.status_code}\n{resp.text[:2000]}", language="html")
        if resp.status_code != 200:
            break
        soup  = BeautifulSoup(resp.text, "html.parser")
        texts = parse_reviews_from_soup(soup, debug=debug)
        for t in texts:
            if t not in seen:
                seen.add(t); reviews.append(t)
        progress.progress(min(int(len(reviews)/target*100), 99), text=f"📥 {len(reviews)}/{target}")
        if len(reviews) >= target:
            break
        lm = soup.find("div", class_="load-more-data")
        if lm and lm.get("data-key"):
            page_key = lm["data-key"]
        else:
            break
    progress.progress(100, text=f"✅ AJAX: {len(reviews)} reviews")
    return reviews


def fetch_via_requests(title_id, target, debug):
    progress = st.progress(0, text="🌐 Strategy 2: Direct requests…")
    try:
        resp = requests.get(f"https://www.imdb.com/title/{title_id}/reviews",
                            headers=HEADERS, timeout=20)
    except Exception as e:
        st.error(f"Request error: {e}"); return []
    if debug:
        st.code(f"Requests status={resp.status_code}\n{resp.text[:3000]}", language="html")
    if resp.status_code != 200:
        return []
    soup   = BeautifulSoup(resp.text, "html.parser")
    texts  = parse_reviews_from_soup(soup, debug=debug)
    reviews = list(dict.fromkeys(t for t in texts if len(t) > 30))
    progress.progress(100, text=f"✅ Requests: {len(reviews)} reviews")
    return reviews


def fetch_via_selenium(title_id, target, debug):
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import (NoSuchElementException,
        ElementClickInterceptedException, StaleElementReferenceException, TimeoutException)
    from webdriver_manager.chrome import ChromeDriverManager
    import time

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"})

    reviews, seen = [], set()
    progress = st.progress(0, text="🤖 Strategy 3: Selenium…")

    LOAD_MORE = [
        (By.CSS_SELECTOR, 'button[data-testid="load-more-btn"]'),
        (By.CSS_SELECTOR, 'button.ipc-btn--see-more'),
        (By.ID, 'load-more-trigger'),
        (By.XPATH, '//button[contains(normalize-space(),"Load More")]'),
        (By.XPATH, '//button[contains(normalize-space(),"See more")]'),
        (By.XPATH, '//button[contains(normalize-space(),"25 more")]'),
    ]

    try:
        driver.get(f"https://www.imdb.com/title/{title_id}/reviews")
        time.sleep(5)
        if debug:
            st.code(driver.page_source[:3000], language="html")

        for attempt in range(max(20, target // 13 + 5)):
            soup  = BeautifulSoup(driver.page_source, "html.parser")
            texts = parse_reviews_from_soup(soup, debug=(debug and attempt == 0))
            for t in texts:
                if t not in seen:
                    seen.add(t); reviews.append(t)
            progress.progress(min(int(len(reviews)/target*100), 99),
                              text=f"📥 {len(reviews)}/{target} (click {attempt+1})")
            if len(reviews) >= target:
                break
            clicked = False
            for by, sel in LOAD_MORE:
                try:
                    btn = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((by, sel)))
                    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
                    time.sleep(0.4)
                    driver.execute_script("arguments[0].click();", btn)
                    clicked = True; time.sleep(3); break
                except (NoSuchElementException, StaleElementReferenceException,
                        ElementClickInterceptedException, TimeoutException):
                    continue
            if not clicked:
                if debug:
                    btns = driver.find_elements(By.TAG_NAME, "button")
                    info = "\n".join(f"  text='{b.text.strip()}' testid='{b.get_attribute('data-testid')}'"
                                     for b in btns)
                    st.code(f"All buttons on page:\n{info}")
                break
    finally:
        driver.quit()

    progress.progress(100, text=f"✅ Selenium: {len(reviews)} reviews")
    return reviews


def fetch_imdb_reviews(url, target=100, debug=False):
    title_id = extract_title_id(url)
    if not title_id:
        st.error("❌ Could not find ttXXXXXXX in the URL.")
        return pd.DataFrame()

    st.info(f"🎯 Title ID: `{title_id}` — fetching up to **{target}** reviews")

    for label, fn in [
        ("1/3 — AJAX",     lambda: fetch_via_ajax(title_id, target, debug)),
        ("2/3 — Requests", lambda: fetch_via_requests(title_id, target, debug)),
        ("3/3 — Selenium", lambda: fetch_via_selenium(title_id, target, debug)),
    ]:
        st.markdown(f"**Strategy {label}**")
        reviews = fn()
        if len(reviews) >= 5:
            return pd.DataFrame({"Description": reviews[:target]})

    st.error("❌ All 3 strategies returned 0 reviews. Enable 🛠️ Debug mode and try again.")
    return pd.DataFrame()


# ================= SARCASM — ML MODEL + RULE HYBRID =================
@st.cache_resource
def load_sarcasm_model():
    """
    Uses a dedicated irony/sarcasm detection model trained on Twitter data.
    Falls back to rule-based if model fails to load.
    """
    try:
        return pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-irony",
            return_all_scores=False
        )
    except Exception:
        return None


def detect_sarcasm_rules(text):
    """
    Broad rule-based sarcasm: catches common patterns in movie reviews.
    Intentionally WIDER than before to actually flag things.
    """
    t = text.lower()

    # Pattern 1: positive adjective immediately followed or preceded by strong negative
    pos_words = r"(great|amazing|fantastic|wonderful|brilliant|masterpiece|perfect|incredible|outstanding|superb|excellent)"
    neg_words = r"(worst|terrible|awful|boring|waste|trash|garbage|disappointing|dull|bad|horrible|pathetic|unwatchable)"

    # Sarcastic structures common in reviews
    patterns = [
        # "oh great", "yeah sure", "oh wonderful"
        rf"\b(oh|yeah|sure|right)\s+{pos_words}",
        # "what a great waste", "what a brilliant mess"
        rf"\bwhat\s+a\s+{pos_words}.{{0,30}}{neg_words}",
        # positive + "but" + strong negative in short window
        rf"{pos_words}.{{0,40}}\bbut\b.{{0,60}}{neg_words}",
        # "totally worth it ... not"
        r"\b(totally|definitely|absolutely)\s+(worth|recommend).{0,50}\b(not|never)\b",
        # Exclamation + positive + immediately negative context
        rf"!.{{0,20}}{neg_words}",
        # "not" + positive adjective (e.g. "not great", "not amazing")
        rf"\bnot\s+{pos_words}",
        # "could have been" + positive = implicit sarcasm about failure
        rf"\bcould\s+have\s+been\s+{pos_words}",
        # "calling this" + positive is sarcasm
        rf"\bcalling\s+this\s+(a\s+)?{pos_words}",
        # Direct markers
        r"\b(sarcasm|sarcastically|ironically|i\s+guess)\b",
        r"\b(they\s+call\s+this|they\s+call\s+it)\b",
    ]

    return any(re.search(p, t) for p in patterns)


def detect_sarcasm_batch(texts):
    """
    Uses ML model (irony detection) when available.
    Falls back to rule-based otherwise.
    Returns list of booleans.
    """
    model = load_sarcasm_model()

    if model is not None:
        try:
            # Model labels: IRONY or NON_IRONY
            results = model(
                [t[:512] for t in texts],
                truncation=True,
                batch_size=16
            )
            # Flag as sarcastic if model says IRONY with confidence > 55%
            flags = [
                r["label"].upper() in ("IRONY", "LABEL_1") and r["score"] > 0.55
                for r in results
            ]
            # Also OR with rules to catch what the model misses
            rule_flags = [detect_sarcasm_rules(t) for t in texts]
            return [ml or rule for ml, rule in zip(flags, rule_flags)]
        except Exception:
            pass  # fall through to rules only

    # Pure rule-based fallback
    return [detect_sarcasm_rules(t) for t in texts]


# ================= NLP =================
@st.cache_resource
def load_models():
    return (
        pipeline("sentiment-analysis"),
        pipeline("text-classification",
                 model="j-hartmann/emotion-english-distilroberta-base")
    )


def analyze_reviews(reviews):
    reviews = reviews.copy()
    sentiment_model, emotion_model = load_models()
    reviews["Short"] = reviews["Description"].apply(lambda x: str(x)[:500])

    with st.spinner("🤖 Running sentiment analysis…"):
        sent = sentiment_model(reviews["Short"].tolist(), truncation=True, batch_size=16)
    reviews["Sentiment"]  = [r["label"] for r in sent]
    reviews["Confidence"] = [round(r["score"] * 100, 2) for r in sent]

    with st.spinner("🎭 Running emotion analysis…"):
        emo = emotion_model(reviews["Short"].tolist(), truncation=True, batch_size=16)
    reviews["Emotion"] = [r["label"] for r in emo]

    with st.spinner("🔍 Detecting sarcasm (ML model + rules)…"):
        sarcasm_flags = detect_sarcasm_batch(reviews["Short"].tolist())
    reviews["Sarcasm"] = sarcasm_flags

    # If ML sarcasm AND original sentiment is POSITIVE → flip to NEGATIVE
    reviews["Final Sentiment"] = reviews.apply(
        lambda row: "NEGATIVE" if row["Sarcasm"] and row["Sentiment"] == "POSITIVE"
                    else row["Sentiment"],
        axis=1
    )
    return reviews


# ================= WORD CLOUD =================
def show_wordcloud(reviews):
    text = re.sub(r"[^\w\s]", "", " ".join(reviews["Description"].tolist()))
    stop_words = set(stopwords.words("english"))
    text = " ".join(w for w in text.split() if w.lower() not in stop_words)
    if not text.strip():
        st.warning("Not enough text."); return
    wc = WordCloud(width=800, height=400, background_color="black").generate(text)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc); ax.axis("off")
    st.pyplot(fig)


# ================= CHARTS =================
def show_sentiment_chart(reviews):
    data   = Counter(reviews["Final Sentiment"])
    colors = {"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c", "NEUTRAL": "#f39c12"}
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(data.keys(), data.values(),
                  color=[colors.get(k, "#3498db") for k in data])
    ax.set_title("Sentiment Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sentiment"); ax.set_ylabel("Count")
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                str(int(b.get_height())), ha="center", va="bottom", fontsize=11)
    st.pyplot(fig)


def show_emotion_chart(reviews):
    data   = Counter(reviews["Emotion"])
    colors = ["#3498db","#e74c3c","#2ecc71","#f39c12","#9b59b6","#1abc9c","#e67e22"]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(data.keys(), data.values(), color=colors[:len(data)])
    ax.set_title("Emotion Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Emotion"); ax.set_ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                str(int(b.get_height())), ha="center", va="bottom", fontsize=10)
    st.pyplot(fig)


def show_sarcasm_table(reviews):
    """Show only the reviews flagged as sarcastic."""
    sarcastic = reviews[reviews["Sarcasm"] == True][["Description", "Sentiment", "Final Sentiment"]]
    if sarcastic.empty:
        st.info("No sarcastic reviews detected in this batch.")
    else:
        st.markdown(f"**{len(sarcastic)} sarcastic reviews detected:**")
        st.dataframe(sarcastic, use_container_width=True)


# ================= INPUT =================
reviews = None

if input_type == "IMDb URL":
    url = st.text_input("Enter IMDb Reviews URL",
                        placeholder="https://www.imdb.com/title/tt1375666/reviews")
    if st.button("Fetch Reviews"):
        if url:
            reviews = fetch_imdb_reviews(url, target=target_count, debug=debug_mode)
            if reviews is not None and not reviews.empty:
                st.success(f"✅ Fetched {len(reviews)} reviews!")
        else:
            st.warning("Please enter a URL first.")

elif input_type == "Upload CSV":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        reviews = pd.read_csv(file)
        for old in ["text", "review", "Review"]:
            if old in reviews.columns and "Description" not in reviews.columns:
                reviews.rename(columns={old: "Description"}, inplace=True)
        if "Description" not in reviews.columns:
            st.error("CSV must have a 'Description', 'text', or 'review' column.")
            st.stop()
        st.success(f"✅ Loaded {len(reviews)} rows.")

elif input_type == "Type Your Review":
    user_input = st.text_area("Enter your review here", height=180)
    if user_input:
        reviews = pd.DataFrame({"Description": [user_input]})

# ================= PROCESS =================
if reviews is not None and not reviews.empty:
    reviews = analyze_reviews(reviews)
    st.markdown("---")

    if analysis_type == "Sentiment Analysis":
        st.subheader("📊 Sentiment Analysis")
        total   = len(reviews)
        pos     = (reviews["Final Sentiment"] == "POSITIVE").sum()
        neg     = (reviews["Final Sentiment"] == "NEGATIVE").sum()
        sarcasm = int(reviews["Sarcasm"].sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Reviews",  total)
        c2.metric("Positive",  pos,  f"{pos/total*100:.1f}%")
        c3.metric("Negative",  neg,  f"{neg/total*100:.1f}%")
        c4.metric("Sarcasm Flags", sarcasm)

        st.dataframe(
            reviews[["Description", "Final Sentiment", "Confidence", "Sarcasm"]],
            use_container_width=True
        )

        # Show sarcastic reviews separately
        if sarcasm > 0:
            with st.expander(f"🎭 View {sarcasm} sarcastic reviews"):
                show_sarcasm_table(reviews)

        show_sentiment_chart(reviews)

    elif analysis_type == "Emotion Analysis":
        st.subheader("🎭 Emotion Analysis")
        st.dataframe(reviews[["Description", "Emotion"]], use_container_width=True)
        show_emotion_chart(reviews)

    elif analysis_type == "Word Cloud":
        st.subheader("☁️ Word Cloud")
        show_wordcloud(reviews)

else:
    st.info("👈 Select an input type from the sidebar and provide your data.")