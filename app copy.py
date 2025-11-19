import streamlit as st
import pandas as pd
import os
import sys
import tempfile
from datetime import datetime, timedelta
import gspread
from google.oauth2.service_account import Credentials
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import logging
import feedparser
import urllib.parse
import pytz
import re

# Load environment variables from .env file in secretcontainer
load_dotenv(dotenv_path=".secretcontainer/.env")
openai_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL", "gpt-40-mini")
client = OpenAI(api_key=openai_key)

# Setup logging
os.makedirs("logs", exist_ok=True)
log_filename = datetime.now().strftime("logs/log_%Y%m%d_%H%M%S.txt")
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

# Media yang disetujui
allowed_sources = [
    "jakarta post", "jakartaglobe", "detik", "liputan6", "cnn indonesia", "tvone", "republika",
    "kompas tv", "sindo news", "bisnis indonesia", "tempo", "investor daily", "kompas", "kontan",
    "metrotv", "antara", "kumparan", "idn times", "batampos", "cnbc indonesia", "media indonesia",
    "tirto", "voi"
]

#HAPUS
#path_ref_file = "Masterlist Media Tier.xlsx"
#st.write("Coba cek file referensi:", os.path.exists(path_ref_file))
#st.write("CWD:", os.getcwd())


def enrich_with_media_tier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raw:
      - Kolom URL yang dipakai: 'Link URL' dan 'Url' (keduanya, berurutan)
    Masterlist:
      - File: Masterlist Media Tier.xlsx (sheet 'Full Media')
      - Kolom: 'Media Url' (wajib), 'Media Tier' (wajib), 'Status' (opsional)
    Step:
      - Step 1 (opsional): match by 'Media' / 'Media Name' -> isi Media Tier
      - Step 2 pass#1: URL match hanya untuk Media Tier == 0
      - Step 2 pass#2: override hasil sebelumnya KECUALI baris dengan tier 1 & 2
    """
    import re
    from urllib.parse import urlparse

    path_ref_file = "Masterlist Media Tier.xlsx"

    # --- pastikan minimal salah satu kolom URL ada ---
    candidate_url_cols = [c for c in ["Link URL", "Url"] if c in df.columns]
    if not candidate_url_cols:
        raise ValueError("Raw data harus punya minimal salah satu kolom: 'Link URL' atau 'Url'.")

    df = df.copy()
    if "Media Tier" not in df.columns:
        df["Media Tier"] = 0

    # ---------- helpers ----------
    def clean_invisibles(s: str) -> str:
        if not isinstance(s, str):
            s = "" if s is None else str(s)
        return s.replace("\u200b", "").replace("\ufeff", "")

    def norm_host(u: str) -> str:
        u = clean_invisibles(u).strip()
        if not u:
            return ""
        # pastikan urlparse bisa baca
        if not re.match(r"^[a-z]+://", u, flags=re.I):
            u = "http://" + u
        host = urlparse(u).netloc.lower()
        # samakan prefix umum
        host = re.sub(r"^(www\.|m\.|amp\.)", "", host)
        return host

    def normalize_media(text):
        text = clean_invisibles(str(text)).lower()
        text = re.sub(r'[\s\.\-]', '', text)
        return text

    # ---------- load masterlist ----------
    try:
        ref_df = pd.read_excel(path_ref_file, sheet_name="Full Media")
    except Exception as e:
        st.error(f"[ERROR] Gagal baca Masterlist: {e}")
        return df

    # filter status (longgar, kalau ada)
    if "Status" in ref_df.columns:
        status_norm = ref_df["Status"].astype(str).str.strip().str.lower()
        ok_values = {"yes", "ya", "true", "1", "y", "ok", "approved"}
        ref_df = ref_df[status_norm.isin(ok_values) | status_norm.eq("yes")]

    # pastikan kolom masterlist
    media_url_col = "Media Url" if "Media Url" in ref_df.columns else (
        "Media URL" if "Media URL" in ref_df.columns else None
    )
    if media_url_col is None:
        raise ValueError("Masterlist harus punya kolom 'Media Url' atau 'Media URL'.")
    if "Media Tier" not in ref_df.columns:
        raise ValueError("Masterlist harus punya kolom 'Media Tier'.")

    # siapkan mapping host -> tier
    host_tiers = []
    for _, row in ref_df.iterrows():
        u_raw = row.get(media_url_col)
        t = row.get("Media Tier")
        if pd.notna(u_raw) and pd.notna(t):
            h = norm_host(str(u_raw))
            if h:
                try:
                    host_tiers.append((h, int(t)))
                except:
                    continue
    host_tiers.sort(key=lambda x: len(x[0]), reverse=True)

    def url_to_tier(u):
        h = norm_host(u)
        if not h:
            return None
        for mh, tier in host_tiers:
            # exact atau subdomain dari master host
            if h == mh or h.endswith("." + mh):
                return tier
        return None

    # ---------- Step 1: by Media Name (opsional) ----------
    media_col = "Media" if "Media" in df.columns else ("Media Name" if "Media Name" in df.columns else None)
    if media_col and "Media Name" in ref_df.columns:
        name_map = {}
        for _, row in ref_df.iterrows():
            mn = row.get("Media Name")
            mt = row.get("Media Tier")
            if pd.notna(mn) and pd.notna(mt):
                name_map[normalize_media(mn)] = int(mt)
        raw_norm = df[media_col].fillna("").map(lambda x: normalize_media(str(x)))
        mapped = raw_norm.map(name_map)
        df["Media Tier"] = mapped.fillna(df["Media Tier"]).fillna(0).astype(int)

    # ---------- Step 2 (pass #1): URL match hanya untuk tier == 0 ----------
    def fill_by_cols_pass1(df, cols):
        need = df["Media Tier"].fillna(0).astype(int).eq(0)
        if not need.any():
            return df
        for c in cols:
            # skip kalau kolom ada tapi seluruhnya NaN/kosong
            if c not in df.columns or not df[c].notna().any():
                continue
            s = df.loc[need, c].map(url_to_tier)
            df.loc[need & s.notna(), "Media Tier"] = s.dropna().astype(int)
            need = df["Media Tier"].fillna(0).astype(int).eq(0)
            if not need.any():
                break
        return df

    df = fill_by_cols_pass1(df, candidate_url_cols)

    # ---------- Step 2 (pass #2): override semua KECUALI tier 1 & 2 ----------
    keep_mask = df["Media Tier"].isin([1, 2])
    override_mask = ~keep_mask
    if override_mask.any():
        # cari tier dari kolom-kolom URL, ambil hasil pertama yang ketemu
        new_tiers2 = pd.Series(index=df.index, dtype="float64")
        for c in candidate_url_cols:
            if c not in df.columns or not df[c].notna().any():
                continue
            s = df.loc[override_mask, c].map(url_to_tier)
            # isi hanya yang masih NaN
            new_tiers2 = new_tiers2.where(new_tiers2.notna(), s)
        df.loc[override_mask & new_tiers2.notna(), "Media Tier"] = new_tiers2.dropna().astype(int)

    # ---------- rapikan posisi kolom ----------
    cols = df.columns.tolist()
    if "Media Tier" in cols:
        cols.remove("Media Tier")
        insert_after = "Link URL" if "Link URL" in df.columns else ("Url" if "Url" in df.columns else None)
        insert_idx = df.columns.get_loc(insert_after) + 1 if insert_after else len(cols)
        cols.insert(insert_idx, "Media Tier")
        df = df[cols]

    return df




def normalize_media(text):
    text = text.lower()
    text = re.sub(r'[\s\.\-]', '', text)
    return text

def gpt_normalize_labels(label_series: pd.Series, label_type: str) -> pd.Series:
    unique_labels = sorted(label_series.dropna().unique().tolist())
    joined_labels = "\n".join(f"- {lbl}" for lbl in unique_labels)

    prompt = (
        f"Saya memiliki daftar {label_type} hasil klasifikasi yang mengandung banyak istilah berbeda tapi memiliki makna yang mirip. "
        f"Tugas Anda adalah mengelompokkan istilah-istilah yang memiliki arti sama atau mirip ke dalam satu label final yang paling representatif dan ringkas. "
        f"Gabungkan semua variasi penulisan, singkatan, atau bentuk lain yang sebenarnya berarti sama.\n\n"
        f"Contoh:\n"
        f"- Dampak terhadap Rakyat dan Pekerjaan → Dampak Sosial\n"
        f"- Dampak pada Rakyat Miskin → Dampak Sosial\n"
        f"- Ruas Jalan Tol Pandaan-Malang → Ruas Jalan Pandaan-Malang\n"
        f"- Ruas Jalan Pandaan-Malang 2025 → Ruas Jalan Pandaan-Malang\n\n"
        f"Berikut daftarnya:\n{joined_labels}\n\n"
        f"Berikan hasil mapping dalam format:\n<Label Awal> → <Label Final>\n"
        f"Jangan sisakan label yang tidak dipetakan, dan usahakan maksimal 10 label final saja."
    )

    try:
        response = client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        output = response.choices[0].message.content
    except Exception as e:
        st.error(f"ERROR saat normalisasi GPT ({label_type}): {str(e)}")
        logging.error(f"[NORMALISASI ERROR - {label_type}] {str(e)}")
        return label_series  # fallback

    mapping = {}
    for line in output.splitlines():
        if "→" in line:
            parts = line.split("→")
            original = parts[0].strip()
            final = parts[1].strip()
            mapping[original] = final

    return label_series.apply(lambda x: mapping.get(x, x))

def search_news_from_google(keyword_list: list, project_name: str):
    all_results = []

    for keyword in keyword_list:
        query = urllib.parse.quote_plus(keyword)
        url = f"https://news.google.com/rss/search?q={query}&hl=id&gl=ID&ceid=ID:id"
        feed = feedparser.parse(url)

        logging.info(f"[DEBUG] Keyword: {keyword}, Total Entries: {len(feed.entries)}")

        count = 0
        for entry in feed.entries:
            if count >= 20:
                break

            try:
                published = datetime(*entry.published_parsed[:6], tzinfo=pytz.utc)
                media_name = entry.source.title if 'source' in entry else 'Unknown'
                all_results.append({
                    
                    "Project Name": project_name,
                    "Judul Berita": entry.title,
                    "Link Berita": entry.link,
                    "Media": media_name,
                    "Tanggal Publikasi": published.strftime("%Y-%m-%d %H:%M")
                })
                count += 1
            except Exception as e:
                logging.error(f"[ERROR parsing entry] {str(e)}")
                continue

        logging.info(f"[DEBUG] → Diambil {count} berita untuk '{keyword}'\n")

    # Ambil minimal 10 jika ada
    if len(all_results) < 10:
        logging.warning("Hanya sedikit berita yang tersedia dari semua keyword.")

    return pd.DataFrame(all_results[:max(10, len(all_results))])


# ======= Topic Extraction Function =======
def extract_topic_from_file(file_path: str, keyword_config, selected_project: str, include_news: bool) -> str:
    df = pd.read_excel(file_path)
    if "Campaign" in df.columns and "Campaigns" not in df.columns:
        df.rename(columns={"Campaign": "Campaigns"}, inplace=True)

    for col in ['Campaigns', 'Title', 'Content']:
        if col not in df.columns:
            df[col] = ""

    topics, sub_topics = [], []
    new_sentiments = []
    confidence_levels = []
    summaries = []

    total = len(df)

    progress_bar = st.progress(0, text="Memulai proses klasifikasi...")

    for i, row in df.iterrows():
        prompt = (
            "Tentukan topik utama (Topic), sub-topik (Sub Topic), dan rangkuman singkat (Summary) dari konten berikut berdasarkan campaign, judul, dan isi kontennya.\n\n"
            "- Topic adalah topik umum dan ringkas (maksimal 5 kata)\n"
            "- Sub Topic adalah penjabaran lebih spesifik dari Topic (maksimal 10 kata)\n"
            "- Summary adalah rangkuman isi konten dalam 4–20 kata yang menjelaskan isi konten, maksud, atau tujuannya secara jelas\n"
            "- Semua jawaban wajib diisi\n\n"
            f"Campaigns: {row['Campaigns']}\n"
            f"Title: {row['Title']}\n"
            f"Content: {row['Content']}\n\n"
            "Jawab dalam format:\nTopic: <isi Topic>\nSub Topic: <isi Sub Topic>\nSummary: <isi Summary>\nNew Sentiment: <positive/negative/neutral>\nConfidence: <0-100>"
        )

        try:
            response = client.chat.completions.create(
                model=openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            result = response.choices[0].message.content
            logging.info(f"Row {i} - SUCCESS\nPrompt: {prompt}\nResponse: {result}")
        except Exception as e:
            st.error(f"[Row {i}] GPT ERROR: {str(e)}")
            logging.error(f"Row {i} - ERROR: {str(e)}")
            result = "Topic: error\nSub Topic: error"

        topic_text, sub_topic_text, summary_text, new_sentiment = "ambiguous", "ambiguous", "", "neutral"
        confidence = 0

        for line in result.splitlines():
            if line.lower().startswith("topic:"):
                topic_text = line.split(":", 1)[1].strip()
            elif line.lower().startswith("sub topic:"):
                sub_topic_text = line.split(":", 1)[1].strip()
            elif line.lower().startswith("new sentiment:"):
                new_sentiment = line.split(":", 1)[1].strip()
            elif line.lower().startswith("summary:"):
                summary_text = line.split(":", 1)[1].strip()
            elif line.lower().startswith("confidence:"):
                try:
                    confidence = int(line.split(":", 1)[1].strip())
                except:
                    confidence = 0

        if confidence < 85:
            new_sentiment = "neutral"

        topics.append(topic_text)
        sub_topics.append(sub_topic_text)
        summaries.append(summary_text)
        new_sentiments.append(new_sentiment)

        progress_percent = int((i + 1) / total * 100)
        progress_bar.progress(min(progress_percent, 100), text=f"Memproses data... {progress_percent}%")

    progress_bar.progress(100, text="Menyiapkan data akhir...")

    df['Topic'] = topics
    df['Sub Topic'] = sub_topics
    df['Summary'] = summaries
    df["New Sentiment"] = new_sentiments
    df["New Sentiment"] = df["New Sentiment"].str.lower()
    
    # Urutkan kolom agar Topic, Sub Topic, New Sentiment, Summary berada setelah Content
    cols = df.columns.tolist()
    insert_after = "Content"

    new_cols = ["Topic", "Sub Topic", "New Sentiment", "Summary"]
    existing_new_cols = [col for col in new_cols if col in cols]

    if insert_after in cols:
        idx = cols.index(insert_after)
        # Hapus kolom baru dulu
        for col in existing_new_cols:
            cols.remove(col)
        # Sisipkan kolom baru setelah Content
        for i, col in enumerate(existing_new_cols):
            cols.insert(idx + 1 + i, col)

        df = df[cols]



    st.info("Menormalisasi Topic dan Sub Topic dengan GPT...")
    df['Topic'] = gpt_normalize_labels(df['Topic'], "Topic")
    df['Sub Topic'] = gpt_normalize_labels(df['Sub Topic'], "Sub Topic")

    # Tambahkan Media Tier ke file utama jika kolom media tersedia
    if "Media" in df.columns or "Media Name" in df.columns:
        df = enrich_with_media_tier(df)

    keyword_list = keyword_config['Keyword'].dropna().tolist()
    df_news = pd.DataFrame()  # default empty

    if include_news:
        df_news = search_news_from_google(keyword_list, selected_project)

    if not df_news.empty:
        df_news = enrich_with_media_tier(df_news)


    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = file_path.replace(".xlsx", f"_processed_{now}.xlsx")

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="Hasil Klasifikasi")
        if include_news:
            df_news.to_excel(writer, index=False, sheet_name="Berita External")

    return output_path

# ======= Google Sheets Config Loader =======
def load_keyword_config():
    json_path = Path(".secretcontainer/insightsautomation-460807-acdad1ee7590.json")
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file(json_path, scopes=scope)
    gc = gspread.authorize(creds)
    spreadsheet = gc.open_by_key("18_cjWzEUochd14i1t_eU2GvVVSZqB6y0paLyiHdCTyM")
    sheet = spreadsheet.get_worksheet(0)
    data = sheet.get_all_records()
    return pd.DataFrame(data)

# ======= Streamlit UI =======
st.title("Insights Topics Automation")

# Load keyword config dan pilih project
keyword_df = load_keyword_config()
project_list = keyword_df['Project Name'].unique().tolist()
selected_project = st.selectbox("Pilih Project Name", ["Pilih Project Terlebih Dahulu"] + project_list)
if selected_project == "Pilih Project Terlebih Dahulu":
    st.warning("Silakan pilih Project Name terlebih dahulu untuk melanjutkan.")
    st.stop()

project_keywords = keyword_df[keyword_df['Project Name'] == selected_project]

# Upload file
uploaded_file = st.file_uploader("Upload file Excel", type=[".xlsx"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    #include_news = st.checkbox("Sertakan pencarian Berita External")
    col1, col2 = st.columns([3, 1])
    with col1:
        include_news = st.checkbox("Sertakan pencarian Berita External")
    with col2:
        st.markdown(
            '<a href="https://docs.google.com/spreadsheets/d/18_cjWzEUochd14i1t_eU2GvVVSZqB6y0paLyiHdCTyM/edit#gid=0" target="_blank">📝 Edit Keyword External News</a>',
            unsafe_allow_html=True
        )


    if st.button("Proses"):
        with st.spinner("Memproses file..."):
            keyword_config = project_keywords
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = Path(uploaded_file.name).stem
            output_file = f"{base_name}_{now}_proced.xlsx"
            result_path = extract_topic_from_file(tmp_path, keyword_config, selected_project, include_news)

            final_path = os.path.join(os.path.dirname(result_path), output_file)
            os.rename(result_path, final_path)

        st.success("Selesai! File berhasil diproses.")
        with open(final_path, "rb") as f:
            st.download_button("Download Hasil", f, file_name=output_file)

    st.markdown(f"Nama file: `{uploaded_file.name}`")
