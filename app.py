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
openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
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

    issues, sub_issues = [], []
    new_sentiments = []
    confidence_levels = []


    total = len(df)

    progress_bar = st.progress(0, text="Memulai proses klasifikasi...")

    for i, row in df.iterrows():
        prompt = (
            "Tentukan topik utama (Issue) dan sub-topik (Sub Issue) dari konten berikut berdasarkan campaign, judul, dan isi kontennya.\n\n"
            "- Issue adalah topik umum dan ringkas (maksimal 4 kata)\n"
            "- Sub Issue adalah penjabaran lebih spesifik dari Issue (maksimal 6 kata)\n"
            "- Keduanya wajib diisi\n\n"
            f"Campaigns: {row['Campaigns']}\n"
            f"Title: {row['Title']}\n"
            f"Content: {row['Content']}\n\n"
            "Jawab dalam format:\nIssue: <isi issue>\nSub Issue: <isi sub issue>\nNew Sentiment: <positive/negative/neutral>\nConfidence: <0-100>"


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
            result = "Issue: error\nSub Issue: error"

        issue_text, sub_issue_text, new_sentiment = "ambiguous", "ambiguous", "neutral"
        confidence = 0

        for line in result.splitlines():
            if line.lower().startswith("issue:"):
                issue_text = line.split(":", 1)[1].strip()
            elif line.lower().startswith("sub issue:"):
                sub_issue_text = line.split(":", 1)[1].strip()
            elif line.lower().startswith("new sentiment:"):
                new_sentiment = line.split(":", 1)[1].strip()
            elif line.lower().startswith("confidence:"):
                try:
                    confidence = int(line.split(":", 1)[1].strip())
                except:
                    confidence = 0

        if confidence < 85:
            new_sentiment = "neutral"

        issues.append(issue_text)
        sub_issues.append(sub_issue_text)
        new_sentiments.append(new_sentiment)

        progress_percent = int((i + 1) / total * 100)
        progress_bar.progress(min(progress_percent, 100), text=f"Memproses data... {progress_percent}%")

    progress_bar.progress(100, text="Menyiapkan data akhir...")

    df['Issue'] = issues
    df['Sub Issue'] = sub_issues
    df["New Sentiment"] = new_sentiments
    df["New Sentiment"] = df["New Sentiment"].str.lower()



    st.info("Menormalisasi Issue dan Sub Issue dengan GPT...")
    df['Issue'] = gpt_normalize_labels(df['Issue'], "Issue")
    df['Sub Issue'] = gpt_normalize_labels(df['Sub Issue'], "Sub Issue")

    keyword_list = keyword_config['Keyword'].dropna().tolist()
    df_news = pd.DataFrame()  # default empty

    if include_news:
        df_news = search_news_from_google(keyword_list, selected_project)

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
st.title("Topic Extraction Automation (Mainstream Media)")

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

    include_news = st.checkbox("Sertakan pencarian Berita External")

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
