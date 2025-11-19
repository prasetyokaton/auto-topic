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
import time
import json
import math

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

# Eligible channels untuk Summary dan Spokesperson
ELIGIBLE_CHANNELS = ['online media', 'printmedia', 'tv', 'newspaper', 'printed']


def validate_required_columns(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validasi kolom wajib yang harus ada di file upload.
    Returns: (is_valid, error_message)
    """
    required_cols = {
        'campaigns': ['Campaigns', 'Campaign'],
        'content': ['Content'],
        'title': ['Title'],
        'channel': ['Channel', 'Media type', 'Media Type']
    }
    
    missing = []
    
    # Check Campaigns/Campaign
    if not any(col in df.columns for col in required_cols['campaigns']):
        missing.append("Campaigns atau Campaign")
    
    # Check Content
    if 'Content' not in df.columns:
        missing.append("Content")
    
    # Check Title
    if 'Title' not in df.columns:
        missing.append("Title")
    
    # Check Channel/Media type
    if not any(col in df.columns for col in required_cols['channel']):
        missing.append("Channel atau Media type")
    
    if missing:
        error_msg = f"❌ Kolom wajib yang hilang: {', '.join(missing)}"
        return False, error_msg
    
    return True, ""


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
        # Tidak ada kolom URL, skip enrichment
        return df

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
        logging.warning(f"[WARN] Gagal baca Masterlist: {e}")
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
        logging.warning("[WARN] Masterlist tidak punya kolom 'Media Url'")
        return df
    if "Media Tier" not in ref_df.columns:
        logging.warning("[WARN] Masterlist tidak punya kolom 'Media Tier'")
        return df

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


def process_batch_with_gpt(batch_df: pd.DataFrame, batch_num: int, total_batches: int) -> pd.DataFrame:
    """
    Proses 1 batch (max 10 baris) dengan GPT.
    Retry logic: 3 attempts total (1 try + 2 retry)
    
    Returns: DataFrame dengan kolom baru atau empty values jika gagal
    """
    max_attempts = 3
    batch_size = len(batch_df)
    
    # Prepare batch data untuk prompt
    batch_data = []
    for idx, row in batch_df.iterrows():
        channel_lower = str(row.get('Channel', '')).lower().strip()
        is_eligible = channel_lower in ELIGIBLE_CHANNELS
        
        batch_data.append({
            "row": idx,
            "campaigns": str(row.get('Campaigns', '')),
            "title": str(row.get('Title', '')),
            "content": str(row.get('Content', '')),
            "channel": channel_lower,
            "is_eligible": is_eligible
        })
    
    # Build prompt
    prompt = f"""Proses {batch_size} baris data berita berikut. Return dalam format JSON array yang VALID.

RULES:
- Topic: dari Campaigns + Title + Content (maksimal 5 kata, ringkas dan umum)
- Sub Topic: dari Campaigns + Title + Content (maksimal 10 kata, lebih spesifik dari Topic)
- New Sentiment: dari Campaigns + Title + Content + Topic yang dihasilkan (hanya: positive/negative/neutral)
- New Summary: dari Title + Content (4-20 kata menjelaskan isi konten) - HANYA jika channel = online media/printmedia/tv/newspaper/printed, selain itu kosongkan dengan ""
- Spokesperson: dari Title + Content (nama orang yang menjadi juru bicara atau narasumber yang dikutip) - HANYA jika channel = online media/printmedia/tv/newspaper/printed, selain itu kosongkan dengan ""

PENTING UNTUK SPOKESPERSON:
- FORMAT WAJIB: "Nama Lengkap (Jabatan/Posisi)"
- Jika ADA LEBIH DARI 1 orang yang dikutip, pisahkan dengan koma: "Nama1 (Jabatan1), Nama2 (Jabatan2)"
- Jika tidak ada jabatan/posisi yang disebutkan, tetap tulis namanya saja dalam kurung kosong atau tanpa kurung
- Jika TIDAK ADA yang dikutip → kosongkan dengan "" (jangan isi "nan" atau "none")

CONTOH BENAR:
- 1 spokesperson: "Doddy Hanggodo (Menteri PUPR)"
- 2 spokesperson: "Doddy Hanggodo (Menteri PUPR), Agus Harimurti (Menko Infrastruktur)"
- Tanpa jabatan jelas: "John Doe (Direktur)", "Jane Smith"
- Tidak ada: ""

CONTOH SALAH:
- "Menteri PUPR Doddy Hanggodo" ❌ (format salah, harus: "Doddy Hanggodo (Menteri PUPR)")
- "Doddy Hanggodo, Menteri PUPR" ❌ (format salah)
- Hanya jabatan tanpa nama ❌

PENTING UNTUK NEW SUMMARY:
- Harus ringkas 4-20 kata menjelaskan ISI berita
- Jika channel BUKAN eligible, kosongkan dengan ""
- JANGAN isi dengan "nan", "none", atau "N/A"

Data:
{json.dumps(batch_data, indent=2, ensure_ascii=False)}

WAJIB return format JSON array seperti ini (tanpa markdown, tanpa backticks):
[
  {{
    "row": <index_baris>,
    "topic": "...",
    "sub_topic": "...",
    "new_sentiment": "positive/negative/neutral",
    "new_summary": "...",
    "spokesperson": "Nama (Jabatan)" atau "Nama1 (Jabatan1), Nama2 (Jabatan2)" atau ""
  }}
]

CRITICAL: 
- Pastikan JSON valid (gunakan double quotes)
- Spokesperson WAJIB format: "Nama (Jabatan)" - pisahkan multiple dengan koma
- New Summary dan Spokesperson: gunakan "" (empty string) jika tidak ada, JANGAN gunakan "nan", "none", "null", "N/A", "Unknown"
- Jika channel BUKAN online media/printmedia/tv/newspaper/printed, wajib kosongkan New Summary dan Spokesperson dengan ""
- Sentiment hanya boleh: positive, negative, atau neutral (lowercase)
- PROSES SEMUA {batch_size} BARIS dengan teliti, jangan skip!
"""
    
    # Retry logic
    for attempt in range(max_attempts):
        try:
            logging.info(f"[BATCH {batch_num}/{total_batches}] Attempt {attempt + 1}/{max_attempts}")
            
            response = client.chat.completions.create(
                model=openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Lower temperature for consistency
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Clean markdown if present
            result_text = re.sub(r'^```json\s*', '', result_text)
            result_text = re.sub(r'^```\s*', '', result_text)
            result_text = re.sub(r'\s*```$', '', result_text)
            result_text = result_text.strip()
            
            # Parse JSON
            results = json.loads(result_text)
            
            if not isinstance(results, list):
                raise ValueError("Response is not a JSON array")
            
            # Map results back to DataFrame
            result_dict = {item['row']: item for item in results}
            
            topics = []
            sub_topics = []
            new_summaries = []
            spokespersons = []
            sentiments = []
            
            for idx in batch_df.index:
                if idx in result_dict:
                    item = result_dict[idx]
                    topics.append(item.get('topic', 'ambiguous'))
                    sub_topics.append(item.get('sub_topic', 'ambiguous'))
                    
                    # Clean 'nan' strings from GPT response - expanded list
                    invalid_values = ['nan', 'none', 'null', 'n/a', 'na', 'unknown', 'tidak ada', '-']
                    
                    summary_val = item.get('new_summary', '')
                    if str(summary_val).lower().strip() in invalid_values:
                        summary_val = ''
                    new_summaries.append(summary_val)
                    
                    spokesperson_val = item.get('spokesperson', '')
                    if str(spokesperson_val).lower().strip() in invalid_values:
                        spokesperson_val = ''
                    spokespersons.append(spokesperson_val)
                    
                    sentiments.append(item.get('new_sentiment', 'neutral').lower())
                else:
                    # Fallback jika row tidak ada di response
                    topics.append('ambiguous')
                    sub_topics.append('ambiguous')
                    new_summaries.append('')
                    spokespersons.append('')
                    sentiments.append('neutral')
            
            batch_df = batch_df.copy()
            batch_df['Topic'] = topics
            batch_df['Sub Topic'] = sub_topics
            batch_df['New Sentiment'] = sentiments
            batch_df['New Summary'] = new_summaries
            batch_df['New Spokesperson'] = spokespersons
            
            logging.info(f"[BATCH {batch_num}/{total_batches}] SUCCESS")
            return batch_df
            
        except json.JSONDecodeError as e:
            logging.error(f"[BATCH {batch_num}/{total_batches}] Attempt {attempt + 1} - JSON Parse Error: {str(e)}")
            logging.error(f"Response text: {result_text[:500]}")
        except Exception as e:
            logging.error(f"[BATCH {batch_num}/{total_batches}] Attempt {attempt + 1} - Error: {str(e)}")
        
        # Wait before retry (except on last attempt)
        if attempt < max_attempts - 1:
            time.sleep(2)
    
    # If all attempts failed, return empty results
    logging.error(f"[BATCH {batch_num}/{total_batches}] FAILED after {max_attempts} attempts")
    batch_df = batch_df.copy()
    batch_df['Topic'] = ''
    batch_df['Sub Topic'] = ''
    batch_df['New Sentiment'] = 'neutral'
    batch_df['New Summary'] = ''
    batch_df['New Spokesperson'] = ''
    
    return batch_df


def gpt_normalize_labels(label_series: pd.Series, label_type: str) -> pd.Series:
    """
    Normalisasi labels yang similar menggunakan GPT.
    Optimized untuk Topic normalization.
    """
    unique_labels = sorted(label_series.dropna().unique().tolist())
    
    # Skip jika terlalu sedikit
    if len(unique_labels) <= 1:
        return label_series
    
    joined_labels = "\n".join(f"- {lbl}" for lbl in unique_labels)

    prompt = f"""Saya memiliki daftar {label_type} hasil klasifikasi yang mengandung banyak istilah berbeda tapi memiliki makna yang mirip atau sama.

TUGAS ANDA:
1. Kelompokkan istilah-istilah yang memiliki arti sama atau sangat mirip
2. Pilih 1 label final yang paling representatif, ringkas, dan jelas untuk setiap kelompok
3. Gabungkan semua variasi penulisan, singkatan, atau bentuk lain yang sebenarnya berarti sama
4. Usahakan maksimal 10-15 label final saja

CONTOH PENGELOMPOKAN:
- "Dampak terhadap Rakyat dan Pekerjaan" → "Dampak Sosial"
- "Dampak pada Rakyat Miskin" → "Dampak Sosial"
- "Pengaruh Ekonomi Masyarakat" → "Dampak Ekonomi"
- "Ruas Jalan Tol Pandaan-Malang" → "Infrastruktur Jalan Pandaan-Malang"
- "Ruas Jalan Pandaan-Malang 2025" → "Infrastruktur Jalan Pandaan-Malang"
- "Tol Pandaan Malang" → "Infrastruktur Jalan Pandaan-Malang"

Daftar {label_type}:
{joined_labels}

INSTRUKSI OUTPUT:
- Berikan hasil mapping dalam format: <Label Awal> → <Label Final>
- Setiap baris adalah 1 mapping
- Jangan sisakan label yang tidak dipetakan
- Label final harus ringkas, jelas, dan konsisten
- Jika label sudah bagus, boleh map ke dirinya sendiri

Format output:
Label A → Label Final 1
Label B → Label Final 1
Label C → Label Final 2
..."""

    try:
        response = client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Lower for consistency
        )
        output = response.choices[0].message.content
        
        logging.info(f"[NORMALISASI {label_type}] GPT Response:\n{output}")
    except Exception as e:
        st.error(f"ERROR saat normalisasi GPT ({label_type}): {str(e)}")
        logging.error(f"[NORMALISASI ERROR - {label_type}] {str(e)}")
        return label_series  # fallback

    mapping = {}
    for line in output.splitlines():
        if "→" in line:
            parts = line.split("→")
            if len(parts) == 2:
                original = parts[0].strip()
                final = parts[1].strip()
                # Remove leading "- " if present
                original = original.lstrip("- ")
                final = final.lstrip("- ")
                if original and final:
                    mapping[original] = final

    logging.info(f"[NORMALISASI {label_type}] Mapping: {mapping}")
    
    return label_series.apply(lambda x: mapping.get(x, x) if pd.notna(x) else x)


def gpt_normalize_spokesperson(spokesperson_series: pd.Series) -> pd.Series:
    """
    Normalisasi spokesperson yang merujuk orang yang sama.
    Format: Nama (Jabatan/Posisi), support multiple spokesperson.
    """
    unique_spokespersons = sorted(spokesperson_series.dropna().unique().tolist())
    
    # Filter hanya yang tidak kosong
    unique_spokespersons = [s for s in unique_spokespersons if s.strip() != '']
    
    # Skip jika terlalu sedikit
    if len(unique_spokespersons) <= 1:
        return spokesperson_series
    
    joined_spokespersons = "\n".join(f"- {sp}" for sp in unique_spokespersons)

    prompt = f"""Saya memiliki daftar SPOKESPERSON (juru bicara/narasumber) dari berbagai berita yang mengandung nama orang yang sama tetapi dengan berbagai variasi penulisan atau format.

FORMAT STANDAR YANG DIINGINKAN: "Nama Lengkap (Jabatan/Posisi)"

TUGAS ANDA:
1. Identifikasi nama-nama yang merujuk pada ORANG YANG SAMA
2. Gabungkan variasi nama tersebut ke dalam 1 nama final dengan FORMAT STANDAR
3. Jika nama berbeda orang, jangan digabung
4. PERTAHANKAN format "Nama (Jabatan)" dalam hasil akhir
5. Jika ada multiple spokesperson dalam 1 entry, pertahankan semua dengan format yang benar

CONTOH PENGELOMPOKAN:
Input:
- "Prabowo" → Output: "Prabowo Subianto (Presiden Indonesia)"
- "Presiden Prabowo" → Output: "Prabowo Subianto (Presiden Indonesia)"
- "Prabowo Subianto" → Output: "Prabowo Subianto (Presiden Indonesia)"
- "Presiden Indonesia Prabowo Subianto" → Output: "Prabowo Subianto (Presiden Indonesia)"
- "Jokowi" → Output: "Joko Widodo (Presiden RI periode sebelumnya)"
- "Doddy Hanggodo" → Output: "Doddy Hanggodo (Menteri PUPR)"
- "Menteri PUPR Doddy Hanggodo" → Output: "Doddy Hanggodo (Menteri PUPR)"

Multiple spokesperson:
- "Doddy Hanggodo, Agus Harimurti" → "Doddy Hanggodo (Menteri PUPR), Agus Harimurti (Menko Infrastruktur)"

PENTING:
- Pilih nama yang paling lengkap
- Jabatan harus dalam kurung setelah nama
- Jangan gabungkan nama orang yang berbeda hanya karena mirip
- Konsisten dengan ejaan formal Indonesia
- Untuk multiple spokesperson, pisahkan dengan koma dan spasi

Daftar Spokesperson:
{joined_spokespersons}

INSTRUKSI OUTPUT:
- Format: <Spokesperson Awal> → <Spokesperson Final>
- Setiap baris adalah 1 mapping
- Spokesperson final harus format: "Nama Lengkap (Jabatan/Posisi)"
- Jika sudah benar formatnya dan lengkap, boleh map ke dirinya sendiri
- Untuk multiple spokesperson, pastikan setiap nama punya format "(Jabatan)"

Format output:
Prabowo → Prabowo Subianto (Presiden Indonesia)
Presiden Prabowo → Prabowo Subianto (Presiden Indonesia)
Doddy Hanggodo → Doddy Hanggodo (Menteri PUPR)
Menteri PUPR Doddy → Doddy Hanggodo (Menteri PUPR)
..."""

    try:
        response = client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Lower for consistency
        )
        output = response.choices[0].message.content
        
        logging.info(f"[NORMALISASI SPOKESPERSON] GPT Response:\n{output}")
    except Exception as e:
        st.error(f"ERROR saat normalisasi Spokesperson: {str(e)}")
        logging.error(f"[NORMALISASI SPOKESPERSON ERROR] {str(e)}")
        return spokesperson_series  # fallback

    mapping = {}
    for line in output.splitlines():
        if "→" in line:
            parts = line.split("→")
            if len(parts) == 2:
                original = parts[0].strip()
                final = parts[1].strip()
                # Remove leading "- " if present
                original = original.lstrip("- ")
                final = final.lstrip("- ")
                if original and final:
                    mapping[original] = final

    logging.info(f"[NORMALISASI SPOKESPERSON] Mapping: {mapping}")
    
    return spokesperson_series.apply(lambda x: mapping.get(x, x) if pd.notna(x) and x.strip() != '' else x)


def search_news_from_google(keyword_list: list, project_name: str):
    """Search news from Google News RSS"""
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


def extract_topic_from_file(file_path: str, keyword_config, selected_project: str, include_news: bool) -> str:
    """
    Main function untuk proses batch extraction dengan GPT.
    """
    # Load data
    df = pd.read_excel(file_path)
    
    # Normalize Campaign -> Campaigns
    if "Campaign" in df.columns and "Campaigns" not in df.columns:
        df.rename(columns={"Campaign": "Campaigns"}, inplace=True)
    
    # Normalize Media type -> Channel
    if "Media type" in df.columns and "Channel" not in df.columns:
        df.rename(columns={"Media type": "Channel"}, inplace=True)
    elif "Media Type" in df.columns and "Channel" not in df.columns:
        df.rename(columns={"Media Type": "Channel"}, inplace=True)
    
    # Validate required columns
    is_valid, error_msg = validate_required_columns(df)
    if not is_valid:
        st.error(error_msg)
        raise ValueError(error_msg)
    
    # Ensure required columns exist
    for col in ['Campaigns', 'Title', 'Content', 'Channel']:
        if col not in df.columns:
            df[col] = ""
    
    total_rows = len(df)
    batch_size = 10
    total_batches = math.ceil(total_rows / batch_size)
    
    progress_bar = st.progress(0, text="Memulai proses batch...")
    
    # Process in batches
    processed_batches = []
    failed_batches = []
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_rows)
        
        batch_df = df.iloc[start_idx:end_idx].copy()
        
        progress_text = f"Memproses Batch {batch_num + 1}/{total_batches} (baris {start_idx + 1}-{end_idx})..."
        progress_bar.progress(int((batch_num / total_batches) * 90), text=progress_text)
        
        # Process batch
        result_batch = process_batch_with_gpt(batch_df, batch_num + 1, total_batches)
        
        # Check if batch failed (all empty)
        if result_batch['Topic'].eq('').all():
            failed_batches.append(batch_num + 1)
        
        processed_batches.append(result_batch)
    
    # Combine all batches
    df_processed = pd.concat(processed_batches, ignore_index=True)
    
    # Show warnings for failed batches
    if failed_batches:
        batch_ranges = []
        for fb in failed_batches:
            start = (fb - 1) * batch_size + 1
            end = min(fb * batch_size, total_rows)
            batch_ranges.append(f"Batch {fb} (baris {start}-{end})")
        
        st.warning(f"⚠️ Batch gagal setelah 3 percobaan: {', '.join(batch_ranges)}")
        logging.warning(f"[FAILED BATCHES] {failed_batches}")
    
    # === RETRY LOGIC FOR EMPTY NEW SPOKESPERSON ===
    progress_bar.progress(85, text="Memeriksa New Spokesperson yang kosong...")
    
    # Identify rows with empty spokesperson but eligible channel
    eligible_mask = df_processed['Channel'].str.lower().isin(ELIGIBLE_CHANNELS)
    empty_spokesperson_mask = (
        eligible_mask & 
        ((df_processed['New Spokesperson'].isna()) | 
         (df_processed['New Spokesperson'].astype(str).str.strip() == ''))
    )
    
    empty_spokesperson_count = empty_spokesperson_mask.sum()
    
    if empty_spokesperson_count > 0:
        st.info(f"🔄 Menemukan {empty_spokesperson_count} baris dengan New Spokesperson kosong. Melakukan retry...")
        logging.info(f"[RETRY SPOKESPERSON] {empty_spokesperson_count} rows to retry")
        
        # Get rows that need retry
        retry_indices = df_processed[empty_spokesperson_mask].index.tolist()
        retry_df = df_processed.loc[retry_indices].copy()
        
        # Process retry in batches of 10
        retry_batches = []
        for i in range(0, len(retry_df), batch_size):
            retry_batch = retry_df.iloc[i:i+batch_size]
            retry_batches.append(retry_batch)
        
        total_retry_batches = len(retry_batches)
        retry_success_count = 0
        
        for retry_idx, retry_batch in enumerate(retry_batches):
            progress_text = f"Retry Spokesperson - Batch {retry_idx + 1}/{total_retry_batches}..."
            progress_bar.progress(85 + int((retry_idx / total_retry_batches) * 5), text=progress_text)
            
            # Build focused prompt for spokesperson only
            batch_data = []
            for idx, row in retry_batch.iterrows():
                batch_data.append({
                    "row": idx,
                    "title": str(row.get('Title', '')),
                    "content": str(row.get('Content', '')),
                    "channel": str(row.get('Channel', '')).lower().strip()
                })
            
            focused_prompt = f"""Ekstrak SPOKESPERSON dari {len(batch_data)} berita berikut.

TUGAS ANDA:
Identifikasi SEMUA nama orang yang MEMBERIKAN PERNYATAAN atau DIKUTIP LANGSUNG dalam berita.

FORMAT WAJIB: "Nama Lengkap (Jabatan/Posisi)"

RULES:
- Cari kata kunci seperti: "mengatakan", "mengungkapkan", "menyatakan", "mengaku", "menjelaskan", "menegaskan", "menuturkan"
- Gunakan format: "Nama (Jabatan)"
- Jika ada LEBIH DARI 1 orang yang dikutip, pisahkan dengan koma: "Nama1 (Jabatan1), Nama2 (Jabatan2)"
- Jika tidak ada jabatan yang disebutkan, tulis nama saja atau dengan kurung kosong
- Jika TIDAK ADA yang dikutip → kosongkan dengan ""
- JANGAN isi dengan "nan", "none", "unknown", "tidak ada"

CONTOH BENAR:
- 1 orang: "Doddy Hanggodo (Menteri PUPR)"
- 2 orang: "Doddy Hanggodo (Menteri PUPR), Agus Harimurti (Menko Infrastruktur)"
- Tanpa jabatan: "John Doe"
- Tidak ada: ""

CONTOH SALAH:
- "Menteri PUPR Doddy Hanggodo" ❌ (harus: "Doddy Hanggodo (Menteri PUPR)")
- "Doddy, Agus" ❌ (kurang format lengkap)

Data:
{json.dumps(batch_data, indent=2, ensure_ascii=False)}

Return format JSON array (tanpa markdown):
[
  {{
    "row": <index>,
    "spokesperson": "Nama (Jabatan)" atau "Nama1 (Jabatan1), Nama2 (Jabatan2)" atau ""
  }}
]

PENTING: Fokus mencari SEMUA spokesperson, gunakan format yang benar!"""

            try:
                response = client.chat.completions.create(
                    model=openai_model,
                    messages=[{"role": "user", "content": focused_prompt}],
                    temperature=0.1,
                )
                
                result_text = response.choices[0].message.content.strip()
                result_text = re.sub(r'^```json\s*', '', result_text)
                result_text = re.sub(r'^```\s*', '', result_text)
                result_text = re.sub(r'\s*```$', '', result_text)
                result_text = result_text.strip()
                
                results = json.loads(result_text)
                
                if isinstance(results, list):
                    result_dict = {item['row']: item for item in results}
                    
                    for idx in retry_batch.index:
                        if idx in result_dict:
                            spokesperson_val = result_dict[idx].get('spokesperson', '')
                            
                            # Clean invalid values
                            invalid_values = ['nan', 'none', 'null', 'n/a', 'na', 'unknown', 'tidak ada', '-']
                            if str(spokesperson_val).lower().strip() not in invalid_values and spokesperson_val.strip():
                                df_processed.loc[idx, 'New Spokesperson'] = spokesperson_val
                                retry_success_count += 1
                                logging.info(f"[RETRY SUCCESS] Row {idx}: {spokesperson_val}")
                
            except Exception as e:
                logging.error(f"[RETRY ERROR] Batch {retry_idx + 1}: {str(e)}")
                continue
        
        if retry_success_count > 0:
            st.success(f"✅ Retry berhasil mengisi {retry_success_count} New Spokesperson yang tadinya kosong!")
            logging.info(f"[RETRY SUMMARY] {retry_success_count}/{empty_spokesperson_count} berhasil diisi")
        else:
            st.info(f"ℹ️ Retry selesai, namun tidak ada New Spokesperson tambahan yang bisa diekstrak")
    
    progress_bar.progress(90, text="Normalisasi Topic...")
    
    # Normalize Topic
    st.info("🔄 Menormalisasi Topic dengan GPT...")
    df_processed['Topic'] = gpt_normalize_labels(df_processed['Topic'], "Topic")
    
    progress_bar.progress(93, text="Normalisasi Sub Topic...")
    
    # Normalize Sub Topic
    st.info("🔄 Menormalisasi Sub Topic dengan GPT...")
    df_processed['Sub Topic'] = gpt_normalize_labels(df_processed['Sub Topic'], "Sub Topic")
    
    progress_bar.progress(96, text="Normalisasi Spokesperson...")
    
    # Normalize Spokesperson (only for eligible channels)
    eligible_mask = df_processed['Channel'].str.lower().isin(ELIGIBLE_CHANNELS)
    spokesperson_series = df_processed.loc[eligible_mask, 'New Spokesperson']
    
    if spokesperson_series.notna().any() and (spokesperson_series != '').any():
        st.info("🔄 Menormalisasi Spokesperson dengan GPT...")
        normalized_spokesperson = gpt_normalize_spokesperson(spokesperson_series)
        df_processed.loc[eligible_mask, 'New Spokesperson'] = normalized_spokesperson
    
    # Reorder columns: Content → Topic → Sub Topic → New Sentiment → New Summary → New Spokesperson
    cols = df_processed.columns.tolist()
    new_order_cols = ["Topic", "Sub Topic", "New Sentiment", "New Summary", "New Spokesperson"]
    
    # Remove new columns from current position
    for col in new_order_cols:
        if col in cols:
            cols.remove(col)
    
    # Insert after Content
    if "Content" in cols:
        content_idx = cols.index("Content")
        for i, col in enumerate(new_order_cols):
            if col in df_processed.columns:
                cols.insert(content_idx + 1 + i, col)
    
    df_processed = df_processed[cols]
    
    progress_bar.progress(98, text="Menambahkan Media Tier...")
    
    # Add Media Tier if media columns available
    if "Media" in df_processed.columns or "Media Name" in df_processed.columns:
        df_processed = enrich_with_media_tier(df_processed)
    
    # Search external news if requested
    df_news = pd.DataFrame()
    if include_news:
        progress_bar.progress(99, text="Mencari berita external...")
        keyword_list = keyword_config['Keyword'].dropna().tolist()
        df_news = search_news_from_google(keyword_list, selected_project)
        
        if not df_news.empty:
            df_news = enrich_with_media_tier(df_news)
    
    progress_bar.progress(100, text="Menyimpan hasil...")
    
    # Save to Excel
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = file_path.replace(".xlsx", f"_processed_{now}.xlsx")

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df_processed.to_excel(writer, index=False, sheet_name="Hasil Klasifikasi")
        if include_news and not df_news.empty:
            df_news.to_excel(writer, index=False, sheet_name="Berita External")

    return output_path


def load_keyword_config():
    """Load keyword configuration from Google Sheets"""
    json_path = Path(".secretcontainer/insightsautomation-460807-acdad1ee7590.json")
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file(json_path, scopes=scope)
    gc = gspread.authorize(creds)
    spreadsheet = gc.open_by_key("18_cjWzEUochd14i1t_eU2GvVVSZqB6y0paLyiHdCTyM")
    sheet = spreadsheet.get_worksheet(0)
    data = sheet.get_all_records()
    return pd.DataFrame(data)


# ======= Streamlit UI =======
st.title("🤖 Topics Automation v2.0")

# Load keyword config dan pilih project
try:
    keyword_df = load_keyword_config()
    project_list = keyword_df['Project Name'].unique().tolist()
    selected_project = st.selectbox("Pilih Project Name", ["Pilih Project Terlebih Dahulu"] + project_list)
    
    if selected_project == "Pilih Project Terlebih Dahulu":
        st.warning("⚠️ Silakan pilih Project Name terlebih dahulu untuk melanjutkan.")
        st.stop()
    
    project_keywords = keyword_df[keyword_df['Project Name'] == selected_project]
except Exception as e:
    st.error(f"❌ Error loading keyword config: {str(e)}")
    st.stop()

# Upload file
uploaded_file = st.file_uploader("📁 Upload file Excel", type=[".xlsx"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    # Preview validation
    try:
        preview_df = pd.read_excel(tmp_path)
        
        # Normalize column names
        if "Campaign" in preview_df.columns and "Campaigns" not in preview_df.columns:
            preview_df.rename(columns={"Campaign": "Campaigns"}, inplace=True)
        if "Media type" in preview_df.columns and "Channel" not in preview_df.columns:
            preview_df.rename(columns={"Media type": "Channel"}, inplace=True)
        elif "Media Type" in preview_df.columns and "Channel" not in preview_df.columns:
            preview_df.rename(columns={"Media Type": "Channel"}, inplace=True)
        
        is_valid, error_msg = validate_required_columns(preview_df)
        
        if is_valid:
            st.success(f"✅ File valid! Total: {len(preview_df)} baris")
        else:
            st.error(error_msg)
            st.stop()
    except Exception as e:
        st.error(f"❌ Error membaca file: {str(e)}")
        st.stop()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        include_news = st.checkbox("📰 Sertakan pencarian Berita External")
    with col2:
        st.markdown(
            '<a href="https://docs.google.com/spreadsheets/d/18_cjWzEUochd14i1t_eU2GvVVSZqB6y0paLyiHdCTyM/edit#gid=0" target="_blank">📝 Edit Keyword</a>',
            unsafe_allow_html=True
        )
    
    if st.button("🚀 Proses Sekarang", type="primary"):
        with st.spinner("⏳ Memproses file dengan batch processing..."):
            try:
                keyword_config = project_keywords
                now = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = Path(uploaded_file.name).stem
                output_file = f"{base_name}_{now}_processed.xlsx"
                result_path = extract_topic_from_file(tmp_path, keyword_config, selected_project, include_news)
                
                final_path = os.path.join(os.path.dirname(result_path), output_file)
                os.rename(result_path, final_path)
                
                st.success("✅ Selesai! File berhasil diproses.")
                st.balloons()
                
                with open(final_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Hasil",
                        data=f,
                        file_name=output_file,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            except Exception as e:
                st.error(f"❌ Error saat memproses: {str(e)}")
                logging.error(f"[MAIN ERROR] {str(e)}")
    
    st.markdown(f"**Nama file:** `{uploaded_file.name}`")

# Footer
st.markdown("---")
st.markdown("Powered by GPT-4o-mini • Batch Processing")