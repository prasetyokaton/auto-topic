# ── JINJA2 LRU CACHE PATCH (unhashable dict key fix) ────────────────────────
try:
    from jinja2.utils import LRUCache as _LRUCache
    
    def _make_hashable(key):
        if isinstance(key, dict):
            return tuple(sorted(key.items()))
        if isinstance(key, (list, tuple)):
            return tuple(_make_hashable(k) for k in key)
        return key
    
    _orig_getitem = _LRUCache.__getitem__
    _orig_setitem = _LRUCache.__setitem__
    _orig_get     = _LRUCache.get
    
    def _patched_getitem(self, key):
        return _orig_getitem(self, _make_hashable(key))
    
    def _patched_setitem(self, key, value):
        return _orig_setitem(self, _make_hashable(key), value)
    
    def _patched_get(self, key):
        return _orig_get(self, _make_hashable(key))
    
    _LRUCache.__getitem__ = _patched_getitem
    _LRUCache.__setitem__ = _patched_setitem
    _LRUCache.get         = _patched_get
    
    print("✅ Jinja2 LRUCache patch applied")
except Exception as e:
    print(f"⚠️ Jinja2 patch failed: {e}")

import gradio as gr
import pandas as pd
import os
import json
import re
import time
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
import tempfile
import math
import random
from collections import Counter
import hashlib

# Google Sheets integration
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import gradio_client.utils as gradio_utils
    
    _original_get_type = gradio_utils.get_type
    _original_json_schema_to_python_type = gradio_utils._json_schema_to_python_type
    
    def patched_get_type(schema):
        if isinstance(schema, bool):
            return "Any"
        if not isinstance(schema, dict):
            return "Any"
        return _original_get_type(schema)
    
    def patched_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        if not isinstance(schema, dict):
            return "Any"
        
        if "additionalProperties" in schema:
            if isinstance(schema["additionalProperties"], bool):
                schema = schema.copy()
                schema["additionalProperties"] = {"type": "string"}
        
        return _original_json_schema_to_python_type(schema, defs)
    
    gradio_utils.get_type = patched_get_type
    gradio_utils._json_schema_to_python_type = patched_json_schema_to_python_type
    
    print("✅ Gradio patch applied successfully")
    
except Exception as e:
    print(f"⚠️ Warning: Failed to patch Gradio: {e}")

load_dotenv(dotenv_path=".secretcontainer/.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ ERROR: OPENAI_API_KEY not found in .env file!")
    raise ValueError("Missing OPENAI_API_KEY")

async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ASYNC SETTINGS
PARALLEL_WORKERS = 10  # Process 10 rows simultaneously
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# CONTENT LIMITS
MIN_CONTENT_WORDS = 4  # Skip if < 4 words
MAX_TOKENS_OUTPUT = 150  # Max tokens per response
TRUNCATE_WORDS = 350  # Input truncation

# ENGAGEMENT WEIGHTS (for pillar generation)
SIMILARITY_THRESHOLD = 0.40

MAINSTREAM_CHANNELS = [
    'tv', 'radio', 'newspaper', 'online', 'printmedia', 'site',
    'printed', 'printedmedia', 'print', 'online media'
]

SOCIAL_CHANNELS = [
    'tiktok', 'instagram', 'youtube', 'facebook', 'twitter', 'x', 
    'threads', 'blog', 'forum'
]

LANGUAGE_CONFIGS = {
    "Indonesia": {
        "code": "id",
        "name": "Bahasa Indonesia",
        "prompt_instruction": "Use Bahasa Indonesia",
        "topic_range": "5-15 kata",
        "pillar_range": "2-6 kata"
    },
    "English": {
        "code": "en",
        "name": "English",
        "prompt_instruction": "Use English",
        "topic_range": "5-15 words",
        "pillar_range": "2-6 words"
    },
    "Thailand": {
        "code": "th",
        "name": "ภาษาไทย (Thai)",
        "prompt_instruction": "Use Thai language (ภาษาไทย)",
        "topic_range": "5-15 คำ",
        "pillar_range": "2-6 คำ"
    },
    "China": {
        "code": "zh",
        "name": "简体中文 (Simplified Chinese)",
        "prompt_instruction": "Use Simplified Chinese (简体中文)",
        "topic_range": "5-15 个词",
        "pillar_range": "2-6 个词"
    }
}

PRICING = {
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4o": {"input": 2.500, "output": 10.000},
    "gpt-3.5-turbo": {"input": 0.500, "output": 1.500},
}

# ── GSHEET CACHE CONFIG ─────────────────────────────────────────────────────
GSHEET_URL = "https://docs.google.com/spreadsheets/d/1oWq0j03boWJySrQCD14xqLBjO0E6-T_7q1EYb2FcdsU"
CACHE_DIR = "download"
CACHE_FILE = os.path.join(CACHE_DIR, "pillar_cache.json")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
log_filename = datetime.now().strftime("logs/gradio_log_%Y%m%d_%H%M%S.txt")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# ── GSHEET CACHE FUNCTIONS ──────────────────────────────────────────────────

def download_gsheet_cache():
    """Download Project List & Pillar Setup from GSheet via HTTP export, save to local cache."""
    try:
        import io as _io
        url = f"{GSHEET_URL}/export?format=xlsx"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 403:
            return False, "❌ Google Sheet tidak public. Share dulu dengan 'Anyone with the link can view'"
        elif response.status_code != 200:
            return False, f"❌ HTTP {response.status_code}: {response.reason}"
        
        # Parse xlsx from memory
        xl = pd.ExcelFile(_io.BytesIO(response.content))
        
        # Read Project List
        if 'Project List' not in xl.sheet_names:
            return False, "❌ Sheet 'Project List' tidak ditemukan"
        
        df_projects = xl.parse('Project List')
        projects = []
        if 'Project Name' in df_projects.columns:
            for val in df_projects['Project Name'].dropna():
                name = str(val).strip()
                if name and name != 'nan':
                    projects.append(name)
        
        # Read Pillar Setup
        if 'Pillar Setup' not in xl.sheet_names:
            return False, "❌ Sheet 'Pillar Setup' tidak ditemukan"
        
        df_pillars = xl.parse('Pillar Setup')
        pillar_setup = {}
        for _, row in df_pillars.iterrows():
            project = str(row.get("Project", "")).strip()
            pillar = str(row.get("Pillar", "")).strip()
            description = str(row.get("Description", "")).strip()
            if project and pillar and project != 'nan' and pillar != 'nan':
                if project not in pillar_setup:
                    pillar_setup[project] = []
                pillar_setup[project].append({
                    "pillar": pillar,
                    "description": description if description != 'nan' else ""
                })
        
        # Save to cache
        cache_data = {
            "projects": projects,
            "pillar_setup": pillar_setup,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        total_pillars = sum(len(v) for v in pillar_setup.values())
        logging.info(f"✅ Cache updated: {len(projects)} projects, {total_pillars} pillar entries")
        return True, f"✅ Cache updated: {len(projects)} projects, {total_pillars} pillar entries\n📅 {cache_data['last_updated']}"
    
    except requests.exceptions.Timeout:
        return False, "❌ Timeout. Cek koneksi internet."
    except requests.exceptions.ConnectionError:
        return False, "❌ Koneksi gagal. Cek network/firewall."
    except Exception as e:
        logging.error(f"GSheet download error: {e}")
        return False, f"❌ Error: {str(e)}"


def load_cache() -> dict:
    """Load cache from local file. Returns empty structure if not found."""
    if not os.path.exists(CACHE_FILE):
        return {"projects": [], "pillar_setup": {}, "last_updated": None}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"projects": [], "pillar_setup": {}, "last_updated": None}


def get_project_choices():
    """Return dropdown choices: Default + project names from cache."""
    cache = load_cache()
    projects = cache.get("projects", [])
    return ["Default (Auto)"] + projects


def get_pillar_setup(project_name):
    """Return list of {pillar, description} for a given project from cache."""
    if not project_name or project_name == "Default (Auto)":
        return []
    cache = load_cache()
    return cache.get("pillar_setup", {}).get(project_name, [])


# ── MANUAL PILLAR CLASSIFICATION (ASYNC) ────────────────────────────────────

async def classify_pillar_single_row_async(
    row_idx: int,
    content: str,
    pillars: list[dict],
    token_tracker,
    semaphore: asyncio.Semaphore
) -> dict:
    """Classify a single row to one of the predefined pillars. No retry."""
    async with semaphore:
        try:
            # Build pillar list for prompt
            pillar_lines = "\n".join(
                f"- **{p['pillar']}**: {p['description']}" for p in pillars
            )
            pillar_names = [p['pillar'] for p in pillars]
            
            prompt = f"""Anda adalah analis insights profesional.

Tugas: Tentukan PILLAR yang paling sesuai untuk konten berikut berdasarkan daftar pillar yang tersedia.

DAFTAR PILLAR:
{pillar_lines}

KONTEN:
{content}

ATURAN:
- Pilih SATU pillar yang paling sesuai dari daftar di atas
- Jika konten tidak cocok dengan pillar manapun atau kamu ragu, isi dengan "Other"
- Output HANYA nama pillar saja, tanpa penjelasan tambahan

OUTPUT:"""

            response = await async_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=30
            )
            
            if hasattr(response, 'usage'):
                token_tracker.add(response.usage.prompt_tokens, response.usage.completion_tokens)
            
            raw = response.choices[0].message.content.strip()
            
            # Validate: must be one of the pillar names or "Other"
            matched = None
            for name in pillar_names:
                if name.lower() == raw.lower() or name.lower() in raw.lower():
                    matched = name
                    break
            
            if not matched:
                matched = "Other"
            
            return {"row_idx": row_idx, "success": True, "pillar": matched}
        
        except Exception as e:
            logging.error(f"❌ Pillar classify row {row_idx} failed: {e}")
            token_tracker.add_failed()
            return {"row_idx": row_idx, "success": False, "pillar": "Other"}


async def classify_pillars_manual_async(
    df: pd.DataFrame,
    title_col: str,
    content_col: str,
    pillars: list[dict],
    token_tracker,
    progress_callback=None
) -> pd.DataFrame:
    """Classify all rows to predefined pillars using async parallel workers."""
    
    logging.info("\n" + "="*80)
    logging.info("[STEP 4] MANUAL PILLAR CLASSIFICATION (from Google Sheet)")
    logging.info(f"  Pillars: {[p['pillar'] for p in pillars]}")
    logging.info("="*80)
    
    if progress_callback is not None:
        progress_callback(0.90, desc="[STEP 4] Classifying pillars (manual mode)...")
    
    df['Pillar'] = ''
    semaphore = asyncio.Semaphore(PARALLEL_WORKERS)
    
    tasks = []
    for idx in df.index:
        # Only process master rows with topic
        if not df.at[idx, '_is_master']:
            continue
        topic = str(df.at[idx, 'Topic']).strip()
        if not topic:
            continue
        
        content = combine_title_content_row(df.loc[idx], title_col, content_col)
        task = classify_pillar_single_row_async(idx, content, pillars, token_tracker, semaphore)
        tasks.append(task)
    
    total_tasks = len(tasks)
    logging.info(f"📊 Classifying {total_tasks} rows into {len(pillars)} pillars")
    
    results = []
    for i in range(0, total_tasks, PARALLEL_WORKERS):
        batch = tasks[i:i + PARALLEL_WORKERS]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)
        
        if progress_callback is not None:
            pct = min(1.0, (i + len(batch)) / max(total_tasks, 1))
            progress_callback(0.90 + pct * 0.08, desc=f"[STEP 4] Classifying {i + len(batch)}/{total_tasks} rows")
    
    # Apply results
    for result in results:
        idx = result['row_idx']
        df.at[idx, 'Pillar'] = result['pillar']
    
    # Copy to duplicate rows
    for hash_val in df['_dedup_hash'].unique():
        group = df[df['_dedup_hash'] == hash_val]
        if len(group) > 1:
            master_idx = group[group['_is_master']].index[0]
            for dup_idx in group[~group['_is_master']].index:
                df.at[dup_idx, 'Pillar'] = df.at[master_idx, 'Pillar']
    
    filled = df['Pillar'].notna() & (df['Pillar'].astype(str).str.strip() != '')
    unique_pillars = df['Pillar'].nunique()
    token_tracker.add_step_stat("Pillar (Manual)", filled.sum(), len(df), unique=unique_pillars)
    logging.info(f"✅ Pillar classification done: {filled.sum()}/{len(df)} rows, {unique_pillars} unique pillars")
    
    return df


class TokenTracker:
    def __init__(self):
        self.total_input = 0
        self.total_output = 0
        self.api_calls = 0
        self.failed_calls = 0
        self.step_stats = {}
        self.row_errors = []
    
    def add(self, input_tokens, output_tokens):
        self.total_input += input_tokens
        self.total_output += output_tokens
        self.api_calls += 1
    
    def add_failed(self):
        self.failed_calls += 1
    
    def add_row_error(self, row_idx, error_msg):
        self.row_errors.append({"row": row_idx, "error": error_msg})
    
    def add_step_stat(self, step_name, success_count, total_count, **kwargs):
        success = int(success_count) if pd.notna(success_count) else 0
        total = int(total_count) if pd.notna(total_count) else 0
        rate = round(float(success / total * 100), 1) if total > 0 else 0.0
        
        self.step_stats[step_name] = {
            "success": success,
            "total": total,
            "rate": rate,
            **kwargs
        }
    
    def get_cost(self, model_name):
        if model_name not in PRICING:
            return 0.0
        pricing = PRICING[model_name]
        input_cost = (self.total_input / 1_000_000) * pricing["input"]
        output_cost = (self.total_output / 1_000_000) * pricing["output"]
        return input_cost + output_cost
    
    def get_summary(self, model_name):
        cost = self.get_cost(model_name)
        return {
            "input_tokens": int(self.total_input),
            "output_tokens": int(self.total_output),
            "total_tokens": int(self.total_input + self.total_output),
            "api_calls": int(self.api_calls),
            "failed_calls": int(self.failed_calls),
            "estimated_cost_usd": float(cost),
            "step_stats": self.step_stats,
            "row_errors": self.row_errors[:100]  # Max 100 errors in report
        }

def safe_text(x):
    return "" if pd.isna(x) else str(x).strip()

def truncate_to_first_n_words(text: str, n: int = TRUNCATE_WORDS) -> str:
    words = text.split()
    return " ".join(words[:n])

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    column_mapping = {}
    
    has_channel = any(col.lower().strip() == 'channel' for col in df.columns)
    has_media_type = any(col.lower().strip() in ['media type', 'mediatype'] for col in df.columns)
    
    for col in df.columns:
        col_lower = col.lower().strip()
        
        if col_lower in ['campaign', 'campaigns']:
            column_mapping[col] = 'Campaigns'
        elif col_lower == 'channel':
            column_mapping[col] = 'Channel'
        elif col_lower in ['media type', 'mediatype'] and not has_channel:
            column_mapping[col] = 'Channel'
        elif col_lower in ['title', 'judul']:
            column_mapping[col] = 'Title'
        elif col_lower in ['content', 'konten', 'isi']:
            column_mapping[col] = 'Content'
        elif col_lower in ['engagement']:
            column_mapping[col] = 'Engagement'
        elif col_lower in ['buzz']:
            column_mapping[col] = 'Buzz'
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
        logging.info(f"✅ Normalized columns: {column_mapping}")
    
    return df

def clean_content_for_analysis(text: str) -> str:
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001F900-\U0001F9FF"
        u"\U0001FA70-\U0001FAFF"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def count_meaningful_words(text: str) -> int:
    cleaned = clean_content_for_analysis(text)
    words = cleaned.split()
    return len(words)

def get_col(df, name_candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in name_candidates:
        c = cols.get(cand.lower())
        if c:
            return c
    return None

def combine_title_content_row(row, title_col, content_col):
    title = safe_text(row.get(title_col, "")) if title_col else ""
    content = safe_text(row.get(content_col, "")) if content_col else ""
    return " ".join([p for p in [title, content] if p]).strip()

def validate_required_columns(df: pd.DataFrame) -> tuple:
    if 'Channel' not in df.columns:
        return False, "❌ Error: Kolom 'Channel' harus ada!"
    
    required = {
        'content': ['Content', 'Konten', 'Isi'],
        'title': ['Title', 'Judul'],
        'campaigns': ['Campaigns', 'Campaign']
    }
    
    title_col = get_col(df, required['title'])
    content_col = get_col(df, required['content'])
    campaigns_col = get_col(df, required['campaigns'])
    
    if not title_col and not content_col:
        return False, "❌ Kolom 'Title' atau 'Content' harus ada!"
    
    if not campaigns_col:
        return False, "❌ Kolom 'Campaigns' harus ada!"
    
    return True, ""

def is_mainstream(channel: str) -> bool:
    if not channel or pd.isna(channel):
        return False
    return str(channel).strip().lower() in MAINSTREAM_CHANNELS

def is_social(channel: str) -> bool:
    if not channel or pd.isna(channel):
        return False
    return str(channel).strip().lower() in SOCIAL_CHANNELS

def create_dedup_hash(row, title_col, content_col):
    combined = combine_title_content_row(row, title_col, content_col)
    return hashlib.md5(combined.encode()).hexdigest()

def extract_json_from_response(s: str):
    if not s:
        return None
    
    s = re.sub(r'^```json\s*', '', s, flags=re.MULTILINE)
    s = re.sub(r'^```\s*', '', s, flags=re.MULTILINE)
    s = re.sub(r'\s*```$', '', s, flags=re.MULTILINE)
    s = s.strip()
    
    try:
        return json.loads(s)
    except:
        pass
    
    array_match = re.search(r'\[[\s\S]*\]', s)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except:
            pass
    
    object_match = re.search(r'\{[\s\S]*\}', s)
    if object_match:
        try:
            return json.loads(object_match.group(0))
        except:
            pass
    
    return None

async def process_single_row_async(
    row_idx: int,
    row_data: dict,
    language: str,
    conf_threshold: int,
    is_mainstream_row: bool,
    generate_spokesperson: bool,
    token_tracker,
    semaphore: asyncio.Semaphore
) -> dict:
    """Process single row with async API call"""
    
    async with semaphore:
        try:
            content = row_data['content']
            
            # Clean and truncate
            content_cleaned = clean_content_for_analysis(content)
            content_truncated = truncate_to_first_n_words(content_cleaned, TRUNCATE_WORDS)
            
            lang_config = LANGUAGE_CONFIGS[language]
            nonce = random.randint(100000, 999999)
            
            # Build prompt based on channel type
            if is_mainstream_row and generate_spokesperson:
                task_description = f"""Extract:
1. **TOPIC** ({lang_config['topic_range']}): Detailed, specific description with WHO/WHAT/WHERE
2. **SENTIMENT**: positive | negative | neutral
3. **CONFIDENCE**: 0-100 (how certain you are)
4. **SPOKESPERSON**: Person quoted in format "Name (Position)" or "-" if none"""
                
                output_format = """{{
  "topic": "...",
  "sentiment": "positive|negative|neutral",
  "confidence": 85,
  "spokesperson": "Name (Position)" or "-"
}}"""
            else:
                task_description = f"""Extract:
1. **TOPIC** ({lang_config['topic_range']}): Detailed, specific description with WHO/WHAT/WHERE
2. **SENTIMENT**: positive | negative | neutral
3. **CONFIDENCE**: 0-100 (how certain you are)"""
                
                output_format = """{{
  "topic": "...",
  "sentiment": "positive|negative|neutral",
  "confidence": 85
}}"""
            
            prompt = f"""You are an ELITE insights analyst.

[Request ID: {nonce}]
[OUTPUT LANGUAGE: {language}]

CONTENT (may be in ANY language):
{content_truncated}

TASK:
{task_description}

CRITICAL RULES:
- Content can be ANY language → Understand it → Output in {language}
- Topic MUST be {lang_config['topic_range']} with SPECIFIC details
- Include WHO + WHAT + WHERE when relevant
- Be SPECIFIC, not generic
- If cannot extract meaningful insights, use "-" for topic

OUTPUT FORMAT (JSON only):
{output_format}

YOUR OUTPUT (JSON only):"""

            # API call with retry logic
            for attempt in range(MAX_RETRIES):
                try:
                    response = await async_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": f"You are an ELITE insights analyst. {lang_config['prompt_instruction']}. Handle multi-language input. Output in {language}."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=MAX_TOKENS_OUTPUT
                    )
                    
                    # Track tokens
                    if hasattr(response, 'usage'):
                        token_tracker.add(response.usage.prompt_tokens, response.usage.completion_tokens)
                    
                    # Parse response
                    raw = response.choices[0].message.content.strip()
                    result = extract_json_from_response(raw)
                    
                    if not result:
                        raise ValueError("Failed to parse JSON response")
                    
                    # Extract fields
                    topic = str(result.get('topic', '')).strip()
                    sentiment = str(result.get('sentiment', 'neutral')).lower().strip()
                    confidence = int(result.get('confidence', 0))
                    spokesperson = str(result.get('spokesperson', '')).strip() if is_mainstream_row and generate_spokesperson else ''
                    
                    # Validate sentiment
                    if sentiment not in ['positive', 'negative', 'neutral']:
                        sentiment = 'neutral'
                    
                    # Confidence threshold
                    confidence = max(0, min(100, confidence))
                    if confidence < conf_threshold and sentiment != 'neutral':
                        sentiment = 'neutral'
                    
                    # Clean topic
                    if topic == '-' or not topic:
                        topic = ''
                    
                    # Clean spokesperson
                    if spokesperson == '-' or not spokesperson:
                        spokesperson = ''
                    
                    return {
                        'row_idx': row_idx,
                        'success': True,
                        'topic': topic,
                        'sentiment': sentiment,
                        'confidence': confidence,
                        'spokesperson': spokesperson
                    }
                    
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                    else:
                        raise
            
            # If all retries failed
            raise Exception("Max retries exceeded")
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"❌ Row {row_idx} failed: {error_msg}")
            token_tracker.add_failed()
            token_tracker.add_row_error(row_idx, error_msg)
            
            return {
                'row_idx': row_idx,
                'success': False,
                'topic': '',
                'sentiment': 'neutral',
                'confidence': 0,
                'spokesperson': '',
                'error': error_msg
            }

async def process_all_rows_async(
    df: pd.DataFrame,
    title_col: str,
    content_col: str,
    language: str,
    conf_threshold: int,
    generate_spokesperson: bool,
    token_tracker,
    progress_callback=None
) -> pd.DataFrame:
    """Process all eligible rows with async parallel execution"""
    
    logging.info("\n" + "="*80)
    logging.info("[STEP 1] ASYNC PER-ROW PROCESSING")
    logging.info("="*80)
    
    # Initialize columns
    df['Topic'] = ''
    df['New Sentiment'] = 'neutral'
    df['New Sentiment Level'] = 0
    if generate_spokesperson:
        df['New Spokesperson'] = ''
    
    # Prepare tasks for eligible master rows
    tasks = []
    row_mapping = {}  # Map task index to DataFrame index
    
    for idx in df.index:
        if not df.at[idx, '_is_master']:
            continue
        
        if not df.at[idx, '_eligible_for_topic']:
            continue
        
        content = combine_title_content_row(df.loc[idx], title_col, content_col)
        is_mainstream_row = df.at[idx, '_is_mainstream']
        
        row_data = {'content': content}
        
        task = process_single_row_async(
            row_idx=idx,
            row_data=row_data,
            language=language,
            conf_threshold=conf_threshold,
            is_mainstream_row=is_mainstream_row,
            generate_spokesperson=generate_spokesperson,
            token_tracker=token_tracker,
            semaphore=asyncio.Semaphore(PARALLEL_WORKERS)
        )
        
        tasks.append(task)
        row_mapping[len(tasks) - 1] = idx
    
    total_tasks = len(tasks)
    logging.info(f"📊 Processing {total_tasks} master rows with {PARALLEL_WORKERS} parallel workers")
    
    # Process with progress tracking
    results = []
    for i in range(0, total_tasks, PARALLEL_WORKERS):
        batch_tasks = tasks[i:i + PARALLEL_WORKERS]
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)
        
        # Update progress
        progress = min(1.0, (i + len(batch_tasks)) / total_tasks)
        if progress_callback is not None:
            progress_callback(0.1 + progress * 0.6, desc=f"[STEP 1] Processing {i + len(batch_tasks)}/{total_tasks} rows")
    
    # Apply results to DataFrame
    success_count = 0
    failed_count = 0
    
    for result in results:
        row_idx = result['row_idx']
        
        if result['success']:
            df.at[row_idx, 'Topic'] = result['topic']
            df.at[row_idx, 'New Sentiment'] = result['sentiment']
            df.at[row_idx, 'New Sentiment Level'] = result['confidence']
            if generate_spokesperson:
                df.at[row_idx, 'New Spokesperson'] = result['spokesperson']
            success_count += 1
        else:
            failed_count += 1
    
    # Copy results to duplicate rows
    logging.info("📋 Copying results to duplicate rows...")
    for hash_val in df['_dedup_hash'].unique():
        group = df[df['_dedup_hash'] == hash_val]
        if len(group) > 1:
            master_idx = group[group['_is_master']].index[0]
            duplicate_indices = group[~group['_is_master']].index
            
            for dup_idx in duplicate_indices:
                df.at[dup_idx, 'Topic'] = df.at[master_idx, 'Topic']
                df.at[dup_idx, 'New Sentiment'] = df.at[master_idx, 'New Sentiment']
                df.at[dup_idx, 'New Sentiment Level'] = df.at[master_idx, 'New Sentiment Level']
                if generate_spokesperson:
                    df.at[dup_idx, 'New Spokesperson'] = df.at[master_idx, 'New Spokesperson']
    
    logging.info(f"✅ Step 1 complete: {success_count} success, {failed_count} failed")
    
    # Track stats
    topic_filled = df['Topic'].notna() & (df['Topic'].astype(str).str.strip() != '')
    token_tracker.add_step_stat("Topic", topic_filled.sum(), len(df))
    
    sentiment_filled = df['New Sentiment'].notna()
    token_tracker.add_step_stat("Sentiment", sentiment_filled.sum(), len(df))
    
    if generate_spokesperson:
        spokes_filled = df['New Spokesperson'].notna() & (df['New Spokesperson'].astype(str).str.strip() != '')
        token_tracker.add_step_stat("Spokesperson", spokes_filled.sum(), len(df))
    
    return df

def normalize_topics_per_campaign(
    df: pd.DataFrame,
    language: str,
    token_tracker,
    progress_callback=None
) -> pd.DataFrame:
    """Normalize similar topics within each campaign"""
    
    logging.info("\n" + "="*80)
    logging.info("[STEP 2] NORMALIZE TOPICS PER CAMPAIGN")
    logging.info("="*80)
    
    if progress_callback is not None:
        progress_callback(0.75, desc="[STEP 2] Normalizing topics...")
    
    campaigns = df['Campaigns'].unique()
    
    for campaign in campaigns:
        campaign_mask = df['Campaigns'] == campaign
        topics = df[campaign_mask]['Topic'].dropna()
        topics = topics[topics.astype(str).str.strip() != '']
        unique_topics = sorted(topics.unique().tolist())
        
        if len(unique_topics) <= 1:
            continue
        
        logging.info(f"📊 Campaign '{campaign}': {len(unique_topics)} unique topics")
        
        # Call LLM to normalize
        topic_list = "\n".join(f"- {t}" for t in unique_topics)
        nonce = random.randint(100000, 999999)
        
        prompt = f"""You are a topic normalization expert.

[Request ID: {nonce}]
[Campaign: {campaign}]

INPUT: {len(unique_topics)} unique topics

Topics:
{topic_list}

TASK:
Identify and MERGE topics that refer to the SAME thing/event.

RULES:
- Merge only if truly the same
- Keep the most descriptive name
- Preserve distinct topics

OUTPUT FORMAT:
For each merge:
<Original Topic> → <Normalized Topic>

For unchanged topics:
<Topic> → <Topic>

YOUR OUTPUT:"""
        
        try:
            # Sync call for normalization (simpler)
            import openai
            sync_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            
            response = sync_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            
            if hasattr(response, 'usage'):
                token_tracker.add(response.usage.prompt_tokens, response.usage.completion_tokens)
            
            output = response.choices[0].message.content
            mapping = {}
            
            for line in output.splitlines():
                if "→" in line:
                    parts = line.split("→")
                    if len(parts) == 2:
                        original = parts[0].strip().lstrip("- ")
                        normalized = parts[1].strip().lstrip("- ")
                        if original and normalized:
                            mapping[original] = normalized
            
            # Fill missing
            for t in unique_topics:
                if t not in mapping:
                    mapping[t] = t
            
            # Apply mapping
            for idx in df[campaign_mask].index:
                topic = df.at[idx, 'Topic']
                if topic and str(topic).strip():
                    normalized = mapping.get(topic, topic)
                    df.at[idx, 'Topic'] = normalized
            
            merged_count = len(unique_topics) - len(set(mapping.values()))
            logging.info(f"  └─ Merged {merged_count} topics")
            
        except Exception as e:
            logging.error(f"Error normalizing topics for campaign '{campaign}': {e}")
    
    return df

def normalize_spokesperson(
    df: pd.DataFrame,
    token_tracker,
    progress_callback=None
) -> pd.DataFrame:
    """Normalize spokesperson names"""
    
    logging.info("\n" + "="*80)
    logging.info("[STEP 3] NORMALIZE SPOKESPERSON")
    logging.info("="*80)
    
    if progress_callback is not None:
        progress_callback(0.85, desc="[STEP 3] Normalizing spokesperson...")
    
    spokespersons = df['New Spokesperson'].dropna()
    spokespersons = spokespersons[spokespersons.astype(str).str.strip() != '']
    unique_spokespersons = sorted(spokespersons.unique().tolist())
    
    if len(unique_spokespersons) <= 1:
        logging.info("No spokesperson to normalize")
        return df
    
    logging.info(f"📊 {len(unique_spokespersons)} unique spokespersons")
    
    joined = "\n".join(f"- {sp}" for sp in unique_spokespersons)
    nonce = random.randint(100000, 999999)
    
    prompt = f"""Normalize spokesperson names referring to the SAME person.

[Request ID: {nonce}]

Spokespersons:
{joined}

TASK: Merge names referring to the same person

OUTPUT FORMAT:
<Original> → <Normalized>

YOUR OUTPUT:"""
    
    try:
        import openai
        sync_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        response = sync_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        
        if hasattr(response, 'usage'):
            token_tracker.add(response.usage.prompt_tokens, response.usage.completion_tokens)
        
        output = response.choices[0].message.content
        mapping = {}
        
        for line in output.splitlines():
            if "→" in line:
                parts = line.split("→")
                if len(parts) == 2:
                    original = parts[0].strip().lstrip("- ")
                    normalized = parts[1].strip().lstrip("- ")
                    if original and normalized:
                        mapping[original] = normalized
        
        for sp in unique_spokespersons:
            if sp not in mapping:
                mapping[sp] = sp
        
        # Apply mapping
        df['New Spokesperson'] = df['New Spokesperson'].apply(
            lambda x: mapping.get(x, x) if pd.notna(x) and str(x).strip() else x
        )
        
        merged_count = len(unique_spokespersons) - len(set(mapping.values()))
        logging.info(f"✅ Merged {merged_count} spokespersons")
        
    except Exception as e:
        logging.error(f"Error normalizing spokesperson: {e}")
    
    return df

def generate_pillars_per_campaign(
    df: pd.DataFrame,
    language: str,
    token_tracker,
    progress_callback=None
) -> pd.DataFrame:
    """Generate pillars from topics per campaign"""
    
    logging.info("\n" + "="*80)
    logging.info("[STEP 4] GENERATE PILLARS PER CAMPAIGN")
    logging.info("="*80)
    
    if progress_callback is not None:
        progress_callback(0.90, desc="[STEP 4] Generating pillars...")
    
    df['Pillar'] = ''
    campaigns = df['Campaigns'].unique()
    lang_config = LANGUAGE_CONFIGS[language]
    
    for idx, campaign in enumerate(campaigns, 1):
        campaign_mask = df['Campaigns'] == campaign
        topics = df[campaign_mask]['Topic'].dropna()
        topics = topics[topics.astype(str).str.strip() != '']
        unique_topics = sorted(topics.unique().tolist())
        
        if len(unique_topics) == 0:
            continue
        
        logging.info(f"\n[CAMPAIGN {idx}/{len(campaigns)}] {campaign}")
        logging.info(f"📊 {len(unique_topics)} unique topics")
        
        topic_list = "\n".join(f"- {t}" for t in unique_topics[:50])  # Max 50 topics
        nonce = random.randint(100000, 999999)
        
        prompt = f"""You are a strategic categorization expert.

[Request ID: {nonce}]
[Campaign: {campaign}]
[OUTPUT LANGUAGE: {language}]

INPUT: {len(unique_topics)} topics

Topics:
{topic_list}

TASK:
For EACH topic, assign a Pillar ({lang_config['pillar_range']}) in {language}.

Pillar = Strategic category (broader than topic)

EXAMPLES:
Topic: "Resep nasi goreng kampung khas Semarang spesial pedas"
→ Pillar: "Resep Nasi Goreng"

Topic: "Pertanyaan tentang isu IMIP di Morowali"
→ Pillar: "Isu IMIP"

OUTPUT FORMAT (JSON):
{{
  "mappings": [
    {{"topic": "...", "pillar": "..."}},
    ...
  ]
}}

YOUR OUTPUT (JSON only):"""
        
        try:
            import openai
            sync_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            
            response = sync_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=3000
            )
            
            if hasattr(response, 'usage'):
                token_tracker.add(response.usage.prompt_tokens, response.usage.completion_tokens)
            
            raw = response.choices[0].message.content.strip()
            result = extract_json_from_response(raw)
            
            if result and 'mappings' in result:
                for mapping in result['mappings']:
                    topic = mapping.get('topic', '')
                    pillar = mapping.get('pillar', '')
                    
                    if topic and pillar:
                        # Apply to all rows with this topic
                        mask = campaign_mask & (df['Topic'] == topic)
                        df.loc[mask, 'Pillar'] = pillar
                
                unique_pillars = df[campaign_mask]['Pillar'].nunique()
                logging.info(f"  └─ Generated {unique_pillars} unique pillars")
            
        except Exception as e:
            logging.error(f"Error generating pillars for campaign '{campaign}': {e}")
    
    pillar_filled = df['Pillar'].notna() & (df['Pillar'].astype(str).str.strip() != '')
    unique_pillars = df['Pillar'].nunique()
    token_tracker.add_step_stat("Pillar", pillar_filled.sum(), len(df), unique=unique_pillars)
    
    return df

def process_file(
    file_path: str,
    sheet_name: str,
    language: str,
    generate_topic: bool,
    generate_sentiment: bool,
    generate_spokesperson: bool,
    conf_threshold: int,
    selected_project: str = "Default (Auto)",
    progress=gr.Progress()
) -> tuple:
    
    try:
        progress(0.05, desc="Loading file...")
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        original_row_count = len(df)
        logging.info(f"📊 Original rows: {original_row_count}")
        
        df = normalize_column_names(df)
        
        if "Campaign" in df.columns and "Campaigns" not in df.columns:
            df.rename(columns={"Campaign": "Campaigns"}, inplace=True)
        
        is_valid, error_msg = validate_required_columns(df)
        if not is_valid:
            return None, {}, error_msg
        
        title_col = get_col(df, ["Title", "Judul"])
        content_col = get_col(df, ["Content", "Konten", "Isi"])
        channel_col = "Channel"
        
        if 'Noise Tag' in df.columns:
            df['Noise Tag'] = df['Noise Tag'].astype(str)
        
        if 'Engagement' not in df.columns:
            df['Engagement'] = 0
        
        logging.info(f"✅ All {original_row_count} rows will be processed")
        
        # Prepare metadata
        df['_original_index'] = df.index
        df['_channel_lower'] = df[channel_col].astype(str).str.lower().str.strip()
        df['_is_mainstream'] = df['_channel_lower'].apply(is_mainstream)
        
        # Deduplication
        logging.info("\n" + "="*80)
        logging.info("[DEDUPLICATION] Creating groups")
        logging.info("="*80)
        
        df['_dedup_hash'] = df.apply(lambda row: create_dedup_hash(row, title_col, content_col), axis=1)
        df['_is_master'] = False
        dedup_groups = df.groupby('_dedup_hash').head(1).index
        df.loc[dedup_groups, '_is_master'] = True
        
        master_rows = df['_is_master'].sum()
        duplicate_rows = len(df) - master_rows
        
        logging.info(f"✅ {len(df)} rows → {master_rows} unique + {duplicate_rows} duplicates")
        
        # Content eligibility (>= 4 words)
        logging.info("\n" + "="*80)
        logging.info("[CONTENT FILTER] Checking word count")
        logging.info("="*80)
        
        df['_word_count'] = df.apply(
            lambda row: count_meaningful_words(combine_title_content_row(row, title_col, content_col)),
            axis=1
        )
        
        df['_eligible_for_topic'] = df['_word_count'] >= MIN_CONTENT_WORDS
        
        eligible_count = df['_eligible_for_topic'].sum()
        skipped_count = (~df['_eligible_for_topic']).sum()
        
        logging.info(f"✅ {eligible_count} eligible, {skipped_count} skipped (< {MIN_CONTENT_WORDS} words)")
        
        tracker = TokenTracker()
        start_time = time.time()
        
        # STEP 1: Async per-row processing
        if generate_topic or generate_sentiment or generate_spokesperson:
            df = asyncio.run(process_all_rows_async(
                df, title_col, content_col, language, conf_threshold,
                generate_spokesperson, tracker,
                progress_callback=progress
            ))
        
        # STEP 2: Normalize topics
        if generate_topic:
            df = normalize_topics_per_campaign(df, language, tracker, progress_callback=progress)
        
        # STEP 3: Normalize spokesperson
        if generate_spokesperson:
            df = normalize_spokesperson(df, tracker, progress_callback=progress)
        
        # STEP 4: Generate pillars
        if generate_topic:
            is_manual_mode = selected_project and selected_project != "Default (Auto)"
            
            if is_manual_mode:
                # Manual mode: classify per row to predefined pillars from GSheet
                pillars = get_pillar_setup(selected_project)
                if pillars:
                    logging.info(f"🎯 Manual Pillar Mode: project='{selected_project}', pillars={[p['pillar'] for p in pillars]}")
                    df = asyncio.run(classify_pillars_manual_async(
                        df, title_col, content_col, pillars, tracker, progress_callback=progress
                    ))
                else:
                    logging.warning(f"⚠️ No pillar setup found for '{selected_project}', fallback to Auto mode")
                    df = generate_pillars_per_campaign(df, language, tracker, progress_callback=progress)
            else:
                # Auto mode: existing AI clustering
                df = generate_pillars_per_campaign(df, language, tracker, progress_callback=progress)
        
        # Finalization
        logging.info("\n" + "="*80)
        logging.info("[FINALIZATION] Preparing output")
        logging.info("="*80)
        
        df = df.drop(['_channel_lower', '_original_index', '_dedup_hash', '_is_master', 
                      '_word_count', '_eligible_for_topic', '_is_mainstream'], axis=1, errors='ignore')
        
        # Reorder columns
        cols = df.columns.tolist()
        new_cols = []
        
        if generate_topic:
            new_cols.extend(['Pillar', 'Topic'])
        if generate_sentiment:
            new_cols.extend(['New Sentiment', 'New Sentiment Level'])
        if generate_spokesperson:
            new_cols.append('New Spokesperson')
        
        for col in new_cols:
            if col in cols:
                cols.remove(col)
        
        insert_idx = cols.index(content_col) + 1 if content_col in cols else len(cols)
        for i, col in enumerate(new_cols):
            if col in df.columns:
                cols.insert(insert_idx + i, col)
        
        df = df[cols]
        
        final_row_count = len(df)
        if final_row_count != original_row_count:
            logging.warning(f"⚠️ Row count mismatch! Original: {original_row_count}, Final: {final_row_count}")
        else:
            logging.info(f"✅ Row count verified: {original_row_count} → {final_row_count}")
        
        progress(0.98, desc="Saving file...")
        
        # Save to Excel
        original_filename = Path(file_path).stem
        output_filename = f"{original_filename}_v14_async.xlsx"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name="Processed")
            
            duration = time.time() - start_time
            token_summary = tracker.get_summary(MODEL_NAME)
            
            meta_data = [
                {"key": "processed_at", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                {"key": "version", "value": "v14.0 - Async Per-Row Processing"},
                {"key": "model", "value": MODEL_NAME},
                {"key": "output_language", "value": f"{language} ({LANGUAGE_CONFIGS[language]['name']})"},
                {"key": "duration_sec", "value": f"{duration:.2f}"},
                {"key": "parallel_workers", "value": PARALLEL_WORKERS},
                {"key": "original_rows", "value": int(original_row_count)},
                {"key": "final_rows", "value": int(final_row_count)},
                {"key": "deduplication_groups", "value": int(master_rows)},
                {"key": "duplicate_rows", "value": int(duplicate_rows)},
                {"key": "eligible_rows", "value": int(eligible_count)},
                {"key": "skipped_rows", "value": int(skipped_count)},
                {"key": "min_content_words", "value": MIN_CONTENT_WORDS},
                {"key": "input_tokens", "value": int(token_summary["input_tokens"])},
                {"key": "output_tokens", "value": int(token_summary["output_tokens"])},
                {"key": "total_tokens", "value": int(token_summary["total_tokens"])},
                {"key": "api_calls", "value": int(token_summary["api_calls"])},
                {"key": "failed_calls", "value": int(token_summary["failed_calls"])},
                {"key": "cost_usd", "value": f"${token_summary['estimated_cost_usd']:.6f}"},
            ]
            
            for step_name, step_data in token_summary['step_stats'].items():
                key = f"success_rate_{step_name.lower().replace(' ', '_')}"
                if 'unique' in step_data:
                    value = f"{step_data['success']}/{step_data['total']} ({step_data['rate']:.1f}%) | Unique: {step_data['unique']}"
                else:
                    value = f"{step_data['success']}/{step_data['total']} ({step_data['rate']:.1f}%)"
                meta_data.append({"key": key, "value": value})
            
            meta = pd.DataFrame(meta_data)
            meta.to_excel(writer, index=False, sheet_name="Meta")
            
            # Error log sheet
            if token_summary['row_errors']:
                errors_df = pd.DataFrame(token_summary['row_errors'])
                errors_df.to_excel(writer, index=False, sheet_name="Errors")
        
        stats = {
            "total_rows": int(len(df)),
            "unchanged": "YES ✅" if original_row_count == final_row_count else f"NO ❌",
            "deduplication": {
                "unique_groups": int(master_rows),
                "duplicate_rows": int(duplicate_rows)
            },
            "content_filter": {
                "eligible": int(eligible_count),
                "skipped": int(skipped_count),
                "min_words": MIN_CONTENT_WORDS
            },
            "processing": {
                "method": "Async per-row",
                "parallel_workers": PARALLEL_WORKERS,
                "max_retries": MAX_RETRIES
            },
            "duration": f"{duration:.2f}s",
            "cost": f"${token_summary['estimated_cost_usd']:.6f}",
            "success_rates": token_summary['step_stats'],
            "failed_calls": int(token_summary['failed_calls']),
            "row_errors": len(token_summary['row_errors'])
        }
        
        logging.info("\n" + "="*80)
        logging.info("✅ PROCESSING COMPLETE - v14.0 Async")
        logging.info("="*80)
        logging.info(f"Rows: {original_row_count} → {final_row_count}")
        logging.info(f"Duration: {duration:.2f}s | Cost: ${token_summary['estimated_cost_usd']:.6f}")
        logging.info(f"API Calls: {token_summary['api_calls']} | Failed: {token_summary['failed_calls']}")
        
        for step_name, step_data in token_summary['step_stats'].items():
            if 'unique' in step_data:
                logging.info(f"{step_name}: {step_data['success']}/{step_data['total']} ({step_data['rate']:.1f}%) | Unique: {step_data['unique']}")
            else:
                logging.info(f"{step_name}: {step_data['success']}/{step_data['total']} ({step_data['rate']:.1f}%)")
        
        progress(1.0, desc="Complete!")
        return output_path, stats, None
        
    except Exception as e:
        logging.error(f"[ERROR] {str(e)}", exc_info=True)
        return None, {}, f"❌ Error: {str(e)}"

def create_gradio_interface():
    with gr.Blocks(title="Insights Generator v15.0 Async", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📊 Insights Generator v15.0 - Async Per-Row Processing ⚡")
        gr.Markdown(f"**NEW:** Per-row processing with {PARALLEL_WORKERS} parallel workers | Pillar Mode: Auto / Manual (Google Sheet)")
        
        with gr.Row():
            with gr.Column(scale=2):
                file_input = gr.File(label="📁 Upload Excel", file_types=[".xlsx"], type="filepath")
                
                # Project selector (above sheet selector)
                with gr.Row():
                    project_selector = gr.Dropdown(
                        label="🎯 Pillar Mode / Project",
                        choices=get_project_choices(),
                        value="Default (Auto)",
                        interactive=True,
                        scale=4
                    )
                    refresh_btn = gr.Button("🔄 Refresh", variant="secondary", size="sm", scale=1)
                
                refresh_status = gr.Markdown("", visible=True)
                
                cache = load_cache()
                if cache.get("last_updated"):
                    refresh_status = gr.Markdown(f"📦 Cache: {cache['last_updated']}", visible=True)
                
                sheet_selector = gr.Dropdown(label="📊 Sheet", choices=[], interactive=True)
                
                def load_sheets(file_path):
                    if file_path:
                        try:
                            xl = pd.ExcelFile(file_path)
                            return gr.Dropdown(choices=xl.sheet_names, value=xl.sheet_names[0])
                        except Exception as e:
                            return gr.Dropdown(choices=[])
                    return gr.Dropdown(choices=[])
                
                def do_refresh():
                    success, msg = download_gsheet_cache()
                    new_choices = get_project_choices()
                    return gr.Dropdown(choices=new_choices, value="Default (Auto)"), gr.Markdown(msg, visible=True)
                
                file_input.change(load_sheets, inputs=file_input, outputs=sheet_selector)
                refresh_btn.click(do_refresh, outputs=[project_selector, refresh_status])
            
            with gr.Column(scale=1):
                gr.Markdown("### 🌍 Language")
                language_selector = gr.Dropdown(
                    label="Output Language",
                    choices=list(LANGUAGE_CONFIGS.keys()),
                    value="Indonesia"
                )
                
                gr.Markdown("### ⚙️ Config")
                conf_threshold = gr.Slider(label="Sentiment Confidence Threshold", minimum=0, maximum=100, value=85, step=5)
                
                gr.Markdown("### ✅ Features")
                gen_topic = gr.Checkbox(label="📌 Topic & Pillar", value=False)
                gen_sentiment = gr.Checkbox(label="😊 Sentiment", value=False)
                gen_spokesperson = gr.Checkbox(label="🎤 Spokesperson (mainstream)", value=False)
        
        validation_error = gr.Markdown("", visible=True)
        process_btn = gr.Button("🚀 Process (Async)", variant="primary", size="lg", interactive=False)
        
        with gr.Row():
            with gr.Column():
                output_file = gr.File(label="📥 Download")
            with gr.Column():
                stats_output = gr.Textbox(label="📊 Stats", lines=18, interactive=False)
        
        error_output = gr.Textbox(label="⚠️ Status", lines=3, visible=True)
        
        def validate_features(topic, sentiment, spokesperson):
            if not any([topic, sentiment, spokesperson]):
                return gr.Button(interactive=False), gr.Markdown("⚠️ **Select at least one feature**", visible=True)
            else:
                return gr.Button(interactive=True), gr.Markdown("", visible=False)
        
        for checkbox in [gen_topic, gen_sentiment, gen_spokesperson]:
            checkbox.change(
                validate_features,
                inputs=[gen_topic, gen_sentiment, gen_spokesperson],
                outputs=[process_btn, validation_error]
            )
        
        def process_wrapper(file_path, sheet_name, language, topic, sentiment, spokesperson, conf, project, progress=gr.Progress()):
            try:
                if not file_path or not sheet_name:
                    return None, "", "❌ Please upload file and select sheet"
                
                if not any([topic, sentiment, spokesperson]):
                    return None, "", "❌ Select at least one feature"
                
                result_path, stats, error = process_file(
                    file_path, sheet_name, language, topic, sentiment, spokesperson, conf,
                    selected_project=project,
                    progress=progress
                )
                
                if error:
                    return None, "", error
                
                if not result_path:
                    return None, "", "❌ Processing failed"
                
                stats_str = json.dumps(stats, indent=2, ensure_ascii=False) if stats else ""
                return result_path, stats_str, "✅ Processing completed!"
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, "", f"❌ Error: {str(e)}"
        
        process_btn.click(
            process_wrapper,
            inputs=[file_input, sheet_selector, language_selector, gen_topic, gen_sentiment, gen_spokesperson, conf_threshold, project_selector],
            outputs=[output_file, stats_output, error_output]
        )
    
    return app

if __name__ == "__main__":
    app = create_gradio_interface()
    app.queue(max_size=10, default_concurrency_limit=4)
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)