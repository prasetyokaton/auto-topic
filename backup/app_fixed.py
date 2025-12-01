import gradio as gr
import pandas as pd
import os
import json
import re
import time
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import tempfile
import math
import random

# ==== ROBUST PATCH FOR GRADIO BUG ====
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

# ====== CONFIGURATION ======
load_dotenv(dotenv_path=".secretcontainer/.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ ERROR: OPENAI_API_KEY not found in .env file!")
    raise ValueError("Missing OPENAI_API_KEY")

print(f"✅ OpenAI API Key loaded: {OPENAI_API_KEY[:8]}...{OPENAI_API_KEY[-4:]}")

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

BATCH_SIZE = 50
MAX_RETRIES = 2
TRUNCATE_WORDS = 200

MAINSTREAM_CHANNELS = ['tv', 'radio', 'printed', 'printedmedia', 'print', 'printmedia', 'newspaper', 'online media']
SOCIAL_CHANNELS = ['tiktok', 'instagram', 'youtube', 'facebook', 'twitter', 'x', 'blog', 'forum']
INVALID_VALUES = ['nan', 'none', 'null', 'n/a', 'na', 'unknown', 'tidak ada', '-', 'tidak diketahui', 'undefined', 'not available']

PRICING = {
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4o": {"input": 2.500, "output": 10.000},
    "gpt-3.5-turbo": {"input": 0.500, "output": 1.500},
}

# ====== LOGGING SETUP ======
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

# ====== API VALIDATION ======
def validate_openai_api():
    try:
        logging.info("[API TEST] Testing OpenAI connection...")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        logging.info(f"[API TEST] ✅ SUCCESS - Model: {MODEL_NAME}")
        return True, "API connection successful"
    except Exception as e:
        error_msg = str(e)
        logging.error(f"[API TEST] ❌ FAILED: {error_msg}")
        return False, f"API connection failed: {error_msg}"

api_ok, api_msg = validate_openai_api()
if not api_ok:
    print(f"\n{'='*60}")
    print(f"⚠️  WARNING: OpenAI API Test Failed!")
    print(f"{'='*60}")
    print(f"Error: {api_msg}")
    print(f"{'='*60}\n")

# ====== TOKEN TRACKER ======
class TokenTracker:
    def __init__(self):
        self.total_input = 0
        self.total_output = 0
        self.api_calls = 0
        self.toon_success = 0
        self.json_fallback = 0
        self.parse_failed = 0
    
    def add(self, input_tokens, output_tokens):
        self.total_input += input_tokens
        self.total_output += output_tokens
        self.api_calls += 1
    
    def add_parse_result(self, format_type):
        if format_type == "toon":
            self.toon_success += 1
        elif format_type == "json":
            self.json_fallback += 1
        else:
            self.parse_failed += 1
    
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
            "input_tokens": self.total_input,
            "output_tokens": self.total_output,
            "total_tokens": self.total_input + self.total_output,
            "api_calls": self.api_calls,
            "estimated_cost_usd": cost,
            "toon_success": self.toon_success,
            "json_fallback": self.json_fallback,
            "parse_failed": self.parse_failed
        }

# ====== HELPER FUNCTIONS ======
def safe_text(x):
    return "" if pd.isna(x) else str(x).strip()

def truncate_to_first_n_words(text: str, n: int = TRUNCATE_WORDS) -> str:
    words = text.split()
    return " ".join(words[:n])

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
    
    return None

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
    has_channel = 'Channel' in df.columns
    has_media_type = 'Media Type' in df.columns or 'Media type' in df.columns
    
    if has_channel and has_media_type:
        return False, "❌ Error: Kolom 'Channel' dan 'Media Type' tidak boleh ada bersamaan! Pilih salah satu."
    
    if not has_channel and not has_media_type:
        return False, "❌ Error: Harus ada kolom 'Channel' atau 'Media Type'!"
    
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

def is_invalid_value(value: str) -> bool:
    if not value or pd.isna(value):
        return True
    value_lower = str(value).strip().lower()
    if value_lower == '' or len(value_lower) < 3:
        return True
    for invalid in INVALID_VALUES:
        if invalid in value_lower:
            return True
    return False

# ====== TOON FORMAT FUNCTIONS ======
def build_toon_input(batch_df, title_col, content_col, batch_size):
    lines = [f"batch[{batch_size}]{{row|text}}:"]
    
    for idx, row in batch_df.iterrows():
        combined = combine_title_content_row(row, title_col, content_col)
        truncated = truncate_to_first_n_words(combined, TRUNCATE_WORDS)
        text = truncated.replace('|', '⎮')
        lines.append(f"{idx}|{text}")
    
    return "\n".join(lines)

def parse_toon_output(response: str, expected_fields: list) -> list:
    lines = response.strip().split('\n')
    
    data_start = 0
    for i, line in enumerate(lines):
        if '{' in line and '}' in line and ':' in line:
            data_start = i + 1
            break
    
    data = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        values = line.split('|')
        
        if len(values) == len(expected_fields):
            values = [v.replace('⎮', '|').strip() for v in values]
            data.append(dict(zip(expected_fields, values)))
    
    return data

def parse_gpt_response(response: str, expected_fields: list, batch_size: int, tracker: TokenTracker):
    try:
        data = parse_toon_output(response, expected_fields)
        
        if len(data) == batch_size:
            logging.info(f"✅ TOON parse SUCCESS - {len(data)} rows")
            tracker.add_parse_result("toon")
            return data, "toon"
        else:
            logging.warning(f"⚠️ TOON count mismatch: expected {batch_size}, got {len(data)}")
            raise ValueError("Row count mismatch")
    
    except Exception as e:
        logging.warning(f"⚠️ TOON parse FAILED: {e}")
    
    try:
        data = extract_json_from_response(response)
        if isinstance(data, list) and len(data) > 0:
            logging.info(f"✅ JSON fallback SUCCESS - {len(data)} rows")
            tracker.add_parse_result("json")
            return data, "json"
    except Exception as e:
        logging.warning(f"⚠️ JSON parse FAILED: {e}")
    
    logging.error(f"❌ All parsing failed")
    tracker.add_parse_result("failed")
    return [], "failed"

# ====== OPENAI API CALL ======
def chat_create(model, messages, token_tracker=None, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            logging.info(f"[API CALL] Attempt {attempt + 1}/{max_retries} - Model: {model}")
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7
            )
            
            if token_tracker and hasattr(response, 'usage'):
                input_tok = response.usage.prompt_tokens
                output_tok = response.usage.completion_tokens
                token_tracker.add(input_tok, output_tok)
                logging.info(f"[API CALL] ✅ SUCCESS - Tokens: {input_tok} in, {output_tok} out")
            
            return response
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            logging.error(f"[API CALL] ❌ Attempt {attempt + 1}/{max_retries} FAILED")
            logging.error(f"[API CALL] Error: {error_type}: {error_msg}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logging.info(f"[API CALL] Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                logging.error(f"[API CALL] ❌ ALL RETRIES FAILED")
                raise
    
    return None

# ====== STEP 1: SUB TOPIC (ALL CHANNELS) ======
def process_batch_sub_topic(
    batch_df: pd.DataFrame,
    batch_num: int,
    total_batches: int,
    title_col: str,
    content_col: str,
    token_tracker: TokenTracker,
    progress=gr.Progress()
) -> pd.DataFrame:
    
    batch_size = len(batch_df)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"[SUB TOPIC BATCH {batch_num}/{total_batches}] Starting - {batch_size} rows")
    logging.info(f"{'='*60}")
    
    batch_df['Sub Topic'] = ''
    
    input_toon = build_toon_input(batch_df, title_col, content_col, batch_size)
    
    nonce = random.randint(100000, 999999)
    
    prompt = f"""You are an insights professional. Analyze content in TOON format.

[Request ID: {nonce}]

INPUT (TOON format with pipe delimiter |):
{input_toon}

TASK: Generate sub_topic (3-7 words, clear & specific) for each row.

CRITICAL RULES:
- NEVER use "unknown", "nan", "none", "tidak jelas"
- Sub topic MUST be clear and specific (3-7 words)
- Use language from the content
- Extract main topic from the content

OUTPUT FORMAT (TOON with pipe delimiter |):
result[{batch_size}]{{row|sub_topic}}:
<row_index>|<sub_topic>
<row_index>|<sub_topic>
...

Example:
result[3]{{row|sub_topic}}:
1|Kenaikan harga BBM
2|Protes warga Jakarta
3|Kenaikan inflasi

YOUR OUTPUT (TOON format only):"""
    
    try:
        progress(0.5, desc=f"🔄 Sub Topic Batch {batch_num}/{total_batches}...")
        
        response = chat_create(
            MODEL_NAME,
            [
                {"role": "system", "content": "You are an insights professional. Output TOON format only. NEVER use 'unknown' for sub_topic."},
                {"role": "user", "content": prompt}
            ],
            token_tracker=token_tracker
        )
        
        if not response:
            raise Exception("API call failed")
        
        raw = response.choices[0].message.content.strip()
        data, format_type = parse_gpt_response(raw, ['row', 'sub_topic'], batch_size, token_tracker)
        
        if not data:
            logging.error(f"[SUB TOPIC BATCH {batch_num}] No data parsed")
            return batch_df
        
        logging.info(f"[SUB TOPIC BATCH {batch_num}] Parsed {len(data)} items ({format_type})")
        
        result_dict = {}
        for item in data:
            try:
                row_idx = int(item.get('row', -1))
                result_dict[row_idx] = item
            except:
                continue
        
        success_count = 0
        warning_count = 0
        
        for idx in batch_df.index:
            if idx in result_dict:
                item = result_dict[idx]
                sub_topic = str(item.get('sub_topic', '')).strip()
                
                if is_invalid_value(sub_topic):
                    logging.warning(f"[SUB TOPIC] Row {idx}: REJECTED invalid '{sub_topic}'")
                    warning_count += 1
                    continue
                
                words = sub_topic.split()
                if len(words) >= 3 and len(words) <= 10:
                    batch_df.at[idx, 'Sub Topic'] = sub_topic
                    success_count += 1
                    logging.info(f"[SUB TOPIC] Row {idx}: '{sub_topic}'")
                else:
                    logging.warning(f"[SUB TOPIC] Row {idx}: REJECTED word count {len(words)} - '{sub_topic}'")
                    warning_count += 1
            else:
                logging.warning(f"[SUB TOPIC] Row {idx}: Missing in response")
                warning_count += 1
        
        logging.info(f"[SUB TOPIC BATCH {batch_num}] ✅ Success: {success_count}/{batch_size} | ⚠️ Warnings: {warning_count}")
        return batch_df
        
    except Exception as e:
        logging.error(f"[SUB TOPIC BATCH {batch_num}] ❌ FAILED: {str(e)}")
        return batch_df

# ====== STEP 2: SENTIMENT (ALL CHANNELS) ======
def process_batch_sentiment(
    batch_df: pd.DataFrame,
    batch_num: int,
    total_batches: int,
    title_col: str,
    content_col: str,
    conf_threshold: int,
    token_tracker: TokenTracker,
    progress=gr.Progress()
) -> pd.DataFrame:
    
    batch_size = len(batch_df)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"[SENTIMENT BATCH {batch_num}/{total_batches}] Starting - {batch_size} rows")
    logging.info(f"{'='*60}")
    
    batch_df['New Sentiment'] = 'neutral'
    batch_df['New Sentiment Level'] = 0
    
    input_toon = build_toon_input(batch_df, title_col, content_col, batch_size)
    
    nonce = random.randint(100000, 999999)
    
    prompt = f"""You are an insights professional. Analyze sentiment in TOON format.

[Request ID: {nonce}]

INPUT (TOON format with pipe delimiter |):
{input_toon}

TASK: Generate sentiment & confidence for each row.

SENTIMENT RULES:
- positive: clear positive emotion, praise, satisfaction
- negative: clear negative emotion, complaint, disappointment
- neutral: no clear emotion or ambiguous

CONFIDENCE: 0-100 (your confidence level)

OUTPUT FORMAT (TOON with pipe delimiter |):
result[{batch_size}]{{row|sentiment|confidence}}:
<row_index>|<sentiment>|<confidence_0-100>

Example:
result[3]{{row|sentiment|confidence}}:
1|negative|85
2|negative|90
3|neutral|75

YOUR OUTPUT (TOON format only):"""
    
    try:
        progress(0.5, desc=f"🔄 Sentiment Batch {batch_num}/{total_batches}...")
        
        response = chat_create(
            MODEL_NAME,
            [
                {"role": "system", "content": "You are an insights professional. Output TOON format only."},
                {"role": "user", "content": prompt}
            ],
            token_tracker=token_tracker
        )
        
        if not response:
            raise Exception("API call failed")
        
        raw = response.choices[0].message.content.strip()
        data, format_type = parse_gpt_response(raw, ['row', 'sentiment', 'confidence'], batch_size, token_tracker)
        
        if not data:
            logging.error(f"[SENTIMENT BATCH {batch_num}] No data parsed")
            return batch_df
        
        logging.info(f"[SENTIMENT BATCH {batch_num}] Parsed {len(data)} items ({format_type})")
        
        result_dict = {}
        for item in data:
            try:
                row_idx = int(item.get('row', -1))
                result_dict[row_idx] = item
            except:
                continue
        
        success_count = 0
        
        for idx in batch_df.index:
            if idx in result_dict:
                item = result_dict[idx]
                sentiment = str(item.get('sentiment', 'neutral')).lower().strip()
                
                if sentiment in ['positive', 'negative', 'neutral']:
                    try:
                        conf = int(item.get('confidence', 0))
                        conf = max(0, min(100, conf))
                        
                        if conf < conf_threshold and sentiment != 'neutral':
                            sentiment = 'neutral'
                        
                        batch_df.at[idx, 'New Sentiment'] = sentiment
                        batch_df.at[idx, 'New Sentiment Level'] = conf
                        success_count += 1
                    except:
                        pass
        
        logging.info(f"[SENTIMENT BATCH {batch_num}] ✅ Success: {success_count}/{batch_size}")
        return batch_df
        
    except Exception as e:
        logging.error(f"[SENTIMENT BATCH {batch_num}] ❌ FAILED: {str(e)}")
        return batch_df

# ====== STEP 3: SPOKESPERSON (MAINSTREAM ONLY) ======
def process_batch_spokesperson(
    batch_df: pd.DataFrame,
    batch_num: int,
    total_batches: int,
    title_col: str,
    content_col: str,
    token_tracker: TokenTracker,
    progress=gr.Progress()
) -> pd.DataFrame:
    
    batch_size = len(batch_df)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"[SPOKESPERSON BATCH {batch_num}/{total_batches}] Starting - {batch_size} rows")
    logging.info(f"{'='*60}")
    
    batch_df['New Spokesperson'] = ''
    
    input_toon = build_toon_input(batch_df, title_col, content_col, batch_size)
    
    nonce = random.randint(100000, 999999)
    
    prompt = f"""You are an insights professional. Extract spokesperson from TOON format.

[Request ID: {nonce}]

INPUT (TOON format with pipe delimiter |):
{input_toon}

TASK: Extract spokesperson (person who is quoted) from each content.

FORMAT RULES:
- Format: "Nama Lengkap (Jabatan/Posisi)"
- Multiple: "Nama1 (Jabatan1), Nama2 (Jabatan2)"
- If no one is quoted: leave empty (use "-" as placeholder)
- DO NOT use "unknown" or "nan"

OUTPUT FORMAT (TOON with pipe delimiter |):
result[{batch_size}]{{row|spokesperson}}:
<row_index>|<spokesperson or ->

Example:
result[3]{{row|spokesperson}}:
1|Ahmad Yani (Direktur Utama)
2|Budi Santoso (Kepala Dinas), Siti Aisyah (Juru Bicara)
3|-

YOUR OUTPUT (TOON format only):"""
    
    try:
        progress(0.5, desc=f"🔄 Spokesperson Batch {batch_num}/{total_batches}...")
        
        response = chat_create(
            MODEL_NAME,
            [
                {"role": "system", "content": "You are an insights professional. Output TOON format only."},
                {"role": "user", "content": prompt}
            ],
            token_tracker=token_tracker
        )
        
        if not response:
            raise Exception("API call failed")
        
        raw = response.choices[0].message.content.strip()
        data, format_type = parse_gpt_response(raw, ['row', 'spokesperson'], batch_size, token_tracker)
        
        if not data:
            logging.error(f"[SPOKESPERSON BATCH {batch_num}] No data parsed")
            return batch_df
        
        logging.info(f"[SPOKESPERSON BATCH {batch_num}] Parsed {len(data)} items ({format_type})")
        
        result_dict = {}
        for item in data:
            try:
                row_idx = int(item.get('row', -1))
                result_dict[row_idx] = item
            except:
                continue
        
        success_count = 0
        
        for idx in batch_df.index:
            if idx in result_dict:
                item = result_dict[idx]
                spokesperson_val = str(item.get('spokesperson', '')).strip()
                
                if spokesperson_val == '-':
                    spokesperson_val = ''
                
                if not is_invalid_value(spokesperson_val):
                    batch_df.at[idx, 'New Spokesperson'] = spokesperson_val
                    success_count += 1
        
        logging.info(f"[SPOKESPERSON BATCH {batch_num}] ✅ Success: {success_count}/{batch_size}")
        return batch_df
        
    except Exception as e:
        logging.error(f"[SPOKESPERSON BATCH {batch_num}] ❌ FAILED: {str(e)}")
        return batch_df

# ====== RETRY UNKNOWN ======
def retry_unknown_batch(
    batch_df: pd.DataFrame,
    title_col: str,
    content_col: str,
    field_name: str,
    token_tracker: TokenTracker,
    progress=gr.Progress()
) -> pd.DataFrame:
    
    batch_size = len(batch_df)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"[RETRY {field_name.upper()}] Starting - {batch_size} rows")
    logging.info(f"{'='*60}")
    
    input_toon = build_toon_input(batch_df, title_col, content_col, batch_size)
    
    nonce = random.randint(100000, 999999)
    
    if field_name == 'Sub Topic':
        prompt = f"""CRITICAL RETRY: You previously FAILED to identify sub_topic for these contents.

[Request ID: {nonce}]

INPUT (TOON format with pipe delimiter |):
{input_toon}

TASK: You MUST provide clear sub_topic (3-7 words) for EACH row.

STRICT RULES:
- READ the content VERY CAREFULLY
- EXTRACT the main topic from content
- NEVER EVER use "unknown", "tidak jelas", "topik umum"
- If content is unclear, use keywords from the content itself

OUTPUT FORMAT (TOON with pipe delimiter |):
result[{batch_size}]{{row|sub_topic}}:

YOUR OUTPUT (TOON format only):"""
    
    else:
        return batch_df
    
    try:
        progress(0.95, desc=f"🔄 Retrying {field_name}...")
        
        response = chat_create(
            MODEL_NAME,
            [
                {"role": "system", "content": "You are an insights professional. This is a RETRY. Be MORE careful. Output TOON format only. NEVER use 'unknown'."},
                {"role": "user", "content": prompt}
            ],
            token_tracker=token_tracker
        )
        
        if not response:
            raise Exception("API call failed")
        
        raw = response.choices[0].message.content.strip()
        data, format_type = parse_gpt_response(raw, ['row', 'sub_topic'], batch_size, token_tracker)
        
        if not data:
            logging.error(f"[RETRY {field_name}] No data parsed")
            return batch_df
        
        logging.info(f"[RETRY {field_name}] Parsed {len(data)} items ({format_type})")
        
        result_dict = {}
        for item in data:
            try:
                row_idx = int(item.get('row', -1))
                result_dict[row_idx] = item
            except:
                continue
        
        recovered_count = 0
        
        for idx in batch_df.index:
            if idx in result_dict:
                item = result_dict[idx]
                sub_topic = str(item.get('sub_topic', '')).strip()
                
                if is_invalid_value(sub_topic):
                    logging.warning(f"[RETRY] Row {idx}: Still invalid '{sub_topic}'")
                    continue
                
                words = sub_topic.split()
                if len(words) >= 3 and len(words) <= 10:
                    batch_df.at[idx, field_name] = sub_topic
                    recovered_count += 1
                    logging.info(f"[RETRY] Row {idx}: RECOVERED '{sub_topic}'")
        
        logging.info(f"[RETRY {field_name}] ✅ Recovered: {recovered_count}/{batch_size}")
        return batch_df
        
    except Exception as e:
        logging.error(f"[RETRY {field_name}] ❌ FAILED: {str(e)}")
        return batch_df

# ====== NORMALIZATION ======
def normalize_sub_topics_to_topics(
    unique_sub_topics: list,
    token_tracker: TokenTracker,
    progress=gr.Progress()
) -> dict:
    
    if len(unique_sub_topics) <= 1:
        return {st: st for st in unique_sub_topics}
    
    progress(0.85, desc="🔄 Normalizing Topics...")
    
    joined = "\n".join(f"- {st}" for st in unique_sub_topics)
    
    nonce = random.randint(100000, 999999)
    
    prompt = f"""You are an insights professional. Group sub-topics into broader TOPIC categories.

[Request ID: {nonce}]

TASK:
1. Group similar/related sub-topics together
2. Create 1 TOPIC label (2-5 words) for each group
3. Maximum 10-15 TOPIC categories

Sub Topics:
{joined}

Return format:
<Sub Topic> → <Topic (2-5 words)>

Output:"""
    
    try:
        response = chat_create(
            MODEL_NAME,
            [
                {"role": "system", "content": "You are an insights professional."},
                {"role": "user", "content": prompt}
            ],
            token_tracker=token_tracker
        )
        
        if not response:
            return {st: st for st in unique_sub_topics}
        
        output = response.choices[0].message.content
        
        mapping = {}
        for line in output.splitlines():
            if "→" in line:
                parts = line.split("→")
                if len(parts) == 2:
                    original = parts[0].strip().lstrip("- ")
                    category = parts[1].strip().lstrip("- ")
                    
                    category_words = category.split()
                    if len(category_words) >= 2 and len(category_words) <= 5:
                        if original and category:
                            mapping[original] = category
        
        for st in unique_sub_topics:
            if st not in mapping:
                words = st.split()
                if len(words) > 5:
                    mapping[st] = " ".join(words[:5])
                else:
                    mapping[st] = st
        
        return mapping
        
    except Exception as e:
        logging.error(f"[NORMALIZE TOPICS ERROR] {str(e)}")
        return {st: st for st in unique_sub_topics}

def normalize_spokesperson(
    unique_spokespersons: list,
    token_tracker: TokenTracker,
    progress=gr.Progress()
) -> dict:
    
    unique_spokespersons = [s for s in unique_spokespersons if s and s.strip()]
    
    if len(unique_spokespersons) <= 1:
        return {sp: sp for sp in unique_spokespersons}
    
    progress(0.90, desc="🔄 Normalizing Spokesperson...")
    
    joined = "\n".join(f"- {sp}" for sp in unique_spokespersons)
    
    nonce = random.randint(100000, 999999)
    
    prompt = f"""You are an insights professional. Normalize spokesperson names referring to the SAME person.

[Request ID: {nonce}]

FORMAT: "Nama Lengkap (Jabatan/Posisi)"

Spokesperson:
{joined}

Return format:
<Original> → <Normalized>

Output:"""
    
    try:
        response = chat_create(
            MODEL_NAME,
            [
                {"role": "system", "content": "You are an insights professional."},
                {"role": "user", "content": prompt}
            ],
            token_tracker=token_tracker
        )
        
        if not response:
            return {sp: sp for sp in unique_spokespersons}
        
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
        
        return mapping
        
    except Exception as e:
        logging.error(f"[NORMALIZE SPOKESPERSON ERROR] {str(e)}")
        return {sp: sp for sp in unique_spokespersons}

# ====== MAIN PROCESSING ======
def process_file(
    file_path: str,
    sheet_name: str,
    generate_topic: bool,
    generate_sentiment: bool,
    generate_spokesperson: bool,
    conf_threshold: int,
    progress=gr.Progress()
) -> tuple:
    
    try:
        progress(0.05, desc="📂 Loading file...")
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        if "Campaign" in df.columns and "Campaigns" not in df.columns:
            df.rename(columns={"Campaign": "Campaigns"}, inplace=True)
        
        if "Media type" in df.columns and "Channel" not in df.columns:
            df.rename(columns={"Media type": "Channel"}, inplace=True)
        elif "Media Type" in df.columns and "Channel" not in df.columns:
            df.rename(columns={"Media Type": "Channel"}, inplace=True)
        
        is_valid, error_msg = validate_required_columns(df)
        if not is_valid:
            return None, {}, error_msg
        
        title_col = get_col(df, ["Title", "Judul"])
        content_col = get_col(df, ["Content", "Konten", "Isi"])
        channel_col = "Channel"
        
        df[channel_col] = df[channel_col].astype(str).str.lower().str.strip()
        
        empty_channels = df[channel_col].isna() | (df[channel_col] == '') | (df[channel_col] == 'nan')
        if empty_channels.any():
            return None, {}, f"❌ Error: {empty_channels.sum()} baris memiliki Channel kosong!"
        
        logging.info(f"[START] Processing {len(df)} rows")
        logging.info(f"[FLOW] 1.SubTopic(all) → 2.Sentiment(all) → 3.Spokesperson(mainstream) → 4.Normalize → 5.Retry")
        
        df['_original_index'] = df.index
        
        mainstream_mask = df[channel_col].apply(is_mainstream)
        social_mask = df[channel_col].apply(is_social)
        
        df_mainstream = df[mainstream_mask].copy()
        df_social = df[social_mask].copy()
        
        mainstream_count = len(df_mainstream)
        social_count = len(df_social)
        
        logging.info(f"[SPLIT] Mainstream: {mainstream_count}, Social: {social_count}")
        
        tracker = TokenTracker()
        start_time = time.time()
        
        # ===== STEP 1: SUB TOPIC (ALL CHANNELS) =====
        if generate_topic:
            logging.info("\n" + "="*60)
            logging.info("STEP 1: GENERATING SUB TOPIC (ALL CHANNELS) - TOON FORMAT")
            logging.info("="*60)
            
            all_batches = []
            total_batches = math.ceil(len(df) / BATCH_SIZE)
            
            for batch_num in range(total_batches):
                start_idx = batch_num * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, len(df))
                batch_df = df.iloc[start_idx:end_idx].copy()
                
                progress_val = 0.1 + (batch_num / total_batches) * 0.2
                progress(progress_val, desc=f"⚙️ Sub Topic {batch_num + 1}/{total_batches}")
                
                result_batch = process_batch_sub_topic(
                    batch_df, batch_num + 1, total_batches,
                    title_col, content_col, tracker, progress
                )
                
                all_batches.append(result_batch)
            
            df = pd.concat(all_batches, ignore_index=False)
            df = df.sort_values('_original_index').reset_index(drop=True)
            
            df_mainstream = df[df[channel_col].apply(is_mainstream)].copy()
            df_social = df[df[channel_col].apply(is_social)].copy()
        
        # ===== STEP 2: SENTIMENT (ALL CHANNELS) =====
        if generate_sentiment:
            logging.info("\n" + "="*60)
            logging.info("STEP 2: GENERATING SENTIMENT (ALL CHANNELS) - TOON FORMAT")
            logging.info("="*60)
            
            all_batches = []
            total_batches = math.ceil(len(df) / BATCH_SIZE)
            
            for batch_num in range(total_batches):
                start_idx = batch_num * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, len(df))
                batch_df = df.iloc[start_idx:end_idx].copy()
                
                progress_val = 0.35 + (batch_num / total_batches) * 0.2
                progress(progress_val, desc=f"⚙️ Sentiment {batch_num + 1}/{total_batches}")
                
                result_batch = process_batch_sentiment(
                    batch_df, batch_num + 1, total_batches,
                    title_col, content_col, conf_threshold, tracker, progress
                )
                
                all_batches.append(result_batch)
            
            df = pd.concat(all_batches, ignore_index=False)
            df = df.sort_values('_original_index').reset_index(drop=True)
            
            df_mainstream = df[df[channel_col].apply(is_mainstream)].copy()
            df_social = df[df[channel_col].apply(is_social)].copy()
        
        # ===== STEP 3: SPOKESPERSON (MAINSTREAM ONLY) =====
        if generate_spokesperson and mainstream_count > 0:
            logging.info("\n" + "="*60)
            logging.info("STEP 3: GENERATING SPOKESPERSON (MAINSTREAM ONLY) - TOON FORMAT")
            logging.info("="*60)
            
            mainstream_batches = []
            total_batches = math.ceil(mainstream_count / BATCH_SIZE)
            
            for batch_num in range(total_batches):
                start_idx = batch_num * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, mainstream_count)
                batch_df = df_mainstream.iloc[start_idx:end_idx].copy()
                
                progress_val = 0.6 + (batch_num / total_batches) * 0.15
                progress(progress_val, desc=f"⚙️ Spokesperson {batch_num + 1}/{total_batches}")
                
                result_batch = process_batch_spokesperson(
                    batch_df, batch_num + 1, total_batches,
                    title_col, content_col, tracker, progress
                )
                
                mainstream_batches.append(result_batch)
            
            df_mainstream = pd.concat(mainstream_batches, ignore_index=False)
        
        # ===== COMBINE =====
        progress(0.8, desc="🔄 Combining...")
        
        if mainstream_count > 0 and social_count > 0:
            if generate_spokesperson and 'New Spokesperson' not in df_social.columns:
                df_social['New Spokesperson'] = ''
            df_processed = pd.concat([df_mainstream, df_social])
        elif mainstream_count > 0:
            df_processed = df_mainstream
        elif social_count > 0:
            if generate_spokesperson:
                df_social['New Spokesperson'] = ''
            df_processed = df_social
        else:
            return None, {}, "❌ No data"
        
        df_processed = df_processed.sort_values('_original_index').reset_index(drop=True)
        df_processed = df_processed.drop('_original_index', axis=1)
        
        # ===== FORCE SOCIAL = EMPTY SPOKESPERSON =====
        if generate_spokesperson:
            social_final_mask = df_processed[channel_col].apply(is_social)
            df_processed.loc[social_final_mask, 'New Spokesperson'] = ''
            logging.info("[FINAL CHECK] ✅ Forced social: Spokesperson = empty")
        
        # ===== STEP 4: RETRY UNKNOWN SUB TOPICS =====
        if generate_topic:
            logging.info("\n" + "="*60)
            logging.info("STEP 4: RETRY UNKNOWN SUB TOPICS")
            logging.info("="*60)
            
            unknown_mask = (df_processed['Sub Topic'].isna()) | \
                          (df_processed['Sub Topic'].astype(str).str.strip() == '') | \
                          (df_processed['Sub Topic'].apply(lambda x: is_invalid_value(str(x))))
            
            df_unknown = df_processed[unknown_mask].copy()
            unknown_count = len(df_unknown)
            
            logging.info(f"[RETRY] Found {unknown_count} unknown sub topics")
            
            if unknown_count > 0:
                progress(0.82, desc=f"🔄 Retrying {unknown_count} unknowns...")
                
                retry_batches = []
                total_batches = math.ceil(unknown_count / BATCH_SIZE)
                
                for batch_num in range(total_batches):
                    start_idx = batch_num * BATCH_SIZE
                    end_idx = min(start_idx + BATCH_SIZE, unknown_count)
                    batch_df = df_unknown.iloc[start_idx:end_idx].copy()
                    
                    result_batch = retry_unknown_batch(
                        batch_df, title_col, content_col, 'Sub Topic', tracker, progress
                    )
                    
                    retry_batches.append(result_batch)
                
                df_unknown = pd.concat(retry_batches, ignore_index=False)
                
                for idx in df_unknown.index:
                    df_processed.at[idx, 'Sub Topic'] = df_unknown.at[idx, 'Sub Topic']
        
        # ===== STEP 5: NORMALIZATION =====
        if generate_topic:
            logging.info("\n" + "="*60)
            logging.info("STEP 5: NORMALIZING SUB TOPIC → TOPIC (2-5 words)")
            logging.info("="*60)
            
            progress(0.88, desc="🔄 Normalizing Topics...")
            
            sub_topics = df_processed['Sub Topic'].dropna()
            sub_topics = sub_topics[sub_topics.astype(str).str.strip() != '']
            sub_topics = sub_topics[~sub_topics.apply(lambda x: is_invalid_value(str(x)))]
            unique_sub_topics = sorted(sub_topics.unique().tolist())
            
            logging.info(f"[NORMALIZE] Found {len(unique_sub_topics)} unique sub topics")
            
            if unique_sub_topics:
                topic_mapping = normalize_sub_topics_to_topics(unique_sub_topics, tracker, progress)
                df_processed['Topic'] = df_processed['Sub Topic'].apply(
                    lambda x: topic_mapping.get(x, x) if x and str(x).strip() and not is_invalid_value(str(x)) else ''
                )
            else:
                df_processed['Topic'] = ''
        
        if generate_spokesperson:
            logging.info("\n" + "="*60)
            logging.info("STEP 6: NORMALIZING SPOKESPERSON")
            logging.info("="*60)
            
            progress(0.93, desc="🔄 Normalizing Spokesperson...")
            
            spokespersons = df_processed['New Spokesperson'].dropna()
            spokespersons = spokespersons[spokespersons.astype(str).str.strip() != '']
            spokespersons = spokespersons[~spokespersons.apply(lambda x: is_invalid_value(str(x)))]
            unique_spokespersons = sorted(spokespersons.unique().tolist())
            
            logging.info(f"[NORMALIZE] Found {len(unique_spokespersons)} unique spokespersons")
            
            if unique_spokespersons:
                spokesperson_mapping = normalize_spokesperson(unique_spokespersons, tracker, progress)
                df_processed['New Spokesperson'] = df_processed['New Spokesperson'].apply(
                    lambda x: spokesperson_mapping.get(x, x) if pd.notna(x) and str(x).strip() and not is_invalid_value(str(x)) else x
                )
        
        # ===== REORDER COLUMNS =====
        cols = df_processed.columns.tolist()
        new_cols = []
        
        if generate_topic:
            new_cols.extend(['Topic', 'Sub Topic'])
        if generate_sentiment:
            new_cols.extend(['New Sentiment', 'New Sentiment Level'])
        if generate_spokesperson:
            new_cols.append('New Spokesperson')
        
        for col in new_cols:
            if col in cols:
                cols.remove(col)
        
        insert_idx = cols.index(content_col) + 1 if content_col in cols else len(cols)
        for i, col in enumerate(new_cols):
            if col in df_processed.columns:
                cols.insert(insert_idx + i, col)
        
        df_processed = df_processed[cols]
        
        # ===== SAVE =====
        progress(0.98, desc="💾 Saving...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"processed_{timestamp}.xlsx"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            df_processed.to_excel(writer, index=False, sheet_name="Processed")
            
            duration = time.time() - start_time
            token_summary = tracker.get_summary(MODEL_NAME)
            
            unknown_sub_topics = df_processed[
                df_processed['Sub Topic'].astype(str).str.lower().str.contains('unknown', na=False)
            ].shape[0] if 'Sub Topic' in df_processed.columns else 0
            
            empty_sub_topics = df_processed[
                (df_processed['Sub Topic'].isna()) | 
                (df_processed['Sub Topic'].astype(str).str.strip() == '')
            ].shape[0] if 'Sub Topic' in df_processed.columns else 0
            
            toon_rate = (token_summary['toon_success'] / token_summary['api_calls'] * 100) if token_summary['api_calls'] > 0 else 0
            
            meta = pd.DataFrame([
                {"key": "processed_at", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                {"key": "model", "value": MODEL_NAME},
                {"key": "duration_sec", "value": f"{duration:.2f}"},
                {"key": "total_rows", "value": len(df_processed)},
                {"key": "mainstream_rows", "value": mainstream_count},
                {"key": "social_rows", "value": social_count},
                {"key": "batch_size", "value": BATCH_SIZE},
                {"key": "input_tokens", "value": token_summary["input_tokens"]},
                {"key": "output_tokens", "value": token_summary["output_tokens"]},
                {"key": "total_tokens", "value": token_summary["total_tokens"]},
                {"key": "api_calls", "value": token_summary["api_calls"]},
                {"key": "cost_usd", "value": f"${token_summary['estimated_cost_usd']:.6f}"},
                {"key": "format", "value": "TOON with pipe delimiter"},
                {"key": "toon_success", "value": token_summary["toon_success"]},
                {"key": "json_fallback", "value": token_summary["json_fallback"]},
                {"key": "parse_failed", "value": token_summary["parse_failed"]},
                {"key": "toon_success_rate", "value": f"{toon_rate:.1f}%"},
                {"key": "flow", "value": "1.SubTopic(all) → 2.Sentiment(all) → 3.Spokesperson(mainstream) → 4.Normalize → 5.Retry"},
                {"key": "verification_unknown_sub_topics", "value": unknown_sub_topics},
                {"key": "verification_empty_sub_topics", "value": empty_sub_topics},
            ])
            meta.to_excel(writer, index=False, sheet_name="Meta")
        
        stats = {
            "total_rows": len(df_processed),
            "mainstream_rows": mainstream_count,
            "social_rows": social_count,
            "duration": f"{duration:.2f}s",
            "cost": f"${token_summary['estimated_cost_usd']:.6f}",
            "format": {
                "toon_success": token_summary["toon_success"],
                "json_fallback": token_summary["json_fallback"],
                "toon_rate": f"{toon_rate:.1f}%"
            },
            "verification": {
                "unknown_sub_topics": unknown_sub_topics,
                "empty_sub_topics": empty_sub_topics
            }
        }
        
        logging.info("\n" + "="*60)
        logging.info("✅ PROCESSING COMPLETE")
        logging.info("="*60)
        logging.info(f"Total: {len(df_processed)} rows | Duration: {duration:.2f}s | Cost: ${token_summary['estimated_cost_usd']:.6f}")
        logging.info(f"TOON Success: {token_summary['toon_success']}/{token_summary['api_calls']} ({toon_rate:.1f}%)")
        logging.info(f"Verification: Unknown={unknown_sub_topics}, Empty={empty_sub_topics}")
        
        progress(1.0, desc="✅ Complete!")
        return output_path, stats, None
        
    except Exception as e:
        logging.error(f"[ERROR] {str(e)}", exc_info=True)
        return None, {}, f"❌ Error: {str(e)}"

# ====== GRADIO UI ======
def create_gradio_interface():
    with gr.Blocks(title="Insights Generator v6 TOON", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📊 Insights Generator v6 - TOON Format Edition")
        gr.Markdown(f"""
        **Model:** {MODEL_NAME} | **Batch:** {BATCH_SIZE} rows | **Format:** TOON (pipe delimiter) | **Temp:** 0.7
        
        **✅ NEW FEATURES:**
        - 🎯 **TOON Format** - 60% token savings
        - 🔄 **Smart Fallback** - TOON → JSON automatic
        - ⚠️ **Lenient Validation** - Warning + continue
        - 🔁 **Retry Unknown** - Second chance for failed rows
        
        **Flow:**
        1. **Sub Topic** (ALL) - 3-7 words from content
        2. **Sentiment** (ALL) - positive/negative/neutral
        3. **Spokesperson** (MAINSTREAM only)
        4. **Normalize Topic** (2-5 words categories)
        5. **Normalize Spokesperson**
        6. **Retry Unknowns**
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                file_input = gr.File(label="📁 Upload Excel", file_types=[".xlsx"], type="filepath")
                sheet_selector = gr.Dropdown(label="📊 Sheet", choices=[], interactive=True)
                
                def load_sheets(file_path):
                    if file_path:
                        try:
                            xl = pd.ExcelFile(file_path)
                            return gr.Dropdown(choices=xl.sheet_names, value=xl.sheet_names[0])
                        except:
                            return gr.Dropdown(choices=[])
                    return gr.Dropdown(choices=[])
                
                file_input.change(load_sheets, inputs=file_input, outputs=sheet_selector)
            
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Config")
                conf_threshold = gr.Slider(label="Confidence Threshold", minimum=0, maximum=100, value=85, step=5)
                
                gr.Markdown("### ✅ Features")
                gen_topic = gr.Checkbox(label="📌 Topic & Sub Topic (all)", value=True)
                gen_sentiment = gr.Checkbox(label="😊 Sentiment (all)", value=True)
                gen_spokesperson = gr.Checkbox(label="🎤 Spokesperson (mainstream only)", value=True)
        
        process_btn = gr.Button("🚀 Process with TOON Format", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                output_file = gr.File(label="📥 Download")
            with gr.Column():
                stats_output = gr.Textbox(label="📊 Stats", lines=12, interactive=False)
        
        error_output = gr.Textbox(label="⚠️ Errors", visible=False)
        
        def process_wrapper(file_path, sheet_name, topic, sentiment, spokesperson, conf, progress=gr.Progress()):
            if not file_path:
                return None, "", "❌ Upload file"
            if not any([topic, sentiment, spokesperson]):
                return None, "", "❌ Select feature"
            
            result_path, stats, error = process_file(
                file_path, sheet_name, topic, sentiment, spokesperson, conf, progress
            )
            
            stats_str = json.dumps(stats, indent=2, ensure_ascii=False) if stats else ""
            return result_path, stats_str, error if error else ""
        
        process_btn.click(
            process_wrapper,
            inputs=[file_input, sheet_selector, gen_topic, gen_sentiment, gen_spokesperson, conf_threshold],
            outputs=[output_file, stats_output, error_output]
        )
        
        gr.Markdown("""
        ---
        ### 📖 TOON Format Benefits:
        
        **💰 Cost Savings:**
        - JSON: ~4,200 tokens per 50 rows
        - TOON: ~1,700 tokens per 50 rows
        - **Savings: 60%** 🎉
        
        **🎯 Better Accuracy:**
        - Built-in validation with [N] row count
        - Self-documenting with {field|names}
        - Benchmark: 73.9% vs 69.7% (JSON)
        
        **🔄 Smart Processing:**
        - Try TOON format first (optimal)
        - Fallback to JSON if parsing fails
        - Retry unknowns with aggressive prompt
        - Lenient validation (warn + continue)
        
        **Key Points:**
        - Sub Topic: 3-7 kata from Title+Content
        - Topic: 2-5 kata kategori umum
        - Social: NO Spokesperson
        - Mainstream: Gets Spokesperson
        """)
    
    return app

if __name__ == "__main__":
    app = create_gradio_interface()
    app.queue(max_size=10, default_concurrency_limit=4)
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)