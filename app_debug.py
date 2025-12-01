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
from collections import Counter

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

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

BATCH_SIZE = 40
RETRY_BATCH_SIZE = 30
MAX_RETRIES = 2
TRUNCATE_WORDS = 200
RETRY_TRUNCATE_WORDS = 400

# Content validation
MIN_CONTENT_CHARS = 30
MIN_CONTENT_WORDS = 5

MAINSTREAM_CHANNELS = [
    'tv', 'radio', 'newspaper', 'online', 'printmedia', 'site',
    'printed', 'printedmedia', 'print', 'online media'
]

SOCIAL_CHANNELS = ['tiktok', 'instagram', 'youtube', 'facebook', 'twitter', 'x', 'blog', 'forum']
INVALID_VALUES = ['nan', 'none', 'null', 'n/a', 'na', 'unknown', 'tidak ada', '-', 'tidak diketahui', 'undefined', 'not available']

LANGUAGE_CONFIGS = {
    "Indonesia": {
        "code": "id",
        "name": "Bahasa Indonesia",
        "prompt_instruction": "Use Bahasa Indonesia for topic and sub_topic",
        "word_count": "3-7 kata"
    },
    "English": {
        "code": "en",
        "name": "English",
        "prompt_instruction": "Use English for topic and sub_topic",
        "word_count": "3-10 words"
    },
    "Thailand": {
        "code": "th",
        "name": "ภาษาไทย (Thai)",
        "prompt_instruction": "Use Thai language (ภาษาไทย) for topic and sub_topic",
        "word_count": "3-7 คำ"
    },
    "China": {
        "code": "zh",
        "name": "简体中文 (Simplified Chinese)",
        "prompt_instruction": "Use Simplified Chinese (简体中文) for topic and sub_topic",
        "word_count": "3-7 个词"
    }
}

GENERIC_PLACEHOLDERS = {
    "English": "Media Content Topic",
    "Indonesia": "Topik Konten Media",
    "Thailand": "หัวข้อเนื้อหาสื่อ",
    "China": "媒体内容主题"
}

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
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        return True, "API connection successful"
    except Exception as e:
        return False, f"API connection failed: {str(e)}"

api_ok, api_msg = validate_openai_api()

# ====== TOKEN TRACKER ======
class TokenTracker:
    def __init__(self):
        self.total_input = 0
        self.total_output = 0
        self.api_calls = 0
        self.toon_success = 0
        self.json_fallback = 0
        self.parse_failed = 0
        
        self.step_stats = {}
    
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
    
    def add_step_stat(self, step_name, success_count, total_count):
        success = int(success_count) if pd.notna(success_count) else 0
        total = int(total_count) if pd.notna(total_count) else 0
        rate = round(float(success / total * 100), 1) if total > 0 else 0.0
        
        self.step_stats[step_name] = {
            "success": success,
            "total": total,
            "rate": rate
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
            "estimated_cost_usd": float(cost),
            "toon_success": int(self.toon_success),
            "json_fallback": int(self.json_fallback),
            "parse_failed": int(self.parse_failed),
            "step_stats": self.step_stats
        }

# ====== HELPER FUNCTIONS ======
def safe_text(x):
    return "" if pd.isna(x) else str(x).strip()

def truncate_to_first_n_words(text: str, n: int = TRUNCATE_WORDS) -> str:
    words = text.split()
    return " ".join(words[:n])

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    column_mapping = {}
    
    # First pass: check what columns exist
    has_channel = any(col.lower().strip() == 'channel' for col in df.columns)
    has_media_type = any(col.lower().strip() in ['media type', 'mediatype'] for col in df.columns)
    
    for col in df.columns:
        col_lower = col.lower().strip()
        
        if col_lower in ['campaign', 'campaigns']:
            column_mapping[col] = 'Campaigns'
        elif col_lower == 'channel':
            column_mapping[col] = 'Channel'
        elif col_lower in ['media type', 'mediatype'] and not has_channel:
            # Only use Media Type as Channel if Channel column doesn't exist
            column_mapping[col] = 'Channel'
        elif col_lower in ['title', 'judul']:
            column_mapping[col] = 'Title'
        elif col_lower in ['content', 'konten', 'isi']:
            column_mapping[col] = 'Content'
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
        logging.info(f"✅ Normalized columns: {column_mapping}")
    
    return df

def clean_content_for_retry(text: str) -> str:
    import re
    
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    hashtag_pos = text.find('#')
    if hashtag_pos > 20:
        text = text[:hashtag_pos].strip()
    else:
        text = re.sub(r'#\w+', '', text)
    
    text = re.sub(r'@\w+', '', text)
    
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
    
    if len(text.strip()) < 20:
        return text
    
    return text.strip()

def extract_keywords_fallback(content: str, max_words: int = 5, output_language: str = "English") -> str:
    import re
    
    has_thai = bool(re.search(r'[\u0E00-\u0E7F]', content))
    has_chinese = bool(re.search(r'[\u4E00-\u9FFF]', content))
    has_indonesian = any(word in content.lower() for word in ['yang', 'dan', 'dengan', 'untuk', 'dari', 'akan'])
    has_english = bool(re.search(r'\b[a-zA-Z]{4,}\b', content))
    
    if has_thai:
        content_lang = "Thailand"
    elif has_chinese:
        content_lang = "China"
    elif has_indonesian and not has_english:
        content_lang = "Indonesia"
    elif has_english:
        content_lang = "English"
    else:
        content_lang = "Unknown"
    
    if content_lang != output_language and content_lang != "Unknown":
        placeholder = GENERIC_PLACEHOLDERS.get(output_language, "Media Content Topic")
        logging.info(f"🌍 Language mismatch detected: {content_lang} → {output_language}, using placeholder: {placeholder}")
        return placeholder
    
    stopwords = set([
        'yang', 'dan', 'di', 'dari', 'ke', 'untuk', 'dengan', 'pada',
        'ini', 'itu', 'adalah', 'akan', 'atau', 'juga', 'tidak', 'bisa',
        'ada', 'sudah', 'nya', 'si', 'the', 'a', 'an', 'in', 'on', 'at',
        'to', 'of', 'for', 'is', 'are', 'was', 'were', 'be', 'been'
    ])
    
    content = re.sub(r'http\S+|www\.\S+|@\w+|#\w+', '', content)
    words = re.findall(r'\b\w+\b', content.lower())
    words = [w for w in words if len(w) > 3 and w not in stopwords and not w.isdigit()]
    counter = Counter(words)
    top_words = [w for w, _ in counter.most_common(max_words)]
    
    if len(top_words) >= 3:
        return " ".join(top_words[:5]).title()
    elif len(top_words) > 0:
        return " ".join(top_words).title()
    else:
        return GENERIC_PLACEHOLDERS.get(output_language, "Media Content Topic")

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
def build_toon_input(batch_df, title_col, content_col, batch_size, truncate_words=TRUNCATE_WORDS, clean_content=False):
    lines = [f"batch[{batch_size}]{{row|text}}:"]
    
    for idx, row in batch_df.iterrows():
        combined = combine_title_content_row(row, title_col, content_col)
        
        if clean_content:
            combined = clean_content_for_retry(combined)
        
        truncated = truncate_to_first_n_words(combined, truncate_words)
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
        
        if len(data) > 0:
            tracker.add_parse_result("toon")
            return data, "toon"
        else:
            raise ValueError("No data parsed")
    
    except Exception as e:
        pass
    
    try:
        data = extract_json_from_response(response)
        if isinstance(data, list) and len(data) > 0:
            tracker.add_parse_result("json")
            return data, "json"
    except Exception as e:
        pass
    
    tracker.add_parse_result("failed")
    return [], "failed"

# ====== OPENAI API CALL ======
def chat_create(model, messages, token_tracker=None, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7
            )
            
            if token_tracker and hasattr(response, 'usage'):
                input_tok = response.usage.prompt_tokens
                output_tok = response.usage.completion_tokens
                token_tracker.add(input_tok, output_tok)
            
            return response
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                raise
    
    return None

# ====== STEP 1: COMBINED SUB TOPIC + SENTIMENT (ALL CHANNELS) ======
def process_batch_combined(
    batch_df: pd.DataFrame,
    batch_num: int,
    total_batches: int,
    title_col: str,
    content_col: str,
    language: str,
    conf_threshold: int,
    token_tracker: TokenTracker,
    progress=gr.Progress()
) -> pd.DataFrame:
    
    batch_size = len(batch_df)
    lang_config = LANGUAGE_CONFIGS[language]
    
    # Initialize columns only if they don't exist
    if 'Sub Topic' not in batch_df.columns:
        batch_df['Sub Topic'] = ''
    if 'New Sentiment' not in batch_df.columns:
        batch_df['New Sentiment'] = 'neutral'
    if 'New Sentiment Level' not in batch_df.columns:
        batch_df['New Sentiment Level'] = 0
    
    input_toon = build_toon_input(batch_df, title_col, content_col, batch_size)
    
    nonce = random.randint(100000, 999999)
    
    prompt = f"""You are an insights professional with MULTI-LANGUAGE expertise.

[Request ID: {nonce}]
[OUTPUT LANGUAGE: {language} - {lang_config['prompt_instruction']}]

INPUT (TOON format, content may be in ANY language):
{input_toon}

TASK: Analyze content and extract insights
- Content may be in: Thai (ภาษาไทย), English, Chinese (中文), Indonesian, mixed languages, etc.
- You MUST UNDERSTAND content regardless of input language
- EXTRACT core concept/topic ({lang_config['word_count']})
- ANALYZE sentiment (positive/negative/neutral)
- ASSESS confidence (0-100)
- OUTPUT everything in {language}

EXAMPLES (Multi-language → {language}):
Thai input: "ธนาคารออมสินเปิดตัวบริการดิจิทัลใหม่"
→ Output: Banking Digital Service Launch|positive|85

Mixed input: "GSB launches บริการใหม่ for SME customers"
→ Output: Bank SME Service Launch|positive|80

English input: "Customer complains about slow service"
→ Output: Customer Service Complaint|negative|90

CRITICAL RULES:
- Sub topic MUST be in {language} ({lang_config['word_count']})
- Capture core meaning, not literal word-by-word translation
- Sub topic: clear and specific
- Sentiment: positive/negative/neutral based on emotion
- Confidence: 0-100 (your certainty level)
- NEVER use "unknown", "nan", "none", "tidak jelas"
- NEVER output in source language if different from {language}

SENTIMENT RULES:
- positive: clear positive emotion, praise, satisfaction, achievement
- negative: clear negative emotion, complaint, disappointment, criticism
- neutral: factual, informational, no clear emotion, or ambiguous

OUTPUT FORMAT (TOON with pipe delimiter |):
result[{batch_size}]{{row|sub_topic|sentiment|confidence}}:
<row_index>|<sub_topic in {language}>|<sentiment>|<confidence_0-100>

YOUR OUTPUT (TOON format only):"""
    
    try:
        progress(0.5, desc=f"Sub Topic+Sentiment {batch_num}/{total_batches}")
        
        response = chat_create(
            MODEL_NAME,
            [
                {"role": "system", "content": f"You are an insights professional. Output TOON format only. {lang_config['prompt_instruction']}. Handle multi-language input, output in {language}."},
                {"role": "user", "content": prompt}
            ],
            token_tracker=token_tracker
        )
        
        if not response:
            raise Exception("API call failed")
        
        raw = response.choices[0].message.content.strip()
        data, format_type = parse_gpt_response(raw, ['row', 'sub_topic', 'sentiment', 'confidence'], batch_size, token_tracker)
        
        if not data:
            return batch_df
        
        result_dict = {}
        for item in data:
            try:
                row_idx = int(item.get('row', -1))
                result_dict[row_idx] = item
            except:
                continue
        
        for idx in batch_df.index:
            if idx in result_dict:
                item = result_dict[idx]
                
                # Only update Sub Topic if currently empty
                current_sub_topic = batch_df.at[idx, 'Sub Topic']
                if pd.isna(current_sub_topic) or str(current_sub_topic).strip() == '':
                    sub_topic = str(item.get('sub_topic', '')).strip()
                    if not is_invalid_value(sub_topic):
                        words = sub_topic.split()
                        if len(words) >= 3 and len(words) <= 10:
                            batch_df.at[idx, 'Sub Topic'] = sub_topic
                
                sentiment = str(item.get('sentiment', 'neutral')).lower().strip()
                if sentiment in ['positive', 'negative', 'neutral']:
                    try:
                        conf = int(item.get('confidence', 0))
                        conf = max(0, min(100, conf))
                        
                        if conf < conf_threshold and sentiment != 'neutral':
                            sentiment = 'neutral'
                        
                        batch_df.at[idx, 'New Sentiment'] = sentiment
                        batch_df.at[idx, 'New Sentiment Level'] = conf
                    except:
                        pass
        
        return batch_df
        
    except Exception as e:
        logging.error(f"Error in combined processing: {e}")
        return batch_df

# ====== STEP 2: SPOKESPERSON (MAINSTREAM ONLY) ======
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

YOUR OUTPUT (TOON format only):"""
    
    try:
        progress(0.5, desc=f"Spokesperson {batch_num}/{total_batches}")
        
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
            return batch_df
        
        result_dict = {}
        for item in data:
            try:
                row_idx = int(item.get('row', -1))
                result_dict[row_idx] = item
            except:
                continue
        
        for idx in batch_df.index:
            if idx in result_dict:
                item = result_dict[idx]
                spokesperson_val = str(item.get('spokesperson', '')).strip()
                
                if spokesperson_val == '-':
                    spokesperson_val = ''
                
                if not is_invalid_value(spokesperson_val):
                    batch_df.at[idx, 'New Spokesperson'] = spokesperson_val
        
        return batch_df
        
    except Exception as e:
        return batch_df

# ====== RETRY SUB TOPIC WITH FALLBACK ======
def retry_sub_topic_batch(
    batch_df: pd.DataFrame,
    title_col: str,
    content_col: str,
    language: str,
    token_tracker: TokenTracker,
    progress=gr.Progress()
) -> pd.DataFrame:
    
    batch_size = len(batch_df)
    lang_config = LANGUAGE_CONFIGS[language]
    
    input_toon = build_toon_input(
        batch_df, 
        title_col, 
        content_col, 
        batch_size,
        truncate_words=RETRY_TRUNCATE_WORDS,
        clean_content=True
    )
    
    nonce = random.randint(100000, 999999)
    
    prompt = f"""🚨🚨🚨 FINAL WARNING - MANDATORY EXTRACTION 🚨🚨🚨

THIS IS YOUR ABSOLUTE LAST CHANCE!

[Request ID: {nonce}]
[OUTPUT LANGUAGE: {language}]

INPUT (TOON format, content may be ANY language):
{input_toon}

MANDATORY RULES:
1. Content may be in ANY language (Thai, English, Chinese, mixed, etc.)
2. You MUST UNDERSTAND and EXTRACT main topic from EVERY row
3. OUTPUT in {language} ({lang_config['word_count']})
4. Even if unclear, extract KEYWORDS or main concept
5. NEVER use: "unknown", "tidak jelas", "nan", empty
6. NEVER output in source language - ALWAYS use {language}

EXAMPLES:
- Promotional content → extract product/service name
- Social post → extract main theme/topic
- News article → extract main subject
- Hashtag-heavy → extract text before hashtags or main keywords

OUTPUT (TOON format):
result[{batch_size}]{{row|sub_topic}}:
<row_index>|<sub_topic in {language}>

YOUR OUTPUT:"""
    
    try:
        progress(0.95, desc=f"Retrying Sub Topics...")
        
        response = chat_create(
            MODEL_NAME,
            [
                {"role": "system", "content": f"CRITICAL RETRY. {lang_config['prompt_instruction']}. Handle multi-language input. NEVER use 'unknown'."},
                {"role": "user", "content": prompt}
            ],
            token_tracker=token_tracker
        )
        
        if not response:
            raise Exception("API call failed")
        
        raw = response.choices[0].message.content.strip()
        data, format_type = parse_gpt_response(raw, ['row', 'sub_topic'], batch_size, token_tracker)
        
        if not data:
            data = []
        
        result_dict = {}
        for item in data:
            try:
                row_idx = int(item.get('row', -1))
                result_dict[row_idx] = item
            except:
                continue
        
        for idx in batch_df.index:
            if idx in result_dict:
                item = result_dict[idx]
                sub_topic = str(item.get('sub_topic', '')).strip()
                
                if is_invalid_value(sub_topic):
                    continue
                
                words = sub_topic.split()
                if len(words) >= 3 and len(words) <= 10:
                    batch_df.at[idx, 'Sub Topic'] = sub_topic
        
        still_empty_mask = (batch_df['Sub Topic'].isna()) | (batch_df['Sub Topic'].astype(str).str.strip() == '')
        
        if still_empty_mask.sum() > 0:
            for idx in batch_df[still_empty_mask].index:
                row = batch_df.loc[idx]
                combined = combine_title_content_row(row, title_col, content_col)
                fallback_topic = extract_keywords_fallback(combined, output_language=language)
                batch_df.at[idx, 'Sub Topic'] = fallback_topic
        
        return batch_df
        
    except Exception as e:
        still_empty_mask = (batch_df['Sub Topic'].isna()) | (batch_df['Sub Topic'].astype(str).str.strip() == '')
        
        for idx in batch_df[still_empty_mask].index:
            row = batch_df.loc[idx]
            combined = combine_title_content_row(row, title_col, content_col)
            fallback_topic = extract_keywords_fallback(combined, output_language=language)
            batch_df.at[idx, 'Sub Topic'] = fallback_topic
        
        return batch_df

# ====== IMPROVED TWO-STEP NORMALIZATION ======
def normalize_sub_topics_to_topics_v2(
    unique_sub_topics: list,
    language: str,
    token_tracker: TokenTracker,
    progress=gr.Progress()
) -> dict:
    """
    Two-step normalization:
    1. Identify main themes (10-15 themes)
    2. Map each sub topic to a theme
    """
    
    if len(unique_sub_topics) <= 1:
        return {st: st for st in unique_sub_topics}
    
    progress(0.90, desc="Step 1: Identifying main themes...")
    
    lang_config = LANGUAGE_CONFIGS[language]
    
    # Remove fallback placeholder from normalization
    placeholder = GENERIC_PLACEHOLDERS.get(language, "Media Content Topic")
    sub_topics_to_normalize = [st for st in unique_sub_topics if st != placeholder]
    
    if len(sub_topics_to_normalize) == 0:
        return {st: st for st in unique_sub_topics}
    
    joined = "\n".join(f"- {st}" for st in sub_topics_to_normalize)
    nonce = random.randint(100000, 999999)
    
    # STEP 1: Identify main themes
    prompt_step1 = f"""You are an insights professional. Identify MAIN THEMES from these sub-topics.

[Request ID: {nonce}]
[LANGUAGE: {lang_config['prompt_instruction']}]

Sub Topics ({len(sub_topics_to_normalize)} items):
{joined}

TASK:
Analyze these sub-topics and identify 10-15 MAIN THEMES that can group them.

CRITICAL RULES:
1. Create ONLY 10-15 broad themes (NOT one theme per sub-topic!)
2. Each theme should be 2-4 words in {language}
3. Themes should be broad enough to group multiple sub-topics
4. Think about categories like: "Credit Products", "Banking Services", "Community Development", etc.

EXAMPLES OF GOOD THEMES:
- Credit Products (can group: loans, credit cards, financing)
- Banking Services (can group: accounts, transfers, banking events)
- Community Development (can group: community projects, empowerment, support)
- Economic Issues (can group: economy, inflation, financial problems)

BAD EXAMPLES (too specific):
❌ "Easy Credit Building Loan" (this is a sub-topic, not a theme!)
❌ "New Credit Card Launch by Central Group" (too specific!)

OUTPUT FORMAT:
List 10-15 themes only, one per line:
<Theme 1>
<Theme 2>
...
<Theme 10-15>

YOUR OUTPUT (10-15 themes in {language}):"""
    
    try:
        response = chat_create(
            MODEL_NAME,
            [{"role": "system", "content": f"{lang_config['prompt_instruction']}."},
             {"role": "user", "content": prompt_step1}],
            token_tracker=token_tracker
        )
        
        if not response:
            logging.warning("Step 1 failed, using fallback")
            return {st: st for st in unique_sub_topics}
        
        themes_output = response.choices[0].message.content.strip()
        
        # Extract themes
        themes = []
        for line in themes_output.splitlines():
            line = line.strip().lstrip("- ").lstrip("* ").lstrip("1234567890.").strip()
            if line and len(line.split()) >= 2 and len(line.split()) <= 4:
                themes.append(line)
        
        if len(themes) < 5:
            logging.warning(f"Too few themes identified: {len(themes)}, using fallback")
            return {st: st for st in unique_sub_topics}
        
        # Limit to 15 themes max
        themes = themes[:15]
        
        logging.info(f"✅ Identified {len(themes)} main themes: {themes}")
        
        progress(0.95, desc="Step 2: Mapping sub topics to themes...")
        
        # STEP 2: Map each sub topic to a theme
        themes_list = "\n".join(f"{i+1}. {theme}" for i, theme in enumerate(themes))
        
        prompt_step2 = f"""You are an insights professional. Map each sub-topic to ONE main theme.

[Request ID: {nonce}]
[LANGUAGE: {language}]

MAIN THEMES ({len(themes)} themes):
{themes_list}

SUB TOPICS TO MAP ({len(sub_topics_to_normalize)} items):
{joined}

TASK:
Map EACH sub-topic to the MOST appropriate theme from the list above.

CRITICAL RULES:
1. MUST use EXACTLY one of the {len(themes)} themes listed above
2. Group similar concepts together under same theme
3. If sub-topic doesn't fit perfectly, choose closest theme
4. NEVER create new themes - ONLY use themes from the list above

EXAMPLES:
"Easy Credit Building Loan Application" → Credit Products
"Credit Card Launch by Central Group" → Credit Products
"Community Product Development" → Community Development
"Community Organization Empowerment" → Community Development

OUTPUT FORMAT:
<Sub Topic> → <Theme from list above>

One mapping per line. Map ALL {len(sub_topics_to_normalize)} sub-topics.

YOUR OUTPUT:"""
        
        response2 = chat_create(
            MODEL_NAME,
            [{"role": "system", "content": f"{lang_config['prompt_instruction']}."},
             {"role": "user", "content": prompt_step2}],
            token_tracker=token_tracker
        )
        
        if not response2:
            logging.warning("Step 2 failed, using fallback")
            return {st: st for st in unique_sub_topics}
        
        mapping_output = response2.choices[0].message.content
        mapping = {}
        
        # Parse mappings
        for line in mapping_output.splitlines():
            if "→" in line:
                parts = line.split("→")
                if len(parts) == 2:
                    original = parts[0].strip().lstrip("- ")
                    theme = parts[1].strip().lstrip("- ")
                    
                    # Validate theme is in our theme list
                    if theme in themes:
                        if original and theme:
                            mapping[original] = theme
        
        # Add placeholder mapping (don't normalize fallback)
        mapping[placeholder] = placeholder
        
        # Fill in any missing with closest theme or original
        for st in unique_sub_topics:
            if st not in mapping:
                if st == placeholder:
                    mapping[st] = st
                else:
                    # Try to find best matching theme
                    st_words = set(st.lower().split())
                    best_theme = st
                    best_score = 0
                    
                    for theme in themes:
                        theme_words = set(theme.lower().split())
                        overlap = len(st_words & theme_words)
                        if overlap > best_score:
                            best_score = overlap
                            best_theme = theme
                    
                    if best_score > 0:
                        mapping[st] = best_theme
                    else:
                        # Shorten to 2-4 words
                        words = st.split()
                        if len(words) > 4:
                            mapping[st] = " ".join(words[:4])
                        else:
                            mapping[st] = st
        
        # Validate: check grouping efficiency
        unique_topics = len(set(mapping.values()))
        grouping_rate = (1 - unique_topics / len(unique_sub_topics)) * 100
        
        logging.info(f"✅ Normalization result: {len(unique_sub_topics)} sub topics → {unique_topics} topics ({grouping_rate:.1f}% reduction)")
        
        # If grouping is too weak, retry with stronger prompt
        if grouping_rate < 20 and len(unique_sub_topics) > 10:
            logging.warning(f"⚠️ Weak grouping ({grouping_rate:.1f}%), but proceeding...")
        
        return mapping
        
    except Exception as e:
        logging.error(f"Normalization error: {e}")
        # Fallback: just shorten
        mapping = {}
        for st in unique_sub_topics:
            words = st.split()
            if len(words) > 4:
                mapping[st] = " ".join(words[:4])
            else:
                mapping[st] = st
        return mapping

def normalize_spokesperson(
    unique_spokespersons: list,
    token_tracker: TokenTracker,
    progress=gr.Progress()
) -> dict:
    
    unique_spokespersons = [s for s in unique_spokespersons if s and s.strip()]
    
    if len(unique_spokespersons) <= 1:
        return {sp: sp for sp in unique_spokespersons}
    
    progress(0.97, desc="Normalizing Spokesperson...")
    
    joined = "\n".join(f"- {sp}" for sp in unique_spokespersons)
    nonce = random.randint(100000, 999999)
    
    prompt = f"""Normalize spokesperson names referring to the SAME person.

[Request ID: {nonce}]

Spokesperson:
{joined}

Return format:
<Original> → <Normalized>

Output:"""
    
    try:
        response = chat_create(MODEL_NAME, [{"role": "user", "content": prompt}], token_tracker=token_tracker)
        
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
        return {sp: sp for sp in unique_spokespersons}

# ====== MAIN PROCESSING ======
def process_file(
    file_path: str,
    sheet_name: str,
    language: str,
    generate_topic: bool,
    generate_sentiment: bool,
    generate_spokesperson: bool,
    conf_threshold: int,
    progress=gr.Progress()
) -> tuple:
    
    try:
        progress(0.05, desc="Loading...")
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
        
        original_count = len(df)
        df = df.drop_duplicates(subset=['Title', 'Content'], keep='first')
        removed_dup = original_count - len(df)
        if removed_dup > 0:
            logging.info(f"✅ Removed {removed_dup} duplicate rows")
        
        empty_content = (df[content_col].isna()) | \
                       (df[content_col].astype(str).str.len() < MIN_CONTENT_CHARS) | \
                       (df[content_col].astype(str).str.split().str.len() < MIN_CONTENT_WORDS)
        
        if empty_content.any():
            removed_empty = empty_content.sum()
            df = df[~empty_content]
            logging.info(f"✅ Removed {removed_empty} rows with empty/short content (< {MIN_CONTENT_CHARS} chars or < {MIN_CONTENT_WORDS} words)")
            df = df.reset_index(drop=True)
        
        final_row_count = len(df)
        total_removed = original_row_count - final_row_count
        
        logging.info(f"📊 Final rows: {final_row_count} (removed {total_removed} rows)")
        logging.info(f"📝 Output language: {language} ({LANGUAGE_CONFIGS[language]['name']})")
        
        df['_channel_original'] = df[channel_col].copy()
        df['_channel_lower'] = df[channel_col].astype(str).str.lower().str.strip()
        
        empty_channels = df['_channel_lower'].isna() | (df['_channel_lower'] == '') | (df['_channel_lower'] == 'nan')
        if empty_channels.any():
            return None, {}, f"❌ Error: {empty_channels.sum()} baris memiliki Channel kosong!"
        
        df['_original_index'] = df.index
        
        # Convert Noise Tag to text if exists
        if 'Noise Tag' in df.columns:
            df['Noise Tag'] = df['Noise Tag'].astype(str)
            logging.info("✅ Converted Noise Tag to text")
        
        # Count channels (will create actual masks after Step 1)
        mainstream_count = df['_channel_lower'].apply(is_mainstream).sum()
        social_count = df['_channel_lower'].apply(is_social).sum()
        
        logging.info(f"📊 Channel split: Mainstream={mainstream_count}, Social={social_count}")
        
        tracker = TokenTracker()
        start_time = time.time()
        
        # STEP 1: COMBINED SUB TOPIC + SENTIMENT (ALL CHANNELS)
        if generate_topic or generate_sentiment:
            logging.info("\n" + "="*80)
            logging.info("[STEP 1/4] SUB TOPIC + SENTIMENT (ALL CHANNELS)")
            logging.info("="*80)
            
            # Filter rows to process
            rows_to_process_mask = pd.Series([True] * len(df), index=df.index)
            
            # Skip rows with existing Topic AND Sub Topic
            if 'Topic' in df.columns and 'Sub Topic' in df.columns:
                has_both = (df['Topic'].notna() & (df['Topic'].astype(str).str.strip() != '')) & \
                          (df['Sub Topic'].notna() & (df['Sub Topic'].astype(str).str.strip() != ''))
                rows_to_process_mask = rows_to_process_mask & ~has_both
                skipped_existing = has_both.sum()
                if skipped_existing > 0:
                    logging.info(f"⏭️ Skipped {skipped_existing} rows (already have Topic + Sub Topic)")
            
            # Skip rows with Noise Tag = "2"
            if 'Noise Tag' in df.columns:
                noise_tag_2 = df['Noise Tag'] == "2"
                rows_to_process_mask = rows_to_process_mask & ~noise_tag_2
                skipped_noise = noise_tag_2.sum()
                if skipped_noise > 0:
                    logging.info(f"⏭️ Skipped {skipped_noise} rows (Noise Tag = 2)")
            
            df_to_process = df[rows_to_process_mask].copy()
            df_skipped = df[~rows_to_process_mask].copy()
            
            logging.info(f"📊 Processing {len(df_to_process)} rows (skipped {len(df_skipped)} rows)")
            
            all_batches = []
            total_batches = math.ceil(len(df_to_process) / BATCH_SIZE)
            
            for batch_num in range(total_batches):
                start_idx = batch_num * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, len(df_to_process))
                batch_df = df_to_process.iloc[start_idx:end_idx].copy()
                
                progress_val = 0.1 + (batch_num / total_batches) * 0.30
                progress(progress_val, desc=f"[STEP 1/4] Sub Topic+Sentiment {batch_num + 1}/{total_batches}")
                
                result_batch = process_batch_combined(
                    batch_df, batch_num + 1, total_batches,
                    title_col, content_col, language, conf_threshold, tracker, progress
                )
                
                all_batches.append(result_batch)
            
            # Merge processed batches
            df_processed_batches = pd.concat(all_batches, ignore_index=False)
            
            # Combine processed and skipped rows
            df = pd.concat([df_processed_batches, df_skipped], ignore_index=False)
            df = df.sort_values('_original_index').reset_index(drop=True)
            
            # ✅ FIX: Recreate masks after reset_index
            mainstream_mask = df['_channel_lower'].apply(is_mainstream)
            social_mask = df['_channel_lower'].apply(is_social)
            
            if generate_topic:
                sub_topic_filled = df['Sub Topic'].notna() & (df['Sub Topic'].astype(str).str.strip() != '')
                success_count = sub_topic_filled.sum()
                tracker.add_step_stat("Sub Topic (initial)", success_count, len(df))
                logging.info(f"[STEP 1/4] ✅ Sub Topic (initial): {success_count}/{len(df)} ({success_count/len(df)*100:.1f}%)")
            
            if generate_sentiment:
                sentiment_filled = df['New Sentiment'].notna()
                success_count = sentiment_filled.sum()
                tracker.add_step_stat("Sentiment", success_count, len(df))
                logging.info(f"[STEP 1/4] ✅ Sentiment: {success_count}/{len(df)} ({success_count/len(df)*100:.1f}%)")
        
        # STEP 2: SPOKESPERSON (MAINSTREAM ONLY)
        if generate_spokesperson and mainstream_count > 0:
            logging.info("\n" + "="*80)
            logging.info("[STEP 2/4] SPOKESPERSON (MAINSTREAM ONLY)")
            logging.info("="*80)
            
            df['New Spokesperson'] = ''
            df_mainstream = df[mainstream_mask].copy()
            
            mainstream_batches = []
            total_batches = math.ceil(mainstream_count / BATCH_SIZE)
            
            for batch_num in range(total_batches):
                start_idx = batch_num * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, mainstream_count)
                batch_df = df_mainstream.iloc[start_idx:end_idx].copy()
                
                progress_val = 0.45 + (batch_num / total_batches) * 0.15
                progress(progress_val, desc=f"[STEP 2/4] Spokesperson {batch_num + 1}/{total_batches}")
                
                result_batch = process_batch_spokesperson(
                    batch_df, batch_num + 1, total_batches,
                    title_col, content_col, tracker, progress
                )
                
                mainstream_batches.append(result_batch)
            
            df_mainstream = pd.concat(mainstream_batches, ignore_index=False)
            
            for idx in df_mainstream.index:
                if 'New Spokesperson' in df_mainstream.columns:
                    spokes_val = df_mainstream.at[idx, 'New Spokesperson']
                    if pd.notna(spokes_val) and str(spokes_val).strip() and str(spokes_val) != 'nan':
                        df.at[idx, 'New Spokesperson'] = spokes_val
            
            spokes_filled = df_mainstream['New Spokesperson'].notna() & \
                           (df_mainstream['New Spokesperson'].astype(str).str.strip() != '')
            success_count = spokes_filled.sum()
            tracker.add_step_stat("Spokesperson", success_count, mainstream_count)
            
            logging.info(f"[STEP 2/4] ✅ Spokesperson: {success_count}/{mainstream_count} ({success_count/mainstream_count*100:.1f}%)")
        
        df_processed = df.copy()
        
        if generate_spokesperson:
            social_final_mask = df_processed['_channel_lower'].apply(is_social)
            if 'New Spokesperson' not in df_processed.columns:
                df_processed['New Spokesperson'] = ''
            df_processed.loc[social_final_mask, 'New Spokesperson'] = ''
        
        # STEP 3: RETRY FAILED SUB TOPICS
        if generate_topic:
            logging.info("\n" + "="*80)
            logging.info("[STEP 3/4] RETRY UNKNOWN SUB TOPICS")
            logging.info("="*80)
            
            unknown_mask = (df_processed['Sub Topic'].isna()) | \
                          (df_processed['Sub Topic'].astype(str).str.strip() == '') | \
                          (df_processed['Sub Topic'].apply(lambda x: is_invalid_value(str(x))))
            
            df_unknown = df_processed[unknown_mask].copy()
            unknown_count = len(df_unknown)
            
            logging.info(f"[STEP 3/4] Found {unknown_count} unknown sub topics")
            
            if unknown_count > 0:
                progress(0.65, desc=f"[STEP 3/4] Retrying {unknown_count} Sub Topics...")
                
                retry_batches = []
                total_batches = math.ceil(unknown_count / RETRY_BATCH_SIZE)
                
                for batch_num in range(total_batches):
                    start_idx = batch_num * RETRY_BATCH_SIZE
                    end_idx = min(start_idx + RETRY_BATCH_SIZE, unknown_count)
                    batch_df = df_unknown.iloc[start_idx:end_idx].copy()
                    
                    result_batch = retry_sub_topic_batch(
                        batch_df, title_col, content_col, language, tracker, progress
                    )
                    
                    retry_batches.append(result_batch)
                
                df_unknown = pd.concat(retry_batches, ignore_index=False)
                
                for idx in df_unknown.index:
                    df_processed.at[idx, 'Sub Topic'] = df_unknown.at[idx, 'Sub Topic']
                
                sub_topic_filled = df_processed['Sub Topic'].notna() & \
                                  (df_processed['Sub Topic'].astype(str).str.strip() != '')
                final_success = sub_topic_filled.sum()
                tracker.add_step_stat("Sub Topic (after retry)", final_success, len(df_processed))
                
                logging.info(f"[STEP 3/4] ✅ Sub Topic (after retry): {final_success}/{len(df_processed)} ({final_success/len(df_processed)*100:.1f}%)")
        
        # STEP 4: IMPROVED NORMALIZATION (PER CAMPAIGN)
        logging.info("\n" + "="*80)
        logging.info("[STEP 4/4] TWO-STEP NORMALIZATION (PER CAMPAIGN)")
        logging.info("="*80)
        
        if generate_topic:
            progress(0.85, desc="[STEP 4/4] Normalizing Topics (per campaign)...")
            
            # Initialize Topic column if not exists
            if 'Topic' not in df_processed.columns:
                df_processed['Topic'] = ''
            
            # Check if Campaigns column exists
            if 'Campaigns' not in df_processed.columns:
                logging.warning("⚠️ 'Campaigns' column not found, performing global normalization")
                # Fallback to global normalization
                sub_topics = df_processed['Sub Topic'].dropna()
                sub_topics = sub_topics[sub_topics.astype(str).str.strip() != '']
                sub_topics = sub_topics[~sub_topics.apply(lambda x: is_invalid_value(str(x)))]
                unique_sub_topics = sorted(sub_topics.unique().tolist())
                
                if unique_sub_topics:
                    topic_mapping = normalize_sub_topics_to_topics_v2(unique_sub_topics, language, tracker, progress)
                    df_processed['Topic'] = df_processed['Sub Topic'].apply(
                        lambda x: topic_mapping.get(x, x) if x and str(x).strip() and not is_invalid_value(str(x)) else ''
                    )
            else:
                # Normalize per campaign
                unique_campaigns = df_processed['Campaigns'].dropna().unique()
                logging.info(f"[STEP 4/4] Found {len(unique_campaigns)} unique campaigns")
                
                all_topic_mappings = {}
                
                for campaign_idx, campaign in enumerate(unique_campaigns):
                    campaign_mask = df_processed['Campaigns'] == campaign
                    campaign_df = df_processed[campaign_mask]
                    
                    sub_topics = campaign_df['Sub Topic'].dropna()
                    sub_topics = sub_topics[sub_topics.astype(str).str.strip() != '']
                    sub_topics = sub_topics[~sub_topics.apply(lambda x: is_invalid_value(str(x)))]
                    unique_sub_topics = sorted(sub_topics.unique().tolist())
                    
                    if len(unique_sub_topics) > 0:
                        logging.info(f"[STEP 4/4] Campaign '{campaign}': {len(unique_sub_topics)} sub topics")
                        
                        # Progress update per campaign
                        progress_val = 0.85 + (campaign_idx / len(unique_campaigns)) * 0.10
                        progress(progress_val, desc=f"[STEP 4/4] Normalizing campaign {campaign_idx+1}/{len(unique_campaigns)}")
                        
                        topic_mapping = normalize_sub_topics_to_topics_v2(unique_sub_topics, language, tracker, progress)
                        
                        # Apply mapping to this campaign's rows
                        for idx in campaign_df.index:
                            sub_topic_val = df_processed.at[idx, 'Sub Topic']
                            if sub_topic_val and str(sub_topic_val).strip() and not is_invalid_value(str(sub_topic_val)):
                                mapped_topic = topic_mapping.get(sub_topic_val, sub_topic_val)
                                df_processed.at[idx, 'Topic'] = mapped_topic
                                
                                # Store in global mapping for stats
                                all_topic_mappings[sub_topic_val] = mapped_topic
                
                # Calculate stats
                unique_sub_topics_count = len(all_topic_mappings)
                unique_topics_final = len(set(all_topic_mappings.values()))
                
                logging.info(f"[STEP 4/4] Total: {unique_sub_topics_count} sub topics → {unique_topics_final} topics across all campaigns")
            
            topic_filled = df_processed['Topic'].notna() & (df_processed['Topic'].astype(str).str.strip() != '')
            topic_success = topic_filled.sum()
            tracker.add_step_stat("Topic", topic_success, len(df_processed))
            
            # Calculate grouping efficiency
            sub_topics_all = df_processed['Sub Topic'].dropna()
            sub_topics_all = sub_topics_all[sub_topics_all.astype(str).str.strip() != '']
            sub_topics_all = sub_topics_all[~sub_topics_all.apply(lambda x: is_invalid_value(str(x)))]
            unique_sub_topics_count = len(sub_topics_all.unique())
            unique_topics_final = df_processed['Topic'].nunique()
            grouping_rate = (1 - unique_topics_final / unique_sub_topics_count) * 100 if unique_sub_topics_count > 0 else 0
            
            logging.info(f"[STEP 4/4] ✅ Topic: {topic_success}/{len(df_processed)} ({topic_success/len(df_processed)*100:.1f}%)")
            logging.info(f"[STEP 4/4] 📊 Grouping: {unique_sub_topics_count} sub topics → {unique_topics_final} topics ({grouping_rate:.1f}% reduction)")
        else:
            df_processed['Topic'] = ''
        
        if generate_spokesperson:
            progress(0.95, desc="[STEP 4/4] Normalizing Spokesperson...")
            
            spokespersons = df_processed['New Spokesperson'].dropna()
            spokespersons = spokespersons[spokespersons.astype(str).str.strip() != '']
            spokespersons = spokespersons[~spokespersons.apply(lambda x: is_invalid_value(str(x)))]
            unique_spokespersons = sorted(spokespersons.unique().tolist())
            
            if unique_spokespersons:
                spokesperson_mapping = normalize_spokesperson(unique_spokespersons, tracker, progress)
                df_processed['New Spokesperson'] = df_processed['New Spokesperson'].apply(
                    lambda x: spokesperson_mapping.get(x, x) if pd.notna(x) and str(x).strip() and not is_invalid_value(str(x)) else x
                )
        
        df_processed[channel_col] = df_processed['_channel_original']
        df_processed = df_processed.drop(['_channel_original', '_channel_lower', '_original_index'], axis=1, errors='ignore')
        
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
        
        progress(0.98, desc="Saving...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"processed_{timestamp}.xlsx"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            df_processed.to_excel(writer, index=False, sheet_name="Processed")
            
            duration = time.time() - start_time
            token_summary = tracker.get_summary(MODEL_NAME)
            
            if generate_topic and 'Sub Topic' in df_processed.columns:
                sub_topics = df_processed['Sub Topic'].dropna()
                sub_topics = sub_topics[sub_topics.astype(str).str.strip() != '']
                avg_sub_topic_words = sub_topics.astype(str).str.split().str.len().mean() if len(sub_topics) > 0 else 0
            else:
                avg_sub_topic_words = 0
            
            if generate_sentiment and 'New Sentiment' in df_processed.columns:
                sentiment_dist = df_processed['New Sentiment'].value_counts(normalize=True) * 100
                sentiment_str = ", ".join([f"{k}({v:.1f}%)" for k, v in sentiment_dist.items()])
            else:
                sentiment_str = "N/A"
            
            toon_rate = (token_summary['toon_success'] / token_summary['api_calls'] * 100) if token_summary['api_calls'] > 0 else 0
            
            if generate_topic:
                unique_topics_final = df_processed['Topic'].nunique()
                unique_sub_topics_count = df_processed['Sub Topic'].nunique()
                grouping_rate = (1 - unique_topics_final / unique_sub_topics_count) * 100 if unique_sub_topics_count > 0 else 0
            else:
                unique_topics_final = 0
                unique_sub_topics_count = 0
                grouping_rate = 0
            
            meta_data = [
                {"key": "processed_at", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                {"key": "model", "value": MODEL_NAME},
                {"key": "output_language", "value": f"{language} ({LANGUAGE_CONFIGS[language]['name']})"},
                {"key": "duration_sec", "value": f"{duration:.2f}"},
                {"key": "original_rows", "value": int(original_row_count)},
                {"key": "removed_duplicates", "value": int(removed_dup) if 'removed_dup' in locals() else 0},
                {"key": "removed_empty_content", "value": int(removed_empty) if 'removed_empty' in locals() else 0},
                {"key": "final_rows", "value": int(len(df_processed))},
                {"key": "mainstream_rows", "value": int(mainstream_count)},
                {"key": "social_rows", "value": int(social_count)},
                {"key": "batch_size", "value": int(BATCH_SIZE)},
                {"key": "retry_batch_size", "value": int(RETRY_BATCH_SIZE)},
                {"key": "input_tokens", "value": int(token_summary["input_tokens"])},
                {"key": "output_tokens", "value": int(token_summary["output_tokens"])},
                {"key": "total_tokens", "value": int(token_summary["total_tokens"])},
                {"key": "api_calls", "value": int(token_summary["api_calls"])},
                {"key": "cost_usd", "value": f"${token_summary['estimated_cost_usd']:.6f}"},
                {"key": "format", "value": "TOON with pipe delimiter"},
                {"key": "toon_success_rate", "value": f"{toon_rate:.1f}%"},
                {"key": "unique_sub_topics", "value": int(unique_sub_topics_count)},
                {"key": "unique_topics", "value": int(unique_topics_final)},
                {"key": "grouping_efficiency", "value": f"{grouping_rate:.1f}%"},
            ]
            
            for step_name, step_data in token_summary['step_stats'].items():
                meta_data.append({
                    "key": f"success_rate_{step_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}",
                    "value": f"{step_data['success']}/{step_data['total']} ({step_data['rate']:.1f}%)"
                })
            
            meta_data.extend([
                {"key": "avg_sub_topic_words", "value": f"{avg_sub_topic_words:.1f}"},
                {"key": "sentiment_distribution", "value": sentiment_str},
            ])
            
            meta = pd.DataFrame(meta_data)
            meta.to_excel(writer, index=False, sheet_name="Meta")
        
        stats = {
            "total_rows": int(len(df_processed)),
            "mainstream_rows": int(mainstream_count),
            "social_rows": int(social_count),
            "removed": {
                "duplicates": int(removed_dup) if 'removed_dup' in locals() else 0,
                "empty_content": int(removed_empty) if 'removed_empty' in locals() else 0
            },
            "language": {
                "output": f"{language} ({LANGUAGE_CONFIGS[language]['name']})"
            },
            "normalization": {
                "unique_sub_topics": int(unique_sub_topics_count),
                "unique_topics": int(unique_topics_final),
                "grouping_efficiency": f"{grouping_rate:.1f}%"
            },
            "duration": f"{duration:.2f}s",
            "cost": f"${token_summary['estimated_cost_usd']:.6f}",
            "success_rates": token_summary['step_stats']
        }
        
        logging.info("\n" + "="*80)
        logging.info("✅ PROCESSING COMPLETE")
        logging.info("="*80)
        logging.info(f"Original: {original_row_count} rows")
        logging.info(f"Removed: {total_removed} rows (duplicates: {removed_dup if 'removed_dup' in locals() else 0}, empty: {removed_empty if 'removed_empty' in locals() else 0})")
        logging.info(f"Final: {len(df_processed)} rows")
        logging.info(f"Duration: {duration:.2f}s | Cost: ${token_summary['estimated_cost_usd']:.6f}")
        logging.info(f"Language: {language}")
        
        if generate_topic:
            logging.info(f"Normalization: {unique_sub_topics_count} sub topics → {unique_topics_final} topics ({grouping_rate:.1f}% reduction)")
        
        for step_name, step_data in token_summary['step_stats'].items():
            logging.info(f"{step_name}: {step_data['success']}/{step_data['total']} ({step_data['rate']:.1f}%)")
        
        progress(1.0, desc="Complete!")
        return output_path, stats, None
        
    except Exception as e:
        logging.error(f"[ERROR] {str(e)}", exc_info=True)
        return None, {}, f"❌ Error: {str(e)}"

# ====== GRADIO UI ======
def create_gradio_interface():
    with gr.Blocks(title="Insights Generator v10.4", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📊 Insights Generator v10.4 - Smart Processing")
        gr.Markdown("""
        #**✨ New in v10.4:**
        #- 🎯 **Smart Skip**: Preserves existing Topic + Sub Topic (won't overwrite!)
        #- 🔇 **Noise Filter**: Automatically skips rows with Noise Tag = 2
        #- 🎪 **Per-Campaign Normalization**: More accurate topic grouping per campaign
        
        #**Previous features:**
        #- ✅ Index mismatch bug fix (v10.3)
        #- 🎯 Two-step normalization (30+ sub topics → 10-15 topics)
        #- 🌍 Language-aware fallback
        #- 🚀 Combined processing (50% faster)
        
        **Mainstream:** tv, radio, newspaper, online, printmedia, site  
        **Social:** tiktok, instagram, youtube, facebook, twitter, x, blog            
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
                        except Exception as e:
                            return gr.Dropdown(choices=[])
                    return gr.Dropdown(choices=[])
                
                file_input.change(load_sheets, inputs=file_input, outputs=sheet_selector)
            
            with gr.Column(scale=1):
                gr.Markdown("### 🌍 Language")
                language_selector = gr.Dropdown(
                    label="Output Language",
                    choices=list(LANGUAGE_CONFIGS.keys()),
                    value="Indonesia",
                    info="Content can be ANY language, output uses your selection"
                )
                
                gr.Markdown("### ⚙️ Config")
                conf_threshold = gr.Slider(label="Sentiment Confidence Threshold", minimum=0, maximum=100, value=85, step=5)
                
                gr.Markdown("### ✅ Features")
                gen_topic = gr.Checkbox(label="📌 Topic & Sub Topic (all channels)", value=True)
                gen_sentiment = gr.Checkbox(label="😊 Sentiment (all channels)", value=True)
                gen_spokesperson = gr.Checkbox(label="🎤 Spokesperson (mainstream only)", value=True)
        
        process_btn = gr.Button("🚀 Process", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                output_file = gr.File(label="📥 Download")
            with gr.Column():
                stats_output = gr.Textbox(label="📊 Stats", lines=16, interactive=False)
        
        error_output = gr.Textbox(label="⚠️ Status", lines=3, visible=True)
        
        def process_wrapper(file_path, sheet_name, language, topic, sentiment, spokesperson, conf, progress=gr.Progress()):
            try:
                if not file_path:
                    return None, "", "❌ Please upload an Excel file"
                
                if not sheet_name:
                    return None, "", "❌ Please select a sheet"
                
                if not any([topic, sentiment, spokesperson]):
                    return None, "", "❌ Please select at least one feature"
                
                result_path, stats, error = process_file(
                    file_path, sheet_name, language, topic, sentiment, spokesperson, conf, progress
                )
                
                if error:
                    return None, "", error
                
                if not result_path:
                    return None, "", "❌ Processing failed"
                
                try:
                    stats_str = json.dumps(stats, indent=2, ensure_ascii=False) if stats else ""
                except (TypeError, ValueError) as json_err:
                    logging.warning(f"JSON serialization warning: {json_err}, converting types...")
                    
                    def make_serializable(obj):
                        if isinstance(obj, dict):
                            return {k: make_serializable(v) for k, v in obj.items()}
                        elif isinstance(obj, (list, tuple)):
                            return [make_serializable(item) for item in obj]
                        elif hasattr(obj, 'item'):
                            return obj.item()
                        elif hasattr(obj, 'tolist'):
                            return obj.tolist()
                        elif pd.isna(obj):
                            return None
                        else:
                            return obj
                    
                    stats_clean = make_serializable(stats)
                    stats_str = json.dumps(stats_clean, indent=2, ensure_ascii=False)
                
                return result_path, stats_str, "✅ Processing completed!"
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                import traceback
                traceback.print_exc()
                return None, "", error_msg
        
        process_btn.click(
            process_wrapper,
            inputs=[file_input, sheet_selector, language_selector, gen_topic, gen_sentiment, gen_spokesperson, conf_threshold],
            outputs=[output_file, stats_output, error_output]
        )
    
    return app

if __name__ == "__main__":
    app = create_gradio_interface()
    app.queue(max_size=10, default_concurrency_limit=4)
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)