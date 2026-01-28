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
import hashlib

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

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# BATCH SIZES
BATCH_SIZE = 20
MAINSTREAM_BATCH_SIZE = 20
RETRY_BATCH_SIZE = 15
MAX_RETRIES = 1  # Only 1 retry now
TRUNCATE_WORDS = 350
MAINSTREAM_TRUNCATE_WORDS = 350
RETRY_TRUNCATE_WORDS = 500  # More context for retry

MIN_CONTENT_WORDS_FOR_TOPIC = 3

# RETRY THRESHOLDS
RETRY_ENGAGEMENT_THRESHOLD = 5000  # Configurable
RETRY_MAINSTREAM_MAX_ROWS = 100  # Max rows to retry for mainstream

# ENGAGEMENT WEIGHTS
TOPIC_ENGAGEMENT_WEIGHT = 0.6
SIMILARITY_THRESHOLD = 0.40
TARGET_PILLARS_PER_CAMPAIGN = 15  # Flexible target

MAINSTREAM_CHANNELS = [
    'tv', 'radio', 'newspaper', 'online', 'printmedia', 'site',
    'printed', 'printedmedia', 'print', 'online media'
]

SOCIAL_CHANNELS = [
    'tiktok', 'instagram', 'youtube', 'facebook', 'twitter', 'x', 
    'threads', 'blog', 'forum'
]

INVALID_VALUES = [
    'nan', 'none', 'null', 'n/a', 'na', 'unknown', 'tidak ada', '-', 
    'tidak diketahui', 'undefined', 'not available', 'tidak jelas'
]

LANGUAGE_CONFIGS = {
    "Indonesia": {
        "code": "id",
        "name": "Bahasa Indonesia",
        "prompt_instruction": "Use Bahasa Indonesia for pillar and topic",
        "pillar_word_count": "2-6 kata",
        "topic_word_count": "5-15 kata",
        "stopwords": ['yang', 'dan', 'di', 'dari', 'ke', 'untuk', 'dengan', 'pada',
                     'ini', 'itu', 'adalah', 'akan', 'atau', 'juga', 'tidak', 'bisa',
                     'ada', 'sudah', 'nya', 'si', 'oleh', 'dalam', 'sebagai', 'telah']
    },
    "English": {
        "code": "en",
        "name": "English",
        "prompt_instruction": "Use English for pillar and topic",
        "pillar_word_count": "2-6 words",
        "topic_word_count": "5-15 words",
        "stopwords": ['the', 'a', 'an', 'in', 'on', 'at', 'to', 'of', 'for', 'is', 
                     'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                     'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can']
    },
    "Thailand": {
        "code": "th",
        "name": "ภาษาไทย (Thai)",
        "prompt_instruction": "Use Thai language (ภาษาไทย) for pillar and topic",
        "pillar_word_count": "2-6 คำ",
        "topic_word_count": "5-15 คำ",
        "stopwords": ['ที่', 'และ', 'ใน', 'เป็น', 'ของ', 'กับ', 'ได้', 'มี', 'ให้', 'จาก']
    },
    "China": {
        "code": "zh",
        "name": "简体中文 (Simplified Chinese)",
        "prompt_instruction": "Use Simplified Chinese (简体中文) for pillar and topic",
        "pillar_word_count": "2-6 个词",
        "topic_word_count": "5-15 个词",
        "stopwords": ['的', '是', '在', '了', '和', '有', '为', '也', '与', '或']
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
            "estimated_cost_usd": float(cost),
            "toon_success": int(self.toon_success),
            "json_fallback": int(self.json_fallback),
            "parse_failed": int(self.parse_failed),
            "step_stats": self.step_stats
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

def normalize_topic_text(text: str, language: str) -> str:
    if not text or pd.isna(text):
        return ""
    
    text = str(text).strip()
    
    if not text or text == "-":
        return ""
    
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
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
    text = emoji_pattern.sub('', text)
    
    text = re.sub(r'[^\w\s\u0E00-\u0E7F\u4E00-\u9FFF]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    noise_words = {
        'Indonesia': ['berita', 'info', 'informasi', 'artikel', 'konten', 'posting', 'post', 'viral', 'trending', 'terbaru', 'update'],
        'English': ['news', 'info', 'information', 'article', 'content', 'post', 'posting', 'viral', 'trending', 'latest', 'update'],
        'Thailand': ['ข่าว', 'ข้อมูล', 'บทความ', 'โพสต์'],
        'China': ['新闻', '信息', '文章', '内容', '帖子']
    }
    
    words = text.split()
    noise = set(noise_words.get(language, []))
    
    filtered_words = [w for w in words if w.lower() not in noise]
    
    if not filtered_words:
        return ""
    
    text = " ".join(filtered_words)
    
    if language in ['Thailand', 'China']:
        return text
    else:
        return text.title()

def validate_and_normalize_topic(text: str, language: str, min_words: int = 1, max_words: int = 15) -> str:
    """Topic validation with 5-15 words"""
    normalized = normalize_topic_text(text, language)
    
    if not normalized:
        return ""
    
    words = normalized.split()
    word_count = len(words)
    
    if word_count < min_words:
        return ""
    
    if word_count > max_words:
        return " ".join(words[:max_words])
    
    return normalized

def validate_and_normalize_pillar(text: str, language: str, min_words: int = 1, max_words: int = 6) -> str:
    """Pillar validation with 2-6 words"""
    normalized = normalize_topic_text(text, language)
    
    if not normalized:
        return ""
    
    words = normalized.split()
    word_count = len(words)
    
    if word_count < min_words:
        return ""
    
    if word_count > max_words:
        return " ".join(words[:max_words])
    
    return normalized

def count_meaningful_words(text: str) -> int:
    cleaned = clean_content_for_analysis(text)
    words = cleaned.split()
    return len(words)

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

def create_dedup_hash(row, title_col, content_col):
    combined = combine_title_content_row(row, title_col, content_col)
    return hashlib.md5(combined.encode()).hexdigest()

def build_toon_input(batch_df, title_col, content_col, batch_size, truncate_words=TRUNCATE_WORDS, clean_content=False):
    lines = [f"batch[{batch_size}]{{row|text}}:"]
    
    for idx, row in batch_df.iterrows():
        combined = combine_title_content_row(row, title_col, content_col)
        
        if clean_content:
            combined = clean_content_for_analysis(combined)
        
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

def chat_create(model, messages, token_tracker=None, max_retries=3):
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

def process_batch_topic_sentiment(
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
    """Step 1: Extract Topic + Sentiment only (no Pillar yet)"""
    
    batch_size = len(batch_df)
    lang_config = LANGUAGE_CONFIGS[language]
    
    if 'Topic' not in batch_df.columns:
        batch_df['Topic'] = ''
    if 'New Sentiment' not in batch_df.columns:
        batch_df['New Sentiment'] = 'neutral'
    if 'New Sentiment Level' not in batch_df.columns:
        batch_df['New Sentiment Level'] = 0
    
    input_toon = build_toon_input(batch_df, title_col, content_col, batch_size, 
                                  truncate_words=TRUNCATE_WORDS, clean_content=True)
    
    nonce = random.randint(100000, 999999)
    
    prompt = f"""You are an ELITE insights analyst with CRITICAL THINKING skills.

[Request ID: {nonce}]
[OUTPUT LANGUAGE: {language}]

INPUT (TOON format, content may be in ANY language):
{input_toon}

🎯 YOUR MISSION: Extract TOPIC + SENTIMENT with DEEP INSIGHTS

You MUST analyze content with FORENSIC DETAIL and capture:

**TOPIC** ({lang_config['topic_word_count']}):
- DETAILED and SPECIFIC description
- Include WHO + WHAT + WHERE when relevant
- Preserve important entities (names, places, organizations)
- ✅ Examples: "Pertanyaan tentang isu IMIP di Morowali", "Perkelahian pekerja di kawasan IMIP"
- ❌ NEVER: Generic terms like "berita", "info", "update", "viral"

**SENTIMENT**:
- positive: Clear positive emotion, praise, satisfaction
- negative: Clear negative emotion, complaint, criticism, concern
- neutral: Factual, informational, question without emotion

**CONFIDENCE** (0-100):
- How certain are you about topic and sentiment?
- Be honest about ambiguity

CRITICAL RULES:
- Content can be in ANY language → You MUST understand it → Output in {language}
- NEVER use noise words: berita, info, informasi, viral, trending, update, artikel
- NEVER output "unknown", "tidak jelas", "nan"
- If you cannot extract meaningful insights, use "-" for that field
- BE SPECIFIC - avoid generic categorization
- Topic MUST be 5-15 words with clear details

OUTPUT FORMAT (TOON with pipe delimiter |):
result[{batch_size}]{{row|topic|sentiment|confidence}}:
<row_index>|<topic in {language} or ->|<sentiment>|<confidence_0-100>

YOUR OUTPUT (TOON format only):"""
    
    try:
        progress_val = 0.1 + (batch_num / total_batches) * 0.30
        progress(progress_val, desc=f"[STEP 1/5] Topic+Sentiment {batch_num}/{total_batches}")
        
        response = chat_create(
            MODEL_NAME,
            [
                {"role": "system", "content": f"You are an ELITE insights analyst. {lang_config['prompt_instruction']}. Handle multi-language input. Output in {language}. NEVER use generic terms."},
                {"role": "user", "content": prompt}
            ],
            token_tracker=token_tracker
        )
        
        if not response:
            raise Exception("API call failed")
        
        raw = response.choices[0].message.content.strip()
        data, format_type = parse_gpt_response(raw, ['row', 'topic', 'sentiment', 'confidence'], 
                                               batch_size, token_tracker)
        
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
                
                # Process TOPIC
                if pd.isna(batch_df.at[idx, 'Topic']) or str(batch_df.at[idx, 'Topic']).strip() == '':
                    topic = str(item.get('topic', '')).strip()
                    
                    if topic != '-' and not is_invalid_value(topic):
                        topic = validate_and_normalize_topic(topic, language)
                        if topic:
                            batch_df.at[idx, 'Topic'] = topic
                
                # Process SENTIMENT
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
        logging.error(f"Error in topic+sentiment processing: {e}")
        return batch_df

def retry_topic_batch_smart(
    batch_df: pd.DataFrame,
    title_col: str,
    content_col: str,
    language: str,
    token_tracker: TokenTracker,
    progress=gr.Progress()
) -> pd.DataFrame:
    """Step 2: Smart retry for empty topics with more context"""
    
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
    
    prompt = f"""🚨 RETRY ATTEMPT - YOU MUST SUCCEED! 🚨

[Request ID: {nonce}]
[OUTPUT LANGUAGE: {language}]
[MORE CONTEXT PROVIDED: {RETRY_TRUNCATE_WORDS} words]

INPUT (TOON format, content may be ANY language):
{input_toon}

⚡ THIS IS YOUR FINAL CHANCE - EXTRACT TOPIC NOW! ⚡

PREVIOUS ATTEMPT FAILED - This time you MUST:
1. Read FULL context carefully ({RETRY_TRUNCATE_WORDS} words provided)
2. Understand content in ANY language (Thai, English, Chinese, Indonesian, mixed)
3. Extract SPECIFIC details - WHO, WHAT, WHERE
4. NEVER use generic terms
5. Output detailed insights in {language}

MANDATORY EXTRACTION RULES:

**TOPIC** ({lang_config['topic_word_count']}):
- DETAILED with specific entities
- Include WHO + WHAT + WHERE
- Examples: "Pertanyaan tentang isu IMIP di Morowali", "Perkelahian pekerja di kawasan IMIP"

CRITICAL:
- Content language can be ANYTHING → You MUST understand → Output in {language}
- NEVER: "berita", "info", "viral", "trending", "unknown", "tidak jelas"
- If truly impossible, use "-" but TRY EVERYTHING FIRST
- Extract KEYWORDS if unclear
- BE SPECIFIC not generic

OUTPUT (TOON format):
result[{batch_size}]{{row|topic}}:
<row_index>|<topic in {language} or ->

YOUR OUTPUT:"""
    
    try:
        progress(0.45, desc=f"[STEP 2/5] Retry Topics (high engagement)...")
        
        response = chat_create(
            MODEL_NAME,
            [
                {"role": "system", "content": f"CRITICAL RETRY. {lang_config['prompt_instruction']}. Multi-language expert. NEVER generic terms. BE SPECIFIC."},
                {"role": "user", "content": prompt}
            ],
            token_tracker=token_tracker
        )
        
        if not response:
            raise Exception("API call failed")
        
        raw = response.choices[0].message.content.strip()
        data, format_type = parse_gpt_response(raw, ['row', 'topic'], batch_size, token_tracker)
        
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
                
                # Update TOPIC if still empty
                if pd.isna(batch_df.at[idx, 'Topic']) or str(batch_df.at[idx, 'Topic']).strip() == '':
                    topic = str(item.get('topic', '')).strip()
                    
                    if topic != '-' and not is_invalid_value(topic):
                        topic = validate_and_normalize_topic(topic, language)
                        if topic:
                            batch_df.at[idx, 'Topic'] = topic
        
        return batch_df
        
    except Exception as e:
        logging.error(f"Retry attempt failed: {e}")
        return batch_df

def prepare_engagement_data(df, campaign_col='Campaigns'):
    engagement_col = 'Engagement' if 'Engagement' in df.columns else None
    
    if not engagement_col:
        logging.warning("⚠️ 'Engagement' column not found, using frequency only")
        engagement_map = df.groupby([campaign_col, 'Topic']).agg({
            'Title': 'count'
        }).reset_index()
        engagement_map.columns = [campaign_col, 'Topic', 'Frequency']
        engagement_map['Total_Engagement'] = 0
        engagement_map['Weight_Score'] = engagement_map['Frequency']
    else:
        engagement_map = df.groupby([campaign_col, 'Topic']).agg({
            engagement_col: 'sum',
            'Title': 'count'
        }).reset_index()
        
        engagement_map.columns = [campaign_col, 'Topic', 'Total_Engagement', 'Frequency']
        
        engagement_map['Weight_Score'] = (
            engagement_map['Total_Engagement'] * TOPIC_ENGAGEMENT_WEIGHT +
            engagement_map['Frequency'] * (1 - TOPIC_ENGAGEMENT_WEIGHT)
        )
    
    return engagement_map

def extract_significant_words(text: str, language: str) -> list:
    stopwords = set(LANGUAGE_CONFIGS.get(language, {}).get('stopwords', []))
    
    words = text.lower().split()
    keywords = [w for w in words if w not in stopwords and len(w) > 2 and not w.isdigit()]
    
    return keywords

def pre_cluster_topics_with_engagement(topics: list, 
                                       engagement_data: pd.DataFrame,
                                       language: str,
                                       threshold: float = SIMILARITY_THRESHOLD) -> dict:
    """Step 3: Cluster topics by similarity with engagement priority"""
    engagement_lookup = dict(zip(
        engagement_data['Topic'], 
        engagement_data['Weight_Score']
    ))
    
    keyword_map = {}
    for topic in topics:
        keywords = extract_significant_words(topic, language)
        keyword_map[topic] = set(keywords)
    
    groups = {}
    used = set()
    group_id = 0
    
    sorted_topics = sorted(
        topics, 
        key=lambda x: engagement_lookup.get(x, 0), 
        reverse=True
    )
    
    for topic1 in sorted_topics:
        if topic1 in used:
            continue
        
        current_group = {
            'topics': [topic1],
            'engagement_scores': [engagement_lookup.get(topic1, 0)],
            'total_engagement': engagement_lookup.get(topic1, 0)
        }
        used.add(topic1)
        
        for topic2 in topics:
            if topic2 in used:
                continue
            
            if not keyword_map[topic1] or not keyword_map[topic2]:
                continue
                
            similarity = len(keyword_map[topic1] & keyword_map[topic2]) / \
                        len(keyword_map[topic1] | keyword_map[topic2])
            
            if similarity >= threshold:
                current_group['topics'].append(topic2)
                current_group['engagement_scores'].append(
                    engagement_lookup.get(topic2, 0)
                )
                current_group['total_engagement'] += engagement_lookup.get(topic2, 0)
                used.add(topic2)
        
        current_group['avg_engagement'] = (
            current_group['total_engagement'] / len(current_group['topics'])
        )
        
        groups[f"group_{group_id}"] = current_group
        group_id += 1
    
    sorted_groups = dict(
        sorted(groups.items(), 
               key=lambda x: x[1]['total_engagement'], 
               reverse=True)
    )
    
    return sorted_groups

def generate_pillars_from_topic_groups(
    campaign: str,
    groups: dict,
    language: str,
    token_tracker: TokenTracker,
    progress=gr.Progress()
) -> dict:
    """Step 4a: Generate Pillar names from topic groups using LLM"""
    
    lang_config = LANGUAGE_CONFIGS[language]
    
    # Prepare groups for LLM
    regular_groups = []
    singleton_topics = []
    
    for group_id, group_data in groups.items():
        topics = group_data['topics']
        
        if len(topics) == 1:
            # Singleton - will batch process separately
            singleton_topics.append({
                'topic': topics[0],
                'engagement': group_data['total_engagement']
            })
        else:
            # Regular group - process individually
            sorted_topics = sorted(
                zip(topics, group_data['engagement_scores']),
                key=lambda x: x[1],
                reverse=True
            )
            
            regular_groups.append({
                'group_id': group_id,
                'topics': [t for t, _ in sorted_topics[:10]],  # Top 10 topics
                'total_engagement': group_data['total_engagement'],
                'topic_count': len(topics)
            })
    
    pillar_mapping = {}
    
    # Process regular groups (batch by campaign)
    if regular_groups:
        nonce = random.randint(100000, 999999)
        
        groups_summary = []
        for g in regular_groups:
            groups_summary.append({
                'group_id': g['group_id'],
                'sample_topics': g['topics'][:5],
                'topic_count': g['topic_count'],
                'engagement': g['total_engagement']
            })
        
        prompt = f"""You are a strategic categorization expert.

[Request ID: {nonce}]
[Campaign: {campaign}]
[OUTPUT LANGUAGE: {language}]

INPUT: {len(regular_groups)} topic groups (sorted by engagement - highest first!)

Groups:
{json.dumps(groups_summary, indent=2, ensure_ascii=False)}

TASK:
For EACH group, generate a Pillar name ({lang_config['pillar_word_count']}) in {language}.

Pillar should be:
- Strategic CATEGORIZATION level (broader than topics)
- Capture main theme of the group
- Based on high-engagement topics in the group

EXAMPLES:
✅ "Resiko IMIP Tutup" (specific issue category)
✅ "Perkelahian Pekerja" (specific event category)
✅ "Membuat Kue" (activity category)

❌ "Berita" (too generic!)
❌ "Informasi" (noise word!)

OUTPUT FORMAT (JSON):
{{
  "pillars": [
    {{
      "group_id": "group_0",
      "pillar_name": "...",
      "reason": "Why this categorization"
    }},
    ...
  ]
}}

OUTPUT (JSON only):"""
        
        try:
            response = chat_create(
                MODEL_NAME,
                [{"role": "user", "content": prompt}],
                token_tracker=token_tracker
            )
            
            if response:
                raw = response.choices[0].message.content.strip()
                result = extract_json_from_response(raw)
                
                if result and 'pillars' in result:
                    for pillar_info in result['pillars']:
                        group_id = pillar_info.get('group_id')
                        pillar_name = pillar_info.get('pillar_name', '')
                        
                        if group_id and pillar_name:
                            # Validate and normalize
                            pillar_name = validate_and_normalize_pillar(pillar_name, language)
                            
                            if pillar_name:
                                # Map all topics in this group to this pillar
                                if group_id in groups:
                                    for topic in groups[group_id]['topics']:
                                        pillar_mapping[topic] = pillar_name
        
        except Exception as e:
            logging.error(f"Error generating pillars for regular groups: {e}")
    
    # Process singletons (batch all together)
    if singleton_topics:
        nonce = random.randint(100000, 999999)
        
        singleton_list = [
            f"{i+1}. \"{s['topic']}\" (engagement: {s['engagement']:.0f})"
            for i, s in enumerate(singleton_topics[:20])  # Max 20 at a time
        ]
        
        prompt = f"""You are a strategic categorization expert.

[Request ID: {nonce}]
[Campaign: {campaign}]
[OUTPUT LANGUAGE: {language}]

INPUT: {len(singleton_list)} standalone topics (no similar topics found)

Topics:
{chr(10).join(singleton_list)}

TASK:
For EACH topic, generate a Pillar name ({lang_config['pillar_word_count']}) in {language}.

Pillar should:
- Categorize the topic (broader, strategic level)
- Be 2-6 words
- Capture the essence

EXAMPLES:
Topic: "Resep nasi goreng kampung khas Semarang spesial pedas"
→ Pillar: "Resep Nasi Goreng"

Topic: "Cara membuat minuman soda sprite segar"
→ Pillar: "Membuat Minuman Soda"

OUTPUT FORMAT (JSON):
{{
  "pillars": [
    {{"number": 1, "pillar": "..."}},
    {{"number": 2, "pillar": "..."}},
    ...
  ]
}}

OUTPUT (JSON only):"""
        
        try:
            response = chat_create(
                MODEL_NAME,
                [{"role": "user", "content": prompt}],
                token_tracker=token_tracker
            )
            
            if response:
                raw = response.choices[0].message.content.strip()
                result = extract_json_from_response(raw)
                
                if result and 'pillars' in result:
                    for pillar_info in result['pillars']:
                        number = pillar_info.get('number')
                        pillar_name = pillar_info.get('pillar', '')
                        
                        if number and pillar_name and number <= len(singleton_topics):
                            pillar_name = validate_and_normalize_pillar(pillar_name, language)
                            
                            if pillar_name:
                                topic = singleton_topics[number - 1]['topic']
                                pillar_mapping[topic] = pillar_name
        
        except Exception as e:
            logging.error(f"Error generating pillars for singletons: {e}")
    
    return pillar_mapping

def normalize_pillars_per_campaign(
    pillars: list,
    campaign: str,
    language: str,
    token_tracker: TokenTracker,
    progress=gr.Progress()
) -> dict:
    """Step 4b: Normalize similar pillars within campaign (LLM-based)"""
    
    unique_pillars = sorted(list(set(pillars)))
    
    if len(unique_pillars) <= 1:
        return {p: p for p in unique_pillars}
    
    nonce = random.randint(100000, 999999)
    
    pillar_list = "\n".join(f"- {p}" for p in unique_pillars)
    
    prompt = f"""You are a Pillar normalization expert.

[Request ID: {nonce}]
[Campaign: {campaign}]
[OUTPUT LANGUAGE: {language}]

INPUT: {len(unique_pillars)} unique Pillars from this campaign

Pillars:
{pillar_list}

TASK:
Identify and MERGE similar Pillars that refer to the SAME category.

RULES:
- Merge only if they are truly the same category
- Keep the most descriptive/common name
- Preserve distinct Pillars

EXAMPLES:
"Membuat Kue Lebaran" + "Resep Kue Lebaran" → "Membuat Kue Lebaran"
"Minuman Segar" + "Resep Minuman" → Keep separate (different focus)

OUTPUT FORMAT:
For each merge, output:
<Original Pillar> → <Normalized Pillar>

For pillars that stay unchanged, output:
<Pillar> → <Pillar>

YOUR OUTPUT:"""
    
    try:
        response = chat_create(
            MODEL_NAME,
            [{"role": "user", "content": prompt}],
            token_tracker=token_tracker
        )
        
        if not response:
            return {p: p for p in unique_pillars}
        
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
        
        # Fill in any missing pillars
        for p in unique_pillars:
            if p not in mapping:
                mapping[p] = p
        
        return mapping
        
    except Exception as e:
        logging.error(f"Error normalizing pillars: {e}")
        return {p: p for p in unique_pillars}

def process_batch_spokesperson(
    batch_df: pd.DataFrame,
    batch_num: int,
    total_batches: int,
    title_col: str,
    content_col: str,
    token_tracker: TokenTracker,
    progress=gr.Progress()
) -> pd.DataFrame:
    """Step 5: Extract spokesperson (mainstream only)"""
    
    batch_size = len(batch_df)
    
    batch_df['New Spokesperson'] = ''
    
    input_toon = build_toon_input(batch_df, title_col, content_col, batch_size, 
                                  truncate_words=MAINSTREAM_TRUNCATE_WORDS)
    
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
- DO NOT use "unknown", "tidak jelas" or "nan"

OUTPUT FORMAT (TOON with pipe delimiter |):
result[{batch_size}]{{row|spokesperson}}:
<row_index>|<spokesperson or ->

YOUR OUTPUT (TOON format only):"""
    
    try:
        progress_val = 0.90 + (batch_num / total_batches) * 0.08
        progress(progress_val, desc=f"[STEP 5/5] Spokesperson {batch_num}/{total_batches}")
        
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
                
                if spokesperson_val == '-' or is_invalid_value(spokesperson_val):
                    spokesperson_val = ''
                
                if spokesperson_val:
                    batch_df.at[idx, 'New Spokesperson'] = spokesperson_val
        
        return batch_df
        
    except Exception as e:
        logging.error(f"Error in spokesperson extraction: {e}")
        return batch_df

def normalize_spokesperson(
    unique_spokespersons: list,
    token_tracker: TokenTracker,
    progress=gr.Progress()
) -> dict:
    """Normalize spokesperson names referring to the SAME person"""
    
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
        response = chat_create(MODEL_NAME, [{"role": "user", "content": prompt}], 
                             token_tracker=token_tracker)
        
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
        logging.error(f"Error normalizing spokesperson: {e}")
        return {sp: sp for sp in unique_spokespersons}

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
        
        if 'Noise Tag' in df.columns:
            df['Noise Tag'] = df['Noise Tag'].astype(str)
            logging.info("✅ Converted Noise Tag to text")
        
        # Check for Engagement/Buzz columns
        has_engagement = 'Engagement' in df.columns
        has_buzz = 'Buzz' in df.columns
        
        if not has_engagement and not has_buzz:
            logging.warning("⚠️ No 'Engagement' or 'Buzz' column found")
        
        if not has_engagement:
            df['Engagement'] = 0
        
        logging.info(f"✅ NO DELETION - All {original_row_count} rows will be processed")
        
        df['_original_index'] = df.index
        df['_channel_original'] = df[channel_col].copy()
        df['_channel_lower'] = df[channel_col].astype(str).str.lower().str.strip()
        
        empty_channels = df['_channel_lower'].isna() | (df['_channel_lower'] == '') | (df['_channel_lower'] == 'nan')
        if empty_channels.any():
            return None, {}, f"❌ Error: {empty_channels.sum()} baris memiliki Channel kosong!"
        
        logging.info("\n" + "="*80)
        logging.info("[DEDUPLICATION] Creating groups for identical content")
        logging.info("="*80)
        
        df['_dedup_hash'] = df.apply(lambda row: create_dedup_hash(row, title_col, content_col), axis=1)
        df['_is_master'] = False
        
        dedup_groups = df.groupby('_dedup_hash').head(1).index
        df.loc[dedup_groups, '_is_master'] = True
        
        total_rows = len(df)
        master_rows = df['_is_master'].sum()
        duplicate_rows = total_rows - master_rows
        
        logging.info(f"✅ Deduplication: {total_rows} rows → {master_rows} unique groups + {duplicate_rows} duplicates")
        
        mainstream_mask = df['_channel_lower'].apply(is_mainstream)
        social_mask = df['_channel_lower'].apply(is_social)
        
        mainstream_count = mainstream_mask.sum()
        social_count = social_mask.sum()
        
        logging.info(f"📊 Channel split: Mainstream={mainstream_count}, Social={social_count}")
        
        logging.info("\n" + "="*80)
        logging.info("[PRE-FILTER] Checking content eligibility")
        logging.info("="*80)
        
        df['_word_count'] = df.apply(
            lambda row: count_meaningful_words(combine_title_content_row(row, title_col, content_col)),
            axis=1
        )
        
        df['_eligible_for_topic'] = df['_word_count'] >= MIN_CONTENT_WORDS_FOR_TOPIC
        
        total_eligible = df['_eligible_for_topic'].sum()
        total_skipped = (~df['_eligible_for_topic']).sum()
        
        logging.info(f"✅ Content filter: {total_eligible} eligible, {total_skipped} skipped (<{MIN_CONTENT_WORDS_FOR_TOPIC} words)")
        
        if generate_topic:
            if 'Topic' not in df.columns:
                df['Topic'] = ''
            if 'Pillar' not in df.columns:
                df['Pillar'] = ''
        
        if generate_sentiment:
            if 'New Sentiment' not in df.columns:
                df['New Sentiment'] = 'neutral'
            if 'New Sentiment Level' not in df.columns:
                df['New Sentiment Level'] = 0
            
            df.loc[~df['_eligible_for_topic'], 'New Sentiment'] = 'neutral'
            df.loc[~df['_eligible_for_topic'], 'New Sentiment Level'] = 0
        
        if generate_spokesperson:
            if 'New Spokesperson' not in df.columns:
                df['New Spokesperson'] = ''
        
        tracker = TokenTracker()
        start_time = time.time()
        
        # ============================================================
        # STEP 1: TOPIC + SENTIMENT EXTRACTION
        # ============================================================
        if generate_topic or generate_sentiment:
            logging.info("\n" + "="*80)
            logging.info("[STEP 1/5] TOPIC + SENTIMENT (NO PILLAR YET)")
            logging.info("="*80)
            
            process_mask = df['_is_master'] & df['_eligible_for_topic']
            
            if 'Noise Tag' in df.columns:
                noise_tag_2 = df['Noise Tag'] == "2"
                process_mask = process_mask & ~noise_tag_2
            
            df_to_process = df[process_mask].copy()
            logging.info(f"📊 Processing {len(df_to_process)} master rows (eligible content only)")
            
            if len(df_to_process) > 0:
                all_batches = []
                total_batches = math.ceil(len(df_to_process) / BATCH_SIZE)
                
                for batch_num in range(total_batches):
                    start_idx = batch_num * BATCH_SIZE
                    end_idx = min(start_idx + BATCH_SIZE, len(df_to_process))
                    batch_df = df_to_process.iloc[start_idx:end_idx].copy()
                    
                    result_batch = process_batch_topic_sentiment(
                        batch_df, batch_num + 1, total_batches,
                        title_col, content_col, language, conf_threshold, tracker, progress
                    )
                    
                    all_batches.append(result_batch)
                
                df_processed = pd.concat(all_batches, ignore_index=False)
                
                for idx in df_processed.index:
                    if generate_topic and 'Topic' in df_processed.columns:
                        df.at[idx, 'Topic'] = df_processed.at[idx, 'Topic']
                    if generate_sentiment:
                        df.at[idx, 'New Sentiment'] = df_processed.at[idx, 'New Sentiment']
                        df.at[idx, 'New Sentiment Level'] = df_processed.at[idx, 'New Sentiment Level']
                
                # Copy to duplicates
                logging.info("📋 Copying results to duplicate rows...")
                for hash_val in df['_dedup_hash'].unique():
                    group = df[df['_dedup_hash'] == hash_val]
                    if len(group) > 1:
                        master_idx = group[group['_is_master']].index[0]
                        duplicate_indices = group[~group['_is_master']].index
                        
                        for dup_idx in duplicate_indices:
                            if generate_topic:
                                if df.at[master_idx, 'Topic'] and str(df.at[master_idx, 'Topic']).strip():
                                    df.at[dup_idx, 'Topic'] = df.at[master_idx, 'Topic']
                            if generate_sentiment:
                                df.at[dup_idx, 'New Sentiment'] = df.at[master_idx, 'New Sentiment']
                                df.at[dup_idx, 'New Sentiment Level'] = df.at[master_idx, 'New Sentiment Level']
                
                if generate_topic:
                    topic_filled = df['Topic'].notna() & (df['Topic'].astype(str).str.strip() != '')
                    topic_success = topic_filled.sum()
                    
                    tracker.add_step_stat("Topic (initial)", topic_success, len(df))
                    logging.info(f"[STEP 1/5] ✅ Topic: {topic_success}/{len(df)} ({topic_success/len(df)*100:.1f}%)")
                
                if generate_sentiment:
                    sentiment_filled = df['New Sentiment'].notna()
                    success_count = sentiment_filled.sum()
                    tracker.add_step_stat("Sentiment", success_count, len(df))
                    logging.info(f"[STEP 1/5] ✅ Sentiment: {success_count}/{len(df)} ({success_count/len(df)*100:.1f}%)")
        
        # ============================================================
        # STEP 2: SMART RETRY FOR EMPTY TOPICS
        # ============================================================
        if generate_topic:
            logging.info("\n" + "="*80)
            logging.info("[STEP 2/5] SMART RETRY FOR EMPTY TOPICS")
            logging.info("="*80)
            
            # Identify rows with empty topics
            empty_topic_mask = (
                df['_is_master'] &
                (df['Topic'].isna() | (df['Topic'].astype(str).str.strip() == ''))
            )
            
            # SOCIAL channels: Check Engagement or Buzz >= threshold
            social_retry_mask = empty_topic_mask & social_mask
            
            if has_engagement:
                social_retry_mask = social_retry_mask & (df['Engagement'] >= RETRY_ENGAGEMENT_THRESHOLD)
                logging.info(f"📊 Social retry based on Engagement >= {RETRY_ENGAGEMENT_THRESHOLD}")
            elif has_buzz:
                social_retry_mask = social_retry_mask & (df['Buzz'] >= RETRY_ENGAGEMENT_THRESHOLD)
                logging.info(f"📊 Social retry based on Buzz >= {RETRY_ENGAGEMENT_THRESHOLD}")
            else:
                social_retry_mask = pd.Series([False] * len(df), index=df.index)
                logging.info(f"📊 Social retry SKIPPED (no Engagement/Buzz column)")
            
            df_social_retry = df[social_retry_mask].copy()
            social_retry_count = len(df_social_retry)
            
            # MAINSTREAM channels: Random sample max 100
            mainstream_empty_mask = empty_topic_mask & mainstream_mask
            df_mainstream_empty = df[mainstream_empty_mask].copy()
            
            if len(df_mainstream_empty) > RETRY_MAINSTREAM_MAX_ROWS:
                df_mainstream_retry = df_mainstream_empty.sample(n=RETRY_MAINSTREAM_MAX_ROWS, random_state=42)
                mainstream_retry_count = RETRY_MAINSTREAM_MAX_ROWS
            else:
                df_mainstream_retry = df_mainstream_empty.copy()
                mainstream_retry_count = len(df_mainstream_retry)
            
            logging.info(f"📊 Retry candidates: Social={social_retry_count}, Mainstream={mainstream_retry_count}")
            
            # Combine and retry
            df_retry = pd.concat([df_social_retry, df_mainstream_retry])
            
            if len(df_retry) > 0:
                retry_batches = []
                total_batches = math.ceil(len(df_retry) / RETRY_BATCH_SIZE)
                
                for batch_num in range(total_batches):
                    start_idx = batch_num * RETRY_BATCH_SIZE
                    end_idx = min(start_idx + RETRY_BATCH_SIZE, len(df_retry))
                    batch_df = df_retry.iloc[start_idx:end_idx].copy()
                    
                    result_batch = retry_topic_batch_smart(
                        batch_df,
                        title_col,
                        content_col,
                        language,
                        tracker,
                        progress
                    )
                    
                    retry_batches.append(result_batch)
                
                df_retried = pd.concat(retry_batches, ignore_index=False)
                
                # Copy back to main df
                for idx in df_retried.index:
                    if 'Topic' in df_retried.columns:
                        topic = df_retried.at[idx, 'Topic']
                        if topic and str(topic).strip():
                            df.at[idx, 'Topic'] = topic
                
                # Copy to duplicates
                logging.info("📋 Copying retried results to duplicate rows...")
                for idx in df_retried.index:
                    hash_val = df.at[idx, '_dedup_hash']
                    duplicate_indices = df[
                        (df['_dedup_hash'] == hash_val) & (~df['_is_master'])
                    ].index
                    
                    for dup_idx in duplicate_indices:
                        if df.at[idx, 'Topic'] and str(df.at[idx, 'Topic']).strip():
                            df.at[dup_idx, 'Topic'] = df.at[idx, 'Topic']
                
                topic_filled = df['Topic'].notna() & (df['Topic'].astype(str).str.strip() != '')
                topic_success = topic_filled.sum()
                
                tracker.add_step_stat("Topic (after retry)", topic_success, len(df))
                logging.info(f"[STEP 2/5] ✅ Topic (after retry): {topic_success}/{len(df)} ({topic_success/len(df)*100:.1f}%)")
            else:
                logging.info("[STEP 2/5] No rows to retry")
        
        # ============================================================
        # STEP 3: TOPIC NORMALIZATION/CLUSTERING PER CAMPAIGN
        # ============================================================
        if generate_topic:
            logging.info("\n" + "="*80)
            logging.info("[STEP 3/5] TOPIC NORMALIZATION/CLUSTERING PER CAMPAIGN")
            logging.info("="*80)
            
            progress(0.50, desc="[STEP 3/5] Clustering topics...")
            
            # Prepare engagement data
            engagement_data = prepare_engagement_data(df, campaign_col='Campaigns')
            
            campaign_groups = {}
            
            for idx, campaign in enumerate(df['Campaigns'].unique(), 1):
                campaign_df = df[df['Campaigns'] == campaign]
                campaign_engagement = engagement_data[engagement_data['Campaigns'] == campaign]
                
                topics = campaign_engagement['Topic'].tolist()
                
                if len(topics) == 0:
                    continue
                
                logging.info(f"📊 Campaign '{campaign}': {len(topics)} topics")
                
                # Cluster topics
                groups = pre_cluster_topics_with_engagement(
                    topics,
                    campaign_engagement,
                    language,
                    threshold=SIMILARITY_THRESHOLD
                )
                
                campaign_groups[campaign] = groups
                
                logging.info(f"  └─ Clustered into {len(groups)} groups")
            
            tracker.add_step_stat("Topic Clustering", len(campaign_groups), len(df['Campaigns'].unique()))
            logging.info(f"[STEP 3/5] ✅ Clustered {len(campaign_groups)} campaigns")
        
        # ============================================================
        # STEP 4: PILLAR GENERATION + NORMALIZATION + MAPPING
        # ============================================================
        if generate_topic:
            logging.info("\n" + "="*80)
            logging.info("[STEP 4/5] PILLAR GENERATION + NORMALIZATION + MAPPING")
            logging.info("="*80)
            
            total_campaigns = len(campaign_groups)
            
            for idx, (campaign, groups) in enumerate(campaign_groups.items(), 1):
                logging.info(f"\n{'='*80}")
                logging.info(f"[CAMPAIGN {idx}/{total_campaigns}] {campaign}")
                logging.info(f"{'='*80}")
                
                progress_val = 0.60 + (idx / total_campaigns) * 0.15
                progress(progress_val, desc=f"[STEP 4/5] Generating Pillars {idx}/{total_campaigns}")
                
                # Step 4a: Generate Pillars from topic groups
                pillar_mapping = generate_pillars_from_topic_groups(
                    campaign,
                    groups,
                    language,
                    tracker,
                    progress
                )
                
                unique_pillars_before = len(set(pillar_mapping.values()))
                logging.info(f"  └─ Generated {unique_pillars_before} unique Pillars")
                
                # Step 4b: Normalize Pillars
                pillars_list = list(pillar_mapping.values())
                pillar_norm_mapping = normalize_pillars_per_campaign(
                    pillars_list,
                    campaign,
                    language,
                    tracker,
                    progress
                )
                
                unique_pillars_after = len(set(pillar_norm_mapping.values()))
                logging.info(f"  └─ After normalization: {unique_pillars_after} unique Pillars")
                
                if unique_pillars_before != unique_pillars_after:
                    logging.info(f"  └─ Merged {unique_pillars_before - unique_pillars_after} Pillars")
                
                # Step 4c: Apply mapping to dataframe
                campaign_mask = df['Campaigns'] == campaign
                
                for idx_row in df[campaign_mask].index:
                    topic_val = df.at[idx_row, 'Topic']
                    
                    if topic_val and str(topic_val).strip() and not is_invalid_value(str(topic_val)):
                        # Get Pillar from topic
                        pillar_raw = pillar_mapping.get(topic_val, '')
                        
                        if pillar_raw:
                            # Normalize Pillar
                            pillar_final = pillar_norm_mapping.get(pillar_raw, pillar_raw)
                            
                            if pillar_final:
                                df.at[idx_row, 'Pillar'] = pillar_final
            
            pillar_filled = df['Pillar'].notna() & (df['Pillar'].astype(str).str.strip() != '')
            pillar_success = pillar_filled.sum()
            unique_pillars_final = df['Pillar'].nunique()
            
            tracker.add_step_stat("Pillar (final)", pillar_success, len(df), unique=unique_pillars_final)
            logging.info(f"[STEP 4/5] ✅ Pillar: {pillar_success}/{len(df)} ({pillar_success/len(df)*100:.1f}%)")
            logging.info(f"[STEP 4/5] ✅ Unique Pillars across all campaigns: {unique_pillars_final}")
        
        # ============================================================
        # STEP 5: SPOKESPERSON EXTRACTION + NORMALIZATION
        # ============================================================
        if generate_spokesperson and mainstream_count > 0:
            logging.info("\n" + "="*80)
            logging.info("[STEP 5/5] SPOKESPERSON (MAINSTREAM ONLY)")
            logging.info("="*80)
            
            mainstream_process_mask = df['_is_master'] & mainstream_mask & df['_eligible_for_topic']
            df_mainstream = df[mainstream_process_mask].copy()
            
            logging.info(f"📊 Processing {len(df_mainstream)} mainstream master rows")
            
            if len(df_mainstream) > 0:
                mainstream_batches = []
                total_batches = math.ceil(len(df_mainstream) / MAINSTREAM_BATCH_SIZE)
                
                for batch_num in range(total_batches):
                    start_idx = batch_num * MAINSTREAM_BATCH_SIZE
                    end_idx = min(start_idx + MAINSTREAM_BATCH_SIZE, len(df_mainstream))
                    batch_df = df_mainstream.iloc[start_idx:end_idx].copy()
                    
                    result_batch = process_batch_spokesperson(
                        batch_df, batch_num + 1, total_batches,
                        title_col, content_col, tracker, progress
                    )
                    
                    mainstream_batches.append(result_batch)
                
                df_mainstream_processed = pd.concat(mainstream_batches, ignore_index=False)
                
                for idx in df_mainstream_processed.index:
                    if 'New Spokesperson' in df_mainstream_processed.columns:
                        df.at[idx, 'New Spokesperson'] = df_mainstream_processed.at[idx, 'New Spokesperson']
                
                # Copy to duplicates
                logging.info("📋 Copying spokesperson to duplicate rows...")
                for hash_val in df[mainstream_mask]['_dedup_hash'].unique():
                    group = df[(df['_dedup_hash'] == hash_val) & mainstream_mask]
                    if len(group) > 1:
                        master_idx = group[group['_is_master']].index[0]
                        duplicate_indices = group[~group['_is_master']].index
                        
                        for dup_idx in duplicate_indices:
                            df.at[dup_idx, 'New Spokesperson'] = df.at[master_idx, 'New Spokesperson']
                
                # Normalize spokesperson
                progress(0.97, desc="[STEP 5/5] Normalizing Spokesperson...")
                
                spokespersons = df['New Spokesperson'].dropna()
                spokespersons = spokespersons[spokespersons.astype(str).str.strip() != '']
                spokespersons = spokespersons[~spokespersons.apply(lambda x: is_invalid_value(str(x)))]
                unique_spokespersons = sorted(spokespersons.unique().tolist())
                
                if unique_spokespersons:
                    spokesperson_mapping = normalize_spokesperson(unique_spokespersons, tracker, progress)
                    df['New Spokesperson'] = df['New Spokesperson'].apply(
                        lambda x: spokesperson_mapping.get(x, x) if pd.notna(x) and str(x).strip() and not is_invalid_value(str(x)) else x
                    )
                
                spokes_filled = df[mainstream_mask]['New Spokesperson'].notna() & \
                               (df[mainstream_mask]['New Spokesperson'].astype(str).str.strip() != '')
                success_count = spokes_filled.sum()
                tracker.add_step_stat("Spokesperson", success_count, mainstream_count)
                
                logging.info(f"[STEP 5/5] ✅ Spokesperson: {success_count}/{mainstream_count} ({success_count/mainstream_count*100:.1f}%)")
            
            df.loc[social_mask, 'New Spokesperson'] = ''
        
        # ============================================================
        # FINALIZATION
        # ============================================================
        logging.info("\n" + "="*80)
        logging.info("[FINALIZATION] Preparing output")
        logging.info("="*80)
        
        df[channel_col] = df['_channel_original']
        
        df = df.drop(['_channel_original', '_channel_lower', '_original_index', '_dedup_hash', 
                      '_is_master', '_word_count', '_eligible_for_topic'], axis=1, errors='ignore')
        
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
            logging.info(f"✅ Row count verified: {original_row_count} → {final_row_count} (unchanged)")
        
        progress(0.98, desc="Saving...")
        
        original_filename = Path(file_path).stem
        output_filename = f"{original_filename}_phase2_v13.xlsx"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name="Processed")
            
            duration = time.time() - start_time
            token_summary = tracker.get_summary(MODEL_NAME)
            
            if generate_topic:
                if 'Topic' in df.columns:
                    topics = df['Topic'].dropna()
                    topics = topics[topics.astype(str).str.strip() != '']
                    avg_topic_words = topics.astype(str).str.split().str.len().mean() if len(topics) > 0 else 0
                else:
                    avg_topic_words = 0
                
                if 'Pillar' in df.columns:
                    pillars = df['Pillar'].dropna()
                    pillars = pillars[pillars.astype(str).str.strip() != '']
                    avg_pillar_words = pillars.astype(str).str.split().str.len().mean() if len(pillars) > 0 else 0
                else:
                    avg_pillar_words = 0
            else:
                avg_topic_words = 0
                avg_pillar_words = 0
            
            if generate_sentiment and 'New Sentiment' in df.columns:
                sentiment_dist = df['New Sentiment'].value_counts(normalize=True) * 100
                sentiment_str = ", ".join([f"{k}({v:.1f}%)" for k, v in sentiment_dist.items()])
            else:
                sentiment_str = "N/A"
            
            toon_rate = (token_summary['toon_success'] / token_summary['api_calls'] * 100) if token_summary['api_calls'] > 0 else 0
            
            if generate_topic:
                unique_pillars_final = df['Pillar'].nunique()
                unique_topics_count = df['Topic'].nunique()
                grouping_rate = (1 - unique_pillars_final / unique_topics_count) * 100 if unique_topics_count > 0 else 0
            else:
                unique_pillars_final = 0
                unique_topics_count = 0
                grouping_rate = 0
            
            meta_data = [
                {"key": "processed_at", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                {"key": "version", "value": "v13.0 - Smart Retry + Topic-to-Pillar Flow"},
                {"key": "model", "value": MODEL_NAME},
                {"key": "output_language", "value": f"{language} ({LANGUAGE_CONFIGS[language]['name']})"},
                {"key": "duration_sec", "value": f"{duration:.2f}"},
                {"key": "original_rows", "value": int(original_row_count)},
                {"key": "final_rows", "value": int(final_row_count)},
                {"key": "row_unchanged", "value": "YES" if original_row_count == final_row_count else "NO"},
                {"key": "deduplication_groups", "value": int(master_rows)},
                {"key": "duplicate_rows", "value": int(duplicate_rows)},
                {"key": "eligible_for_topic", "value": int(total_eligible)},
                {"key": "skipped_short_content", "value": int(total_skipped)},
                {"key": "mainstream_rows", "value": int(mainstream_count)},
                {"key": "social_rows", "value": int(social_count)},
                {"key": "batch_size", "value": int(BATCH_SIZE)},
                {"key": "retry_engagement_threshold", "value": int(RETRY_ENGAGEMENT_THRESHOLD)},
                {"key": "retry_mainstream_max", "value": int(RETRY_MAINSTREAM_MAX_ROWS)},
                {"key": "similarity_threshold", "value": f"{SIMILARITY_THRESHOLD*100}%"},
                {"key": "target_pillars_per_campaign", "value": f"{TARGET_PILLARS_PER_CAMPAIGN} (flexible)"},
                {"key": "topic_engagement_weight", "value": f"{TOPIC_ENGAGEMENT_WEIGHT*100}%"},
                {"key": "input_tokens", "value": int(token_summary["input_tokens"])},
                {"key": "output_tokens", "value": int(token_summary["output_tokens"])},
                {"key": "total_tokens", "value": int(token_summary["total_tokens"])},
                {"key": "api_calls", "value": int(token_summary["api_calls"])},
                {"key": "cost_usd", "value": f"${token_summary['estimated_cost_usd']:.6f}"},
                {"key": "format", "value": "TOON with pipe delimiter"},
                {"key": "toon_success_rate", "value": f"{toon_rate:.1f}%"},
                {"key": "unique_topics", "value": int(unique_topics_count)},
                {"key": "unique_pillars", "value": int(unique_pillars_final)},
                {"key": "grouping_efficiency", "value": f"{grouping_rate:.1f}%"},
            ]
            
            for step_name, step_data in token_summary['step_stats'].items():
                key = f"success_rate_{step_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
                
                if 'unique' in step_data:
                    value = f"{step_data['success']}/{step_data['total']} ({step_data['rate']:.1f}%) | Unique: {step_data['unique']}"
                else:
                    value = f"{step_data['success']}/{step_data['total']} ({step_data['rate']:.1f}%)"
                
                meta_data.append({"key": key, "value": value})
            
            meta_data.extend([
                {"key": "avg_pillar_words", "value": f"{avg_pillar_words:.1f}"},
                {"key": "avg_topic_words", "value": f"{avg_topic_words:.1f}"},
                {"key": "sentiment_distribution", "value": sentiment_str},
            ])
            
            meta = pd.DataFrame(meta_data)
            meta.to_excel(writer, index=False, sheet_name="Meta")
        
        stats = {
            "total_rows": int(len(df)),
            "unchanged": "YES ✅" if original_row_count == final_row_count else f"NO ❌ ({original_row_count} → {final_row_count})",
            "deduplication": {
                "unique_groups": int(master_rows),
                "duplicate_rows": int(duplicate_rows),
                "api_savings": f"{duplicate_rows} calls"
            },
            "pre_filter": {
                "eligible": int(total_eligible),
                "skipped_short": int(total_skipped)
            },
            "channels": {
                "mainstream": int(mainstream_count),
                "social": int(social_count)
            },
            "language": {
                "output": f"{language} ({LANGUAGE_CONFIGS[language]['name']})"
            },
            "normalization": {
                "unique_topics": int(unique_topics_count),
                "unique_pillars": int(unique_pillars_final),
                "grouping_efficiency": f"{grouping_rate:.1f}%",
                "similarity_threshold": f"{SIMILARITY_THRESHOLD*100}%"
            },
            "duration": f"{duration:.2f}s",
            "cost": f"${token_summary['estimated_cost_usd']:.6f}",
            "success_rates": token_summary['step_stats']
        }
        
        logging.info("\n" + "="*80)
        logging.info("✅ PROCESSING COMPLETE - v13.0")
        logging.info("="*80)
        logging.info(f"Rows: {original_row_count} → {final_row_count} (unchanged: {original_row_count == final_row_count})")
        logging.info(f"Duration: {duration:.2f}s | Cost: ${token_summary['estimated_cost_usd']:.6f}")
        
        if generate_topic:
            logging.info(f"Topics: {unique_topics_count} → Pillars: {unique_pillars_final} ({grouping_rate:.1f}% reduction)")
        
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
    with gr.Blocks(title="Insights Generator v13.0", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📊 Insights Generator v13.0 - Smart Retry + Topic-to-Pillar Flow")
        gr.Markdown("**NEW:** Smart retry logic (Engagement/Buzz threshold) + Pillar generated from Topic groups")
        
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
                
                gr.Markdown("### ✅ Features (Select at least 1)")
                gen_topic = gr.Checkbox(label="📌 Topic & Pillar (smart extraction)", value=False)
                gen_sentiment = gr.Checkbox(label="😊 Sentiment", value=False)
                gen_spokesperson = gr.Checkbox(label="🎤 Spokesperson (mainstream only)", value=False)
        
        validation_error = gr.Markdown("", visible=True)
        
        process_btn = gr.Button("🚀 Process", variant="primary", size="lg", interactive=False)
        
        with gr.Row():
            with gr.Column():
                output_file = gr.File(label="📥 Download")
            with gr.Column():
                stats_output = gr.Textbox(label="📊 Stats", lines=18, interactive=False)
        
        error_output = gr.Textbox(label="⚠️ Status", lines=3, visible=True)
        
        def validate_features(topic, sentiment, spokesperson):
            if not any([topic, sentiment, spokesperson]):
                return gr.Button(interactive=False), gr.Markdown("⚠️ **Please select at least one feature to process**", visible=True)
            else:
                return gr.Button(interactive=True), gr.Markdown("", visible=False)
        
        gen_topic.change(
            validate_features,
            inputs=[gen_topic, gen_sentiment, gen_spokesperson],
            outputs=[process_btn, validation_error]
        )
        gen_sentiment.change(
            validate_features,
            inputs=[gen_topic, gen_sentiment, gen_spokesperson],
            outputs=[process_btn, validation_error]
        )
        gen_spokesperson.change(
            validate_features,
            inputs=[gen_topic, gen_sentiment, gen_spokesperson],
            outputs=[process_btn, validation_error]
        )
        
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