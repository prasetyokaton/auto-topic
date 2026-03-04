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

# NEW BATCH SIZES - SMALLER FOR BETTER QUALITY
BATCH_SIZE = 20
MAINSTREAM_BATCH_SIZE = 20
RETRY_BATCH_SIZE = 15
MAX_RETRIES = 1  # Increased from 2 to 3
TRUNCATE_WORDS = 350  # Increased from 100 to 200
MAINSTREAM_TRUNCATE_WORDS = 350  # Increased from 150
RETRY_TRUNCATE_WORDS = 450  # Increased from 200

MIN_CONTENT_WORDS_FOR_TOPIC = 3

# NEW: Separate engagement weights for Pillar and Topic
PILLAR_ENGAGEMENT_WEIGHT = 0.8  # High priority for strategic categorization
TOPIC_ENGAGEMENT_WEIGHT = 0.6   # Medium-high for variety with priority
SIMILARITY_THRESHOLD = 0.40
TARGET_PILLARS_PER_CAMPAIGN = 15  # Target pillars (was topics)
TARGET_TOPICS_PER_PILLAR = 3  # Average topics per pillar
#SKIP_RETRY_THRESHOLD = 0.90  # More aggressive retry (was 0.95)

MAINSTREAM_CHANNELS = [
    'tv', 'radio', 'newspaper', 'online', 'printmedia', 'site',
    'printed', 'printedmedia', 'print', 'online media'
]

SOCIAL_CHANNELS = ['tiktok', 'instagram', 'youtube', 'facebook', 'twitter', 'x', 'blog', 'forum']
INVALID_VALUES = ['nan', 'none', 'null', 'n/a', 'na', 'unknown', 'tidak ada', '-', 'tidak diketahui', 'undefined', 'not available', 'tidak jelas']

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
    """NEW: Topic validation with 5-15 words"""
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
    """NEW: Pillar validation with 2-6 words"""
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

def extract_keywords_fallback(content: str, max_words: int = 5, output_language: str = "English") -> str:
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
        return placeholder
    
    stopwords = set(LANGUAGE_CONFIGS.get(output_language, {}).get('stopwords', []))
    
    content = re.sub(r'http\S+|www\.\S+|#\w+', '', content)
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
    """NEW v12: Enhanced prompt for Pillar + Topic extraction"""
    
    batch_size = len(batch_df)
    lang_config = LANGUAGE_CONFIGS[language]
    
    if 'Topic' not in batch_df.columns:
        batch_df['Topic'] = ''
    if 'Pillar' not in batch_df.columns:
        batch_df['Pillar'] = ''
    if 'New Sentiment' not in batch_df.columns:
        batch_df['New Sentiment'] = 'neutral'
    if 'New Sentiment Level' not in batch_df.columns:
        batch_df['New Sentiment Level'] = 0
    
    input_toon = build_toon_input(batch_df, title_col, content_col, batch_size, truncate_words=TRUNCATE_WORDS, clean_content=True)
    
    nonce = random.randint(100000, 999999)
    
    prompt = f"""You are an ELITE insights analyst with CRITICAL THINKING skills.

[Request ID: {nonce}]
[OUTPUT LANGUAGE: {language} - OPTIMIZED for Bahasa Indonesia]

INPUT (TOON format, content may be in ANY language):
{input_toon}

🎯 YOUR MISSION: Extract DEEP INSIGHTS with CRITICAL ANALYSIS

You MUST analyze content with FORENSIC DETAIL and capture:
1. **WHO**: Named entities (people, organizations, places)
   Examples: IMIP, Morowali, WN RRC, Satgas, specific names
   
2. **WHAT**: Specific actions/events (NOT generic!)
   ✅ GOOD: "Penutupan IMIP", "Perkelahian pekerja", "Pelanggaran kapal ilegal"
   ❌ BAD: "Berita", "Informasi", "Update"
   
3. **INTENT**: What is this content about?
   - Pertanyaan (asking question)
   - Keluhan (complaint)
   - Resiko/Dampak (risk/impact)
   - Pelanggaran (violation)
   - Informasi (information)
   
4. **CONTEXT**: Specific details that matter
   Examples: "85rb+ lapangan kerja hilang", "US$15B+ ekspor anjlok"

OUTPUT STRUCTURE:

**PILLAR** ({lang_config['pillar_word_count']}):
- Strategic CATEGORIZATION
- Capture main issue/theme
- ✅ Examples: "Resiko IMIP Tutup", "Perkelahian Pekerja", "Pelanggaran Hukum"
- ❌ NEVER: "Berita", "Info", "Update", "Viral"

**TOPIC** ({lang_config['topic_word_count']}):
- DETAILED and SPECIFIC description
- Include WHO + WHAT + WHERE when relevant
- Preserve important entities (IMIP, Morowali, WN RRC, etc.)
- ✅ Examples: "Pertanyaan tentang isu IMIP di Morowali", "Perkelahian pekerja di kawasan IMIP", "Pelanggaran hukum kapal ilegal dari WN RRC"
- ❌ NEVER: Generic terms without details

**SENTIMENT**:
- positive: Clear positive emotion, praise, satisfaction
- negative: Clear negative emotion, complaint, criticism, concern
- neutral: Factual, informational, question without emotion

**CONFIDENCE** (0-100):
- How certain are you about pillar, topic, and sentiment?
- Be honest about ambiguity

CRITICAL RULES:
- Content can be in ANY language → You MUST understand it → Output in {language}
- NEVER use noise words: berita, info, informasi, viral, trending, update, artikel
- NEVER output "unknown", "tidak jelas", "nan"
- If you cannot extract meaningful insights, use "-" for that field
- CAPTURE entity names exactly (IMIP, Morowali, etc.)
- BE SPECIFIC - avoid generic categorization
- Topic MUST be 5-15 words with clear details
- Pillar MUST be 2-6 words categorization

OUTPUT FORMAT (TOON with pipe delimiter |):
result[{batch_size}]{{row|pillar|topic|sentiment|confidence}}:
<row_index>|<pillar in {language} or ->|<topic in {language} or ->|<sentiment>|<confidence_0-100>

YOUR OUTPUT (TOON format only):"""
    
    try:
        progress(0.5, desc=f"Pillar+Topic+Sentiment {batch_num}/{total_batches}")
        
        response = chat_create(
            MODEL_NAME,
            [
                {"role": "system", "content": f"You are an ELITE insights analyst with critical thinking. {lang_config['prompt_instruction']}. Handle multi-language input. Output in {language}. NEVER use generic terms."},
                {"role": "user", "content": prompt}
            ],
            token_tracker=token_tracker
        )
        
        if not response:
            raise Exception("API call failed")
        
        raw = response.choices[0].message.content.strip()
        data, format_type = parse_gpt_response(raw, ['row', 'pillar', 'topic', 'sentiment', 'confidence'], batch_size, token_tracker)
        
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
                
                # Process PILLAR (new)
                current_pillar = batch_df.at[idx, 'Pillar']
                # Update PILLAR if still empty
                if pd.isna(batch_df.at[idx, 'Pillar']) or str(batch_df.at[idx, 'Pillar']).strip() == '':
                    pillar = str(item.get('pillar', '')).strip()

                    if pillar != '-' and not is_invalid_value(pillar):
                        pillar = validate_and_normalize_pillar(pillar, language)
                        if pillar:
                            batch_df.at[idx, 'Pillar'] = pillar

                
                # Process TOPIC (enhanced from old sub_topic)
                current_topic = batch_df.at[idx, 'Topic']
                if pd.isna(current_topic) or str(current_topic).strip() == '':
                    topic = str(item.get('topic', '')).strip()
                    
                    if topic == '-' or is_invalid_value(topic):
                        topic = ''
                    
                    if topic:
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
        logging.error(f"Error in combined processing: {e}")
        return batch_df

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
    
    input_toon = build_toon_input(batch_df, title_col, content_col, batch_size, truncate_words=MAINSTREAM_TRUNCATE_WORDS)
    
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
                
                if spokesperson_val == '-' or is_invalid_value(spokesperson_val):
                    spokesperson_val = ''
                
                if spokesperson_val:
                    batch_df.at[idx, 'New Spokesperson'] = spokesperson_val
        
        return batch_df
        
    except Exception as e:
        return batch_df

def retry_topic_batch(
    batch_df: pd.DataFrame,
    retry_attempt: int,
    title_col: str,
    content_col: str,
    language: str,
    token_tracker: TokenTracker,
    progress=gr.Progress()
) -> pd.DataFrame:
    """NEW v12: Enhanced retry with increasing aggression per attempt"""
    
    batch_size = len(batch_df)
    lang_config = LANGUAGE_CONFIGS[language]
    
    # Increase truncate words per attempt
    truncate_words = RETRY_TRUNCATE_WORDS + (retry_attempt * 50)
    
    input_toon = build_toon_input(
        batch_df, 
        title_col, 
        content_col, 
        batch_size,
        truncate_words=truncate_words,
        clean_content=True
    )
    
    nonce = random.randint(100000, 999999)
    
    # More aggressive prompt per attempt
    aggression_level = ["FINAL WARNING"][0] 
    
    prompt = f"""🚨🚨🚨 {aggression_level} - RETRY ATTEMPT {retry_attempt}/{MAX_RETRIES} 🚨🚨🚨

[Request ID: {nonce}]
[OUTPUT LANGUAGE: {language}]
[MORE CONTEXT PROVIDED: {truncate_words} words]

INPUT (TOON format, content may be ANY language):
{input_toon}

⚡ THIS IS YOUR RETRY #{retry_attempt} - YOU MUST SUCCEED! ⚡

PREVIOUS ATTEMPT FAILED - This time you MUST:
1. Read FULL context carefully ({truncate_words} words provided)
2. Understand content in ANY language (Thai, English, Chinese, Indonesian, mixed)
3. Extract SPECIFIC details - WHO, WHAT, WHERE
4. NEVER use generic terms
5. Output detailed insights in {language}

MANDATORY EXTRACTION RULES:

**PILLAR** ({lang_config['pillar_word_count']}):
- Strategic categorization
- Capture MAIN issue/theme
- Examples: "Resiko IMIP Tutup", "Perkelahian Pekerja", "Pelanggaran Hukum"

**TOPIC** ({lang_config['topic_word_count']}):
- DETAILED with specific entities
- Include WHO + WHAT + WHERE
- Examples: "Pertanyaan tentang isu IMIP di Morowali", "Perkelahian pekerja di kawasan IMIP"

CRITICAL:
- Content language can be ANYTHING → You MUST understand → Output in {language}
- NEVER: "berita", "info", "viral", "trending", "unknown", "tidak jelas"
- If truly impossible, use "-" but TRY EVERYTHING FIRST
- Extract KEYWORDS if unclear
- Be SPECIFIC not generic

OUTPUT (TOON format):
result[{batch_size}]{{row|pillar|topic}}:
<row_index>|<pillar in {language} or ->|<topic in {language} or ->

YOUR OUTPUT:"""
    
    try:
        progress(0.95, desc=f"Retry #{retry_attempt}/3 - Extracting Topics...")
        
        response = chat_create(
            MODEL_NAME,
            [
                {"role": "system", "content": f"CRITICAL RETRY ATTEMPT {retry_attempt}. {lang_config['prompt_instruction']}. Multi-language expert. NEVER generic terms. BE SPECIFIC."},
                {"role": "user", "content": prompt}
            ],
            token_tracker=token_tracker
        )
        
        if not response:
            raise Exception("API call failed")
        
        raw = response.choices[0].message.content.strip()
        data, format_type = parse_gpt_response(raw, ['row', 'pillar', 'topic'], batch_size, token_tracker)
        
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
                
                
                # Update TOPIC if still empty
                if pd.isna(batch_df.at[idx, 'Topic']) or str(batch_df.at[idx, 'Topic']).strip() == '':
                    topic = str(item.get('topic', '')).strip()
                    
                    if topic != '-' and not is_invalid_value(topic):
                        topic = validate_and_normalize_topic(topic, language)
                        if topic:
                            batch_df.at[idx, 'Topic'] = topic

                # Update PILLAR if still empty
                if pd.isna(batch_df.at[idx, 'Pillar']) or str(batch_df.at[idx, 'Pillar']).strip() == '':
                    pillar = str(item.get('pillar', '')).strip()

                    if pillar != '-' and not is_invalid_value(pillar):
                        pillar = validate_and_normalize_pillar(pillar, language)
                        if pillar:
                            batch_df.at[idx, 'Pillar'] = pillar

    
                
        
        return batch_df
        
    except Exception as e:
        logging.error(f"Retry attempt {retry_attempt} failed: {e}")
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
        
        # Use TOPIC_ENGAGEMENT_WEIGHT (0.6)
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
    """Cluster topics by similarity with engagement priority"""
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

def consolidate_topics_to_pillars(groups: dict, 
                                  language: str,
                                  token_tracker: TokenTracker,
                                  target_pillars: int = TARGET_PILLARS_PER_CAMPAIGN) -> dict:
    """NEW v12: Consolidate topic groups into Pillars"""
    
    group_summary = []
    
    for group_id, group_data in groups.items():
        topics = group_data['topics']
        engagement_scores = group_data['engagement_scores']
        
        sorted_pairs = sorted(
            zip(topics, engagement_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_samples = [t for t, _ in sorted_pairs[:5]]
        top_engagement = sum([eng for _, eng in sorted_pairs[:5]])
        
        group_summary.append({
            "group_id": group_id,
            "count": len(topics),
            "total_engagement": group_data['total_engagement'],
            "avg_engagement": group_data['avg_engagement'],
            "top_samples": top_samples,
            "sample_engagement": top_engagement
        })
    
    group_summary.sort(key=lambda x: x['total_engagement'], reverse=True)
    
    nonce = random.randint(100000, 999999)
    lang_config = LANGUAGE_CONFIGS[language]
    
    prompt = f"""You are a strategic categorization expert with ENGAGEMENT PRIORITY.

[Request ID: {nonce}]

INPUT: {len(groups)} topic groups (SORTED BY ENGAGEMENT - highest first!)

Groups (in order of importance by engagement):
{json.dumps(group_summary, indent=2, ensure_ascii=False)}

TASK:
1. Analyze groups and CREATE strategic PILLARS
2. Create approximately {target_pillars} PILLARS (12-18 range)
3. Each PILLAR should be {lang_config['pillar_word_count']} in {language}
4. Pillars are CATEGORIZATION level - broad but meaningful

CRITICAL ENGAGEMENT RULES:
⭐ HIGH-ENGAGEMENT groups at TOP are MOST IMPORTANT
⭐ Pillar names should reflect HIGH-ENGAGEMENT content
⭐ When merging groups, prioritize naming from high-engagement samples
⭐ Ensure high-engagement topics get meaningful pillar categorization

GOOD PILLARS:
✅ "Resiko IMIP Tutup" (specific issue category)
✅ "Perkelahian Pekerja" (specific event category)
✅ "Pelanggaran Hukum" (specific problem category)
✅ "Pertanyaan IMIP" (specific intent category)

BAD PILLARS:
❌ "Berita" (too generic!)
❌ "Informasi" (noise word!)
❌ "Update" (too vague!)
❌ "Viral" (not meaningful!)

OUTPUT FORMAT (JSON):
{{
  "pillars": [
    {{
      "pillar_name": "...",
      "merged_groups": ["group_0", "group_3"],
      "estimated_engagement": 75000,
      "description": "Why this categorization (based on high-engagement content)"
    }},
    ...
  ],
  "total_pillars": {target_pillars}
}}

OUTPUT (JSON only):"""
    
    try:
        response = chat_create(
            MODEL_NAME,
            [{"role": "user", "content": prompt}],
            token_tracker=token_tracker
        )
        
        if not response:
            logging.warning("Pillar consolidation API call failed")
            return {"pillars": [], "total_pillars": 0}
        
        raw = response.choices[0].message.content.strip()
        result = extract_json_from_response(raw)
        
        if not result or 'pillars' not in result:
            logging.warning("Invalid pillar consolidation result")
            return {"pillars": [], "total_pillars": 0}
        
        for pillar_info in result['pillars']:
            if 'pillar_name' in pillar_info:
                original = pillar_info['pillar_name']
                normalized = validate_and_normalize_pillar(original, language)
                if normalized:
                    pillar_info['pillar_name'] = normalized
                else:
                    logging.warning(f"Pillar normalization failed for: {original}")
        
        return result
        
    except Exception as e:
        logging.error(f"Pillar consolidation error: {e}")
        return {"pillars": [], "total_pillars": 0}

def validate_engagement_coverage(df, mapping: dict, 
                                campaign: str,
                                engagement_col='Engagement'):
    """Validate pillar engagement coverage"""
    campaign_df = df[df['Campaigns'] == campaign].copy()
    
    campaign_df['Pillar_Mapped'] = campaign_df['Topic'].map(mapping)
    
    if engagement_col not in campaign_df.columns:
        logging.warning("Engagement column not found for validation")
        return pd.DataFrame()
    
    pillar_engagement = campaign_df.groupby('Pillar_Mapped').agg({
        engagement_col: 'sum',
        'Topic': 'count'
    }).reset_index()
    
    pillar_engagement.columns = ['Pillar', 'Total_Engagement', 'Topic_Count']
    pillar_engagement = pillar_engagement.sort_values('Total_Engagement', ascending=False)
    
    total_engagement = campaign_df[engagement_col].sum()
    
    if total_engagement > 0:
        pillar_engagement['Engagement_Share'] = (
            pillar_engagement['Total_Engagement'] / total_engagement * 100
        )
        pillar_engagement['Cumulative_Share'] = pillar_engagement['Engagement_Share'].cumsum()
    else:
        pillar_engagement['Engagement_Share'] = 0
        pillar_engagement['Cumulative_Share'] = 0
    
    logging.info(f"\n{'='*80}")
    logging.info(f"[ENGAGEMENT COVERAGE] Campaign: '{campaign}'")
    logging.info(f"{'='*80}")
    logging.info(f"Total Engagement: {total_engagement:,.0f}")
    logging.info(f"\nTop 10 Pillars by Engagement:")
    
    for idx, (_, row) in enumerate(pillar_engagement.head(10).iterrows(), 1):
        logging.info(
            f"  {idx}. {str(row['Pillar']):<40} | "
            f"Engagement: {row['Total_Engagement']:>10,.0f} ({row['Engagement_Share']:>5.1f}%) | "
            f"Topics: {row['Topic_Count']:>3}"
        )
    
    if total_engagement > 0:
        top_80_pillars = pillar_engagement[pillar_engagement['Cumulative_Share'] <= 80]
        logging.info(f"\n📊 Top {len(top_80_pillars)} pillars cover 80% of engagement")
    
    return pillar_engagement

def normalize_topics_to_pillars_v12(df, 
                                    campaign_col='Campaigns',
                                    engagement_col='Engagement',
                                    language='Indonesia',
                                    target_pillars=TARGET_PILLARS_PER_CAMPAIGN,
                                    similarity_threshold=SIMILARITY_THRESHOLD,
                                    token_tracker=None,
                                    progress=gr.Progress()):
    """NEW v12: Normalize Topics → Pillars with engagement awareness"""
    
    logging.info("[PREP] Calculating engagement weights for Topics → Pillars...")
    engagement_data = prepare_engagement_data(df, campaign_col)
    
    results = {}
    total_campaigns = df[campaign_col].nunique()
    
    for idx, campaign in enumerate(df[campaign_col].unique(), 1):
        logging.info(f"\n{'='*80}")
        logging.info(f"[NORMALIZATION {idx}/{total_campaigns}] Campaign: '{campaign}'")
        logging.info(f"{'='*80}")
        
        campaign_df = df[df[campaign_col] == campaign]
        campaign_engagement = engagement_data[engagement_data[campaign_col] == campaign]
        
        topics = campaign_engagement['Topic'].tolist()
        
        if engagement_col in campaign_df.columns:
            total_engagement = campaign_df[engagement_col].sum()
        else:
            total_engagement = 0
        
        logging.info(f"  └─ Topics: {len(topics)}")
        if total_engagement > 0:
            logging.info(f"  └─ Total Engagement: {total_engagement:,.0f}")
        
        progress_val = 0.85 + (idx / total_campaigns) * 0.10
        progress(progress_val, desc=f"[STEP 4/4] Normalizing campaign {idx}/{total_campaigns}")
        
        # Pre-cluster topics
        groups = pre_cluster_topics_with_engagement(
            topics,
            campaign_engagement,
            language,
            threshold=similarity_threshold
        )
        
        logging.info(f"  └─ Pre-clustering: {len(topics)} topics → {len(groups)} groups")
        logging.info(f"     └─ Groups sorted by engagement (highest first)")
        
        # Consolidate groups to pillars
        pillar_result = consolidate_topics_to_pillars(
            groups,
            language,
            token_tracker,
            target_pillars=target_pillars
        )
        
        if not pillar_result or 'pillars' not in pillar_result or len(pillar_result['pillars']) == 0:
            logging.warning(f"  └─ Pillar consolidation failed for campaign '{campaign}', using fallback")
            
            mapping = {}
            for group_id, group_data in groups.items():
                first_topic = group_data['topics'][0]
                pillar_name = validate_and_normalize_pillar(first_topic, language)
                
                if not pillar_name:
                    pillar_name = ""
                
                for t in group_data['topics']:
                    mapping[t] = pillar_name
            
            final_pillars = len([p for p in set(mapping.values()) if p])
        else:
            final_pillars = len(pillar_result['pillars'])
            
            logging.info(f"  └─ Consolidation: {len(groups)} groups → {final_pillars} pillars")
            
            mapping = {}
            group_to_pillar = {}
            
            for pillar_info in pillar_result['pillars']:
                pillar_name = pillar_info['pillar_name']
                for group_id in pillar_info['merged_groups']:
                    group_to_pillar[group_id] = pillar_name
            
            for group_id, group_data in groups.items():
                pillar = group_to_pillar.get(group_id, "")
                for t in group_data['topics']:
                    mapping[t] = pillar
        
        if engagement_col in df.columns:
            pillar_engagement = validate_engagement_coverage(
                df, mapping, campaign, engagement_col
            )
        else:
            pillar_engagement = pd.DataFrame()
        
        reduction_rate = (1 - final_pillars/len(topics)) * 100 if len(topics) > 0 else 0
        
        results[campaign] = {
            'mapping': mapping,
            'pillars': pillar_result.get('pillars', []),
            'pillar_engagement': pillar_engagement,
            'stats': {
                'original_topics': len(topics),
                'final_pillars': final_pillars,
                'total_engagement': total_engagement,
                'reduction_rate': reduction_rate
            }
        }
        
        logging.info(f"  └─ Reduction: {reduction_rate:.1f}%")
    
    return results

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
        
        if 'Engagement' not in df.columns:
            logging.warning("⚠️ 'Engagement' column not found, normalization will use frequency only")
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
        logging.info(f"💰 API savings from deduplication: {duplicate_rows} calls")
        
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
        logging.info(f"💰 API savings from pre-filter: {total_skipped} topic extractions skipped")
        
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
        
        if generate_topic or generate_sentiment:
            logging.info("\n" + "="*80)
            logging.info("[STEP 1/4] PILLAR + TOPIC + SENTIMENT (MASTER ROWS, ELIGIBLE CONTENT)")
            logging.info("="*80)
            
            process_mask = df['_is_master'] & df['_eligible_for_topic']
            
            if generate_topic and 'Pillar' in df.columns and 'Topic' in df.columns:
                has_both = (df['Pillar'].notna() & (df['Pillar'].astype(str).str.strip() != '')) & \
                          (df['Topic'].notna() & (df['Topic'].astype(str).str.strip() != ''))
                process_mask = process_mask & ~has_both
            
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
                    
                    progress_val = 0.1 + (batch_num / total_batches) * 0.30
                    progress(progress_val, desc=f"[STEP 1/4] Processing {batch_num + 1}/{total_batches}")
                    
                    result_batch = process_batch_combined(
                        batch_df, batch_num + 1, total_batches,
                        title_col, content_col, language, conf_threshold, tracker, progress
                    )
                    
                    all_batches.append(result_batch)
                
                df_processed = pd.concat(all_batches, ignore_index=False)
                
                for idx in df_processed.index:
                    if generate_topic:
                        if 'Pillar' in df_processed.columns:
                            df.at[idx, 'Pillar'] = df_processed.at[idx, 'Pillar']
                        if 'Topic' in df_processed.columns:
                            df.at[idx, 'Topic'] = df_processed.at[idx, 'Topic']
                    if generate_sentiment:
                        df.at[idx, 'New Sentiment'] = df_processed.at[idx, 'New Sentiment']
                        df.at[idx, 'New Sentiment Level'] = df_processed.at[idx, 'New Sentiment Level']
                
                logging.info("📋 Copying results to duplicate rows...")
                for hash_val in df['_dedup_hash'].unique():
                    group = df[df['_dedup_hash'] == hash_val]
                    if len(group) > 1:
                        master_idx = group[group['_is_master']].index[0]
                        duplicate_indices = group[~group['_is_master']].index
                        
                        for dup_idx in duplicate_indices:
                            if generate_topic:
                                if df.at[master_idx, 'Pillar'] and str(df.at[master_idx, 'Pillar']).strip():
                                    df.at[dup_idx, 'Pillar'] = df.at[master_idx, 'Pillar']
                                if df.at[master_idx, 'Topic'] and str(df.at[master_idx, 'Topic']).strip():
                                    df.at[dup_idx, 'Topic'] = df.at[master_idx, 'Topic']

                            if generate_sentiment:
                                df.at[dup_idx, 'New Sentiment'] = df.at[master_idx, 'New Sentiment']
                                df.at[dup_idx, 'New Sentiment Level'] = df.at[master_idx, 'New Sentiment Level']
                
                if generate_topic:
                    pillar_filled = df['Pillar'].notna() & (df['Pillar'].astype(str).str.strip() != '')
                    topic_filled = df['Topic'].notna() & (df['Topic'].astype(str).str.strip() != '')
                    
                    pillar_success = pillar_filled.sum()
                    topic_success = topic_filled.sum()
                    
                    tracker.add_step_stat("Pillar (initial)", pillar_success, len(df))
                    tracker.add_step_stat("Topic (initial)", topic_success, len(df))
                    
                    logging.info(f"[STEP 1/4] ✅ Pillar: {pillar_success}/{len(df)} ({pillar_success/len(df)*100:.1f}%)")
                    logging.info(f"[STEP 1/4] ✅ Topic: {topic_success}/{len(df)} ({topic_success/len(df)*100:.1f}%)")
                
                if generate_sentiment:
                    sentiment_filled = df['New Sentiment'].notna()
                    success_count = sentiment_filled.sum()
                    tracker.add_step_stat("Sentiment", success_count, len(df))
                    logging.info(f"[STEP 1/4] ✅ Sentiment: {success_count}/{len(df)} ({success_count/len(df)*100:.1f}%)")
        
        if generate_spokesperson and mainstream_count > 0:
            logging.info("\n" + "="*80)
            logging.info("[STEP 2/4] SPOKESPERSON (MAINSTREAM MASTER ROWS, ELIGIBLE CONTENT)")
            logging.info("="*80)
            
            mainstream_process_mask = df['_is_master'] & mainstream_mask & df['_eligible_for_topic']
            df_mainstream = df[mainstream_process_mask].copy()
            
            logging.info(f"📊 Processing {len(df_mainstream)} mainstream master rows (eligible content only)")
            
            if len(df_mainstream) > 0:
                mainstream_batches = []
                total_batches = math.ceil(len(df_mainstream) / MAINSTREAM_BATCH_SIZE)
                
                for batch_num in range(total_batches):
                    start_idx = batch_num * MAINSTREAM_BATCH_SIZE
                    end_idx = min(start_idx + MAINSTREAM_BATCH_SIZE, len(df_mainstream))
                    batch_df = df_mainstream.iloc[start_idx:end_idx].copy()
                    
                    progress_val = 0.45 + (batch_num / total_batches) * 0.15
                    progress(progress_val, desc=f"[STEP 2/4] Spokesperson {batch_num + 1}/{total_batches}")
                    
                    result_batch = process_batch_spokesperson(
                        batch_df, batch_num + 1, total_batches,
                        title_col, content_col, tracker, progress
                    )
                    
                    mainstream_batches.append(result_batch)
                
                df_mainstream_processed = pd.concat(mainstream_batches, ignore_index=False)
                
                for idx in df_mainstream_processed.index:
                    if 'New Spokesperson' in df_mainstream_processed.columns:
                        df.at[idx, 'New Spokesperson'] = df_mainstream_processed.at[idx, 'New Spokesperson']
                
                logging.info("📋 Copying spokesperson to duplicate rows...")
                for hash_val in df[mainstream_mask]['_dedup_hash'].unique():
                    group = df[(df['_dedup_hash'] == hash_val) & mainstream_mask]
                    if len(group) > 1:
                        master_idx = group[group['_is_master']].index[0]
                        duplicate_indices = group[~group['_is_master']].index
                        
                        for dup_idx in duplicate_indices:
                            df.at[dup_idx, 'New Spokesperson'] = df.at[master_idx, 'New Spokesperson']
                
                spokes_filled = df[mainstream_mask]['New Spokesperson'].notna() & \
                               (df[mainstream_mask]['New Spokesperson'].astype(str).str.strip() != '')
                success_count = spokes_filled.sum()
                tracker.add_step_stat("Spokesperson", success_count, mainstream_count)
                
                logging.info(f"[STEP 2/4] ✅ Spokesperson: {success_count}/{mainstream_count} ({success_count/mainstream_count*100:.1f}%)")
            
            df.loc[social_mask, 'New Spokesperson'] = ''
        
        # ============================
        # [STEP 3/4] EXHAUSTIVE RETRY FOR EMPTY TOPICS (ALWAYS 3x)
        # ============================

        if generate_topic:
            logging.info("\n" + "="*80)
            logging.info("[STEP 3/4] EXHAUSTIVE RETRY FOR EMPTY TOPIC/PILLAR (MAX 3x)")

            logging.info("="*80)

            for retry_attempt in range(1, MAX_RETRIES + 1):

                # Ambil HANYA master rows yang Topic masih kosong
                empty_mask = (
                    df['_is_master'] &
                    (
                        # Topic kosong
                        df['Topic'].isna() |
                        (df['Topic'].astype(str).str.strip() == '') |
                        (df['Topic'].apply(lambda x: is_invalid_value(str(x))))
                        |
                        # ATAU Pillar kosong
                        df['Pillar'].isna() |
                        (df['Pillar'].astype(str).str.strip() == '') |
                        (df['Pillar'].apply(lambda x: is_invalid_value(str(x))))
                    )
                )


                df_empty = df[empty_mask].copy()
                empty_count = len(df_empty)

                logging.info(f"[STEP 3/4] Retry #{retry_attempt}: {empty_count} empty topic/pillar rows")


                if empty_count == 0:
                    logging.info("[STEP 3/4] ✅ All topics filled, stopping retry early")
                    break

                retry_batches = []
                total_batches = math.ceil(empty_count / RETRY_BATCH_SIZE)

                for batch_num in range(total_batches):
                    start_idx = batch_num * RETRY_BATCH_SIZE
                    end_idx = min(start_idx + RETRY_BATCH_SIZE, empty_count)

                    batch_df = df_empty.iloc[start_idx:end_idx].copy()

                    result_batch = retry_topic_batch(
                        batch_df,
                        retry_attempt,
                        title_col,
                        content_col,
                        language,
                        tracker,
                        progress
                    )

                    retry_batches.append(result_batch)

                df_retried = pd.concat(retry_batches, ignore_index=False)

                # Copy hasil retry ke df utama
                for idx in df_retried.index:
                    if 'Pillar' in df_retried.columns:
                        retried_pillar = df_retried.at[idx, 'Pillar']
                        if retried_pillar and str(retried_pillar).strip():
                            df.at[idx, 'Pillar'] = retried_pillar
                    if 'Topic' in df_retried.columns:
                        df.at[idx, 'Topic'] = df_retried.at[idx, 'Topic']

                # Copy ke duplicate rows
                logging.info("📋 Copying retried results to duplicate rows...")
                for idx in df_retried.index:
                    hash_val = df.at[idx, '_dedup_hash']
                    duplicate_indices = df[
                        (df['_dedup_hash'] == hash_val) & (~df['_is_master'])
                    ].index

                    for dup_idx in duplicate_indices:
                        if df.at[idx, 'Pillar'] and str(df.at[idx, 'Pillar']).strip():
                            df.at[dup_idx, 'Pillar'] = df.at[idx, 'Pillar']
                        if df.at[idx, 'Topic'] and str(df.at[idx, 'Topic']).strip():
                            df.at[dup_idx, 'Topic'] = df.at[idx, 'Topic']


        
        logging.info("\n" + "="*80)
        logging.info("[STEP 4/4] NORMALIZATION (TOPICS → PILLARS PER CAMPAIGN)")
        logging.info("="*80)
        
        if generate_topic:
            progress(0.85, desc="[STEP 4/4] Normalizing Topics → Pillars (per campaign)...")
            
            if 'Campaigns' not in df.columns:
                logging.warning("⚠️ 'Campaigns' column not found, skipping normalization")
            else:
                results = normalize_topics_to_pillars_v12(
                    df,
                    campaign_col='Campaigns',
                    engagement_col='Engagement',
                    language=language,
                    target_pillars=TARGET_PILLARS_PER_CAMPAIGN,
                    similarity_threshold=SIMILARITY_THRESHOLD,
                    token_tracker=tracker,
                    progress=progress
                )
                
                # Apply pillar mapping to original Pillar column
                for campaign, result in results.items():
                    mapping = result['mapping']
                    campaign_mask = df['Campaigns'] == campaign
                    
                    for idx in df[campaign_mask].index:
                        topic_val = df.at[idx, 'Topic']
                        if topic_val and str(topic_val).strip() and not is_invalid_value(str(topic_val)):
                            mapped_pillar = mapping.get(topic_val, '')
                            if mapped_pillar:
                                df.at[idx, 'Pillar'] = mapped_pillar
                
                pillar_filled = df['Pillar'].notna() & (df['Pillar'].astype(str).str.strip() != '')
                pillar_success = pillar_filled.sum()
                tracker.add_step_stat("Pillar (normalized)", pillar_success, len(df))
                
                topics_all = df['Topic'].dropna()
                topics_all = topics_all[topics_all.astype(str).str.strip() != '']
                topics_all = topics_all[~topics_all.apply(lambda x: is_invalid_value(str(x)))]
                unique_topics_count = len(topics_all.unique())
                unique_pillars_final = df['Pillar'].nunique()
                grouping_rate = (1 - unique_pillars_final / unique_topics_count) * 100 if unique_topics_count > 0 else 0
                
                logging.info(f"[STEP 4/4] ✅ Pillar (normalized): {pillar_success}/{len(df)} ({pillar_success/len(df)*100:.1f}%)")
                logging.info(f"[STEP 4/4] 📊 Grouping: {unique_topics_count} topics → {unique_pillars_final} pillars ({grouping_rate:.1f}% reduction)")
        

        logging.info("\n" + "="*80)
        logging.info("[STEP 5/5] SYNCHRONIZE TOPIC & PILLAR (FINAL PASS)")
        logging.info("="*80)

        if generate_topic:
            # Phase A1: Topic kosong, Pillar ada → Topic = Pillar
            mask_topic_empty = (
                (df['Topic'].isna() | (df['Topic'].astype(str).str.strip() == '') | df['Topic'].apply(lambda x: is_invalid_value(str(x)))) &
                (df['Pillar'].notna() & (df['Pillar'].astype(str).str.strip() != '') & ~df['Pillar'].apply(lambda x: is_invalid_value(str(x))))
            )

            topic_filled_from_pillar = mask_topic_empty.sum()
            df.loc[mask_topic_empty, 'Topic'] = df.loc[mask_topic_empty, 'Pillar'].apply(
                lambda x: validate_and_normalize_topic(x, language)
            )


            # Phase A2: Pillar kosong, Topic ada → Pillar = Topic
            mask_pillar_empty = (
                (df['Pillar'].isna() | (df['Pillar'].astype(str).str.strip() == '') | df['Pillar'].apply(lambda x: is_invalid_value(str(x)))) &
                (df['Topic'].notna() & (df['Topic'].astype(str).str.strip() != '') & ~df['Topic'].apply(lambda x: is_invalid_value(str(x))))
            )

            pillar_filled_from_topic = mask_pillar_empty.sum()
            df.loc[mask_pillar_empty, 'Pillar'] = df.loc[mask_pillar_empty, 'Topic'].apply(
                lambda x: validate_and_normalize_pillar(x, language)
            )


            logging.info(
                f"[STEP 5/5] Synchronize filled Topic←Pillar: {topic_filled_from_pillar}, "
                f"Pillar←Topic: {pillar_filled_from_topic}"
            )

        
        # Phase B: Final retry 1x for rows where BOTH Topic & Pillar still empty
        final_retry_mask = (
            df['_is_master'] &
            (
                (df['Topic'].isna() | (df['Topic'].astype(str).str.strip() == '') | df['Topic'].apply(lambda x: is_invalid_value(str(x)))) &
                (df['Pillar'].isna() | (df['Pillar'].astype(str).str.strip() == '') | df['Pillar'].apply(lambda x: is_invalid_value(str(x))))
            )
        )

        final_retry_df = df[final_retry_mask].copy()
        final_retry_count = len(final_retry_df)

        logging.info(f"[STEP 5/5] Final retry candidates (both empty): {final_retry_count}")

        if final_retry_count > 0:
            retry_batches = []
            total_batches = math.ceil(final_retry_count / RETRY_BATCH_SIZE)

            for batch_num in range(total_batches):
                start_idx = batch_num * RETRY_BATCH_SIZE
                end_idx = min(start_idx + RETRY_BATCH_SIZE, final_retry_count)

                batch_df = final_retry_df.iloc[start_idx:end_idx].copy()

                # 🔁 FINAL RETRY → retry_attempt = 1
                result_batch = retry_topic_batch(
                    batch_df,
                    retry_attempt=1,
                    title_col=title_col,
                    content_col=content_col,
                    language=language,
                    token_tracker=tracker,
                    progress=progress
                )

                retry_batches.append(result_batch)

            df_final_retry = pd.concat(retry_batches, ignore_index=False)

            # Copy hasil retry ke df utama
            for idx in df_final_retry.index:
                if df_final_retry.at[idx, 'Topic'] and str(df_final_retry.at[idx, 'Topic']).strip():
                    df.at[idx, 'Topic'] = df_final_retry.at[idx, 'Topic']
                if df_final_retry.at[idx, 'Pillar'] and str(df_final_retry.at[idx, 'Pillar']).strip():
                    df.at[idx, 'Pillar'] = df_final_retry.at[idx, 'Pillar']

            # Copy ke duplicate rows
            for idx in df_final_retry.index:
                hash_val = df.at[idx, '_dedup_hash']
                duplicate_indices = df[
                    (df['_dedup_hash'] == hash_val) & (~df['_is_master'])
                ].index

                for dup_idx in duplicate_indices:
                    if df.at[idx, 'Topic'] and str(df.at[idx, 'Topic']).strip():
                        df.at[dup_idx, 'Topic'] = df.at[idx, 'Topic']
                    if df.at[idx, 'Pillar'] and str(df.at[idx, 'Pillar']).strip():
                        df.at[dup_idx, 'Pillar'] = df.at[idx, 'Pillar']


            logging.info(f"[STEP 5/5] Final retry completed")





        if generate_spokesperson:
            progress(0.95, desc="[STEP 4/4] Normalizing Spokesperson...")
            
            spokespersons = df['New Spokesperson'].dropna()
            spokespersons = spokespersons[spokespersons.astype(str).str.strip() != '']
            spokespersons = spokespersons[~spokespersons.apply(lambda x: is_invalid_value(str(x)))]
            unique_spokespersons = sorted(spokespersons.unique().tolist())
            
            if unique_spokespersons:
                spokesperson_mapping = normalize_spokesperson(unique_spokespersons, tracker, progress)
                df['New Spokesperson'] = df['New Spokesperson'].apply(
                    lambda x: spokesperson_mapping.get(x, x) if pd.notna(x) and str(x).strip() and not is_invalid_value(str(x)) else x
                )
        
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
        output_filename = f"{original_filename}_phase2_v12.xlsx"
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
                {"key": "version", "value": "v12.0 - Enhanced Pillar+Topic with Retry"},
                {"key": "model", "value": MODEL_NAME},
                {"key": "output_language", "value": f"{language} ({LANGUAGE_CONFIGS[language]['name']})"},
                {"key": "duration_sec", "value": f"{duration:.2f}"},
                {"key": "original_rows", "value": int(original_row_count)},
                {"key": "final_rows", "value": int(final_row_count)},
                {"key": "row_unchanged", "value": "YES" if original_row_count == final_row_count else "NO"},
                {"key": "deduplication_groups", "value": int(master_rows)},
                {"key": "duplicate_rows", "value": int(duplicate_rows)},
                {"key": "dedup_api_savings", "value": int(duplicate_rows)},
                {"key": "eligible_for_topic", "value": int(total_eligible)},
                {"key": "skipped_short_content", "value": int(total_skipped)},
                {"key": "prefilter_api_savings", "value": int(total_skipped)},
                {"key": "mainstream_rows", "value": int(mainstream_count)},
                {"key": "social_rows", "value": int(social_count)},
                {"key": "batch_size", "value": int(BATCH_SIZE)},
                {"key": "max_retries", "value": int(MAX_RETRIES)},
                {"key": "similarity_threshold", "value": f"{SIMILARITY_THRESHOLD*100}%"},
                {"key": "target_pillars_per_campaign", "value": int(TARGET_PILLARS_PER_CAMPAIGN)},
                #{"key": "skip_retry_threshold", "value": f"{SKIP_RETRY_THRESHOLD*100}%"},
                {"key": "pillar_engagement_weight", "value": f"{PILLAR_ENGAGEMENT_WEIGHT*100}%"},
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
                meta_data.append({
                    "key": f"success_rate_{step_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}",
                    "value": f"{step_data['success']}/{step_data['total']} ({step_data['rate']:.1f}%)"
                })
            
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
                "skipped_short": int(total_skipped),
                "api_savings": f"{total_skipped} topic extractions"
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
                "similarity_threshold": f"{SIMILARITY_THRESHOLD*100}%",
                "target_pillars": int(TARGET_PILLARS_PER_CAMPAIGN)
            },
            "duration": f"{duration:.2f}s",
            "cost": f"${token_summary['estimated_cost_usd']:.6f}",
            "success_rates": token_summary['step_stats']
        }
        
        logging.info("\n" + "="*80)
        logging.info("✅ PROCESSING COMPLETE - v12.0")
        logging.info("="*80)
        logging.info(f"Rows: {original_row_count} → {final_row_count} (unchanged: {original_row_count == final_row_count})")
        logging.info(f"Deduplication: {master_rows} groups, {duplicate_rows} duplicates (saved {duplicate_rows} calls)")
        logging.info(f"Pre-filter: {total_eligible} eligible, {total_skipped} skipped (saved {total_skipped} topic calls)")
        logging.info(f"Duration: {duration:.2f}s | Cost: ${token_summary['estimated_cost_usd']:.6f}")
        logging.info(f"Language: {language}")
        
        if generate_topic:
            logging.info(f"Normalization: {unique_topics_count} topics → {unique_pillars_final} pillars ({grouping_rate:.1f}% reduction)")
        
        for step_name, step_data in token_summary['step_stats'].items():
            logging.info(f"{step_name}: {step_data['success']}/{step_data['total']} ({step_data['rate']:.1f}%)")
        
        progress(1.0, desc="Complete!")
        return output_path, stats, None
        
    except Exception as e:
        logging.error(f"[ERROR] {str(e)}", exc_info=True)
        return None, {}, f"❌ Error: {str(e)}"

def create_gradio_interface():
    with gr.Blocks(title="Insights Generator v12.0", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📊 Insights Generator v12.0 - Enhanced Pillar+Topic Extraction")
        gr.Markdown("**NEW:** Pillar (2-6 words categorization) + Topic (5-15 words detailed) with 3x retry + engagement-aware normalization")
        
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
                    label="Output Language (Optimized for Indonesia)",
                    choices=list(LANGUAGE_CONFIGS.keys()),
                    value="Indonesia",
                    info="Content can be ANY language, output uses your selection"
                )
                
                gr.Markdown("### ⚙️ Config")
                conf_threshold = gr.Slider(label="Sentiment Confidence Threshold", minimum=0, maximum=100, value=85, step=5)
                
                gr.Markdown("### ✅ Features (Select at least 1)")
                gen_topic = gr.Checkbox(label="📌 Pillar & Topic (all channels) - Enhanced with 3x retry", value=False)
                gen_sentiment = gr.Checkbox(label="😊 Sentiment (all channels)", value=False)
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