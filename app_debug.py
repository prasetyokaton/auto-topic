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

INITIAL_BATCH_SIZE = 20
INITIAL_TRUNCATE = 200

RETRY_CONFIGS = [
    {"batch_size": 15, "truncate": 250},
    {"batch_size": 10, "truncate": 300},
    {"batch_size": 5, "truncate": 400}
]

MAINSTREAM_BATCH_SIZE = 30
MAX_RETRIES = 3

MIN_CONTENT_WORDS_FOR_TOPIC = 5

ENGAGEMENT_WEIGHT = 0.7
NORMALIZATION_BATCH_SIZE = 500
TOPICS_PER_100_SUBTOPICS = 5

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
        "prompt_instruction": "Use Bahasa Indonesia for topic and sub_topic",
        "word_count": "3-8 kata",
        "stopwords": ['yang', 'dan', 'di', 'dari', 'ke', 'untuk', 'dengan', 'pada',
                     'ini', 'itu', 'adalah', 'akan', 'atau', 'juga', 'tidak', 'bisa',
                     'ada', 'sudah', 'nya', 'si', 'oleh', 'dalam', 'sebagai', 'telah']
    },
    "English": {
        "code": "en",
        "name": "English",
        "prompt_instruction": "Use English for topic and sub_topic",
        "word_count": "3-8 words",
        "stopwords": ['the', 'a', 'an', 'in', 'on', 'at', 'to', 'of', 'for', 'is', 
                     'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                     'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can']
    },
    "Thailand": {
        "code": "th",
        "name": "ภาษาไทย (Thai)",
        "prompt_instruction": "Use Thai language (ภาษาไทย) for topic and sub_topic",
        "word_count": "3-8 คำ",
        "stopwords": ['ที่', 'และ', 'ใน', 'เป็น', 'ของ', 'กับ', 'ได้', 'มี', 'ให้', 'จาก']
    },
    "China": {
        "code": "zh",
        "name": "简体中文 (Simplified Chinese)",
        "prompt_instruction": "Use Simplified Chinese (简体中文) for topic and sub_topic",
        "word_count": "3-8 个词",
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

def truncate_to_first_n_words(text: str, n: int) -> str:
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
        elif col_lower == 'type':
            column_mapping[col] = 'Type'
    
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

def validate_and_normalize_subtopic(text: str, language: str, min_words: int = 3, max_words: int = 8) -> str:
    normalized = normalize_topic_text(text, language)
    
    if not normalized:
        return ""
    
    words = normalized.split()
    word_count = len(words)
    
    if word_count < min_words or word_count > max_words:
        return ""
    
    return normalized

def validate_and_normalize_topic(text: str, language: str, min_words: int = 3, max_words: int = 8) -> str:
    normalized = normalize_topic_text(text, language)
    
    if not normalized:
        return ""
    
    words = normalized.split()
    word_count = len(words)
    
    if word_count < min_words or word_count > max_words:
        if word_count > max_words:
            return " ".join(words[:max_words])
        return ""
    
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

def build_toon_input(batch_df, title_col, content_col, batch_size, truncate_words, clean_content=False):
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

def chat_create(model, messages, token_tracker=None, max_retries=2):
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
    batch_size: int,
    truncate_words: int,
    campaign_name: str = "",
    campaign_progress: str = "",
    progress=gr.Progress()
) -> pd.DataFrame:
    
    actual_batch_size = len(batch_df)
    lang_config = LANGUAGE_CONFIGS[language]
    
    if 'Sub Topic' not in batch_df.columns:
        batch_df['Sub Topic'] = ''
    if 'New Sentiment' not in batch_df.columns:
        batch_df['New Sentiment'] = 'neutral'
    if 'New Sentiment Level' not in batch_df.columns:
        batch_df['New Sentiment Level'] = 0
    
    input_toon = build_toon_input(batch_df, title_col, content_col, actual_batch_size, truncate_words, clean_content=True)
    
    nonce = random.randint(100000, 999999)
    
    prompt = f"""You are an insights professional with MULTI-LANGUAGE expertise.

[Request ID: {nonce}]
[OUTPUT LANGUAGE: {language} - {lang_config['prompt_instruction']}]

INPUT (TOON format, content may be in ANY language):
{input_toon}

TASK: Analyze content and extract insights
- Content may be in: Thai (ภาษาไทย), English, Chinese (中文), Indonesian, mixed languages, etc.
- You MUST UNDERSTAND content regardless of input language
- EXTRACT specific sub topic ({lang_config['word_count']})
- ANALYZE sentiment (positive/negative/neutral)
- ASSESS confidence (0-100)
- OUTPUT everything in {language}

MANDATORY SPECIFICITY RULES FOR SUB TOPIC:
1. MUST be CONCRETE and SPECIFIC - include context/event/issue/activity
2. MUST include brand/product name if mentioned prominently
3. AVOID abstract feelings: kesegaran, manfaat, pengaruh, reaksi, kesan
4. PREFER specific: event names, issues, activities, locations
5. Length: {lang_config['word_count']} - be descriptive but concise

GOOD EXAMPLES (SPECIFIC):
✅ "Le Minerale di Jakarta Running Festival" (event + brand)
✅ "Klarifikasi Sumber Air Le Minerale" (specific issue)
✅ "Promo Galon Le Minerale di Alfamart" (specific activity + location)
✅ "Kritik Kualitas Kemasan Le Minerale" (specific criticism)

BAD EXAMPLES (TOO GENERIC):
❌ "Kesegaran Le Minerale" (abstract feeling)
❌ "Manfaat Air Mineral" (too broad)
❌ "Pengaruh Le Minerale" (vague)
❌ "Info Terbaru" (noise words)

SENTIMENT RULES:
- positive: clear positive emotion, praise, satisfaction, achievement
- negative: clear negative emotion, complaint, disappointment, criticism
- neutral: factual, informational, no clear emotion, or ambiguous

CRITICAL:
- NEVER use "unknown", "nan", "none", "tidak jelas"
- NEVER use noise words: berita, info, viral, trending, update, artikel
- If cannot extract SPECIFIC sub topic, use "-"

OUTPUT FORMAT (TOON with pipe delimiter |):
result[{actual_batch_size}]{{row|sub_topic|sentiment|confidence}}:
<row_index>|<specific sub_topic in {language} or ->|<sentiment>|<confidence_0-100>

YOUR OUTPUT (TOON format only):"""
    
    try:
        progress_desc = f"[STEP 1/4] {campaign_progress}Sub Topic+Sentiment {batch_num}/{total_batches}"
        progress(0.5, desc=progress_desc)
        
        response = chat_create(
            MODEL_NAME,
            [
                {"role": "system", "content": f"You are an insights professional. Output TOON format only. {lang_config['prompt_instruction']}. Handle multi-language input, output in {language}. FOCUS ON SPECIFIC CONTEXT."},
                {"role": "user", "content": prompt}
            ],
            token_tracker=token_tracker
        )
        
        if not response:
            raise Exception("API call failed")
        
        raw = response.choices[0].message.content.strip()
        data, format_type = parse_gpt_response(raw, ['row', 'sub_topic', 'sentiment', 'confidence'], actual_batch_size, token_tracker)
        
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
                
                current_sub_topic = batch_df.at[idx, 'Sub Topic']
                if pd.isna(current_sub_topic) or str(current_sub_topic).strip() == '':
                    sub_topic = str(item.get('sub_topic', '')).strip()
                    
                    if sub_topic == '-' or is_invalid_value(sub_topic):
                        sub_topic = ''
                    
                    if sub_topic:
                        sub_topic = validate_and_normalize_subtopic(sub_topic, language)
                        
                        if sub_topic:
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

def process_batch_spokesperson(
    batch_df: pd.DataFrame,
    batch_num: int,
    total_batches: int,
    title_col: str,
    content_col: str,
    token_tracker: TokenTracker,
    campaign_name: str = "",
    campaign_progress: str = "",
    progress=gr.Progress()
) -> pd.DataFrame:
    
    batch_size = len(batch_df)
    
    batch_df['New Spokesperson'] = ''
    
    input_toon = build_toon_input(batch_df, title_col, content_col, batch_size, 150)
    
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
        progress_desc = f"[STEP 2/4] {campaign_progress}Spokesperson {batch_num}/{total_batches}"
        progress(0.5, desc=progress_desc)
        
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

def retry_sub_topic_batch(
    batch_df: pd.DataFrame,
    title_col: str,
    content_col: str,
    language: str,
    token_tracker: TokenTracker,
    batch_size: int,
    truncate_words: int,
    retry_num: int,
    campaign_progress: str = "",
    progress=gr.Progress()
) -> pd.DataFrame:
    
    actual_size = len(batch_df)
    lang_config = LANGUAGE_CONFIGS[language]
    
    input_toon = build_toon_input(
        batch_df, 
        title_col, 
        content_col, 
        actual_size,
        truncate_words,
        clean_content=True
    )
    
    nonce = random.randint(100000, 999999)
    
    prompt = f"""🚨🚨🚨 RETRY #{retry_num} - MANDATORY SPECIFIC EXTRACTION 🚨🚨🚨

[Request ID: {nonce}]
[OUTPUT LANGUAGE: {language}]
[EXTENDED CONTEXT: {truncate_words} words for better analysis]

INPUT (TOON format, content may be ANY language):
{input_toon}

CRITICAL MISSION: Extract SPECIFIC sub topic from EVERY row

MANDATORY SPECIFICITY RULES:
1. Content may be ANY language - you MUST understand it
2. Extract CONCRETE context: WHO/WHAT/WHERE/WHEN if available
3. AVOID abstract: kesegaran, manfaat, pengaruh, reaksi
4. PREFER specific: event names, issues, activities, products
5. Include brand/product if mentioned
6. Length: {lang_config['word_count']}

EXAMPLES:
✅ "Klarifikasi Sumber Air Le Minerale di DPR"
✅ "Promo Galon Le Minerale di Alfamart"
✅ "Jakarta Running Festival dengan Le Minerale"
❌ "Manfaat Le Minerale" (too generic)
❌ "Info Terbaru" (noise words)

RULES:
- NEVER use: "unknown", "tidak jelas", "berita", "info", "viral"
- If really cannot extract SPECIFIC context, use "-"
- OUTPUT in {language} only

OUTPUT (TOON format):
result[{actual_size}]{{row|sub_topic}}:
<row_index>|<specific sub_topic in {language} or ->

YOUR OUTPUT:"""
    
    try:
        progress_desc = f"[STEP 3/4] {campaign_progress}Retry #{retry_num} ({actual_size} rows, {truncate_words}w)..."
        progress(0.95, desc=progress_desc)
        
        response = chat_create(
            MODEL_NAME,
            [
                {"role": "system", "content": f"CRITICAL RETRY. {lang_config['prompt_instruction']}. Multi-language input. EXTRACT SPECIFIC CONTEXT. No generic terms."},
                {"role": "user", "content": prompt}
            ],
            token_tracker=token_tracker
        )
        
        if not response:
            raise Exception("API call failed")
        
        raw = response.choices[0].message.content.strip()
        data, format_type = parse_gpt_response(raw, ['row', 'sub_topic'], actual_size, token_tracker)
        
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
                
                if sub_topic == '-' or is_invalid_value(sub_topic):
                    sub_topic = ''
                
                if sub_topic:
                    sub_topic = validate_and_normalize_subtopic(sub_topic, language)
                    
                    if sub_topic:
                        batch_df.at[idx, 'Sub Topic'] = sub_topic
        
        return batch_df
        
    except Exception as e:
        return batch_df

def calculate_target_topics(unique_count: int) -> int:
    """Calculate target topics: 5 topics per 100 sub topics"""
    target = max(5, (unique_count // 100) * 5)
    if unique_count % 100 >= 50:
        target += 5
    return target

def openai_generate_topics(
    sub_topics_with_engagement: list,
    language: str,
    target_topics: int,
    token_tracker: TokenTracker
) -> list:
    
    lang_config = LANGUAGE_CONFIGS[language]
    nonce = random.randint(100000, 999999)
    
    sub_topics_text = "\n".join([
        f"{i+1}. {st['sub_topic']} ({st['engagement']:,.0f} engagement, {st['frequency']} mentions)"
        for i, st in enumerate(sub_topics_with_engagement)
    ])
    
    prompt = f"""You are a topic normalization expert with focus on SPECIFIC, MEANINGFUL categories.

[Request ID: {nonce}]
[OUTPUT LANGUAGE: {language}]

INPUT: {len(sub_topics_with_engagement)} sub topics (sorted by engagement, highest first)

Sub Topics:
{sub_topics_text}

TASK:
Generate approximately {target_topics} NORMALIZED TOPICS (range: {max(5, target_topics-5)} to {target_topics+5})
- Each topic: 3-8 words in {language}
- Topics must be SPECIFIC, CONCRETE, and MEANINGFUL
- Group similar sub topics under clear category names
- HIGH-ENGAGEMENT sub topics (top of list) are MOST IMPORTANT for naming

MANDATORY RULES:
⭐ Prioritize HIGH-ENGAGEMENT content for topic names
⭐ Topics must be SPECIFIC and ACTIONABLE
⭐ AVOID generic/abstract: manfaat, kesegaran, pengaruh, reaksi
⭐ PREFER concrete: specific events, issues, activities, products
⭐ NO noise words: berita, info, viral, trending, update, artikel
⭐ Each topic must be DISTINCT and NON-OVERLAPPING

GOOD EXAMPLES (SPECIFIC):
✅ "Klarifikasi Sumber Air di DPR" (specific issue)
✅ "Le Minerale di Event Olahraga" (specific context)
✅ "Promo dan Jastip Galon" (specific activity)
✅ "Kritik Kualitas dan Kemasan" (specific problem)

BAD EXAMPLES (TOO GENERIC):
❌ "Manfaat Le Minerale" (abstract)
❌ "Info Produk" (noise words)
❌ "Kesegaran dan Hidrasi" (too vague)

OUTPUT FORMAT (TOON with pipe delimiter |):
topics[{target_topics}]{{topic}}:
<topic_name_1>
<topic_name_2>
...

YOUR OUTPUT (TOON format only):"""
    
    try:
        response = chat_create(
            MODEL_NAME,
            [
                {"role": "system", "content": f"You are a topic expert. {lang_config['prompt_instruction']}. Output TOON format only. FOCUS ON SPECIFIC, CONCRETE CATEGORIES."},
                {"role": "user", "content": prompt}
            ],
            token_tracker=token_tracker
        )
        
        if not response:
            logging.warning("Topic generation API call failed")
            return []
        
        raw = response.choices[0].message.content.strip()
        lines = raw.strip().split('\n')
        
        topics = []
        data_start = False
        
        for line in lines:
            if 'topics[' in line.lower() and '{' in line and '}' in line:
                data_start = True
                continue
            
            if data_start:
                line = line.strip()
                if line and not line.startswith('#'):
                    topic = validate_and_normalize_topic(line, language)
                    if topic:
                        topics.append(topic)
        
        logging.info(f"Generated {len(topics)} normalized topics")
        return topics
        
    except Exception as e:
        logging.error(f"Topic generation error: {e}")
        return []

def openai_map_subtopics(
    all_sub_topics: list,
    normalized_topics: list,
    language: str,
    token_tracker: TokenTracker
) -> dict:
    
    if not normalized_topics:
        logging.warning("No normalized topics provided for mapping")
        return {}
    
    lang_config = LANGUAGE_CONFIGS[language]
    nonce = random.randint(100000, 999999)
    
    topics_text = "\n".join([f"{i+1}. {topic}" for i, topic in enumerate(normalized_topics)])
    sub_topics_text = "\n".join([f"{i+1}. {st}" for i, st in enumerate(all_sub_topics)])
    
    prompt = f"""You are a mapping expert with focus on SEMANTIC SIMILARITY.

[Request ID: {nonce}]

INPUT:
{len(normalized_topics)} Normalized Topics:
{topics_text}

{len(all_sub_topics)} Sub Topics to Map:
{sub_topics_text}

TASK:
Map each sub topic to the MOST SEMANTICALLY SIMILAR normalized topic.

RULES:
- Match based on MEANING and CONTEXT, not just keywords
- If sub topic clearly belongs to a topic category, map it
- If NO good semantic match exists, leave UNMAPPED (use "-")
- NEVER force mapping if similarity is low
- Consider: same event, same issue, same activity type, same product focus

OUTPUT FORMAT (TOON with pipe delimiter |):
mapping[{len(all_sub_topics)}]{{sub_topic|normalized_topic}}:
<sub_topic_1>|<matched_topic or ->
<sub_topic_2>|<matched_topic or ->
...

YOUR OUTPUT (TOON format only):"""
    
    try:
        response = chat_create(
            MODEL_NAME,
            [
                {"role": "system", "content": f"You are a mapping expert. {lang_config['prompt_instruction']}. Output TOON format only. Focus on SEMANTIC SIMILARITY."},
                {"role": "user", "content": prompt}
            ],
            token_tracker=token_tracker
        )
        
        if not response:
            logging.warning("Mapping API call failed")
            return {}
        
        raw = response.choices[0].message.content.strip()
        
        # ========================================
        # MORE ROBUST PARSING - CRITICAL FIX
        # ========================================
        lines = raw.strip().split('\n')
        
        mapping = {}
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines, comments, code blocks
            if not line or line.startswith('#') or line.startswith('```') or line.startswith('**'):
                continue
            
            # Skip header lines (contain 'mapping[' or descriptive text without |)
            if 'mapping[' in line.lower():
                continue
            if '{' in line and '}' in line and ':' in line and '|' not in line:
                continue
            
            # Skip explanation lines (no pipe delimiter)
            if '|' not in line:
                continue
            
            # Parse data lines (must contain |)
            parts = line.split('|', 1)
            if len(parts) == 2:
                sub_topic = parts[0].strip()
                topic = parts[1].strip()
                
                # Remove numbering if present (e.g., "1. Sub Topic" → "Sub Topic")
                if sub_topic and sub_topic[0].isdigit():
                    if '. ' in sub_topic:
                        sub_topic = sub_topic.split('. ', 1)[1].strip()
                    elif ' ' in sub_topic and sub_topic.split()[0].replace('.', '').isdigit():
                        sub_topic = ' '.join(sub_topic.split()[1:]).strip()
                
                # Validate and add to mapping
                if (sub_topic and topic and 
                    len(sub_topic) >= 3 and  # Real sub topic
                    topic != '-' and 
                    not is_invalid_value(topic)):
                    mapping[sub_topic] = topic
        
        mapped_count = len(mapping)
        unmapped_count = len(all_sub_topics) - mapped_count
        
        logging.info(f"Mapped {mapped_count}/{len(all_sub_topics)} sub topics ({unmapped_count} unmapped)")
        
        # ========================================
        # ENHANCED DEBUG LOGGING
        # ========================================
        if mapped_count == 0:
            logging.error(f"❌ ZERO MAPPINGS CREATED!")
            logging.error(f"Raw API response (first 500 chars):")
            logging.error(raw[:500])
            logging.error(f"Response has {len(lines)} lines")
            logging.error(f"Lines preview: {lines[:5]}")
        elif mapped_count < len(all_sub_topics) * 0.3:
            logging.warning(f"⚠️ LOW MAPPING RATE: {mapped_count}/{len(all_sub_topics)} ({mapped_count/len(all_sub_topics)*100:.1f}%)")
            logging.warning(f"Sample unmapped sub topics (first 5):")
            unmapped = [st for st in all_sub_topics if st not in mapping]
            for st in unmapped[:5]:
                logging.warning(f"   - '{st}'")
        
        return mapping
        
    except Exception as e:
        logging.error(f"Mapping error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {}

def openai_consolidate_topics(
    all_topics: list,
    target_final: int,
    language: str,
    token_tracker: TokenTracker
) -> dict:
    """Consolidate multiple topic sets into final unified set"""
    
    lang_config = LANGUAGE_CONFIGS[language]
    nonce = random.randint(100000, 999999)
    
    topics_text = "\n".join([f"{i+1}. {topic}" for i, topic in enumerate(all_topics)])
    
    prompt = f"""You are a topic consolidation expert.

[Request ID: {nonce}]

INPUT: {len(all_topics)} topics from multiple batches (may have duplicates/overlaps)

Topics:
{topics_text}

TASK:
Consolidate into {target_final} FINAL UNIFIED TOPICS (range: {target_final-5} to {target_final+5})

RULES:
- MERGE similar/duplicate topics
- Keep MOST SPECIFIC and MEANINGFUL names
- Maintain 3-8 words in {language}
- NO generic terms: manfaat, kesegaran, info, berita
- Output ONLY consolidated unique topics

OUTPUT FORMAT (TOON):
consolidated_topics[{target_final}]{{topic}}:
<final_topic_1>
<final_topic_2>
...

YOUR OUTPUT:"""
    
    try:
        response = chat_create(
            MODEL_NAME,
            [
                {"role": "system", "content": f"{lang_config['prompt_instruction']}. Output TOON format only."},
                {"role": "user", "content": prompt}
            ],
            token_tracker=token_tracker
        )
        
        if not response:
            return {"topics": all_topics[:target_final]}
        
        raw = response.choices[0].message.content.strip()
        lines = raw.strip().split('\n')
        
        consolidated = []
        data_start = False
        
        for line in lines:
            if 'consolidated_topics[' in line.lower() or 'topics[' in line.lower():
                data_start = True
                continue
            
            if data_start:
                line = line.strip()
                if line and not line.startswith('#'):
                    topic = validate_and_normalize_topic(line, language)
                    if topic:
                        consolidated.append(topic)
        
        if not consolidated:
            return {"topics": all_topics[:target_final]}
        
        old_to_new = {}
        for old_topic in all_topics:
            best_match = None
            for new_topic in consolidated:
                if old_topic == new_topic:
                    best_match = new_topic
                    break
                old_words = set(old_topic.lower().split())
                new_words = set(new_topic.lower().split())
                if len(old_words & new_words) >= 2:
                    best_match = new_topic
                    break
            
            if best_match:
                old_to_new[old_topic] = best_match
            else:
                old_to_new[old_topic] = consolidated[0] if consolidated else old_topic
        
        logging.info(f"Consolidated {len(all_topics)} topics → {len(consolidated)} final topics")
        
        return {
            "topics": consolidated,
            "mapping": old_to_new
        }
        
    except Exception as e:
        logging.error(f"Consolidation error: {e}")
        return {"topics": all_topics[:target_final]}

def normalize_topics_v13_batched(
    df: pd.DataFrame,
    selected_campaigns: list,
    campaign_col: str = 'Campaigns',
    engagement_col: str = 'Engagement',
    language: str = 'Indonesia',
    token_tracker: TokenTracker = None,
    progress=gr.Progress()
) -> dict:
    
    results = {}
    total_campaigns = len(selected_campaigns)
    
    for idx, campaign in enumerate(selected_campaigns, 1):
        logging.info(f"\n{'='*80}")
        logging.info(f"[NORMALIZATION {idx}/{total_campaigns}] Campaign: '{campaign}'")
        logging.info(f"{'='*80}")
        
        campaign_df = df[df[campaign_col] == campaign].copy()
        
        sub_topics = campaign_df['Sub Topic'].dropna()
        sub_topics = sub_topics[sub_topics.astype(str).str.strip() != '']
        sub_topics = sub_topics[~sub_topics.apply(lambda x: is_invalid_value(str(x)))]
        unique_sub_topics = sub_topics.unique().tolist()
        
        if len(unique_sub_topics) == 0:
            logging.warning(f"  └─ No valid sub topics for campaign '{campaign}', skipping")
            results[campaign] = {
                'mapping': {},
                'topics': [],
                'stats': {
                    'original_subtopics': 0,
                    'final_topics': 0,
                    'cost_usd': 0
                }
            }
            continue
        
        if engagement_col in campaign_df.columns:
            engagement_map = campaign_df.groupby('Sub Topic').agg({
                engagement_col: 'sum',
                'Title': 'count'
            }).reset_index()
            engagement_map.columns = ['Sub Topic', 'Total_Engagement', 'Frequency']
            
            engagement_map['Weight_Score'] = (
                engagement_map['Total_Engagement'] * ENGAGEMENT_WEIGHT +
                engagement_map['Frequency'] * (1 - ENGAGEMENT_WEIGHT)
            )
            
            engagement_map = engagement_map.sort_values('Weight_Score', ascending=False)
        else:
            engagement_map = campaign_df.groupby('Sub Topic').agg({
                'Title': 'count'
            }).reset_index()
            engagement_map.columns = ['Sub Topic', 'Frequency']
            engagement_map['Total_Engagement'] = 0
            engagement_map['Weight_Score'] = engagement_map['Frequency']
            engagement_map = engagement_map.sort_values('Weight_Score', ascending=False)
        
        engagement_map = engagement_map[engagement_map['Sub Topic'].isin(unique_sub_topics)]
        
        logging.info(f"  └─ Sub topics: {len(unique_sub_topics)}")
        
        campaign_progress = f"Campaign {campaign} ({idx}/{total_campaigns}) - "
        progress_val = 0.85 + (idx / total_campaigns) * 0.10
        
        if len(unique_sub_topics) <= NORMALIZATION_BATCH_SIZE:
            logging.info(f"  └─ Single batch normalization ({len(unique_sub_topics)} ≤ {NORMALIZATION_BATCH_SIZE})")
            
            target_topics = calculate_target_topics(len(unique_sub_topics))
            logging.info(f"  └─ Target topics: {target_topics} (ratio: 5 per 100)")
            
            sub_topics_with_engagement = [
                {
                    'sub_topic': row['Sub Topic'],
                    'engagement': row['Total_Engagement'],
                    'frequency': row['Frequency']
                }
                for _, row in engagement_map.iterrows()
            ]
            
            progress(progress_val, desc=f"[STEP 4/4] {campaign_progress}Generating Topics...")
            
            normalized_topics = openai_generate_topics(
                sub_topics_with_engagement,
                language,
                target_topics,
                token_tracker
            )
            
            if not normalized_topics:
                logging.warning(f"  └─ Topic generation failed, using fallback")
                mapping = {st: "" for st in unique_sub_topics}
            else:
                logging.info(f"  └─ Generated {len(normalized_topics)} topics")
                
                progress(progress_val + 0.02, desc=f"[STEP 4/4] {campaign_progress}Mapping Sub Topics...")
                
                mapping = openai_map_subtopics(
                    unique_sub_topics,
                    normalized_topics,
                    language,
                    token_tracker
                )
        
        else:
            logging.info(f"  └─ Multi-batch normalization ({len(unique_sub_topics)} > {NORMALIZATION_BATCH_SIZE})")
            
            num_batches = math.ceil(len(unique_sub_topics) / NORMALIZATION_BATCH_SIZE)
            logging.info(f"  └─ Splitting into {num_batches} batches of {NORMALIZATION_BATCH_SIZE}")
            
            all_batch_topics = []
            batch_mappings = {}
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * NORMALIZATION_BATCH_SIZE
                end_idx = min(start_idx + NORMALIZATION_BATCH_SIZE, len(engagement_map))
                
                batch_engagement = engagement_map.iloc[start_idx:end_idx]
                batch_sub_topics = batch_engagement['Sub Topic'].tolist()
                
                batch_target = calculate_target_topics(len(batch_sub_topics))
                
                logging.info(f"  └─ Batch {batch_idx+1}/{num_batches}: {len(batch_sub_topics)} sub topics → target {batch_target} topics")
                
                sub_topics_with_engagement = [
                    {
                        'sub_topic': row['Sub Topic'],
                        'engagement': row['Total_Engagement'],
                        'frequency': row['Frequency']
                    }
                    for _, row in batch_engagement.iterrows()
                ]
                
                progress(progress_val + (batch_idx/num_batches)*0.04, 
                        desc=f"[STEP 4/4] {campaign_progress}Gen Topics Batch {batch_idx+1}/{num_batches}...")
                
                batch_topics = openai_generate_topics(
                    sub_topics_with_engagement,
                    language,
                    batch_target,
                    token_tracker
                )
                
                if batch_topics:
                    all_batch_topics.extend(batch_topics)
                    logging.info(f"     └─ Generated {len(batch_topics)} topics for batch {batch_idx+1}")
                    
                    for st in batch_sub_topics:
                        batch_mappings[st] = batch_topics
            
            if not all_batch_topics:
                logging.warning(f"  └─ All batches failed, using fallback")
                mapping = {st: "" for st in unique_sub_topics}
                normalized_topics = []
            else:
                logging.info(f"  └─ Total topics from all batches: {len(all_batch_topics)}")
                
                final_target = calculate_target_topics(len(unique_sub_topics))
                logging.info(f"  └─ Consolidating {len(all_batch_topics)} → {final_target} final topics")
                
                progress(progress_val + 0.05, desc=f"[STEP 4/4] {campaign_progress}Consolidating Topics...")
                
                consolidation_result = openai_consolidate_topics(
                    all_batch_topics,
                    final_target,
                    language,
                    token_tracker
                )
                
                normalized_topics = consolidation_result.get('topics', all_batch_topics[:final_target])
                batch_to_final = consolidation_result.get('mapping', {})
                
                logging.info(f"  └─ Final consolidated topics: {len(normalized_topics)}")
                
                progress(progress_val + 0.07, desc=f"[STEP 4/4] {campaign_progress}Final Mapping...")
                
                mapping = {}
                for sub_topic, batch_topics in batch_mappings.items():
                    temp_mapping = openai_map_subtopics(
                        [sub_topic],
                        batch_topics,
                        language,
                        token_tracker
                    )
                    
                    if sub_topic in temp_mapping:
                        batch_topic = temp_mapping[sub_topic]
                        final_topic = batch_to_final.get(batch_topic, batch_topic)
                        mapping[sub_topic] = final_topic
        
        final_topics = len([t for t in set(mapping.values()) if t])
        reduction_rate = (1 - final_topics/len(unique_sub_topics)) * 100 if len(unique_sub_topics) > 0 else 0
        
        results[campaign] = {
            'mapping': mapping,
            'topics': normalized_topics if isinstance(normalized_topics, list) else [],
            'stats': {
                'original_subtopics': len(unique_sub_topics),
                'final_topics': final_topics,
                'reduction_rate': reduction_rate
            }
        }
        
        logging.info(f"  └─ Final topics: {final_topics}")
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

def load_campaigns_from_file(file_path, sheet_name):
    """Load campaigns with row counts from Excel file"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df = normalize_column_names(df)
        
        if 'Campaigns' not in df.columns:
            return []
        
        campaign_counts = df['Campaigns'].value_counts().to_dict()
        
        campaigns_with_counts = []
        for campaign, count in sorted(campaign_counts.items(), key=lambda x: x[1], reverse=True):
            campaign_name = str(campaign) if pd.notna(campaign) and str(campaign).strip() else "(Unnamed Campaign)"
            campaigns_with_counts.append({
                'name': campaign_name,
                'count': int(count),
                'label': f"{campaign_name} ({count:,} rows)"
            })
        
        return campaigns_with_counts
        
    except Exception as e:
        logging.error(f"Error loading campaigns: {e}")
        return []

def process_file(
    file_path: str,
    sheet_name: str,
    selected_campaigns: list,
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
        
        total_campaigns_in_file = df['Campaigns'].nunique()
        
        if not selected_campaigns or len(selected_campaigns) == 0:
            return None, {}, "❌ Error: Please select at least one campaign!"
        
        df = df[df['Campaigns'].isin(selected_campaigns)].copy()
        
        if len(df) == 0:
            return None, {}, "❌ Error: No data for selected campaigns!"
        
        filtered_row_count = len(df)
        skipped_campaigns = total_campaigns_in_file - len(selected_campaigns)
        
        logging.info(f"\n{'='*80}")
        logging.info(f"[CAMPAIGN SELECTION]")
        logging.info(f"{'='*80}")
        logging.info(f"Total campaigns in file: {total_campaigns_in_file}")
        logging.info(f"Selected for processing: {len(selected_campaigns)}")
        logging.info(f"Skipped campaigns: {skipped_campaigns}")
        logging.info(f"Original rows: {original_row_count}")
        logging.info(f"Filtered rows (selected campaigns): {filtered_row_count}")
        logging.info(f"Selected campaigns: {', '.join(selected_campaigns)}")
        
        title_col = get_col(df, ["Title", "Judul"])
        content_col = get_col(df, ["Content", "Konten", "Isi"])
        channel_col = "Channel"
        
        if 'Noise Tag' in df.columns:
            df['Noise Tag'] = df['Noise Tag'].astype(str)
            logging.info("✅ Converted Noise Tag to text")
        
        if 'Engagement' not in df.columns:
            logging.warning("⚠️ 'Engagement' column not found, normalization will use frequency only")
            df['Engagement'] = 0
        
        has_type_column = 'Type' in df.columns
        
        if has_type_column:
            type_comment_reply = df['Type'].isin(['Comment', 'Reply'])
            comment_count = (df['Type'] == 'Comment').sum()
            reply_count = (df['Type'] == 'Reply').sum()
            other_count = (~type_comment_reply).sum()
            
            logging.info(f"\n{'='*80}")
            logging.info(f"[TYPE FILTER]")
            logging.info(f"{'='*80}")
            logging.info(f"Comment: {comment_count} rows (will be excluded from topic extraction)")
            logging.info(f"Reply: {reply_count} rows (will be excluded from topic extraction)")
            logging.info(f"Other: {other_count} rows (will process for topics)")
            logging.info(f"Note: Sentiment & Spokesperson will process ALL {filtered_row_count} rows")
        else:
            logging.info(f"\n{'='*80}")
            logging.info(f"[TYPE FILTER]")
            logging.info(f"{'='*80}")
            logging.info(f"'Type' column not found - processing all rows for topics")
            type_comment_reply = pd.Series([False] * len(df), index=df.index)
        
        df['_original_index'] = df.index
        df['_channel_original'] = df[channel_col].copy()
        df['_channel_lower'] = df[channel_col].astype(str).str.lower().str.strip()
        
        empty_channels = df['_channel_lower'].isna() | (df['_channel_lower'] == '') | (df['_channel_lower'] == 'nan')
        if empty_channels.any():
            return None, {}, f"❌ Error: {empty_channels.sum()} baris memiliki Channel kosong!"
        
        logging.info("\n" + "="*80)
        logging.info("[DEDUPLICATION] Creating groups for identical content (selected campaigns only)")
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
        
        if has_type_column:
            df.loc[type_comment_reply, '_eligible_for_topic'] = False
        
        total_eligible = df['_eligible_for_topic'].sum()
        total_skipped = (~df['_eligible_for_topic']).sum()
        
        logging.info(f"✅ Content filter: {total_eligible} eligible, {total_skipped} skipped")
        if has_type_column:
            logging.info(f"   └─ Skipped: {(~df['_eligible_for_topic'] & ~type_comment_reply).sum()} (<{MIN_CONTENT_WORDS_FOR_TOPIC} words)")
            logging.info(f"   └─ Excluded: {type_comment_reply.sum()} (Comment/Reply)")
        else:
            logging.info(f"   └─ Skipped: {total_skipped} (<{MIN_CONTENT_WORDS_FOR_TOPIC} words)")
        logging.info(f"💰 API savings from pre-filter: {total_skipped} topic extractions skipped")
        
        if generate_topic:
            if 'Sub Topic' not in df.columns:
                df['Sub Topic'] = ''
            if 'Topic' not in df.columns:
                df['Topic'] = ''
        
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
        
        per_campaign_stats = {}
        
        for camp_idx, campaign in enumerate(selected_campaigns, 1):
            campaign_start_time = time.time()
            campaign_tracker = TokenTracker()
            
            logging.info(f"\n{'='*80}")
            logging.info(f"[PROCESSING CAMPAIGN {camp_idx}/{len(selected_campaigns)}] '{campaign}'")
            logging.info(f"{'='*80}")
            
            campaign_df_mask = df['Campaigns'] == campaign
            campaign_row_count = campaign_df_mask.sum()
            
            logging.info(f"Campaign rows: {campaign_row_count}")
            
            campaign_progress = f"Campaign {campaign} ({camp_idx}/{len(selected_campaigns)}) - "
            
            if generate_topic or generate_sentiment:
                logging.info("\n" + "="*80)
                logging.info(f"[STEP 1/4] SUB TOPIC + SENTIMENT - {campaign}")
                logging.info(f"Strategy: Aggressive Quality (smaller batches, longer content)")
                logging.info("="*80)
                
                process_mask = df['_is_master'] & df['_eligible_for_topic'] & campaign_df_mask
                
                if generate_topic and 'Topic' in df.columns and 'Sub Topic' in df.columns:
                    has_both = (df['Topic'].notna() & (df['Topic'].astype(str).str.strip() != '')) & \
                              (df['Sub Topic'].notna() & (df['Sub Topic'].astype(str).str.strip() != ''))
                    process_mask = process_mask & ~has_both
                
                if 'Noise Tag' in df.columns:
                    noise_tag_2 = df['Noise Tag'] == "2"
                    process_mask = process_mask & ~noise_tag_2
                
                df_to_process = df[process_mask].copy()
                logging.info(f"📊 Processing {len(df_to_process)} master rows")
                logging.info(f"   Initial: Batch {INITIAL_BATCH_SIZE} × {INITIAL_TRUNCATE} words")
                
                if len(df_to_process) > 0:
                    all_batches = []
                    total_batches = math.ceil(len(df_to_process) / INITIAL_BATCH_SIZE)
                    
                    for batch_num in range(total_batches):
                        start_idx = batch_num * INITIAL_BATCH_SIZE
                        end_idx = min(start_idx + INITIAL_BATCH_SIZE, len(df_to_process))
                        batch_df = df_to_process.iloc[start_idx:end_idx].copy()
                        
                        progress_val = 0.1 + ((camp_idx-1) / len(selected_campaigns)) * 0.30 + (batch_num / total_batches / len(selected_campaigns)) * 0.30
                        
                        result_batch = process_batch_combined(
                            batch_df, batch_num + 1, total_batches,
                            title_col, content_col, language, conf_threshold, 
                            campaign_tracker, INITIAL_BATCH_SIZE, INITIAL_TRUNCATE,
                            campaign, campaign_progress, progress
                        )
                        
                        all_batches.append(result_batch)
                    
                    df_processed = pd.concat(all_batches, ignore_index=False)
                    
                    for idx in df_processed.index:
                        if generate_topic and 'Sub Topic' in df_processed.columns:
                            df.at[idx, 'Sub Topic'] = df_processed.at[idx, 'Sub Topic']
                        if generate_sentiment:
                            df.at[idx, 'New Sentiment'] = df_processed.at[idx, 'New Sentiment']
                            df.at[idx, 'New Sentiment Level'] = df_processed.at[idx, 'New Sentiment Level']
                    
                    logging.info("📋 Copying results to duplicate rows...")
                    for hash_val in df[campaign_df_mask]['_dedup_hash'].unique():
                        group = df[(df['_dedup_hash'] == hash_val) & campaign_df_mask]
                        if len(group) > 1:
                            master_idx = group[group['_is_master']].index[0]
                            duplicate_indices = group[~group['_is_master']].index
                            
                            for dup_idx in duplicate_indices:
                                if generate_topic and 'Sub Topic' in df.columns:
                                    df.at[dup_idx, 'Sub Topic'] = df.at[master_idx, 'Sub Topic']
                                if generate_sentiment:
                                    df.at[dup_idx, 'New Sentiment'] = df.at[master_idx, 'New Sentiment']
                                    df.at[dup_idx, 'New Sentiment Level'] = df.at[master_idx, 'New Sentiment Level']
                    
                    if generate_topic:
                        campaign_sub_filled = df[campaign_df_mask]['Sub Topic'].notna() & \
                                            (df[campaign_df_mask]['Sub Topic'].astype(str).str.strip() != '')
                        success_count = campaign_sub_filled.sum()
                        logging.info(f"[STEP 1/4] ✅ Sub Topic: {success_count}/{campaign_row_count} ({success_count/campaign_row_count*100:.1f}%)")
                    
                    if generate_sentiment:
                        campaign_sent_filled = df[campaign_df_mask]['New Sentiment'].notna()
                        success_count = campaign_sent_filled.sum()
                        logging.info(f"[STEP 1/4] ✅ Sentiment: {success_count}/{campaign_row_count} ({success_count/campaign_row_count*100:.1f}%)")
            
            if generate_spokesperson:
                campaign_mainstream = mainstream_mask & campaign_df_mask
                campaign_mainstream_count = campaign_mainstream.sum()
                
                if campaign_mainstream_count > 0:
                    logging.info("\n" + "="*80)
                    logging.info(f"[STEP 2/4] SPOKESPERSON - {campaign} (Mainstream Only)")
                    logging.info("="*80)
                    
                    mainstream_process_mask = df['_is_master'] & campaign_mainstream & df['_eligible_for_topic']
                    df_mainstream = df[mainstream_process_mask].copy()
                    
                    logging.info(f"📊 Processing {len(df_mainstream)} mainstream master rows")
                    
                    if len(df_mainstream) > 0:
                        mainstream_batches = []
                        total_batches = math.ceil(len(df_mainstream) / MAINSTREAM_BATCH_SIZE)
                        
                        for batch_num in range(total_batches):
                            start_idx = batch_num * MAINSTREAM_BATCH_SIZE
                            end_idx = min(start_idx + MAINSTREAM_BATCH_SIZE, len(df_mainstream))
                            batch_df = df_mainstream.iloc[start_idx:end_idx].copy()
                            
                            progress_val = 0.45 + ((camp_idx-1) / len(selected_campaigns)) * 0.15 + (batch_num / total_batches / len(selected_campaigns)) * 0.15
                            
                            result_batch = process_batch_spokesperson(
                                batch_df, batch_num + 1, total_batches,
                                title_col, content_col, campaign_tracker,
                                campaign, campaign_progress, progress
                            )
                            
                            mainstream_batches.append(result_batch)
                        
                        df_mainstream_processed = pd.concat(mainstream_batches, ignore_index=False)
                        
                        for idx in df_mainstream_processed.index:
                            if 'New Spokesperson' in df_mainstream_processed.columns:
                                df.at[idx, 'New Spokesperson'] = df_mainstream_processed.at[idx, 'New Spokesperson']
                        
                        logging.info("📋 Copying spokesperson to duplicate rows...")
                        for hash_val in df[campaign_mainstream]['_dedup_hash'].unique():
                            group = df[(df['_dedup_hash'] == hash_val) & campaign_mainstream]
                            if len(group) > 1:
                                master_idx = group[group['_is_master']].index[0]
                                duplicate_indices = group[~group['_is_master']].index
                                
                                for dup_idx in duplicate_indices:
                                    df.at[dup_idx, 'New Spokesperson'] = df.at[master_idx, 'New Spokesperson']
                        
                        spokes_filled = df[campaign_mainstream]['New Spokesperson'].notna() & \
                                       (df[campaign_mainstream]['New Spokesperson'].astype(str).str.strip() != '')
                        success_count = spokes_filled.sum()
                        
                        logging.info(f"[STEP 2/4] ✅ Spokesperson: {success_count}/{campaign_mainstream_count} ({success_count/campaign_mainstream_count*100:.1f}%)")
                else:
                    logging.info(f"\n[STEP 2/4] ⚠️ No mainstream content in campaign '{campaign}', skipping spokesperson")
            
            if generate_topic:
                campaign_sub_filled = df[campaign_df_mask]['Sub Topic'].notna() & \
                                     (df[campaign_df_mask]['Sub Topic'].astype(str).str.strip() != '')
                success_rate = campaign_sub_filled.sum() / campaign_row_count
                
                retry_count = 0
                
                while retry_count < MAX_RETRIES and success_rate < 1.0:
                    retry_config = RETRY_CONFIGS[retry_count]
                    
                    logging.info("\n" + "="*80)
                    logging.info(f"[STEP 3/4] RETRY #{retry_count+1} - {campaign}")
                    logging.info(f"Strategy: Batch {retry_config['batch_size']} × {retry_config['truncate']} words")
                    logging.info(f"Current success: {success_rate:.1%}")
                    logging.info("="*80)
                    
                    unknown_mask = df['_is_master'] & df['_eligible_for_topic'] & campaign_df_mask & \
                                  ((df['Sub Topic'].isna()) | \
                                   (df['Sub Topic'].astype(str).str.strip() == '') | \
                                   (df['Sub Topic'].apply(lambda x: is_invalid_value(str(x)))))
                    
                    df_unknown = df[unknown_mask].copy()
                    unknown_count = len(df_unknown)
                    
                    if unknown_count == 0:
                        logging.info(f"[STEP 3/4] ✅ All sub topics extracted, skipping remaining retries")
                        break
                    
                    logging.info(f"[STEP 3/4] Found {unknown_count} rows still missing sub topics")
                    
                    retry_batch_size = min(unknown_count, retry_config['batch_size'])
                    retry_truncation = retry_config['truncate']
                    
                    retry_batches = []
                    total_batches = math.ceil(unknown_count / retry_batch_size)
                    
                    for batch_num in range(total_batches):
                        start_idx = batch_num * retry_batch_size
                        end_idx = min(start_idx + retry_batch_size, unknown_count)
                        batch_df = df_unknown.iloc[start_idx:end_idx].copy()
                        
                        progress_val = 0.65 + ((camp_idx-1) / len(selected_campaigns)) * 0.20 + (batch_num / total_batches / len(selected_campaigns)) * 0.20
                        
                        result_batch = retry_sub_topic_batch(
                            batch_df, title_col, content_col, language, 
                            campaign_tracker, retry_batch_size, retry_truncation, retry_count + 1,
                            campaign_progress, progress
                        )
                        
                        retry_batches.append(result_batch)
                    
                    df_unknown = pd.concat(retry_batches, ignore_index=False)
                    
                    for idx in df_unknown.index:
                        df.at[idx, 'Sub Topic'] = df_unknown.at[idx, 'Sub Topic']
                    
                    logging.info("📋 Copying retried results to duplicate rows...")
                    for idx in df_unknown.index:
                        hash_val = df.at[idx, '_dedup_hash']
                        duplicate_indices = df[(df['_dedup_hash'] == hash_val) & (~df['_is_master']) & campaign_df_mask].index
                        
                        for dup_idx in duplicate_indices:
                            df.at[dup_idx, 'Sub Topic'] = df.at[idx, 'Sub Topic']
                    
                    campaign_sub_filled = df[campaign_df_mask]['Sub Topic'].notna() & \
                                         (df[campaign_df_mask]['Sub Topic'].astype(str).str.strip() != '')
                    success_rate = campaign_sub_filled.sum() / campaign_row_count
                    
                    logging.info(f"[STEP 3/4] ✅ Sub Topic after retry #{retry_count+1}: {campaign_sub_filled.sum()}/{campaign_row_count} ({success_rate:.1%})")
                    
                    retry_count += 1
                
                still_empty_mask = campaign_df_mask & \
                                  ((df['Sub Topic'].isna()) | \
                                   (df['Sub Topic'].astype(str).str.strip() == ''))
                
                if still_empty_mask.sum() > 0:
                    logging.info(f"[STEP 3/4] Applying fallback keyword extraction to {still_empty_mask.sum()} rows...")
                    
                    for idx in df[still_empty_mask].index:
                        row = df.loc[idx]
                        combined = combine_title_content_row(row, title_col, content_col)
                        combined = clean_content_for_analysis(combined)
                        fallback_topic = extract_keywords_fallback(combined, output_language=language)
                        
                        if fallback_topic != GENERIC_PLACEHOLDERS.get(language, "Media Content Topic"):
                            fallback_topic = validate_and_normalize_subtopic(fallback_topic, language)
                            if fallback_topic:
                                df.at[idx, 'Sub Topic'] = fallback_topic
            
            campaign_duration = time.time() - campaign_start_time
            campaign_cost = campaign_tracker.get_cost(MODEL_NAME)
            
            per_campaign_stats[campaign] = {
                'rows': int(campaign_row_count),
                'duration_sec': float(campaign_duration),
                'cost_usd': float(campaign_cost),
                'api_calls': int(campaign_tracker.api_calls)
            }
            
            for k, v in campaign_tracker.get_summary(MODEL_NAME).items():
                if k not in ['step_stats']:
                    tracker.__dict__[k] = tracker.__dict__.get(k, 0) + (v if isinstance(v, (int, float)) else 0)
        
        if generate_topic:
            logging.info("\n" + "="*80)
            logging.info("[STEP 4/4] NORMALIZATION (BATCHED 500 + CONSOLIDATION)")
            logging.info(f"Strategy: Per 100 sub topics = 5 topics (ratio 20:1)")
            logging.info("="*80)
            
            progress(0.85, desc="[STEP 4/4] Normalizing Topics...")
            
            results = normalize_topics_v13_batched(
                df,
                selected_campaigns,
                campaign_col='Campaigns',
                engagement_col='Engagement',
                language=language,
                token_tracker=tracker,
                progress=progress
            )
            
            for campaign, result in results.items():
                mapping = result['mapping']
                campaign_mask = df['Campaigns'] == campaign
                
                for idx in df[campaign_mask].index:
                    sub_topic_val = df.at[idx, 'Sub Topic']
                    if sub_topic_val and str(sub_topic_val).strip() and not is_invalid_value(str(sub_topic_val)):
                        mapped_topic = mapping.get(sub_topic_val, '')
                        if mapped_topic:
                            df.at[idx, 'Topic'] = mapped_topic
                
                if 'stats' in result and campaign in per_campaign_stats:
                    per_campaign_stats[campaign]['unique_sub_topics'] = result['stats']['original_subtopics']
                    per_campaign_stats[campaign]['final_topics'] = result['stats']['final_topics']
                    per_campaign_stats[campaign]['grouping_efficiency'] = f"{result['stats']['reduction_rate']:.1f}%"
            
            topic_filled = df['Topic'].notna() & (df['Topic'].astype(str).str.strip() != '')
            topic_success = topic_filled.sum()
            tracker.add_step_stat("Topic", topic_success, len(df))
            
            logging.info(f"[STEP 4/4] ✅ Topic: {topic_success}/{len(df)} ({topic_success/len(df)*100:.1f}%)")
        
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
            if col in df.columns:
                cols.insert(insert_idx + i, col)
        
        df = df[cols]
        
        final_row_count = len(df)
        
        logging.info(f"✅ Rows after processing: {final_row_count}")
        
        progress(0.98, desc="Saving...")
        
        original_filename = Path(file_path).stem
        output_filename = f"{original_filename}_phase2.xlsx"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name="Processed")
            
            duration = time.time() - start_time
            token_summary = tracker.get_summary(MODEL_NAME)
            
            # FIX: Extract nested f-string
            retry_configs_str = ', '.join([f"B{c['batch_size']}×{c['truncate']}w" for c in RETRY_CONFIGS])
            
            meta_data = [
                {"key": "processed_at", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                {"key": "version", "value": "v13.0 - Aggressive Quality + Batched Normalization"},
                {"key": "model", "value": MODEL_NAME},
                {"key": "output_language", "value": f"{language} ({LANGUAGE_CONFIGS[language]['name']})"},
                {"key": "extraction_strategy", "value": f"Initial: B{INITIAL_BATCH_SIZE}×{INITIAL_TRUNCATE}w, Retry: {retry_configs_str}"},
                {"key": "normalization_strategy", "value": f"Batch {NORMALIZATION_BATCH_SIZE} + Consolidation, Ratio 5:100"},
                {"key": "sub_topic_range", "value": "3-8 words (specific context required)"},
                {"key": "topic_range", "value": "3-8 words"},
                {"key": "duration_sec", "value": f"{duration:.2f}"},
                {"key": "file_campaigns_total", "value": int(total_campaigns_in_file)},
                {"key": "selected_campaigns_count", "value": int(len(selected_campaigns))},
                {"key": "skipped_campaigns", "value": int(skipped_campaigns)},
                {"key": "selected_campaigns", "value": ", ".join(selected_campaigns)},
                {"key": "original_rows_in_file", "value": int(original_row_count)},
                {"key": "filtered_rows_selected", "value": int(filtered_row_count)},
                {"key": "final_rows", "value": int(final_row_count)},
                {"key": "deduplication_groups", "value": int(master_rows)},
                {"key": "duplicate_rows", "value": int(duplicate_rows)},
                {"key": "dedup_api_savings", "value": int(duplicate_rows)},
                {"key": "eligible_for_topic", "value": int(total_eligible)},
                {"key": "skipped_short_content", "value": int(total_skipped)},
                {"key": "prefilter_api_savings", "value": int(total_skipped)},
                {"key": "mainstream_rows", "value": int(mainstream_count)},
                {"key": "social_rows", "value": int(social_count)},
                {"key": "type_filter_enabled", "value": "Yes" if has_type_column else "No"},
                {"key": "input_tokens", "value": int(token_summary["input_tokens"])},
                {"key": "output_tokens", "value": int(token_summary["output_tokens"])},
                {"key": "total_tokens", "value": int(token_summary["total_tokens"])},
                {"key": "api_calls", "value": int(token_summary["api_calls"])},
                {"key": "cost_usd", "value": f"${token_summary['estimated_cost_usd']:.6f}"},
            ]
            
            for step_name, step_data in token_summary['step_stats'].items():
                meta_data.append({
                    "key": f"success_rate_{step_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}",
                    "value": f"{step_data['success']}/{step_data['total']} ({step_data['rate']:.1f}%)"
                })
            
            meta = pd.DataFrame(meta_data)
            meta.to_excel(writer, index=False, sheet_name="Meta")
            
            if per_campaign_stats:
                campaign_stats_data = []
                for campaign, stats in per_campaign_stats.items():
                    campaign_stats_data.append({
                        'Campaign': campaign,
                        'Rows': stats.get('rows', 0),
                        'Unique Sub Topics': stats.get('unique_sub_topics', 0),
                        'Final Topics': stats.get('final_topics', 0),
                        'Grouping Efficiency': stats.get('grouping_efficiency', 'N/A'),
                        'Duration (sec)': f"{stats.get('duration_sec', 0):.2f}",
                        'API Calls': stats.get('api_calls', 0),
                        'Cost (USD)': f"${stats.get('cost_usd', 0):.6f}"
                    })
                
                campaign_stats_df = pd.DataFrame(campaign_stats_data)
                campaign_stats_df.to_excel(writer, index=False, sheet_name="Campaign Stats")
        
        stats = {
            "file_campaigns_total": int(total_campaigns_in_file),
            "selected_campaigns": selected_campaigns,
            "selected_count": int(len(selected_campaigns)),
            "skipped_campaigns": int(skipped_campaigns),
            "original_rows_in_file": int(original_row_count),
            "filtered_rows": int(filtered_row_count),
            "final_rows": int(final_row_count),
            "extraction_strategy": f"Aggressive Quality - Initial B{INITIAL_BATCH_SIZE}×{INITIAL_TRUNCATE}w",
            "normalization_strategy": f"Batched {NORMALIZATION_BATCH_SIZE} + Consolidation",
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
            "type_filter": {
                "enabled": has_type_column,
                "comment_count": int(comment_count) if has_type_column else 0,
                "reply_count": int(reply_count) if has_type_column else 0
            },
            "language": {
                "output": f"{language} ({LANGUAGE_CONFIGS[language]['name']})"
            },
            "per_campaign_stats": per_campaign_stats,
            "duration": f"{duration:.2f}s",
            "cost": f"${token_summary['estimated_cost_usd']:.6f}",
            "success_rates": token_summary['step_stats']
        }
        
        logging.info("\n" + "="*80)
        logging.info("✅ PROCESSING COMPLETE")
        logging.info("="*80)
        logging.info(f"Campaigns: {len(selected_campaigns)}/{total_campaigns_in_file} selected")
        logging.info(f"Rows: {original_row_count} → {filtered_row_count} (selected) → {final_row_count} (final)")
        logging.info(f"Duration: {duration:.2f}s | Cost: ${token_summary['estimated_cost_usd']:.6f}")
        
        for step_name, step_data in token_summary['step_stats'].items():
            logging.info(f"{step_name}: {step_data['success']}/{step_data['total']} ({step_data['rate']:.1f}%)")
        
        progress(1.0, desc="Complete!")
        return output_path, stats, None
        
    except Exception as e:
        logging.error(f"[ERROR] {str(e)}", exc_info=True)
        return None, {}, f"❌ Error: {str(e)}"

def create_gradio_interface():
    with gr.Blocks(title="Insights Generator v13.0", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📊 Insights Generator v13.0 - Aggressive Quality + Specific Sub Topics")
        gr.Markdown("""
        **New in v13.0:**
        - 🎯 **Aggressive Quality** - Batch 20×200w initial, 15×250w, 10×300w, 5×400w retry
        - 🔍 **Specific Sub Topics** - 3-8 words with mandatory context (events/issues/activities)
        - 📊 **Batched Normalization** - Process 500 at a time, consolidate if >500
        - 🎲 **Dynamic Topics** - 5 topics per 100 sub topics (ratio 20:1)
        - 🗂️ **Type Filter** - Exclude Comment/Reply from topic extraction
        - 📈 **Strict Prompts** - Focus on concrete, actionable categories
        - 💾 **Output:** `{original_filename}_phase2.xlsx`
        
        **Requirements:**
        - Excel with: Channel, Campaigns, Title/Content
        - Optional: Engagement, Type columns
        """)
        
        file_state = gr.State(None)
        
        with gr.Row():
            with gr.Column(scale=2):
                file_input = gr.File(label="📁 Upload Excel", file_types=[".xlsx"], type="filepath")
                sheet_selector = gr.Dropdown(label="📊 Sheet", choices=[], interactive=True)
                
                campaign_selector = gr.CheckboxGroup(
                    label="📋 Select Campaigns",
                    choices=[],
                    value=[],
                    interactive=True,
                    visible=False
                )
                
                def load_sheets(file_path):
                    if file_path:
                        try:
                            xl = pd.ExcelFile(file_path)
                            return (
                                gr.Dropdown(choices=xl.sheet_names, value=xl.sheet_names[0]),
                                file_path,
                                gr.CheckboxGroup(visible=False, choices=[], value=[])
                            )
                        except Exception as e:
                            return (
                                gr.Dropdown(choices=[]),
                                None,
                                gr.CheckboxGroup(visible=False, choices=[], value=[])
                            )
                    return (
                        gr.Dropdown(choices=[]),
                        None,
                        gr.CheckboxGroup(visible=False, choices=[], value=[])
                    )
                
                file_input.change(
                    load_sheets, 
                    inputs=file_input, 
                    outputs=[sheet_selector, file_state, campaign_selector]
                )
                
                def load_campaigns(file_path, sheet_name):
                    if not file_path or not sheet_name:
                        return gr.CheckboxGroup(visible=False, choices=[], value=[])
                    
                    campaigns = load_campaigns_from_file(file_path, sheet_name)
                    
                    if campaigns:
                        labels = [c['label'] for c in campaigns]
                        
                        return gr.CheckboxGroup(
                            visible=True,
                            choices=labels,
                            value=labels,
                            label=f"📋 Select Campaigns ({len(campaigns)} total, all selected by default)"
                        )
                    else:
                        return gr.CheckboxGroup(visible=False, choices=[], value=[])
                
                sheet_selector.change(
                    load_campaigns,
                    inputs=[file_state, sheet_selector],
                    outputs=[campaign_selector]
                )
            
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
                gen_topic = gr.Checkbox(label="📌 Topic & Sub Topic (batched normalization)", value=False)
                gen_sentiment = gr.Checkbox(label="😊 Sentiment (all channels)", value=False)
                gen_spokesperson = gr.Checkbox(label="🎤 Spokesperson (mainstream only)", value=False)
        
        validation_error = gr.Markdown("", visible=True)
        
        process_btn = gr.Button("🚀 Process", variant="primary", size="lg", interactive=False)
        
        with gr.Row():
            with gr.Column():
                output_file = gr.File(label="📥 Download")
            with gr.Column():
                stats_output = gr.Textbox(label="📊 Stats", lines=20, interactive=False)
        
        error_output = gr.Textbox(label="⚠️ Status", lines=3, visible=True)
        
        def validate_all(topic, sentiment, spokesperson, campaigns):
            errors = []
            
            if not any([topic, sentiment, spokesperson]):
                errors.append("⚠️ Please select at least one feature")
            
            if not campaigns or len(campaigns) == 0:
                errors.append("⚠️ Please select at least one campaign")
            
            if errors:
                return gr.Button(interactive=False), gr.Markdown("<br>".join(errors), visible=True)
            else:
                return gr.Button(interactive=True), gr.Markdown("", visible=False)
        
        for component in [gen_topic, gen_sentiment, gen_spokesperson, campaign_selector]:
            component.change(
                validate_all,
                inputs=[gen_topic, gen_sentiment, gen_spokesperson, campaign_selector],
                outputs=[process_btn, validation_error]
            )
        
        def process_wrapper(file_path, sheet_name, campaigns_with_counts, language, topic, sentiment, spokesperson, conf, progress=gr.Progress()):
            try:
                if not file_path:
                    return None, "", "❌ Please upload an Excel file"
                
                if not sheet_name:
                    return None, "", "❌ Please select a sheet"
                
                if not campaigns_with_counts or len(campaigns_with_counts) == 0:
                    return None, "", "❌ Please select at least one campaign"
                
                if not any([topic, sentiment, spokesperson]):
                    return None, "", "❌ Please select at least one feature"
                
                selected_campaigns = []
                for label in campaigns_with_counts:
                    campaign_name = label.split(' (')[0] if ' (' in label else label
                    selected_campaigns.append(campaign_name)
                
                result_path, stats, error = process_file(
                    file_path, sheet_name, selected_campaigns, language, topic, sentiment, spokesperson, conf, progress
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
            inputs=[file_state, sheet_selector, campaign_selector, language_selector, gen_topic, gen_sentiment, gen_spokesperson, conf_threshold],
            outputs=[output_file, stats_output, error_output]
        )
    
    return app

if __name__ == "__main__":
    app = create_gradio_interface()
    app.queue(max_size=10, default_concurrency_limit=4)
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)