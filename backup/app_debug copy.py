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
# ==== END PATCH ====



# ====== CONFIGURATION ======
load_dotenv(dotenv_path=".secretcontainer/.env")

# Validate API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ ERROR: OPENAI_API_KEY not found in .env file!")
    print("Please create .secretcontainer/.env with OPENAI_API_KEY=your_key")
    raise ValueError("Missing OPENAI_API_KEY")

print(f"✅ OpenAI API Key loaded: {OPENAI_API_KEY[:8]}...{OPENAI_API_KEY[-4:]}")

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

BATCH_SIZE = 50
MAX_RETRIES = 3
TRUNCATE_WORDS = 200

ELIGIBLE_CHANNELS = ['online media', 'printmedia', 'tv', 'newspaper', 'printed']

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

# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# ====== API VALIDATION ======
def validate_openai_api():
    """Test OpenAI API connection"""
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

# Test API on startup
api_ok, api_msg = validate_openai_api()
if not api_ok:
    print(f"\n{'='*60}")
    print(f"⚠️  WARNING: OpenAI API Test Failed!")
    print(f"{'='*60}")
    print(f"Error: {api_msg}")
    print(f"\nPossible causes:")
    print(f"1. Invalid API key")
    print(f"2. Network connectivity issue")
    print(f"3. OpenAI service is down")
    print(f"4. Model '{MODEL_NAME}' not available")
    print(f"{'='*60}\n")

# ====== TOKEN TRACKER ======
class TokenTracker:
    def __init__(self):
        self.total_input = 0
        self.total_output = 0
        self.api_calls = 0
    
    def add(self, input_tokens, output_tokens):
        self.total_input += input_tokens
        self.total_output += output_tokens
        self.api_calls += 1
    
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
            "estimated_cost_usd": cost
        }

# ====== HELPER FUNCTIONS ======
def safe_text(x):
    return "" if pd.isna(x) else str(x).strip()

def truncate_to_first_n_words(text: str, n: int = TRUNCATE_WORDS) -> str:
    words = text.split()
    return " ".join(words[:n])

def extract_json_from_response(s: str):
    """Extract JSON from response with markdown cleanup"""
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
    """Validate required columns"""
    required = {
        'content': ['Content', 'Konten', 'Isi'],
        'title': ['Title', 'Judul'],
        'channel': ['Channel', 'Media type', 'Media Type']
    }
    
    title_col = get_col(df, required['title'])
    content_col = get_col(df, required['content'])
    channel_col = get_col(df, required['channel'])
    
    if not title_col and not content_col:
        return False, "❌ Kolom 'Title' atau 'Content' harus ada!"
    
    return True, ""

# ====== OPENAI API CALL ======
def chat_create(model, messages, token_tracker=None, max_retries=MAX_RETRIES):
    """API call with retry logic and detailed error logging"""
    for attempt in range(max_retries):
        try:
            logging.info(f"[API CALL] Attempt {attempt + 1}/{max_retries} - Model: {model}")
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2
            )
            
            # Track tokens
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
            logging.error(f"[API CALL] Error Type: {error_type}")
            logging.error(f"[API CALL] Error Message: {error_msg}")
            
            # Print ke console untuk debugging
            print(f"\n❌ API Error (Attempt {attempt + 1}/{max_retries}):")
            print(f"   Type: {error_type}")
            print(f"   Message: {error_msg}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logging.info(f"[API CALL] Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                logging.error(f"[API CALL] ❌ ALL RETRIES FAILED")
                raise
    
    return None

# ====== BATCH PROCESSING ======
def process_batch_all_features(
    batch_df: pd.DataFrame,
    batch_num: int,
    total_batches: int,
    title_col: str,
    content_col: str,
    channel_col: str,
    generate_topic: bool,
    generate_sentiment: bool,
    generate_summary: bool,
    generate_spokesperson: bool,
    conf_threshold: int,
    token_tracker: TokenTracker,
    progress=gr.Progress()
) -> pd.DataFrame:
    """Process 1 batch with all selected features"""
    batch_size = len(batch_df)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"[BATCH {batch_num}/{total_batches}] Starting - {batch_size} rows")
    logging.info(f"{'='*60}")
    
    # Prepare batch data
    batch_data = []
    for idx, row in batch_df.iterrows():
        combined = combine_title_content_row(row, title_col, content_col)
        truncated = truncate_to_first_n_words(combined, TRUNCATE_WORDS)
        
        channel_lower = str(row.get(channel_col, '')).lower().strip() if channel_col else ''
        is_eligible = channel_lower in ELIGIBLE_CHANNELS
        
        batch_data.append({
            "row": idx,
            "text": truncated,
            "title": safe_text(row.get(title_col, '')) if title_col else '',
            "content": safe_text(row.get(content_col, '')) if content_col else '',
            "channel": channel_lower,
            "is_eligible": is_eligible
        })
    
    # Build dynamic prompt
    tasks = []
    output_fields = []
    
    if generate_topic:
        tasks.append("""- sub_topic: 4-15 kata menjelaskan topik spesifik dari berita (gunakan bahasa dari teks)""")
        output_fields.append('"sub_topic":"..."')
    
    if generate_sentiment:
        tasks.append("""- sentiment: "positive" atau "negative" atau "neutral"
- confidence: 0-100 (tingkat keyakinan Anda)
  
  Rules sentiment:
  * positive: emosi positif jelas, pujian, kepuasan
  * negative: emosi negatif jelas, keluhan, kekecewaan
  * neutral: tidak ada emosi jelas atau ambigu""")
        output_fields.extend(['"sentiment":"..."', '"confidence":85'])
    
    if generate_summary:
        tasks.append("""- new_summary: 4-20 kata ringkasan isi berita
  * HANYA untuk channel: online media/printmedia/tv/newspaper/printed
  * Channel lain → kosongkan dengan ""
  * Jangan isi "nan", "none", atau "N/A" """)
        output_fields.append('"new_summary":"..."')
    
    if generate_spokesperson:
        tasks.append("""- spokesperson: nama narasumber yang dikutip
  * FORMAT WAJIB: "Nama Lengkap (Jabatan/Posisi)"
  * Multiple: "Nama1 (Jabatan1), Nama2 (Jabatan2)"
  * HANYA untuk channel: online media/printmedia/tv/newspaper/printed
  * Channel lain → kosongkan dengan ""
  * Jika tidak ada yang dikutip → ""
  * Jangan isi "nan", "none", atau "unknown" """)
        output_fields.append('"spokesperson":"..."')
    
    task_list = "\n".join(tasks)
    output_example = ",\n  ".join(output_fields)
    
    prompt = f"""Analisis {batch_size} berita berikut. Return ONLY valid JSON array.

TASKS:
{task_list}

Data:
{json.dumps(batch_data, indent=2, ensure_ascii=False)}

Return format (exactly {batch_size} objects):
[
  {{
    "row": <index>,
    {output_example}
  }}
]

CRITICAL:
- Valid JSON dengan double quotes
- Kosongkan dengan "" bukan "nan"/"none"/"null"
- Process ALL {batch_size} rows
- Spokesperson format: "Nama (Jabatan)"

JSON array:"""
    
    # API call with retry
    try:
        progress(0.5, desc=f"🔄 Batch {batch_num}/{total_batches} - Calling GPT...")
        
        logging.info(f"[BATCH {batch_num}] Sending API request...")
        
        response = chat_create(
            MODEL_NAME,
            [
                {"role": "system", "content": "You are a JSON generator. Return valid JSON array only."},
                {"role": "user", "content": prompt}
            ],
            token_tracker=token_tracker
        )
        
        if not response:
            raise Exception("API call failed after retries")
        
        raw = response.choices[0].message.content.strip()
        logging.info(f"[BATCH {batch_num}] Response length: {len(raw)} chars")
        
        data = extract_json_from_response(raw)
        
        if not isinstance(data, list):
            logging.error(f"[BATCH {batch_num}] Response is not a JSON array")
            raise ValueError("Response is not a JSON array")
        
        logging.info(f"[BATCH {batch_num}] Parsed {len(data)} items from JSON")
        
        # Map results back to DataFrame
        result_dict = {item['row']: item for item in data}
        
        # Initialize result columns
        if generate_topic:
            batch_df['Sub Topic'] = 'unknown'
        if generate_sentiment:
            batch_df['New Sentiment'] = 'neutral'
            batch_df['New Sentiment Level'] = 0
        if generate_summary:
            batch_df['New Summary'] = ''
        if generate_spokesperson:
            batch_df['New Spokesperson'] = ''
        
        # Fill results
        invalid_values = ['nan', 'none', 'null', 'n/a', 'na', 'unknown', 'tidak ada', '-']
        
        for idx in batch_df.index:
            if idx in result_dict:
                item = result_dict[idx]
                
                if generate_topic:
                    sub_topic = str(item.get('sub_topic', 'unknown')).strip()
                    words = sub_topic.split()
                    if len(words) >= 4 and len(words) <= 15:
                        batch_df.at[idx, 'Sub Topic'] = sub_topic
                
                if generate_sentiment:
                    sentiment = str(item.get('sentiment', 'neutral')).lower().strip()
                    if sentiment in ['positive', 'negative', 'neutral']:
                        conf = int(item.get('confidence', 0))
                        conf = max(0, min(100, conf))
                        
                        if conf < conf_threshold and sentiment != 'neutral':
                            sentiment = 'neutral'
                        
                        batch_df.at[idx, 'New Sentiment'] = sentiment
                        batch_df.at[idx, 'New Sentiment Level'] = conf
                
                if generate_summary:
                    summary_val = str(item.get('new_summary', '')).strip()
                    if summary_val.lower() not in invalid_values:
                        batch_df.at[idx, 'New Summary'] = summary_val
                
                if generate_spokesperson:
                    spokesperson_val = str(item.get('spokesperson', '')).strip()
                    if spokesperson_val.lower() not in invalid_values:
                        batch_df.at[idx, 'New Spokesperson'] = spokesperson_val
        
        logging.info(f"[BATCH {batch_num}/{total_batches}] ✅ SUCCESS - {batch_size} rows")
        return batch_df
        
    except Exception as e:
        logging.error(f"[BATCH {batch_num}/{total_batches}] ❌ FAILED after retries: {str(e)}")
        
        # Return with empty/default values
        if generate_topic:
            batch_df['Sub Topic'] = 'unknown'
        if generate_sentiment:
            batch_df['New Sentiment'] = 'neutral'
            batch_df['New Sentiment Level'] = 0
        if generate_summary:
            batch_df['New Summary'] = ''
        if generate_spokesperson:
            batch_df['New Spokesperson'] = ''
        
        return batch_df

# ====== RETRY B: EMPTY RESULTS ======
def retry_empty_spokesperson(
    df: pd.DataFrame,
    title_col: str,
    content_col: str,
    channel_col: str,
    token_tracker: TokenTracker,
    progress=gr.Progress()
) -> pd.DataFrame:
    """Retry B: Re-process rows with empty spokesperson"""
    
    eligible_mask = df[channel_col].str.lower().isin(ELIGIBLE_CHANNELS) if channel_col else pd.Series([False] * len(df))
    empty_mask = (df['New Spokesperson'].isna()) | (df['New Spokesperson'].astype(str).str.strip() == '')
    
    retry_mask = eligible_mask & empty_mask
    retry_count = retry_mask.sum()
    
    if retry_count == 0:
        return df
    
    logging.info(f"[RETRY B SPOKESPERSON] {retry_count} rows to retry")
    progress(0.92, desc=f"🔄 Retry {retry_count} empty Spokesperson...")
    
    retry_indices = df[retry_mask].index.tolist()
    retry_df = df.loc[retry_indices].copy()
    
    for i in range(0, len(retry_df), BATCH_SIZE):
        retry_batch = retry_df.iloc[i:i+BATCH_SIZE]
        
        batch_data = []
        for idx, row in retry_batch.iterrows():
            combined = combine_title_content_row(row, title_col, content_col)
            truncated = truncate_to_first_n_words(combined, TRUNCATE_WORDS)
            
            batch_data.append({
                "row": idx,
                "text": truncated
            })
        
        prompt = f"""Ekstrak SPOKESPERSON dari {len(batch_data)} berita.

TUGAS: Identifikasi nama orang yang DIKUTIP LANGSUNG.

FORMAT WAJIB: "Nama Lengkap (Jabatan/Posisi)"
Multiple: "Nama1 (Jabatan1), Nama2 (Jabatan2)"

Data:
{json.dumps(batch_data, indent=2, ensure_ascii=False)}

Return JSON:
[
  {{"row": <index>, "spokesperson": "Nama (Jabatan)" atau ""}}
]

Jangan isi "nan"/"none"!

JSON:"""
        
        try:
            response = chat_create(
                MODEL_NAME,
                [
                    {"role": "system", "content": "Return JSON array only."},
                    {"role": "user", "content": prompt}
                ],
                token_tracker=token_tracker,
                max_retries=1
            )
            
            if response:
                raw = response.choices[0].message.content.strip()
                data = extract_json_from_response(raw)
                
                if isinstance(data, list):
                    result_dict = {item['row']: item for item in data}
                    
                    invalid_values = ['nan', 'none', 'null', 'n/a', 'na', 'unknown', 'tidak ada', '-']
                    
                    for idx in retry_batch.index:
                        if idx in result_dict:
                            spokesperson_val = str(result_dict[idx].get('spokesperson', '')).strip()
                            if spokesperson_val and spokesperson_val.lower() not in invalid_values:
                                df.at[idx, 'New Spokesperson'] = spokesperson_val
                                logging.info(f"[RETRY SUCCESS] Row {idx}: {spokesperson_val}")
        
        except Exception as e:
            logging.error(f"[RETRY B ERROR] {str(e)}")
            continue
    
    return df

def retry_empty_summary(
    df: pd.DataFrame,
    title_col: str,
    content_col: str,
    channel_col: str,
    token_tracker: TokenTracker,
    progress=gr.Progress()
) -> pd.DataFrame:
    """Retry B: Re-process rows with empty summary"""
    
    eligible_mask = df[channel_col].str.lower().isin(ELIGIBLE_CHANNELS) if channel_col else pd.Series([False] * len(df))
    empty_mask = (df['New Summary'].isna()) | (df['New Summary'].astype(str).str.strip() == '')
    
    retry_mask = eligible_mask & empty_mask
    retry_count = retry_mask.sum()
    
    if retry_count == 0:
        return df
    
    logging.info(f"[RETRY B SUMMARY] {retry_count} rows to retry")
    progress(0.94, desc=f"🔄 Retry {retry_count} empty Summary...")
    
    retry_indices = df[retry_mask].index.tolist()
    retry_df = df.loc[retry_indices].copy()
    
    for i in range(0, len(retry_df), BATCH_SIZE):
        retry_batch = retry_df.iloc[i:i+BATCH_SIZE]
        
        batch_data = []
        for idx, row in retry_batch.iterrows():
            combined = combine_title_content_row(row, title_col, content_col)
            truncated = truncate_to_first_n_words(combined, TRUNCATE_WORDS)
            
            batch_data.append({
                "row": idx,
                "text": truncated
            })
        
        prompt = f"""Buat ringkasan singkat dari {len(batch_data)} berita.

TUGAS: Ringkasan 4-20 kata menjelaskan ISI berita.

Data:
{json.dumps(batch_data, indent=2, ensure_ascii=False)}

Return JSON:
[
  {{"row": <index>, "summary": "ringkasan 4-20 kata"}}
]

JSON:"""
        
        try:
            response = chat_create(
                MODEL_NAME,
                [
                    {"role": "system", "content": "Return JSON array only."},
                    {"role": "user", "content": prompt}
                ],
                token_tracker=token_tracker,
                max_retries=1
            )
            
            if response:
                raw = response.choices[0].message.content.strip()
                data = extract_json_from_response(raw)
                
                if isinstance(data, list):
                    result_dict = {item['row']: item for item in data}
                    
                    invalid_values = ['nan', 'none', 'null', 'n/a', 'na', 'unknown', 'tidak ada', '-']
                    
                    for idx in retry_batch.index:
                        if idx in result_dict:
                            summary_val = str(result_dict[idx].get('summary', '')).strip()
                            if summary_val and summary_val.lower() not in invalid_values:
                                df.at[idx, 'New Summary'] = summary_val
        
        except Exception as e:
            logging.error(f"[RETRY B SUMMARY ERROR] {str(e)}")
            continue
    
    return df

# ====== NORMALIZATION ======
def normalize_sub_topics_to_topics(
    unique_sub_topics: list,
    token_tracker: TokenTracker,
    progress=gr.Progress()
) -> dict:
    """Normalize Sub Topics into broader Topic categories"""
    
    if len(unique_sub_topics) <= 1:
        return {st: st for st in unique_sub_topics}
    
    progress(0.96, desc="🔄 Normalizing Sub Topics to Topics...")
    
    joined = "\n".join(f"- {st}" for st in unique_sub_topics)
    
    prompt = f"""Kelompokkan sub-topik berikut ke dalam KATEGORI TOPIK yang lebih luas.

TUGAS:
1. Group sub-topik yang serupa/related
2. Buat 1 label TOPIC (kategori umum, 3-8 kata) untuk setiap group
3. Maksimal 10-15 TOPIC categories

CONTOH:
Sub Topics:
- menanam padi oleh pemerintah
- bertanam padi oleh gubernur
- pemerintah melakukan penanaman padi
- subsidi pupuk untuk petani
- bantuan alat pertanian

Output:
menanam padi oleh pemerintah → program pertanian pemerintah
bertanam padi oleh gubernur → program pertanian pemerintah
pemerintah melakukan penanaman padi → program pertanian pemerintah
subsidi pupuk untuk petani → bantuan pertanian
bantuan alat pertanian → bantuan pertanian

Sub Topics:
{joined}

Return format:
<Sub Topic> → <Topic Category>

Output:"""
    
    try:
        response = chat_create(
            MODEL_NAME,
            [
                {"role": "system", "content": "You are a topic categorization expert."},
                {"role": "user", "content": prompt}
            ],
            token_tracker=token_tracker
        )
        
        if not response:
            return {st: st for st in unique_sub_topics}
        
        output = response.choices[0].message.content
        logging.info(f"[NORMALIZE TOPICS] GPT Response:\n{output}")
        
        mapping = {}
        for line in output.splitlines():
            if "→" in line:
                parts = line.split("→")
                if len(parts) == 2:
                    original = parts[0].strip().lstrip("- ")
                    category = parts[1].strip().lstrip("- ")
                    if original and category:
                        mapping[original] = category
        
        for st in unique_sub_topics:
            if st not in mapping:
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
    """Normalize spokesperson names"""
    
    unique_spokespersons = [s for s in unique_spokespersons if s and s.strip()]
    
    if len(unique_spokespersons) <= 1:
        return {sp: sp for sp in unique_spokespersons}
    
    progress(0.98, desc="🔄 Normalizing Spokesperson...")
    
    joined = "\n".join(f"- {sp}" for sp in unique_spokespersons)
    
    prompt = f"""Normalisasi nama SPOKESPERSON yang merujuk ORANG YANG SAMA.

FORMAT STANDAR: "Nama Lengkap (Jabatan/Posisi)"

TUGAS:
1. Identifikasi nama yang merujuk orang yang sama
2. Gabungkan ke format standar
3. Jangan gabung nama orang berbeda

CONTOH:
Prabowo → Prabowo Subianto (Presiden Indonesia)
Presiden Prabowo → Prabowo Subianto (Presiden Indonesia)
Doddy Hanggodo → Doddy Hanggodo (Menteri PUPR)
Menteri PUPR Doddy → Doddy Hanggodo (Menteri PUPR)

Spokesperson:
{joined}

Return format:
<Original> → <Normalized>

Output:"""
    
    try:
        response = chat_create(
            MODEL_NAME,
            [
                {"role": "system", "content": "You are a name normalization expert."},
                {"role": "user", "content": prompt}
            ],
            token_tracker=token_tracker
        )
        
        if not response:
            return {sp: sp for sp in unique_spokespersons}
        
        output = response.choices[0].message.content
        logging.info(f"[NORMALIZE SPOKESPERSON] GPT Response:\n{output}")
        
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

# ====== MAIN PROCESSING FUNCTION ======
def process_file(
    file_path: str,
    sheet_name: str,
    generate_topic: bool,
    generate_sentiment: bool,
    generate_summary: bool,
    generate_spokesperson: bool,
    conf_threshold: int,
    progress=gr.Progress()
) -> tuple:
    """Main processing function"""
    
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
        channel_col = get_col(df, ["Channel", "Media type", "Media Type"])
        
        logging.info(f"[START] Processing {len(df)} rows with {BATCH_SIZE} batch size")
        logging.info(f"[CONFIG] Topic:{generate_topic} Sentiment:{generate_sentiment} Summary:{generate_summary} Spokesperson:{generate_spokesperson}")
        
        total_rows = len(df)
        total_batches = math.ceil(total_rows / BATCH_SIZE)
        
        tracker = TokenTracker()
        start_time = time.time()
        
        processed_batches = []
        failed_batches = []
        
        for batch_num in range(total_batches):
            start_idx = batch_num * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, total_rows)
            
            batch_df = df.iloc[start_idx:end_idx].copy()
            
            batch_progress = (batch_num / total_batches) * 0.85
            progress(batch_progress, desc=f"⚙️ Batch {batch_num + 1}/{total_batches} ({len(batch_df)} rows)...")
            
            result_batch = process_batch_all_features(
                batch_df,
                batch_num + 1,
                total_batches,
                title_col,
                content_col,
                channel_col,
                generate_topic,
                generate_sentiment,
                generate_summary,
                generate_spokesperson,
                conf_threshold,
                tracker,
                progress
            )
            
            if generate_topic and result_batch['Sub Topic'].eq('unknown').all():
                failed_batches.append(batch_num + 1)
            
            processed_batches.append(result_batch)
        
        df_processed = pd.concat(processed_batches, ignore_index=True)
        
        if generate_spokesperson:
            df_processed = retry_empty_spokesperson(
                df_processed,
                title_col,
                content_col,
                channel_col,
                tracker,
                progress
            )
        
        if generate_summary:
            df_processed = retry_empty_summary(
                df_processed,
                title_col,
                content_col,
                channel_col,
                tracker,
                progress
            )
        
        if generate_topic:
            progress(0.95, desc="🔄 Normalizing Topics...")
            
            sub_topics = df_processed['Sub Topic'].dropna()
            sub_topics = sub_topics[sub_topics != 'unknown']
            unique_sub_topics = sorted(sub_topics.unique().tolist())
            
            if unique_sub_topics:
                topic_mapping = normalize_sub_topics_to_topics(unique_sub_topics, tracker, progress)
                df_processed['Topic'] = df_processed['Sub Topic'].apply(
                    lambda x: topic_mapping.get(x, x) if x != 'unknown' else 'unknown'
                )
            else:
                df_processed['Topic'] = 'unknown'
        
        if generate_spokesperson:
            spokespersons = df_processed['New Spokesperson'].dropna()
            spokespersons = spokespersons[spokespersons.astype(str).str.strip() != '']
            unique_spokespersons = sorted(spokespersons.unique().tolist())
            
            if unique_spokespersons:
                spokesperson_mapping = normalize_spokesperson(unique_spokespersons, tracker, progress)
                df_processed['New Spokesperson'] = df_processed['New Spokesperson'].apply(
                    lambda x: spokesperson_mapping.get(x, x) if pd.notna(x) and str(x).strip() else x
                )
        
        cols = df_processed.columns.tolist()
        new_cols = []
        
        if generate_topic:
            new_cols.extend(['Topic', 'Sub Topic'])
        if generate_sentiment:
            new_cols.extend(['New Sentiment', 'New Sentiment Level'])
        if generate_summary:
            new_cols.append('New Summary')
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
        
        progress(0.99, desc="💾 Saving file...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"processed_{timestamp}.xlsx"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            df_processed.to_excel(writer, index=False, sheet_name="Processed")
            
            duration = time.time() - start_time
            token_summary = tracker.get_summary(MODEL_NAME)
            
            operations = []
            if generate_topic: operations.append("Topic & Sub Topic")
            if generate_sentiment: operations.append("Sentiment")
            if generate_summary: operations.append("Summary")
            if generate_spokesperson: operations.append("Spokesperson")
            
            meta = pd.DataFrame([
                {"key": "processed_at", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                {"key": "model", "value": MODEL_NAME},
                {"key": "duration_sec", "value": f"{duration:.2f}"},
                {"key": "total_rows", "value": len(df_processed)},
                {"key": "batch_size", "value": BATCH_SIZE},
                {"key": "operations", "value": ", ".join(operations)},
                {"key": "conf_threshold", "value": conf_threshold},
                {"key": "input_tokens", "value": token_summary["input_tokens"]},
                {"key": "output_tokens", "value": token_summary["output_tokens"]},
                {"key": "total_tokens", "value": token_summary["total_tokens"]},
                {"key": "api_calls", "value": token_summary["api_calls"]},
                {"key": "cost_usd", "value": f"${token_summary['estimated_cost_usd']:.6f}"},
                {"key": "failed_batches", "value": ", ".join(map(str, failed_batches)) if failed_batches else "None"},
            ])
            meta.to_excel(writer, index=False, sheet_name="Meta")
        
        stats = {
            "total_rows": len(df_processed),
            "duration": f"{duration:.2f}s",
            "tokens": token_summary["total_tokens"],
            "cost": f"${token_summary['estimated_cost_usd']:.6f}",
            "api_calls": token_summary["api_calls"],
            "failed_batches": failed_batches
        }
        
        progress(1.0, desc="✅ Complete!")
        
        return output_path, stats, None
        
    except Exception as e:
        logging.error(f"[MAIN ERROR] {str(e)}", exc_info=True)
        return None, {}, f"❌ Error: {str(e)}"

# ====== GRADIO UI ======
def create_gradio_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(title="📊 Insights Generator", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📊 Insights Generator - Topics, Sentiment, Summary & Spokesperson")
        gr.Markdown(f"**Model:** {MODEL_NAME} | **Batch Size:** {BATCH_SIZE} rows | **Token Limit:** {TRUNCATE_WORDS} words/item")
        
        # Show API status
        if not api_ok:
            gr.Markdown(f"""
            <div style="background-color: #fff3cd; border: 1px solid #ffc107; padding: 10px; border-radius: 5px; margin: 10px 0;">
            ⚠️ <strong>WARNING:</strong> OpenAI API test failed!<br>
            Error: {api_msg}<br>
            Please check your API key in .secretcontainer/.env
            </div>
            """)
        
        with gr.Row():
            with gr.Column(scale=2):
                file_input = gr.File(
                    label="📁 Upload Excel File",
                    file_types=[".xlsx"],
                    type="filepath"
                )
                
                sheet_selector = gr.Dropdown(
                    label="📊 Select Sheet",
                    choices=[],
                    interactive=True
                )
                
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
                gr.Markdown("### ⚙️ Configuration")
                
                conf_threshold = gr.Slider(
                    label="Confidence Threshold (Sentiment)",
                    minimum=0,
                    maximum=100,
                    value=85,
                    step=5,
                    info="Sentiment below this confidence → neutral"
                )
                
                gr.Markdown("### ✅ Features to Generate")
                
                gen_topic = gr.Checkbox(label="📌 Topic & Sub Topic", value=True)
                gen_sentiment = gr.Checkbox(label="😊 Sentiment & Confidence", value=True)
                gen_summary = gr.Checkbox(label="📝 Summary (eligible channels)", value=True)
                gen_spokesperson = gr.Checkbox(label="🎤 Spokesperson (eligible channels)", value=True)
        
        process_btn = gr.Button("🚀 Process Now", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                output_file = gr.File(label="📥 Download Result")
            
            with gr.Column():
                stats_output = gr.Textbox(
                    label="📊 Processing Stats (JSON)",
                    lines=10,
                    interactive=False
                )

        error_output = gr.Textbox(label="⚠️ Errors", visible=False)

        
        def process_wrapper(
            file_path, sheet_name, topic, sentiment, summary, spokesperson, conf, progress=gr.Progress()
        ):
            if not file_path:
                return None, "", "Please upload a file"
            
            if not any([topic, sentiment, summary, spokesperson]):
                return None, "", "Please select at least one feature to generate"
            
            result_path, stats, error = process_file(
                file_path,
                sheet_name,
                topic,
                sentiment,
                summary,
                spokesperson,
                conf,
                progress
            )

            stats_str = json.dumps(stats, indent=2, ensure_ascii=False) if stats else ""

            if error:
                return None, stats_str, error
            
            return result_path, stats_str, ""

        
        process_btn.click(
            process_wrapper,
            inputs=[file_input, sheet_selector, gen_topic, gen_sentiment, gen_summary, gen_spokesperson, conf_threshold],
            outputs=[output_file, stats_output, error_output]
        )
        
        gr.Markdown("""
        ---
        ### 📖 How to Use:
        1. Upload Excel file with **Title** and/or **Content** columns
        2. Select sheet to process
        3. Choose features to generate
        4. Click **Process Now**
        5. Download result when complete
        
        ### 📌 Eligible Channels for Summary/Spokesperson:
        `online media`, `printmedia`, `tv`, `newspaper`, `printed`
        
        ### 🔍 Troubleshooting:
        - Check logs in `logs/` folder for detailed error messages
        - Ensure `.secretcontainer/.env` has valid `OPENAI_API_KEY`
        - Check console output for API connection errors
        """)
    
    return app

# ====== LAUNCH ======
if __name__ == "__main__":
    app = create_gradio_interface()
    
    app.queue(
        max_size=10,
        default_concurrency_limit=4
    )
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
    )