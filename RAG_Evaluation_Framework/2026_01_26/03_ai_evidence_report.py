import json
import os
import time
import google.generativeai as genai
import typing_extensions as typing
import PyPDF2
from docx import Document

# --- 1. 定義回傳結構 ---
class DocVerification(typing.TypedDict):
    is_relevant: bool
    evidence_quote: str
    is_sufficient: bool
    ai_reasoning: str

# --- 2. 配置區 ---
GENAI_API_KEY = "API_KEY"
MODEL_NAME = "models/gemini-2.5-flash"
genai.configure(api_key=GENAI_API_KEY)

# 檔案路徑
INDEX_JSON_PATH = "./output/document_index.json"
MAPPING_JSON_PATH = "./output/candidate_mapping.json"
DATA_DIR = "./data/"
OUTPUT_JSON_PATH = "./output/ai_evidence_report.json"

model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    generation_config={
        "temperature": 0,
        "response_mime_type": "application/json",
        "response_schema": DocVerification
    }
)

# --- 3. 工具函式 ---

def load_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 建立 ID 與 檔名 的查找表
def get_file_map(index_data):
    return {doc['doc_id']: doc['file_name'] for doc in index_data.get('documents', [])}

def load_content(file_name):
    path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(path):
        return None
    
    ext = os.path.splitext(file_name)[1].lower()
    try:
        if ext == ".txt":
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == ".pdf":
            text = ""
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += (page.extract_text() or "") + "\n"
            return text
        elif ext == ".docx":
            doc = Document(path)
            return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"❌ 解析檔案 {file_name} 失敗: {e}")
    return None

def ai_verify(question, file_name, content):
    if not content or len(content.strip()) < 20:
        return None
    
    print(f"處理 {question} {file_name}")
    prompt = f"任務：校規關聯性審核\n問題：{question}\n文檔名稱：{file_name}\n內文：\n{content[:15000]}"
    for attempt in range(3):
        print(f"嘗試第{attempt+1}次呼叫AI驗證")
        try:
            response = model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            print(f"發生錯誤 {e}")
            time.sleep(15)
    return None

# --- 4. 主流程 ---

def main():
    index_data = load_json(INDEX_JSON_PATH)
    file_map = get_file_map(index_data)
    tasks = load_json(MAPPING_JSON_PATH)
    
    # 斷點續傳
    results = load_json(OUTPUT_JSON_PATH)
    done_ids = {r['question_id'] for r in results}

    print(f"🚀 開始驗證任務... (已完成: {len(done_ids)} 題)")

    for task in tasks:
        q_id = task['question_id']
        if q_id in done_ids: continue

        print(f"🔎 正在核對 Q{q_id}: {task['question'][:15]}...")
        
        verification_list = []
        for cand in task.get('llm_suggested_candidates', []):
            d_id = cand['doc_id']
            # 從索引中找正確檔名 (含副檔名)
            real_file_name = file_map.get(d_id)
            
            if not real_file_name:
                print(f"  ⚠️ 索引中找不到 {d_id}")
                continue
                
            content = load_content(real_file_name)
            if content:
                audit = ai_verify(task['question'], real_file_name, content)
                if audit:
                    verification_list.append({
                        "doc_id": d_id,
                        "file_name": real_file_name,
                        "ai_audit": audit
                    })
            print("等待15秒")
            time.sleep(15)

        results.append({
            "question_id": q_id,
            "question": task['question'],
            "verification_results": verification_list,
            "expert_final_check": ""
        })
        save_json(results, OUTPUT_JSON_PATH)

    print(f"✨ 報表已生成：{OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()