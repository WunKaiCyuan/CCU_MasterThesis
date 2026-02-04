import json
import os
import time
import google.generativeai as genai
from datetime import timedelta
import typing_extensions as typing

# --- 1. 配置與路徑 ---
GENAI_API_KEY = "API_KEY"
MODEL_NAME = "models/gemini-2.5-flash"
genai.configure(api_key=GENAI_API_KEY)

# 輸入
QUESTIONS_PATH = "./output/generated_questions.json"  
SUMMARIES_PATH = "./output/document_summaries.json"
# 輸出 (檔名符合你的 04 系列習慣)
OUTPUT_PATH = "./output/retrieval_results_full_llm_scan.json"

# --- 2. Schema 定義 ---
class RelevantDoc(typing.TypedDict):
    doc_id: str
    file_name: str

class RetrievalResponse(typing.TypedDict):
    relevant_docs: list[RelevantDoc] # 儲存包含 ID 與 檔名的物件

# --- 3. 工具程式 ---
def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    questions = load_json(QUESTIONS_PATH)
    all_summaries = load_json(SUMMARIES_PATH)
    
    # 建立 ID 到 檔名的對照表，方便輸回格式
    id_to_name = {s['doc_id']: s['doc_name'] for s in all_summaries}
    
    final_results = load_json(OUTPUT_PATH)
    done_ids = {r["question_id"] for r in final_results}
    print(f"🔄 偵測到已完成 {len(done_ids)} 題，剩餘 {len(questions) - len(done_ids)} 題...")

    # 建立快取
    summary_context = "\n".join([
        f"【ID: {s['doc_id']} | 標題: {s['doc_name']}】\n摘要：{s['summary']}\n關鍵字：{', '.join(s.get('keywords', []))}\n---"
        for s in all_summaries
    ])

    print("📂 建立摘要索引快取...")
    cache = genai.caching.CachedContent.create(
        model=MODEL_NAME,
        display_name="ccu_summaries_cache",
        contents=[f"以下是中正大學校規摘要索引：\n{summary_context}"],
        ttl=timedelta(hours=1)
    )

    model = genai.GenerativeModel.from_cached_content(
        cached_content=cache,
        generation_config={
            "temperature": 0,
            "response_mime_type": "application/json",
            "response_schema": RetrievalResponse,
        }
    )

    try:
        for item in questions:
            q_id = item["id"]
            query = item["question"]
            if q_id in done_ids:
                continue

            print(f"🔎 正在處理 Q{q_id}: {query[:15]}...")
            
            prompt = f"使用者問題：{query}\n任務：請從索引中挑選出最相關的 10 個法規。請依相關程度由高到低排序。"

            try:
                response = model.generate_content(prompt)
                res_json = json.loads(response.text)
                
                # 格式轉換，確保 ID 與 檔名正確對應
                candidates = []
                for doc in res_json.get("relevant_docs", []):
                    d_id = doc['doc_id']
                    candidates.append({
                        "doc_id": d_id,
                        "file_name": id_to_name.get(d_id, "Unknown")
                    })

                final_results.append({
                    "question_id": q_id,
                    "question": query,
                    "retrieved_candidates": candidates
                })
                
                save_json(final_results, OUTPUT_PATH)
                time.sleep(1) 

            except Exception as e:
                print(f"❌ Q{q_id} 失敗: {e}")
                time.sleep(5)
                continue
    finally:
        cache.delete()
        print(f"✅ 任務完成！結果儲存於 {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
