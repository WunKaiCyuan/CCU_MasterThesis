import json
import os
import time
import google.generativeai as genai
from datetime import timedelta
from core.document_loader import load_documents
import typing_extensions as typing

# --- 1. 定義 Schema ---
class AuditDetail(typing.TypedDict):
    is_relevant: bool
    evidence_quote: str
    is_sufficient: bool
    ai_reasoning: str

class VerificationResult(typing.TypedDict):
    doc_id: str
    file_name: str
    ai_audit: AuditDetail

class SingleQuestionResponse(typing.TypedDict):
    verification_results: list[VerificationResult]

# --- 2. 配置 ---
GENAI_API_KEY = "API_KEY"
MODEL_NAME = "models/gemini-2.5-flash"
genai.configure(api_key=GENAI_API_KEY)

INDEX_PATH = "./output/document_index.json"
DATA_DIR = "/Volumes/Shared/MasterThesis/RAG_Evaluation_Framework/data"
QUESTIONS_PATH = "./output/generated_questions.json"
OUTPUT_PATH = "./output/ai_evidence_report.json"

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
    # 讀取索引與題目
    index_data = load_json(INDEX_PATH)
    file_map = {doc["file_name"].strip(): doc["doc_id"] for doc in index_data.get("documents", [])}
    questions = load_json(QUESTIONS_PATH)

    # 讀取現有結果 (斷點續傳關鍵)
    final_report = load_json(OUTPUT_PATH)
    done_ids = {r["question_id"] for r in final_report}
    print(f"🔄 偵測到已完成 {len(done_ids)} 題，將從下一題開始...")

    # 1. 載入並準備快取內容 (僅在有新題目要跑時才做)
    if len(done_ids) < len(questions):
        print("📂 載入文檔並建立快取...")
        langchain_docs = load_documents(DATA_DIR, clean=True)
        all_contents = []
        for doc in langchain_docs:
            f_name = os.path.basename(doc.metadata.get("source", "")).strip()
            d_id = file_map.get(f_name, "Unknown")
            all_contents.append(f"【ID: {d_id} | 檔名: {f_name}】\n{doc.page_content}\n---")

        cache = genai.caching.CachedContent.create(
            model=MODEL_NAME,
            display_name="golden_dataset_cache",
            contents=all_contents,
            ttl=timedelta(hours=1)
        )

        model = genai.GenerativeModel.from_cached_content(
            cached_content=cache,
            generation_config={
                "temperature": 0,
                "response_mime_type": "application/json",
                "response_schema": SingleQuestionResponse,
            }
        )

        # 2. 批次詢問迴圈
        for q_item in questions:
            q_id = q_item["id"]
            if q_id in done_ids:
                continue

            print(f"🔎 正在處理 Q{q_id}: {q_item['question'][:15]}...")
            prompt = f"問題：{q_item['question']}\n任務：請找出所有相關校規文件並分析關聯性。"

            try:
                response = model.generate_content(prompt)
                res_json = json.loads(response.text)

                final_report.append({
                    "question_id": q_id,
                    "question": q_item["question"],
                    "verification_results": res_json.get("verification_results", []),
                    "expert_final_check": ""
                })
                
                # 每題跑完立即存檔，防止程式崩潰
                save_json(final_report, OUTPUT_PATH)
                
            except Exception as e:
                print(f"❌ Q{q_id} 失敗: {e}")
                # 遇到錯誤通常是 API 限制，稍微休息長一點
                time.sleep(30)
                continue

            time.sleep(5) # 正常間隔

        print(f"✅ 任務完成！結果儲存於 {OUTPUT_PATH}")
    else:
        print("✨ 所有題目皆已完成。")

if __name__ == "__main__":
    main()
