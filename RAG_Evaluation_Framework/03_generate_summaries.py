import json
import os
import time
import google.generativeai as genai
from datetime import timedelta
from core.document_loader import load_documents
import typing_extensions as typing

# --- 1. 定義 Schema ---
class SummaryResponse(typing.TypedDict):
    summary: str
    doc_name: str
    keywords: list

# --- 2. 配置 ---
GENAI_API_KEY = "API_KEY"
MODEL_NAME = "models/gemini-2.5-flash"
genai.configure(api_key=GENAI_API_KEY)

INDEX_PATH = "./output/document_index.json"
DATA_DIR = "/Volumes/Shared/MasterThesis/RAG_Evaluation_Framework/data"
OUTPUT_PATH = "./output/document_summaries.json"

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
    # 讀取索引與檔案內容
    index_data = load_json(INDEX_PATH)
    file_map = {doc["file_name"].strip(): doc["doc_id"] for doc in index_data.get("documents", [])}

    # 讀取現有結果 (斷點續傳關鍵)
    final_report = load_json(OUTPUT_PATH)
    done_ids = {r["doc_id"] for r in final_report}
    print(f"🔄 偵測到已完成 {len(done_ids)} 題，將從下一題開始...")

    # 1. 載入並準備快取內容 (僅在有新題目要跑時才做)
    if len(done_ids) < len(file_map):
        print("📂 載入文檔並建立快取...")
        langchain_docs = load_documents(DATA_DIR, clean=True)
        all_contents = []
        for doc in langchain_docs:
            f_name = os.path.basename(doc.metadata.get("source", "")).strip()
            d_id = file_map.get(f_name, "Unknown")
            all_contents.append(f"【ID: {d_id} | 檔名: {f_name}】\n{doc.page_content}\n---")

        cache = genai.caching.CachedContent.create(
            model=MODEL_NAME,
            display_name="ccu_rule_documents_cache",
            contents=all_contents,
            ttl=timedelta(hours=1)
        )

        model = genai.GenerativeModel.from_cached_content(
            cached_content=cache,
            generation_config={
                "temperature": 0,
                "response_mime_type": "application/json",
                "response_schema": SummaryResponse,
            }
        )

        # 2. 批次詢問迴圈
        for doc_item in index_data.get("documents", []):
            doc_id = doc_item["doc_id"]
            doc_name = doc_item["file_name"]
            if doc_id in done_ids:
                continue

            print(f"🔎 正在處理 {doc_id} 文檔: {doc_name[:15]}...")
            prompt = f"文檔名稱：{doc_name}\n任務：請針對此文檔內容生成 150-200 字的『檢索專用摘要』。請包含：1.核心規範事項（如：休學、學分抵免、獎勵申請）。2.關鍵限制條件。目的是讓另一個 LLM 能僅憑此摘要判斷該文檔是否與使用者的問題相關。"

            try:
                response = model.generate_content(prompt)
                res_json = json.loads(response.text)

                final_report.append({
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "summary": res_json.get("summary", ""),
                    "keywords": res_json.get("keywords", [])
                })
                
                # 每份文檔跑完立即存檔，防止程式崩潰
                save_json(final_report, OUTPUT_PATH)
                
            except Exception as e:
                print(f"❌ {doc_id} 失敗: {e}")
                # 遇到錯誤通常是 API 限制，稍微休息長一點
                time.sleep(30)
                continue

            time.sleep(5) # 正常間隔

        cache.delete()
        print(f"✅ 任務完成！結果儲存於 {OUTPUT_PATH}")
    else:
        print("✨ 所有題目皆已完成。")

if __name__ == "__main__":
    main()
