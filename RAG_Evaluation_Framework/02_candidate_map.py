import json
import os
import time
import google.generativeai as genai
import typing_extensions as typing

# --- 1. 配置與模型設定 ---

# 定義單個文檔的結構
class CandidateDoc(typing.TypedDict):
    doc_id: str
    file_name: str
    reason: str

# 定義批次回答的結構
class QuestionResult(typing.TypedDict):
    question_id: int
    suggested_docs: list[CandidateDoc]

GENAI_API_KEY = "API_KEY"
MODEL_NAME = "models/gemini-2.5-flash"
genai.configure(api_key=GENAI_API_KEY)

# 初始化模型，強制要求回傳 JSON Array[QuestionResult]
model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    generation_config={
        "temperature": 0,
        "response_mime_type": "application/json",
        "response_schema": list[QuestionResult],
    }
)

# --- 2. 工具函式 ---

def load_json(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 找不到檔案: {file_path}")
        return [] if "questions" in file_path else {}
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"✅ 檔案已儲存至: {file_path}")

def batch_llm_suggest(questions_subset, doc_list):
    """將多個問題打包，一次性詢問 Gemini"""
    
    # 建立文檔清單文本
    doc_context = "\n".join([f"- {d['doc_id']}: {d['file_name']}" for d in doc_list])
    
    # 建立問題清單文本
    questions_context = "\n".join([f"Q{q['id']}: {q['question']}" for q in questions_subset])

    prompt = f"""
你是一位法律文檔專家。我正在處理「中正大學校規」的 RAG 檢索優化研究。
請針對以下「問題清單」中的每一個問題，從「文檔清單」中挑選出 1-5 份最可能包含答案的候選文檔。

【文檔清單】:
{doc_context}

【問題清單】:
{questions_context}

【要求】:
1. 針對每個問題給予推薦理由 reason。
2. 必須嚴格按照提供的 JSON Schema 回傳，確保 question_id 與問題清單對應。
"""
    
    try:
        response = model.generate_content(prompt)
        
        # 檢查是否有內容 (處理安全過濾)
        if not response.text:
            print("⚠️ 模型未回傳內容，可能是被安全機制攔截。")
            return []
            
        return json.loads(response.text)
    except Exception as e:
        print(f"❌ 批次請求出錯: {e}")
        return []

# --- 3. 主程式流程 ---

def main():
    # 設定檔案路徑
    INDEX_PATH = "./output/document_index.json"
    QUESTIONS_PATH = "./output/generated_questions.json"
    OUTPUT_PATH = "./output/candidate_mapping.json"

    # 載入資料
    inventory = load_json(INDEX_PATH)
    questions_all = load_json(QUESTIONS_PATH)
    docs = inventory.get("documents", [])

    if not docs or not questions_all:
        print("資料載入失敗，請確認 JSON 檔案內容與路徑。")
        return

    mapped_results = []
    
    # --- 批次處理設定 ---
    BATCH_SIZE = 10  # 每次處理 10 個問題
    SLEEP_TIME = 15  # 每次請求後休息 15 秒以符合 Free Tier 限制 (5 RPM)

    total_questions = len(questions_all)
    print(f"🚀 開始分析！總計 {total_questions} 個問題，預計分為 { (total_questions // BATCH_SIZE) + 1 } 批次...")

    for i in range(0, total_questions, BATCH_SIZE):
        subset = questions_all[i : i + BATCH_SIZE]
        current_batch_ids = [q['id'] for q in subset]
        
        print(f"\n📦 正在處理批次: Q{current_batch_ids[0]} ~ Q{current_batch_ids[-1]}...")
        
        # 呼叫 Gemini 進行批次處理
        batch_results = batch_llm_suggest(subset, docs)
        
        # 將 LLM 的結果對照回原本的問題清單
        for q in subset:
            # 在 LLM 回傳的清單中尋找對應的 id
            matched_llm_res = next((res for res in batch_results if res.get('question_id') == q['id']), None)
            
            suggested = matched_llm_res.get('suggested_docs', []) if matched_llm_res else []
            
            mapped_results.append({
                "question_id": q["id"],
                "question": q["question"],
                "category": q.get("category", ""),
                "llm_suggested_candidates": suggested,
                "manual_confirmed_doc_ids": [] 
            })
            
        print(f"✔️ 批次處理完成 (成功匹配: {len([r for r in batch_results])} 筆)")
        
        # 如果不是最後一輪，則進入冷卻
        if i + BATCH_SIZE < total_questions:
            print(f"😴 等待 {SLEEP_TIME} 秒以遵守 API 頻率限制...")
            time.sleep(SLEEP_TIME)

    # 儲存最終結果
    save_json(mapped_results, OUTPUT_PATH)
    print("\n✨ 全部任務完成！請查看 output 檔案。")

if __name__ == "__main__":
    main()