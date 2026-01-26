import json
import os
import google.generativeai as genai
import typing_extensions as typing

# --- 1. 定義回傳格式 ---
class GeneratedQuestion(typing.TypedDict):
    category: str
    question: str

class QuestionList(typing.TypedDict):
    items: list[GeneratedQuestion]

# --- 2. 配置區 ---
GENAI_API_KEY = "API_KEY"
MODEL_NAME = "models/gemini-2.5-flash"
genai.configure(api_key=GENAI_API_KEY)

model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    generation_config={
        "temperature": 0.8, # 稍微調高，讓問題更口語、多樣化
        "response_mime_type": "application/json",
        "response_schema": QuestionList
    }
)

OUTPUT_PATH = "./output/generated_questions.json"

# --- 3. 主流程 ---

def main():
    print(f"🚀 正在直接生成 50 個大學校規相關問題...")

    prompt = """
    你是一位熟悉台灣國立大學（特別是中正大學）行政規章的諮詢專家。
    請設計 50 個學生在校園生活中最常遇到的具體問題。
    
    要求：
    1. 問題必須口語化且具體。例如：「如果我期末考因為確診沒辦法去考，補考成績會打折嗎？」而非「詢問考試請假規定」。
    2. 分類請選取：[學籍與成績管理, 獎懲與操行規定, 宿舍與校園生活, 社團、活動與工讀, 教師與教學行政]。
    3. 涵蓋範圍：請假、抵免學分、轉系、雙主修、停修、離校手續、獎學金申請、學生證遺失、操行獎懲、選課規範等。
    4. 請產出 50 題。
    """

    try:
        response = model.generate_content(prompt)
        data = json.loads(response.text)
        
        final_list = []
        for i, item in enumerate(data.get('items', []), 1):
            final_list.append({
                "id": i,
                "category": item['category'],
                "question": item['question']
            })

        # 儲存結果
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(final_list, f, ensure_ascii=False, indent=4)

        print(f"✨ 完成！共生成 {len(final_list)} 個問題。")
        print(f"📁 儲存路徑：{OUTPUT_PATH}")

    except Exception as e:
        print(f"❌ 生成失敗: {e}")

if __name__ == "__main__":
    main()