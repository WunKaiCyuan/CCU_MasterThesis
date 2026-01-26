
import requests
import json
import time
from typing import List, Dict

# 定義測試題組
TEST_CASES = [
    {"id": 1, "question": "學士班修業年限是多少年？", "ground_truths": ["學籍法規_國立中正大學學則.pdf"]},
    {"id": 2, "question": "碩士班研究生修業期限規定為何？", "ground_truths": ["學籍法規_國立中正大學學則.pdf", "學籍法規_國立中正大學研究生學位授予辦法.pdf"]},
    {"id": 3, "question": "申請轉系需要符合什麼資格？", "ground_truths": ["學籍法規_國立中正大學學生轉系（所）辦法.pdf"]},
    {"id": 4, "question": "學生修讀輔系的申請時間與程序為何？", "ground_truths": ["學籍法規_國立中正大學學生修讀輔系（所、學位學程）辦法.pdf", "業務法規_加修輔系申請流程.doc"]},
    {"id": 5, "question": "申請雙主修有什麼條件限制？", "ground_truths": ["學籍法規_國立中正大學學生修讀雙主修辦法.pdf", "業務法規_加修雙主修申請流程.doc"]},
    {"id": 6, "question": "什麼情況下會被勒令退學？", "ground_truths": ["學籍法規_國立中正大學學則.pdf", "業務法規_退學作業流程.doc"]},
    {"id": 7, "question": "休學期限最長可以多久？", "ground_truths": ["學籍法規_國立中正大學學則.pdf", "業務法規_休學作業程序.doc"]},
    {"id": 8, "question": "學生證遺失了要如何申請補發？", "ground_truths": ["其他法規_國立中正大學學生遺失學生證申請補、換發辦法.pdf"]},
    {"id": 9, "question": "碩士學位論文考試不及格怎麼辦？有重考機會嗎？", "ground_truths": ["學籍法規_國立中正大學研究生學位授予辦法.pdf", "學籍法規_國立中正大學學位論文管理要點.pdf"]},
    {"id": 10, "question": "博士班研究生資格考核有幾次機會？", "ground_truths": ["學籍法規_國立中正大學研究生學位授予辦法.pdf", "學籍法規_國立中正大學學生逕修讀博士學位作業規定.pdf"]},
    {"id": 11, "question": "如何申請抵免學分？抵免上限是多少？", "ground_truths": ["成績_國立中正大學辦理學生抵免學分辦法.pdf"]},
    {"id": 12, "question": "操行成績不及格會被退學嗎？", "ground_truths": ["學籍法規_國立中正大學學則.pdf", "成績_國立中正大學學生成績作業要點.pdf"]},
    {"id": 13, "question": "暑期開班授課的選課規定是什麼？", "ground_truths": ["選課法規_國立中正大學暑期授課辦法.pdf"]},
    {"id": 14, "question": "跨校選課的學分如何計算？", "ground_truths": ["選課法規_國立中正大學校際選課實施要點.pdf", "業務法規_校際選課流程.doc"]},
    {"id": 15, "question": "學生擔任教學助理(TA)有什麼資格要求？", "ground_truths": ["其他法規_國立中正大學教學助理制度實施準則.pdf", "其他法規_國立中正大學優良教學助理遴選與獎勵要點.pdf"]},
    {"id": 16, "question": "如何申請校內的獎助學金？", "ground_truths": ["其他法規_國立中正大學獎助學金實施要點.pdf", "其他法規_國立中正大學培育優秀全職本國博士生新生獎學金作業須知.pdf"]},
    {"id": 17, "question": "學術倫理教育課程是必修的嗎？不及格怎麼辦？", "ground_truths": ["選課法規_學術倫理教育實施要點.pdf", "其他法規_國立中正大學學生違反學術倫理與研究誠信案件處理要點.pdf"]},
    {"id": 18, "question": "英文畢業門檻的檢定標準與實施辦法為何？", "ground_truths": ["其他法規_「英文能力」及「資訊能力」畢業資格檢定暨實施辦法 (已終止).pdf"]},
    {"id": 19, "question": "畢業離校手續包含哪些項目？", "ground_truths": ["學籍法規_國立中正大學學生辦理離校作業要點.pdf", "業務法規_畢業離校、領取學位證書辦理流程.pdf"]},
    {"id": 20, "question": "遭遇重大災害影響就學，有什麼彈性修業措施？", "ground_truths": ["其他法規_國立中正大學學士班就學期間專案服役學生彈性修業措施.pdf"]},
]

API_URL = "http://localhost:8001/query"

def evaluate():
    print("="*60)
    print(f"開始執行檢索評估，共 {len(TEST_CASES)} 題")
    print("="*60)
    
    results = []
    total_hits = 0
    
    for case in TEST_CASES:
        print(f"\n[Test Case {case['id']}]: {case['question']}")
        
        try:
            # 發送請求 (啟用 skip_llm 模式)
            start_time = time.time()
            response = requests.post(
                API_URL, 
                json={
                    "question": case["question"],
                    "skip_llm": True
                },
                timeout=30 
            )
            response.raise_for_status()
            data = response.json()
            elapsed = time.time() - start_time
            
            # 獲取檢索到的來源與內容
            sources = data.get('sources', [])
            retrieved_sources = [s.get('file_name', '') for s in sources]
            unique_sources = list(set(retrieved_sources))
            
            # 檢查是否命中 (Hit): 必須包含所有預期的文件
            # 檢查 ground_truths 是否為 unique_sources 的子集
            missing_files = [gt for gt in case["ground_truths"] if gt not in unique_sources]
            is_hit = len(missing_files) == 0
            hit_files = [gt for gt in case["ground_truths"] if gt in unique_sources]
            
            # 記錄結果
            result = {
                "id": case["id"],
                "question": case["question"],
                "is_hit": is_hit,
                "ground_truths": case["ground_truths"],
                "retrieved_sources": unique_sources,
                "sources_details": sources, # 儲存詳細資訊以供印出
                "missing_files": missing_files,
                "latency": round(elapsed, 2)
            }
            results.append(result)
            
            if is_hit:
                total_hits += 1
                print(f"✅ Hit! (All found: {hit_files})")
                # 印出第一個來源的內容片段作為範例
                if sources:
                    print(f"   Snippet: {sources[0].get('content', '')[:100]}...")
            else:
                print(f"❌ Miss")
                print(f"   Expected: {case['ground_truths']}")
                print(f"   Missing: {missing_files}")
                print(f"   Got: {unique_sources}")
                
        except Exception as e:
            print(f"⚠️ Error: {e}")
            results.append({
                "id": case["id"],
                "error": str(e),
                "is_hit": False
            })

    # 統計
    accuracy = (total_hits / len(TEST_CASES)) * 100
    
    print("\n" + "="*60)
    print("評估報告 (Evaluation Report)")
    print("="*60)
    print(f"Total Questions: {len(TEST_CASES)}")
    print(f"Total Hits: {total_hits}")
    print(f"Retrieval Accuracy (Strict Match): {accuracy:.2f}%")
    print("="*60)

    # 輸出詳細失敗案例
    if total_hits < len(TEST_CASES):
        print("\n[Missed Cases]")
        for res in results:
            if not res.get("is_hit", False):
                print(f"{res['id']}. {res.get('question', '')}")
                print(f"   Expected: {res.get('ground_truths', [])}")
                print(f"   Missing:  {res.get('missing_files', [])}")
                print(f"   Got:      {res.get('retrieved_sources', [])}")
                print("-" * 30)

if __name__ == "__main__":
    evaluate()
