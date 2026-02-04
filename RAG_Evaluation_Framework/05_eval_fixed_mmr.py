import json
import os
from datetime import datetime
from core.metrics_utils import calculate_detailed_metrics

# --- 配置區 ---
EXP_CONFIG = {
    "STRATEGY_NAME": "Fixed_Chunking_MMR",
    "EMBEDDING_MODEL": "intfloat/multilingual-e5-small",
    "CHUNK_SIZE": 200,
    "CHUNK_OVERLAP": 20,
    "GROUND_TRUTH_PATH": "./output/ai_evidence_report.json",
    "PREDICTION_PATH": "./output/retrieval_results_parent_doc_mmr.json",
    "OUTPUT_REPORT_PATH": "./output/evaluation_full_report_4-2.json",
    "K_VALUES": [1, 3, 5, 8]
}

def main():
    print(f"🚀 啟動評估流程: {EXP_CONFIG['STRATEGY_NAME']}")

    # 1. 載入資料
    if not os.path.exists(EXP_CONFIG["GROUND_TRUTH_PATH"]) or not os.path.exists(EXP_CONFIG["PREDICTION_PATH"]):
        print("❌ 錯誤: 找不到標準答案或預測結果檔案。")
        return

    with open(EXP_CONFIG["GROUND_TRUTH_PATH"], 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    with open(EXP_CONFIG["PREDICTION_PATH"], 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

    # 2. 建立 Ground Truth 對照表 {q_id: set(doc_ids)}
    gt_map = {}
    for entry in gt_data:
        relevant_ids = [
            res['doc_id'] for res in entry['verification_results'] 
            if res['ai_audit']['is_relevant'] is True
        ]
        gt_map[entry['question_id']] = set(relevant_ids)

    # 3. 建立預測結果對照表 {q_id: [doc_ids]}
    pred_map = {p['question_id']: [c['doc_id'] for c in p['retrieved_candidates']] for p in pred_data}

    # 4. 計算各項指標
    metrics = calculate_detailed_metrics(gt_map, pred_map, EXP_CONFIG["K_VALUES"])

    # 5. 整合完整報告 (Metadata + Summary + Detailed Results)
    full_report = {
        "experiment_metadata": {
            "strategy": EXP_CONFIG["STRATEGY_NAME"],
            "embedding_model": EXP_CONFIG["EMBEDDING_MODEL"],
            "parameters": {
                "chunk_size": EXP_CONFIG["CHUNK_SIZE"],
                "chunk_overlap": EXP_CONFIG["CHUNK_OVERLAP"]
            },
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": len(recalls if 'recalls' in locals() else gt_map)
        },
        "summary_metrics": metrics,
        "detailed_retrieval_results": pred_data # 完整保留當初檢索產生的原始結果
    }

    # 6. 儲存 JSON
    with open(EXP_CONFIG["OUTPUT_REPORT_PATH"], "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=4, ensure_ascii=False)

    print(f"✅ 評估報告已產生: {EXP_CONFIG['OUTPUT_REPORT_PATH']}")

if __name__ == "__main__":
    main()