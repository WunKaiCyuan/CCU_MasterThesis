import json
import os
from datetime import datetime
from core.metrics_utils import calculate_detailed_metrics

# --- 配置區 ---
EXP_CONFIG = {
    "STRATEGY_NAME": "Semantic_Chunking_RERANK",
    "EMBEDDING_MODEL": "intfloat/multilingual-e5-small",
    "GROUND_TRUTH_PATH": "./output/ai_evidence_report.json",
    "PREDICTION_PATH": "./output/retrieval_results_semantic_parent_rerank.json",
    "OUTPUT_REPORT_PATH": "./output/evaluation_full_report_4-5.json",
    "K_VALUES": [1, 3, 5, 8]
}

def main():
    print(f"📊 正在產出整合評估報告: {EXP_CONFIG['STRATEGY_NAME']}")

    # 1. 載入資料
    with open(EXP_CONFIG["GROUND_TRUTH_PATH"], 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    with open(EXP_CONFIG["PREDICTION_PATH"], 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

    # 2. 建立 Ground Truth 對照表
    gt_map = {}
    for entry in gt_data:
        relevant_ids = [
            res['doc_id'] for res in entry['verification_results'] 
            if res['ai_audit']['is_relevant'] is True
        ]
        gt_map[entry['question_id']] = set(relevant_ids)

    # 3. 建立預測對照表
    pred_map = {p['question_id']: [c['doc_id'] for c in p['retrieved_candidates']] for p in pred_data}

    # 4. 計算指標
    metrics = calculate_detailed_metrics(gt_map, pred_map, EXP_CONFIG["K_VALUES"])

    # 5. 整合所有資訊 (基本資訊 + 檢索結果 + 指標)
    full_report = {
        "experiment_metadata": {
            "strategy": EXP_CONFIG["STRATEGY_NAME"],
            "embedding_model": EXP_CONFIG["EMBEDDING_MODEL"],
            "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions_evaluated": len(gt_map)
        },
        "summary_metrics": metrics,
        "detailed_retrieval_results": pred_data  # 直接將原本檢索產生的結果寫入
    }

    # 6. 儲存整合 JSON
    with open(EXP_CONFIG["OUTPUT_REPORT_PATH"], "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=4, ensure_ascii=False)

    print(f"\n✅ 整合報告已產生！檔案路徑: {EXP_CONFIG['OUTPUT_REPORT_PATH']}")

if __name__ == "__main__":
    main()