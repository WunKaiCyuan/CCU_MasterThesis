import json
import os
import pandas as pd

# --- 配置區 ---
EXP_FILES = {
    "01 Fixed (4-0)": "./output/evaluation_full_report_4-0.json",
    "02 Semantic (4-1)": "./output/evaluation_full_report_4-1.json",
    "03 Fixed mmr (4-2)": "./output/evaluation_full_report_4-2.json",
    "04 Semantic mmr (4-3)": "./output/evaluation_full_report_4-3.json",
    "05 Fixed rerank (4-4)": "./output/evaluation_full_report_4-4.json",
    "06 Semantic rerank (4-5)": "./output/evaluation_full_report_4-5.json"
}
OUTPUT_CSV = "./output/final_comparison_table.csv"
OUTPUT_JSON = "./output/final_experiment_comparison.json"

def main():
    print("📊 正在整合實驗數據並產出論文對照表...")
    
    comparison_rows = []
    
    for strategy_label, file_path in EXP_FILES.items():
        if not os.path.exists(file_path):
            print(f"⚠️ 警告: 找不到檔案 {file_path}，跳過此項。")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            metrics = data.get("summary_metrics", {})
            
            # 解析 at_k_1, at_k_3, at_k_5
            for k_key, values in metrics.items():
                k_val = k_key.split('_')[-1] # 取得 1, 3, 5
                comparison_rows.append({
                    "Strategy": strategy_label,
                    "K": int(k_val),
                    "Hit Rate": values.get("hit_rate", 0),
                    "MRR": values.get("mrr", 0),
                    "Mean Recall": values.get("mean_recall", 0),
                    "Mean Precision": values.get("mean_precision", 0),
                    "MAP": values.get("map", 0),
                })

    if not comparison_rows:
        print("❌ 沒有數據可以整合，請確認 05-0 與 05-1 是否已執行。")
        return

    # 1. 轉換為 DataFrame 並排序
    df = pd.DataFrame(comparison_rows)
    df = df.sort_values(by=["K", "Strategy"]) # 先按 K 排序，再按策略排

    # 2. 格式化百分比 (美化輸出，但在儲存 JSON 時保留原始數值)
    df_display = df.copy()
    for col in ["Hit Rate", "Mean Recall", "Mean Precision"]:
        df_display[col] = df_display[col].map(lambda x: f"{x:.2%}")
    df_display["MRR"] = df_display["MRR"].map(lambda x: f"{x:.4f}")
    df_display["MAP"] = df_display["MAP"].map(lambda x: f"{x:.4f}")

    print("\n" + "="*70)
    print("🏆 碩士論文實驗結果對照表 (Unified Evaluation Standards)")
    print("="*70)
    print(df_display.to_string(index=False))
    print("="*70)

    # 3. 儲存結果
    # CSV 適合直接貼進論文表格
    df_display.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    # JSON 適合後續繪圖程式調用
    df.to_json(OUTPUT_JSON, orient="records", indent=4, force_ascii=False)

    print(f"\n✅ 對照表已生成：\n- CSV: {OUTPUT_CSV}\n- JSON: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()