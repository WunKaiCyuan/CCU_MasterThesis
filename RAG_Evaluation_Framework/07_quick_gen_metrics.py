import subprocess
import sys
import time

def run_experiment(module_name):
    """執行指定的 Python 模組並紀錄時間"""
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"🚀 正在啟動實驗模組: {module_name}")
    print(f"{'='*60}")
    
    try:
        # 使用 -m 模式執行，確保路徑與導入正確
        process = subprocess.Popen(
            [sys.executable, "-m", module_name],
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        process.wait()
        
        if process.returncode == 0:
            duration = time.time() - start_time
            print(f"✅ {module_name} 執行成功！ (耗時: {duration:.2f} 秒)")
            return True
        else:
            print(f"❌ {module_name} 執行失敗，錯誤碼: {process.returncode}")
            return False
            
    except Exception as e:
        print(f"💥 發生意外錯誤: {str(e)}")
        return False

def main():
    # 定義實驗流水線順序
    pipeline = [
        # 1. 檢索測試階段 (Retrieval Tests)
        # "04_retrieval_parent_document_mmr",
        # "04_retrieval_parent_document_rerank",
        # "04_retrieval_parent_document",
        # "04_retrieval_semantic_parent_document_mmr",
        # "04_retrieval_semantic_parent_document_rerank",
        # "04_retrieval_semantic_parent_document",
        "04_generate_all_hybrid_results",
        
        # 2. 評估階段 (Evaluation Metrics Calculation)
        "05_eval_fixed_mmr",
        "05_eval_fixed_rerank",
        "05_eval_fixed",
        "05_eval_semantic_mmr",
        "05_eval_semantic_rerank",
        "05_eval_semantic",
        "05_eval_full_llm",
        "05_eval_hybrid",
        
        # 3. 最終圖表與報告生成 (Final Report)
        "06_generate_final_comparison"
    ]

    total_start = time.time()
    success_count = 0

    for module in pipeline:
        if run_experiment(module):
            success_count += 1
        else:
            print(f"\n⚠️  由於 {module} 失敗，停止後續實驗以避免數據污染。")
            break

    total_duration = time.time() - total_start
    print(f"\n{'-'*60}")
    print(f"🏁 總結報告:")
    print(f"   - 成功項目: {success_count}/{len(pipeline)}")
    print(f"   - 總共耗時: {total_duration/60:.2f} 分鐘")
    print(f"{'-'*60}")

if __name__ == "__main__":
    main()
