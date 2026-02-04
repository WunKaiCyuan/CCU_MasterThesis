import json
import os

INDEX_PATH = "./output/document_index.json"
with open(INDEX_PATH, 'r', encoding='utf-8') as f:
        index_data = json.load(f)

# 建立映射表 {"file_name": "doc_id"}
id_to_name = {doc['doc_id']: doc['file_name'] for doc in index_data['documents']}

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f: return json.load(f)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=4, ensure_ascii=False)

def combine_a_and_b(results_a, results_b):
    hybrid_results = []
    question_list = results_a
    dict_a = {item['question_id']: item for item in results_a}
    dict_b = {item['question_id']: item for item in results_b}
    for question in question_list:
        q_id = question['question_id']
        q_name = question['question']
        seen_ids = set()
        result_a = dict_a.get(q_id)
        unique_candidates_a = []
        score_dict = {}

        for index, cand in enumerate(result_a['retrieved_candidates']):
            d_id = cand['doc_id']
            if d_id not in seen_ids:
                unique_candidates_a.append(cand)
                seen_ids.add(d_id)
                current_score = score_dict.get(d_id, 0);
                score_dict[d_id] = current_score + 1 * (8-index) # 向量查詢權重計算

        result_b = dict_b.get(q_id)
        unique_candidates_b = []

        for index, cand in enumerate(result_b['retrieved_candidates']):
            d_id = cand['doc_id']
            if d_id not in seen_ids:
                unique_candidates_b.append(cand)
                seen_ids.add(d_id)
                current_score = score_dict.get(d_id, 0);
                score_dict[d_id] = current_score + 1 * (8-index) # 向量查詢權重計算

        # final_candidates = (unique_candidates_a[:8] + unique_candidates_b[:8])[:8]
        sorted_keys = sorted(score_dict, key=score_dict.get, reverse=True)[:8]
        final_candidates = [ {"doc_id": k, "file_name": id_to_name[k]} for k in sorted_keys]
        
        hybrid_results.append({
            "question_id": q_id,
            "question": q_name,
            "retrieved_candidates": final_candidates
        })
    return hybrid_results

# 執行批量合成
def main():
    path_b = "./output/retrieval_results_full_llm_scan.json"
    res_b = load_json(path_b)
    
    # 定義 6 種 A 路徑的檔名
    a_files = [
        "retrieval_results_parent_doc.json",
        "retrieval_results_parent_doc_mmr.json",
        "retrieval_results_parent_doc_rerank.json",
        "retrieval_results_semantic_parent.json",
        "retrieval_results_semantic_parent_mmr.json",
        "retrieval_results_semantic_parent_rerank.json"
    ]
    
    for f_name in a_files:
        input_path = f"./output/{f_name}"
        if os.path.exists(input_path):
            res_a = load_json(input_path)
            res_ab = combine_a_and_b(res_a, res_b)
            output_name = f_name.replace("retrieval_results_", "retrieval_results_hybrid_")
            save_json(res_ab, f"./output/{output_name}")
            print(f"✅ 已生成補償版本: {output_name}")

if __name__ == "__main__":
    main()
