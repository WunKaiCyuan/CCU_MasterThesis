def calculate_detailed_metrics(gt_map, pred_map, k_values):
    """
    gt_map: {q_id: set(relevant_doc_ids)}
    pred_map: {q_id: [retrieved_doc_ids_from_candidates]}
    k_values: list of int, e.g., [1, 3, 5]
    """
    metrics_results = {}
    
    for k in k_values:
        hits = 0
        recalls = []
        precisions = []
        rr_list = []
        ap_list = []
        
        for q_id, true_ids in gt_map.items():
            if not true_ids:
                # 處理專家標註為無答案的情況：通常不計入檢索指標評估
                continue
            
            predicted_all = pred_map.get(q_id, [])
            predicted_k = []
            seen = set()
            for pid in predicted_all:
                if pid not in seen:
                    predicted_k.append(pid)
                    seen.add(pid)
                if len(predicted_k) == k:
                    break
            
            pred_set = set(predicted_k)
            intersect = true_ids & pred_set

            # 1. Hit Rate@K
            if intersect:
                hits += 1
            
            # 2. Recall@K
            recalls.append(len(intersect) / len(true_ids))
            
            # 3. Precision@K
            precisions.append(len(intersect) / k)
            
            # 4. MRR (Reciprocal Rank)
            rr = 0
            for i, pid in enumerate(predicted_k):
                if pid in true_ids:
                    rr = 1 / (i + 1)
                    break
            rr_list.append(rr)

            # 5. AP (Average Precision) for MAP
            score_ap = 0.0
            num_hits = 0
            for i, pid in enumerate(predicted_k):
                if pid in true_ids:
                    num_hits += 1
                    # 當前位置的 Precision = 到目前為止中的數量 / 當前排名
                    score_ap += num_hits / (i + 1)
            
            # AP 是將累積的 Precision 除以「總共應該找回的數量」
            ap_list.append(score_ap / len(true_ids))

        metrics_results[f"at_k_{k}"] = {
            "hit_rate": hits / len(recalls) if recalls else 0,
            "mean_recall": sum(recalls) / len(recalls) if recalls else 0,
            "mean_precision": sum(precisions) / len(precisions) if precisions else 0,
            "mrr": sum(rr_list) / len(rr_list) if rr_list else 0,
            "map": sum(ap_list) / len(ap_list) if ap_list else 0
        }
        
    return metrics_results