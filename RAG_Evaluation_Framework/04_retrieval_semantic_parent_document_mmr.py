import os
import json
import torch
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from core.serializable_mongodb_byte_store import SerializableMongoDBByteStore
import chromadb

EXP_CONFIG = {
    # 模型設定
    "MODEL_NAME": "intfloat/multilingual-e5-small",
    
    # ChromaDB 連線設定
    "CHROMA_HOST": "localhost",
    "CHROMA_PORT": 8000,
    "CHROMA_COLLECTION": "ccu_rules_semantic_child",
    
    # MongoDB 連線設定
    "MONGODB_URI": "mongodb://admin:UTWi1dCo6jFxNlS0@localhost:27017",
    "MONGODB_DB": "ccu_school_rules",
    "MONGODB_COLLECTION": "ccu_rules_semantic_parent",
    
    # 檔案路徑設定
    "QUESTIONS_PATH": "./output/generated_questions.json", 
    "INDEX_PATH": "./output/document_index.json",
    "OUTPUT_JSON_PATH": "./output/retrieval_results_semantic_parent_mmr.json",
    
    # 檢索參數
    "K": 5
}

def main():
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(EXP_CONFIG["OUTPUT_JSON_PATH"]), exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧪 啟動 4-1 語意切分評估 (對齊 JSON 格式)...環境設備: {device}")

    # 1. 初始化 Embedding
    embeddings = HuggingFaceEmbeddings(
        model_name=EXP_CONFIG["MODEL_NAME"],
        model_kwargs={'device': device}
    )

    # 2. 連接 ChromaDB
    client = chromadb.HttpClient(
        host=EXP_CONFIG["CHROMA_HOST"], 
        port=EXP_CONFIG["CHROMA_PORT"]
    )
    vectorstore = Chroma(
        client=client,
        collection_name=EXP_CONFIG["CHROMA_COLLECTION"],
        embedding_function=embeddings
    )

    # 3. 連接 MongoDB Store
    store = SerializableMongoDBByteStore(
        connection_string=EXP_CONFIG["MONGODB_URI"],
        db_name=EXP_CONFIG["MONGODB_DB"],
        collection_name=EXP_CONFIG["MONGODB_COLLECTION"]
    )

    # 4. 載入測試集
    if not os.path.exists(EXP_CONFIG["QUESTIONS_PATH"]):
        print(f"❌ 錯誤: 找不到測試集檔案 {EXP_CONFIG['QUESTIONS_PATH']}")
        return

    with open(EXP_CONFIG["QUESTIONS_PATH"], 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    with open(EXP_CONFIG["INDEX_PATH"], 'r', encoding='utf-8') as f:
        index_data = json.load(f)

    # 建立映射表，例如: {"成績作業要點.pdf": "D37"}
    name_to_id = {doc['file_name']: doc['doc_id'] for doc in index_data['documents']}

    final_output = []

    # 5. 執行檢索 (加上 tqdm 進度條)
    print(f"📡 正在處理 {len(test_data)} 個問題...")
    
    # 使用 tqdm 顯示進度
    for item in tqdm(test_data, desc="檢索進度", unit="query"):
        query = item['question']
        # 優先從 JSON 拿 question_id，沒有的話則手動編號
        q_id = item.get('question_id', test_data.index(item) + 1)

        # A. 檢索子向量
        results = vectorstore.max_marginal_relevance_search(query, k=EXP_CONFIG["K"])

        # B. 取得 Parent ID 並從 MongoDB 抓取對應資訊
        candidates = []
        for doc in results:
            parent_id = doc.metadata.get("parent_id")
            
            # 從 MongoDB 獲取母文件 metadata
            parent_doc = store.mget([parent_id])[0]
            
            if parent_doc:
                candidates.append({
                    "doc_id": name_to_id[doc.metadata.get("file_name")],
                    "file_name": doc.metadata.get("file_name")
                })

        # C. 封裝結果
        final_output.append({
            "question_id": q_id,
            "question": query,
            "retrieved_candidates": candidates
        })

    # 6. 輸出最終結果 JSON
    with open(EXP_CONFIG["OUTPUT_JSON_PATH"], 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    
    print(f"\n✅ 4-3 評估完成！")
    print(f"💾 結果已存至: {EXP_CONFIG['OUTPUT_JSON_PATH']}")

if __name__ == "__main__":
    main()