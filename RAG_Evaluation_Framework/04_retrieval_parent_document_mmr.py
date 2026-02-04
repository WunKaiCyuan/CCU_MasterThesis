import os
import json
import torch
from tqdm import tqdm

# LangChain 核心與向量庫
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_vector import SearchType

# MongoDB 與 Chroma HTTP 連線
from core.serializable_mongodb_byte_store import SerializableMongoDBByteStore
import chromadb

# --- 1. 實驗配置區 (EXP_CONFIG) ---
EXP_CONFIG = {
    # 檔案路徑設定
    "INDEX_PATH": "./output/document_index.json",
    "QUESTIONS_PATH": "./output/generated_questions.json",
    "OUTPUT_PATH": "./output/retrieval_results_parent_doc_mmr.json",
    
    # MongoDB 連線設定
    "MONGODB_URI": "mongodb://admin:UTWi1dCo6jFxNlS0@localhost:27017",
    "MONGODB_DB": "ccu_school_rules",
    "MONGODB_COLLECTION": "ccu_rules",
    
    # ChromaDB 連線設定
    "CHROMA_HOST": "localhost",
    "CHROMA_PORT": 8000,
    "CHROMA_COLLECTION": "ccu_rules",
    
    # Embedding 模型配置
    "EMBED_MODEL_NAME": "intfloat/multilingual-e5-small",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    
    # 切分參數設定 (4-0 固定切分)
    "CHILD_CHUNK_SIZE": 200,
    "CHILD_CHUNK_OVERLAP": 20,
    "PARENT_CHUNK_SIZE": 1000,
    "PARENT_CHUNK_OVERLAP": 0,
    
    # 檢索參數
    "K": 8
}

def main():
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(EXP_CONFIG["OUTPUT_PATH"]), exist_ok=True)

    # --- 2. 初始化 MongoDB Store ---
    print(f"🔗 Connecting to MongoDB: {EXP_CONFIG['MONGODB_DB']}...")
    mongo_store = SerializableMongoDBByteStore(
        connection_string=EXP_CONFIG["MONGODB_URI"],
        db_name=EXP_CONFIG["MONGODB_DB"],
        collection_name=EXP_CONFIG["MONGODB_COLLECTION"]
    )

    # --- 3. 初始化 Embedding 模型 ---
    print(f"🧬 Loading Embedding Model: {EXP_CONFIG['EMBED_MODEL_NAME']}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EXP_CONFIG["EMBED_MODEL_NAME"],
        model_kwargs={'device': EXP_CONFIG["DEVICE"]},
        encode_kwargs={'normalize_embeddings': True}
    )

    # --- 4. 初始化 Chroma ---
    print(f"📡 Connecting to ChromaDB Server at {EXP_CONFIG['CHROMA_HOST']}:{EXP_CONFIG['CHROMA_PORT']}...")
    client = chromadb.HttpClient(
        host=EXP_CONFIG["CHROMA_HOST"],
        port=EXP_CONFIG["CHROMA_PORT"]
    )
    vectorstore = Chroma(
        client=client,
        collection_name=EXP_CONFIG["CHROMA_COLLECTION"],
        embedding_function=embeddings
    )

    # --- 5. 配置 Parent Document Retriever ---
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=EXP_CONFIG["CHILD_CHUNK_SIZE"], 
        chunk_overlap=EXP_CONFIG["CHILD_CHUNK_OVERLAP"]
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=EXP_CONFIG["PARENT_CHUNK_SIZE"], 
        chunk_overlap=EXP_CONFIG["PARENT_CHUNK_OVERLAP"]
    )

    # 注意：這裡的 retriever 主要是用來 invoke 檢索
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=mongo_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_type=SearchType.mmr
    )

    # --- 6. 載入測試數據與索引 ---
    if not os.path.exists(EXP_CONFIG["QUESTIONS_PATH"]):
        print(f"❌ 錯誤: 找不到測試集檔案 {EXP_CONFIG['QUESTIONS_PATH']}")
        return

    with open(EXP_CONFIG["QUESTIONS_PATH"], 'r', encoding='utf-8') as f:
        questions = json.load(f)

    with open(EXP_CONFIG["INDEX_PATH"], 'r', encoding='utf-8') as f:
        index_data = json.load(f)

    # 建立映射表 {"file_name": "doc_id"}
    name_to_id = {doc['file_name']: doc['doc_id'] for doc in index_data['documents']}

    retrieval_results = []
    print(f"🔎 Running experiment [4-0 Fixed] for {len(questions)} queries...")

    # --- 7. 執行檢索實驗 ---
    for q in tqdm(questions, desc="Retrieving", unit="query"):
        # e5 模型建議 query 前綴加上 "query: "
        query_text = f"query: {q['question']}"
        
        # 檢索並取前 K 個
        # retrieved_docs = retriever.invoke(query_text)[:EXP_CONFIG["K"]]
        retrieved_docs = vectorstore.max_marginal_relevance_search(query_text, k=EXP_CONFIG["K"])
        
        candidates = []
        for doc in retrieved_docs:
            f_name = doc.metadata.get("file_name")
            candidates.append({
                "doc_id": name_to_id.get(f_name, "Unknown"),
                "file_name": f_name
            })
            
        retrieval_results.append({
            "question_id": q.get('id') or q.get('question_id'),
            "question": q['question'],
            "retrieved_candidates": candidates
        })

    # --- 8. 儲存結果 ---
    with open(EXP_CONFIG["OUTPUT_PATH"], 'w', encoding='utf-8') as f:
        json.dump(retrieval_results, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ 4-2 Retrieval JSON generated: {EXP_CONFIG['OUTPUT_PATH']}")

if __name__ == "__main__":
    main()