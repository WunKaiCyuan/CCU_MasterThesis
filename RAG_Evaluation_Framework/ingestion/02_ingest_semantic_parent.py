import os
import torch
import uuid
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
import chromadb

from core.serializable_mongodb_byte_store import SerializableMongoDBByteStore
from core.document_loader import load_documents

SETTINGS = {
    "MODEL_NAME": "intfloat/multilingual-e5-small",
    "CHROMA_HOST": "localhost",
    "CHROMA_PORT": 8000,
    "CHROMA_COLLECTION": "ccu_rules_semantic_child",
    "MONGODB_URI": "mongodb://admin:UTWi1dCo6jFxNlS0@localhost:27017",
    "MONGODB_DB": "ccu_school_rules",
    "MONGODB_COLLECTION": "ccu_rules_semantic_parent",
    "DATA_DIR": "./data/",
    "PARENT_CHUNK_SIZE": 1000,
    "PARENT_CHUNK_OVERLAP": 0,
    "BREAKPOINT_PERCENTILE": 95 
}

def main():
    print("🚀 開始執行 [手動語意切分 + Parent-Child] 入庫流程...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=SETTINGS["MODEL_NAME"],
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

    client = chromadb.HttpClient(host=SETTINGS["CHROMA_HOST"], port=SETTINGS["CHROMA_PORT"])
    vectorstore = Chroma(
        client=client,
        collection_name=SETTINGS["CHROMA_COLLECTION"],
        embedding_function=embeddings
    )
    
    store = SerializableMongoDBByteStore(
        connection_string=SETTINGS["MONGODB_URI"],
        db_name=SETTINGS["MONGODB_DB"],
        collection_name=SETTINGS["MONGODB_COLLECTION"]
    )

    # 1. 初始化切分器
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=SETTINGS["PARENT_CHUNK_SIZE"], 
        chunk_overlap=SETTINGS["PARENT_CHUNK_OVERLAP"],
        separators=["\n?第[一二三四五六七八九十百]+條", "\n?[一二三四五六七八九十百]+、", "\n\n", "\n", "。", " "],
        is_separator_regex=True
    )
    
    # 這裡只用於計算，不直接傳給 retriever
    semantic_chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=SETTINGS["BREAKPOINT_PERCENTILE"]
    )

    # 2. 載入文件
    raw_documents = load_documents(SETTINGS["DATA_DIR"], clean=True)
    
    print("✂️ 正在手動處理語意切分並建立 Parent-Child 關聯...")
    
    for i, doc in enumerate(raw_documents):
        file_name = os.path.basename(doc.metadata.get("source", "unknown"))
        print(f"[{i+1}/{len(raw_documents)}] 處理檔案: {file_name}")
        
        # A. 先切出 Parent Chunks
        parent_chunks = parent_splitter.split_documents([doc])
        
        for p_chunk in parent_chunks:
            # 為這個 Parent Chunk 產生唯一 ID
            parent_id = str(uuid.uuid4())
            p_chunk.page_content = f"[資料來源:{file_name}]\n{p_chunk.page_content.strip()}"
            p_chunk.metadata["file_name"] = file_name
            p_chunk.metadata["doc_id"] = parent_id # 供參考用
            
            # B. 儲存 Parent 到 MongoDB
            store.mset([(parent_id, p_chunk)])
            
            # C. 用語意切分切出 Child Chunks
            child_chunks = semantic_chunker.split_documents([p_chunk])
            
            # D. 為 Child 加上 parent_id 並存入 Chroma
            for c_chunk in child_chunks:
                c_chunk.metadata["parent_id"] = parent_id
            
            vectorstore.add_documents(child_chunks)

    print(f"✅ 入庫完成！")
    print(f"📊 Chroma (語意子向量) 總數: {vectorstore._collection.count()}")

if __name__ == "__main__":
    main()