import os
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

from core.serializable_mongodb_byte_store import SerializableMongoDBByteStore
from core.document_loader import load_documents

# --- 直接設定參數 (Hardcoded Settings) ---
SETTINGS = {
    "MODEL_NAME": "intfloat/multilingual-e5-small",
    "CHROMA_HOST": "localhost",
    "CHROMA_PORT": 8000,
    "CHROMA_COLLECTION": "ccu_rules",
    "MONGODB_URI": "mongodb://admin:UTWi1dCo6jFxNlS0@localhost:27017",
    "MONGODB_DB": "ccu_school_rules",
    "MONGODB_COLLECTION": "ccu_rules",
    "DATA_DIR": "./data/",
    "PARENT_CHUNK_SIZE": 1000,
    "PARENT_CHUNK_OVERLAP": 0,
    "CHILD_CHUNK_SIZE": 200,
    "CHILD_CHUNK_OVERLAP": 20
}

def main():
    print("🚀 開始執行 Parent-Child 入庫流程...")

    # 1. 初始化 Embedding (使用 GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧬 使用設備: {device}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=SETTINGS["MODEL_NAME"],
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. 連接資料庫 (Chroma & MongoDB)
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

    # 3. 配置切分器
    separators = ["\n?第[一二三四五六七八九十百]+條", "\n?[一二三四五六七八九十百]+、", "\n\n", "\n", "。", " "]
    
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=SETTINGS["PARENT_CHUNK_SIZE"], 
        chunk_overlap=SETTINGS["PARENT_CHUNK_OVERLAP"],
        separators=separators, 
        is_separator_regex=True
    )
    
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=SETTINGS["CHILD_CHUNK_SIZE"], 
        chunk_overlap=SETTINGS["CHILD_CHUNK_OVERLAP"],
        separators=separators, 
        is_separator_regex=True
    )

    # 4. 初始化 Retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
    )

    # 5. 載入並預處理文檔
    if not os.path.exists(SETTINGS["DATA_DIR"]):
        print(f"❌ 錯誤: 找不到資料夾 {SETTINGS["DATA_DIR"]}")
        return

    raw_documents = load_documents(SETTINGS["DATA_DIR"], clean=True)
    
    processed_parents = []
    print("✂️ 正在進行父文件切分與 Metadata 注入...")
    for doc in parent_splitter.split_documents(raw_documents):
        # 取得純檔名
        source_path = doc.metadata.get("source", "unknown")
        file_name = os.path.basename(source_path)
        
        # 加上來源標籤，這對後續 RAG 回答很有幫助
        doc.page_content = f"[資料來源:{file_name}]\n{doc.page_content.strip()}"
        doc.metadata["file_name"] = file_name
        processed_parents.append(doc)

    # 6. 執行入庫
    print(f"📦 正在寫入 {len(processed_parents)} 個父文件段落至 MongoDB 與 Chroma...")
    # retriever.add_documents 會自動調用 child_splitter 把父文件切成子文件並存入 Chroma
    retriever.add_documents(processed_parents)
    
    print(f"✅ 入庫完成！")
    print(f"📊 Chroma 目前共有 {vectorstore._collection.count()} 個子向量。")

if __name__ == "__main__":
    main()