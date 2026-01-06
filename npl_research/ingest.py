import os
import chromadb
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredWordDocumentLoader

# ==========================================
# 1. 設定區
# ==========================================
DATA_DIR = "/Users/cyuan/RiderProjects/MasterThesis/DownloadedResources"

# 向量資料庫
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "thesis_collection"

# Embedding模型
# MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"  # 國際高效能模型
# MODEL_NAME = "infuse-ai/taiwan-embedding-v1"  # 台灣在地優化模型
# MODEL_NAME = "intfloat/multilingual-e5-large" # 雖然是國際模型，但繁體中文搜尋極強
MODEL_NAME = "intfloat/multilingual-e5-small" # 雖然是國際模型，但繁體中文搜尋極強

# ==========================================
# 2. 載入目錄下多種格式文件 (PDF, DOCX, ODT)
# ==========================================
print(f"正在讀取資料夾 {DATA_DIR} 中的所有文件...")

# 定義副檔名與對應的解析器
loaders = {
    ".pdf": PyPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
}

def create_loader(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in loaders:
        return loaders[ext](file_path)
    return None

# 使用 DirectoryLoader 的多格式支援 (2026 推薦寫法)
# 注意：glob 必須包含所有副檔名
raw_documents = []
for ext in loaders.keys():
    loader = DirectoryLoader(
        DATA_DIR, 
        glob=f"./*{ext}", 
        loader_cls=loaders[ext],
        show_progress=True # 這也會顯示進度條
    )
    raw_documents.extend(loader.load())

print(f"文件讀取完成，共讀取了 {len(raw_documents)} 個片段。")

# ==========================================
# 3. 初始化 Embedding 模型 (CPU 運作)
# ==========================================
print(f"正在載入 Embedding 模型 ({MODEL_NAME})...")
# Intel Mac 強制指定使用 CPU
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': 'cpu'}
)

# ==========================================
# 4. 語意理解切分 (Semantic Chunking)
# ==========================================
print("正在進行語意切分 (Semantic Chunking)...")
# 這會利用 Embedding 模型分析句子間的相關性，相似度高才會留在同一個區塊
text_splitter = SemanticChunker(
    embeddings, 
    breakpoint_threshold_type="percentile", # 基於百分位數的語意差異切分
    breakpoint_threshold_amount = 95.0
)

semantic_chunks = text_splitter.split_documents(raw_documents)
print(f"切分完成！共產生 {len(semantic_chunks)} 個區塊。")

# ==========================================
# 5. 建立並持久化向量資料庫 (ChromaDB)
# ==========================================
print(f"正在連線至 Docker ChromaDB 服務 ({CHROMA_HOST}:{CHROMA_PORT})...")

# 1. 建立一個持久化的 HTTP 客戶端連接 Docker
persistent_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# 2. 移除舊的 Collection（確保實驗數據乾淨），如果不存在會自動忽略
try:
    persistent_client.delete_collection(name=COLLECTION_NAME)
    print(f"已清理舊有的 Collection: {COLLECTION_NAME}")
except Exception:
    print("尚未存在舊的 Collection，準備建立新資料集。")

# 3. 將文件上傳至 Docker 服務
vectorstore = Chroma.from_documents(
    documents=semantic_chunks,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    client=persistent_client  # 指定使用剛剛建立的客戶端連接
)

print(f"✅ 向量資料庫建立成功！資料已儲存於 Docker 中的 {COLLECTION_NAME}。")
print("您現在可以執行 app.py 進行查詢。")
