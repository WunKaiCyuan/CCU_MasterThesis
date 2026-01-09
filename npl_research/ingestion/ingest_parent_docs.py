"""
Parent-Child 檢索方式資料建置腳本
使用 ParentDocumentRetriever 建立父子文檔結構
子文檔（小塊）用於向量檢索，父文檔（大塊）儲存在 MongoDB 中
"""
import os
import sys
import re
import logging
from pathlib import Path
from typing import List

# 導入配置
try:
    from .config import Config
except ImportError:
    import sys
    import os
    ingestion_dir = os.path.dirname(os.path.abspath(__file__))
    if ingestion_dir not in sys.path:
        sys.path.insert(0, ingestion_dir)
    from config import Config

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.storage import MongoDBByteStore

# 嘗試不同的導入路徑
try:
    from langchain.retrievers import ParentDocumentRetriever
except ImportError:
    try:
        from langchain_community.retrievers import ParentDocumentRetriever
    except ImportError:
        try:
            from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
        except ImportError:
            raise ImportError(
                "無法找到 ParentDocumentRetriever。\n"
                "請確認 LangChain 版本是否 >= 0.3.0，或嘗試安裝：\n"
                "pip install langchain>=0.3.0"
            )


# 設定日誌
log_file = Path(__file__).parent / "ingest_parent.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    清理文本內容
    - 將多個連續換行合併為單一換行
    - 移除多餘的空格（但保留必要的空格）
    """
    # 將多個連續換行 (\n\n\n...) 取代為單一換行 (\n)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 移除行首行尾的多餘空格，但保留行內必要的空格
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines]
    text = '\n'.join(cleaned_lines)
    
    # 移除多餘的連續空格（但保留單一空格）
    text = re.sub(r' {2,}', ' ', text)
    
    return text


def load_documents(data_dir: str) -> List:
    """
    載入目錄下的所有文件（PDF, DOCX）
    
    Args:
        data_dir: 資料目錄路徑
        
    Returns:
        文件列表
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"資料目錄不存在: {data_dir}")
    
    logger.info(f"正在讀取資料夾 {data_dir} 中的所有文件...")
    
    # 定義副檔名與對應的解析器
    loaders_info = {
        ".pdf": {
            "cls": PyPDFLoader,
            "kwargs": {"mode": "single"}
        },
        ".docx": {
            "cls": UnstructuredWordDocumentLoader,
            "kwargs": {"mode": "single"}
        }
    }
    
    raw_documents = []
    file_count = 0
    
    for ext, config in loaders_info.items():
        try:
            loader = DirectoryLoader(
                data_dir,
                glob=f"./*{ext}",
                loader_cls=config["cls"],
                loader_kwargs=config["kwargs"],
                show_progress=True
            )
            docs = loader.load()
            raw_documents.extend(docs)
            unique_files = len(set(doc.metadata.get("source", "") for doc in docs))
            file_count += unique_files
            logger.info(f"  載入 {len(docs)} 個 {ext} 文件片段（來自 {unique_files} 個檔案）")
        except Exception as e:
            logger.warning(f"載入 {ext} 檔案時發生錯誤: {e}")
            continue
    
    if not raw_documents:
        raise ValueError(f"在 {data_dir} 中未找到任何支援的文件（PDF 或 DOCX）")
    
    logger.info(f"文件讀取完成，共讀取了 {len(raw_documents)} 個片段（來自 {file_count} 個檔案）")
    
    # 清理文本內容
    logger.info("正在清理文本內容...")
    for doc in raw_documents:
        doc.page_content = clean_text(doc.page_content)
    
    return raw_documents


def create_parent_child_vectorstore(documents: List):
    """
    使用 Parent-Child 方式建立向量資料庫
    
    Args:
        documents: 原始文件列表
    """
    logger.info("正在初始化 Embedding 模型...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=Config.MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"Embedding 模型 {Config.MODEL_NAME} 載入成功")
    except Exception as e:
        logger.error(f"載入 Embedding 模型失敗: {e}", exc_info=True)
        raise
    
    logger.info(f"正在連接到 Chroma 向量資料庫 ({Config.CHROMA_HOST}:{Config.CHROMA_PORT})...")
    try:
        vectorstore = Chroma(
            embedding_function=embeddings,
            host=Config.CHROMA_HOST,
            port=Config.CHROMA_PORT,
            collection_name=Config.CHROMA_COLLECTION_NAME
        )
        
        # 檢查集合是否已存在
        existing_count = vectorstore._collection.count()
        if existing_count > 0:
            logger.warning(f"集合 '{Config.CHROMA_COLLECTION_NAME}' 已存在，包含 {existing_count} 個文檔")
            response = input("是否要重置集合並重新建立？(y/N): ").strip().lower()
            if response == 'y':
                logger.info("正在重置集合...")
                vectorstore.reset_collection()
            else:
                logger.info("將在現有集合中添加新文檔...")
        
        logger.info("✅ Chroma 向量資料庫連接成功")
    except Exception as e:
        logger.error(f"連接 Chroma 資料庫失敗: {e}", exc_info=True)
        raise
    
    logger.info(f"正在連接到 MongoDB ({Config.MONGODB_DB_NAME})...")
    try:
        store = MongoDBByteStore(
            connection_string=Config.MONGODB_CONNECTION_STRING,
            db_name=Config.MONGODB_DB_NAME,
            collection_name=Config.MONGODB_COLLECTION_NAME
        )
        logger.info("✅ MongoDB 連接成功")
    except Exception as e:
        logger.error(f"連接 MongoDB 失敗: {e}", exc_info=True)
        logger.error("請確認 MongoDB 服務是否正在運行，且連接字串正確")
        raise
    
    # 建立切分器
    logger.info("正在設定 Parent-Child 切分器...")
    logger.info(f"  Parent 切分: chunk_size={Config.PARENT_CHUNK_SIZE}, overlap={Config.PARENT_CHUNK_OVERLAP}")
    logger.info(f"  Child 切分: chunk_size={Config.CHILD_CHUNK_SIZE}, overlap={Config.CHILD_CHUNK_OVERLAP}")
    
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.PARENT_CHUNK_SIZE,
        chunk_overlap=Config.PARENT_CHUNK_OVERLAP,
        separators=[
            "\n第[一二三四五六七八九十百]+條",
            "第[一二三四五六七八九十百]+條",
            "\n\n",
            "\n",
            "。",
            " ",
            ""
        ],
        is_separator_regex=True
    )
    
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHILD_CHUNK_SIZE,
        chunk_overlap=Config.CHILD_CHUNK_OVERLAP,
        separators=[
            "\n第[一二三四五六七八九十百]+條",
            "第[一二三四五六七八九十百]+條",
            "\n\n",
            "\n",
            "。",
            " ",
            ""
        ],
        is_separator_regex=True
    )
    
    # 建立 ParentDocumentRetriever
    logger.info("正在建立 ParentDocumentRetriever...")
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 5}  # 檢索前 5 個 child 片段，然後返回對應的 parent
    )
    
    # 添加文檔
    logger.info(f"正在將 {len(documents)} 個文檔添加到 Parent-Child 結構中...")
    logger.info("（這可能需要一些時間，因為需要建立 Child 向量和 Parent 文檔映射）")
    
    try:
        retriever.add_documents(documents)
        
        # 統計資訊
        child_count = vectorstore._collection.count()
        
        logger.info(f"✅ Parent-Child 資料庫建立成功！")
        logger.info(f"   Chroma (Child 文檔): {child_count} 個向量片段")
        logger.info(f"   MongoDB (Parent 文檔): 已儲存父文檔")
        logger.info(f"   每個 Parent 文檔包含多個 Child 片段用於檢索")
        
        return retriever, vectorstore, store
        
    except Exception as e:
        logger.error(f"添加文檔時發生錯誤: {e}", exc_info=True)
        raise


def main():
    """主函數"""
    try:
        logger.info("=" * 60)
        logger.info("開始 Parent-Child 資料建置流程")
        logger.info("=" * 60)
        
        # 驗證設定
        try:
            Config.validate()
        except (ValueError, FileNotFoundError) as e:
            logger.error(f"設定檔驗證失敗: {e}")
            sys.exit(1)
        
        # 1. 載入文件
        documents = load_documents(Config.DATA_DIR)
        
        # 2. 建立 Parent-Child 向量資料庫
        retriever, vectorstore, store = create_parent_child_vectorstore(documents)
        
        logger.info("=" * 60)
        logger.info("✅ Parent-Child 資料建置完成！")
        logger.info(f"您現在可以使用 ParentDocumentRetriever 進行檢索。")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("\n程式已由用戶中斷")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"資料建置失敗: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
