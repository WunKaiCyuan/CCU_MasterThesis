"""
資料建置腳本
將 PDF/DOCX 文件處理並建立向量資料庫
"""
import os
import sys
import re
import logging
from pathlib import Path
from typing import List
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 導入配置模組
try:
    from .config import Config
except ImportError:
    # 直接運行時
    import sys
    import os
    ingestion_dir = os.path.dirname(os.path.abspath(__file__))
    if ingestion_dir not in sys.path:
        sys.path.insert(0, ingestion_dir)
    from config import Config

# 設定日誌
log_file = Path(__file__).parent / "ingest.log"
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


def load_documents(data_dir: str) -> List[Document]:
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
    
    raw_documents: List[Document] = []
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
            file_count += len(set(doc.metadata.get("source", "") for doc in docs))
            logger.info(f"  載入 {len(docs)} 個 {ext} 文件片段（來自 {len(set(doc.metadata.get('source', '') for doc in docs))} 個檔案）")
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


def split_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    切分文件為較小的區塊
    
    Args:
        documents: 原始文件列表
        chunk_size: 每個區塊的最大字元數
        chunk_overlap: 區塊間的重疊字元數
        
    Returns:
        切分後的文檔列表
    """
    logger.info(f"正在切分文件（chunk_size={chunk_size}, chunk_overlap={chunk_overlap}）...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n第[一二三四五六七八九十百]+條",  # 匹配換行後的「第一條」、「第二十條」等
            "第[一二三四五六七八九十百]+條",    # 匹配行首的條文標記
            "\n\n",
            "\n",
            "。",   # 中文句號也是很好的切分點
            " ",
            ""
        ],
        is_separator_regex=True  # 必須設定為 True，上面的正則表達式才會生效
    )
    
    final_chunks = text_splitter.split_documents(documents)
    
        # 添加檔案來源資訊（如果啟用）
    if Config.INCLUDE_SOURCE:
        logger.info("正在添加檔案來源資訊...")
        for doc in final_chunks:
            full_path = doc.metadata.get("source", "未知法規")
            file_name = os.path.basename(full_path)
            doc.page_content = f"[資料來源:{file_name}]\n{doc.page_content}"
    
    logger.info(f"切分完成！共產生 {len(final_chunks)} 個文檔區塊")
    
    # 統計資訊
    avg_chunk_size = sum(len(doc.page_content) for doc in final_chunks) / len(final_chunks) if final_chunks else 0
    logger.info(f"平均區塊大小: {avg_chunk_size:.1f} 字元")
    
    return final_chunks


def create_vectorstore(chunks: List[Document]) -> Chroma:
    """
    建立並持久化向量資料庫
    
    Args:
        chunks: 切分後的文檔列表
        
    Returns:
        Chroma 向量資料庫實例
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
        
        logger.info(f"正在將 {len(chunks)} 個文檔區塊添加到向量資料庫...")
        vectorstore.add_documents(documents=chunks)
        
        final_count = vectorstore._collection.count()
        logger.info(f"✅ 向量資料庫建立成功！目前包含 {final_count} 個文檔片段")
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"建立向量資料庫失敗: {e}", exc_info=True)
        raise


def main():
    """主函數"""
    try:
        logger.info("=" * 60)
        logger.info("開始資料建置流程")
        logger.info("=" * 60)
        
        # 驗證設定
        try:
            Config.validate()
        except (ValueError, FileNotFoundError) as e:
            logger.error(f"設定檔驗證失敗: {e}")
            sys.exit(1)
        
        # 1. 載入文件
        documents = load_documents(Config.DATA_DIR)
        
        # 2. 切分文件
        chunks = split_documents(
            documents,
            Config.CHUNK_SIZE,
            Config.CHUNK_OVERLAP
        )
        
        # 3. 建立向量資料庫
        vectorstore = create_vectorstore(chunks)
        
        logger.info("=" * 60)
        logger.info("✅ 資料建置完成！")
        logger.info(f"您現在可以執行應用程式進行查詢。")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("\n程式已由用戶中斷")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"資料建置失敗: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
