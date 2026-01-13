"""
Parent-Child 檢索方式資料建置腳本
使用 ParentDocumentRetriever 建立父子文檔結構
子文檔（小塊）用於向量檢索，父文檔（大塊）儲存在 MongoDB 中
"""
import os
import sys
import logging
from pathlib import Path
from typing import List
from core.serializable_mongodb_byte_store import SerializableMongoDBByteStore

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

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
import chromadb
import pickle
from ingestion.document_loader import load_documents


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

def create_parent_child_vectorstore(documents: List):
    """
    使用 Parent-Child 方式建立向量資料庫
    
    Args:
        documents: 原始文件列表
        
    Returns:
        tuple: (retriever, vectorstore, store)
            - retriever: ParentDocumentRetriever 實例
            - vectorstore: Chroma 向量資料庫實例
            - store: MongoDBByteStore 實例
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
        # 在 langchain 0.3.0 中，需要使用 chromadb.HttpClient
        client = chromadb.HttpClient(
            host=Config.CHROMA_HOST,
            port=Config.CHROMA_PORT
        )
        vectorstore = Chroma(
            client=client,
            collection_name=Config.CHROMA_COLLECTION_NAME,
            embedding_function=embeddings
        )
        
        # 檢查集合是否已存在
        existing_count = vectorstore._collection.count()
        should_reset = False
        if existing_count > 0:
            logger.warning(f"集合 '{Config.CHROMA_COLLECTION_NAME}' 已存在，包含 {existing_count} 個文檔")
            response = input("是否要重置集合並重新建立？(y/N): ").strip().lower()
            if response == 'y':
                logger.info("正在重置集合...")
                vectorstore.reset_collection()
                should_reset = True
            else:
                logger.info("將在現有集合中添加新文檔...")
        
        logger.info("✅ Chroma 向量資料庫連接成功")
    except Exception as e:
        logger.error(f"連接 Chroma 資料庫失敗: {e}", exc_info=True)
        raise
    
    logger.info(f"正在連接到 MongoDB ({Config.MONGODB_DB_NAME})...")
    try:
        # 使用自定義的 SerializableMongoDBByteStore 來確保正確序列化
        store = SerializableMongoDBByteStore(
            connection_string=Config.MONGODB_CONNECTION_STRING,
            db_name=Config.MONGODB_DB_NAME,
            collection_name=Config.MONGODB_COLLECTION_NAME
        )
        
        # 檢查 MongoDB 集合是否已有數據
        # 如果 Chroma 集合被重置，我們也應該清理 MongoDB，避免重複數據
        if should_reset:
            logger.info("正在清理 MongoDB 集合（與 Chroma 集合同步重置）...")
            try:
                # MongoDBByteStore 使用 MongoDB 的集合，我們需要直接連接到 MongoDB 來清理
                from pymongo import MongoClient
                mongo_client = MongoClient(Config.MONGODB_CONNECTION_STRING)
                db = mongo_client[Config.MONGODB_DB_NAME]
                collection = db[Config.MONGODB_COLLECTION_NAME]
                deleted_count = collection.delete_many({}).deleted_count
                logger.info(f"已從 MongoDB 刪除 {deleted_count} 個文檔")
                mongo_client.close()
            except Exception as e:
                logger.warning(f"清理 MongoDB 集合時發生錯誤（可能集合不存在或已為空）: {e}")
        elif existing_count > 0:
            logger.warning(f"⚠️  MongoDB 集合可能包含現有數據，重複寫入可能導致重複文檔")
            logger.warning(f"   建議在重置 Chroma 集合時也清理 MongoDB，或手動清理 MongoDB 集合")
        
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
    
    # 預處理和驗證文檔
    logger.info("正在驗證和預處理文檔...")
    processed_documents = []
    total_docs = len(documents)
    
    for i, doc in enumerate(documents, 1):
        try:
            # 驗證文檔內容
            if not doc.page_content or not doc.page_content.strip():
                logger.warning(f"  跳過空文檔 {i}: {doc.metadata.get('filename', '未知')}")
                continue
            
            # 清理和驗證 metadata
            cleaned_metadata = {}
            if doc.metadata:
                for key, value in doc.metadata.items():
                    # MongoDB 和 Chroma 只支援基本類型
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        cleaned_metadata[key] = value
                    else:
                        # 將其他類型轉換為字符串
                        cleaned_metadata[key] = str(value)
            
            # 確保必要的 metadata 欄位存在
            if "source" not in cleaned_metadata:
                cleaned_metadata["source"] = doc.metadata.get("source", f"document_{i}")
            if "filename" not in cleaned_metadata:
                source = cleaned_metadata.get("source", "")
                cleaned_metadata["filename"] = os.path.basename(source) if source else f"document_{i}"
            
            # 添加來源資訊（如果啟用）
            page_content = doc.page_content.strip()
            if Config.INCLUDE_SOURCE:
                file_name = cleaned_metadata.get("filename", "未知")
                page_content = f"[資料來源:{file_name}]\n{page_content}"
            
            # 創建清理後的文檔物件
            clean_doc = Document(
                page_content=page_content,
                metadata=cleaned_metadata
            )
            
            processed_documents.append(clean_doc)
            logger.info(f"  ✅ 預處理文檔 {i}/{total_docs}: {cleaned_metadata.get('filename', '未知')}")
            
        except Exception as prep_error:
            logger.error(f"  ❌ 預處理文檔 {i} 時發生錯誤: {prep_error}")
            logger.error(f"     文檔: {doc.metadata.get('filename', '未知') if doc.metadata else '未知'}")
            continue
    
    if not processed_documents:
        raise ValueError("沒有有效的文檔可以添加！所有文檔都為空或格式不正確。")
    
    logger.info(f"預處理完成，{len(processed_documents)}/{total_docs} 個文檔有效")
    
    # 添加文檔
    logger.info(f"正在將 {len(processed_documents)} 個文檔添加到 Parent-Child 結構中...")
    logger.info("（這可能需要一些時間，因為需要建立 Child 向量和 Parent 文檔映射）")
    
    try:
        # 逐個處理文檔，以便更好地追蹤和處理錯誤
        successfully_added = 0
        failed_docs = []
        
        for i, doc in enumerate(processed_documents, 1):
            try:
                # 逐個添加文檔
                retriever.add_documents([doc])
                successfully_added += 1
                logger.info(f"  ✅ 已處理 {i}/{len(processed_documents)} 個文檔: {doc.metadata.get('filename', '未知')}")
                
            except Exception as doc_error:
                error_msg = str(doc_error)
                logger.error(f"  ❌ 處理文檔 {i} 時發生錯誤: {error_msg}")
                logger.error(f"     文檔: {doc.metadata.get('filename', '未知')}")
                logger.error(f"     來源: {doc.metadata.get('source', '未知')}")
                logger.error(f"     內容長度: {len(doc.page_content)} 字元")
                # 記錄完整的錯誤堆疊（僅在 DEBUG 模式下）
                import traceback
                logger.debug(f"     完整錯誤堆疊:\n{traceback.format_exc()}")
                failed_docs.append((i, error_msg))
                # 繼續處理下一個文檔，不中斷整個流程
                continue
        
        if successfully_added == 0:
            error_summary = "\n".join([f"  文檔 {idx}: {msg}" for idx, msg in failed_docs[:10]])
            raise ValueError(
                f"沒有文檔成功添加！所有 {len(processed_documents)} 個文檔都失敗了。\n"
                f"失敗原因（前10個）：\n{error_summary}"
            )
        
        # 統計資訊
        child_count = vectorstore._collection.count()
        
        # 計算平均統計（只計算成功添加的文檔）
        failed_indices = {idx for idx, _ in failed_docs}
        successful_docs = [doc for i, doc in enumerate(processed_documents, 1) if i not in failed_indices]
        avg_parent_size = sum(len(doc.page_content) for doc in successful_docs) / len(successful_docs) if successful_docs else 0
        
        logger.info("")
        logger.info(f"✅ Parent-Child 資料庫建立成功！")
        logger.info(f"   原始文檔數量: {total_docs} 個")
        logger.info(f"   有效文檔數量: {len(processed_documents)} 個")
        logger.info(f"   成功添加: {successfully_added} 個")
        if failed_docs:
            logger.warning(f"   失敗文檔: {len(failed_docs)} 個")
            if len(failed_docs) <= 5:
                for idx, msg in failed_docs:
                    logger.warning(f"     - 文檔 {idx}: {msg[:100]}")
        logger.info(f"   Chroma (Child 文檔): {child_count} 個向量片段")
        logger.info(f"   MongoDB (Parent 文檔): {successfully_added} 個父文檔")
        logger.info(f"   平均父文檔大小: {avg_parent_size:.1f} 字元")
        logger.info(f"   每個 Parent 文檔包含多個 Child 片段用於檢索")
        logger.info(f"   Child 片段大小: {Config.CHILD_CHUNK_SIZE} 字元")
        logger.info(f"   Parent 片段大小: {Config.PARENT_CHUNK_SIZE} 字元")
        
        return retriever, vectorstore, store
        
    except Exception as e:
        logger.error(f"添加文檔時發生錯誤: {e}", exc_info=True)
        logger.error("請檢查：")
        logger.error("  1. MongoDB 服務是否正常運行")
        logger.error("  2. MongoDB 連接字串是否正確")
        logger.error("  3. MongoDB 用戶是否有寫入權限")
        logger.error("  4. 文檔格式是否正確（metadata 是否包含不可序列化的類型）")
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
        
        # 1. 載入文件（啟用文本清理）
        documents = load_documents(Config.DATA_DIR, clean=True, logger_instance=logger)
        
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
