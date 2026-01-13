"""
Parent-Child 檢索測試腳本
驗證 Parent-Child 檢索方式的效果
"""
import os
import sys
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

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.storage import MongoDBByteStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
import chromadb


def initialize_parent_child_retriever():
    """初始化 Parent-Child 檢索器"""
    # 驗證配置
    try:
        Config.validate()
    except (ValueError, FileNotFoundError) as e:
        print(f"❌ 配置驗證失敗: {e}")
        sys.exit(1)
    
    # 初始化 Embedding 模型
    print(f"正在載入 Embedding 模型 ({Config.MODEL_NAME})...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=Config.MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ Embedding 模型載入成功")
    except Exception as e:
        print(f"❌ 載入 Embedding 模型失敗: {e}")
        sys.exit(1)
    
    # 連接向量資料庫
    print(f"正在連接到 Chroma 向量資料庫 ({Config.CHROMA_HOST}:{Config.CHROMA_PORT})...")
    try:
        # 使用 HttpClient 連接方式（與 ingest_parent_docs.py 一致）
        client = chromadb.HttpClient(
            host=Config.CHROMA_HOST,
            port=Config.CHROMA_PORT
        )
        vectorstore = Chroma(
            client=client,
            collection_name=Config.CHROMA_COLLECTION_NAME,
            embedding_function=embeddings
        )
        child_count = vectorstore._collection.count()
        print(f"✅ Chroma 連接成功，包含 {child_count} 個 Child 文檔片段")
        
        if child_count == 0:
            print("\n⚠️  警告：向量資料庫中沒有文檔！")
            print("   請先執行 ingest_parent_docs.py 建置資料庫。")
            sys.exit(1)
    except Exception as e:
        print(f"❌ 連接 Chroma 資料庫失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 連接 MongoDB（使用與 ingest_parent_docs.py 相同的 SerializableMongoDBByteStore）
    print(f"正在連接到 MongoDB ({Config.MONGODB_DB_NAME})...")
    try:
        # 導入自定義的 SerializableMongoDBByteStore
        from ingest_parent_docs import SerializableMongoDBByteStore
        store = SerializableMongoDBByteStore(
            connection_string=Config.MONGODB_CONNECTION_STRING,
            db_name=Config.MONGODB_DB_NAME,
            collection_name=Config.MONGODB_COLLECTION_NAME
        )
        print("✅ MongoDB 連接成功")
    except Exception as e:
        print(f"❌ 連接 MongoDB 失敗: {e}")
        print("   請確認 MongoDB 服務是否正在運行")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 建立切分器（必須與建置時相同）
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
    print("正在建立 ParentDocumentRetriever...")
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 5}  # 檢索前 5 個 child 片段，然後返回對應的 parent
    )
    print("✅ ParentDocumentRetriever 建立成功\n")
    
    return retriever, vectorstore, store


def test_parent_child_retrieval(queries: List[str], top_k: int = 5):
    """
    測試 Parent-Child 檢索功能
    
    Args:
        queries: 測試查詢列表
        top_k: 每個查詢返回的結果數量
    """
    print("=" * 80)
    print("Parent-Child 檢索測試")
    print("=" * 80)
    
    retriever, vectorstore, store = initialize_parent_child_retriever()
    
    # 執行測試查詢
    for i, query in enumerate(queries, 1):
        print(f"\n【測試 {i}】")
        print(f"查詢: {query}")
        print("=" * 80)
        
        try:
            # 使用 ParentDocumentRetriever 檢索
            print("【Parent-Child 檢索結果】")
            print("（先檢索相關的 Child 片段，然後返回對應的 Parent 文檔）")
            try:
                # ParentDocumentRetriever 使用 invoke 方法
                docs = retriever.invoke(query)
            except Exception as e:
                print(f"❌ 檢索失敗: {e}")
                continue
            
            if not docs:
                print("⚠️  未找到相關文檔")
                continue
            
            print(f"找到 {len(docs)} 個相關 Parent 文檔：\n")
            
            for j, doc in enumerate(docs, 1):
                # ParentDocumentRetriever 返回的是 Parent 文檔（完整上下文）
                source = doc.metadata.get("source", "未知來源")
                file_name = Path(source).name if source != "未知來源" else "未知來源"
                
                # 顯示 Parent 文檔內容（較長的完整上下文）
                content_preview = doc.page_content[:500].replace('\n', ' ')
                if len(doc.page_content) > 500:
                    content_preview += "..."
                
                print(f"  [{j}] Parent 文檔")
                print(f"      來源: {file_name}")
                print(f"      長度: {len(doc.page_content)} 字元")
                print(f"      內容: {content_preview}")
                print()
            
            # 同時顯示 Child 文檔（從向量檢索獲得的）
            print("【對比：僅 Child 文檔檢索（向量相似度）】")
            try:
                child_docs = vectorstore.similarity_search(query, k=top_k)
                if child_docs:
                    print(f"找到 {len(child_docs)} 個相關 Child 文檔片段：\n")
                    for j, doc in enumerate(child_docs, 1):
                        source = doc.metadata.get("source", "未知來源")
                        file_name = Path(source).name if source != "未知來源" else "未知來源"
                        content_preview = doc.page_content[:300].replace('\n', ' ')
                        if len(doc.page_content) > 300:
                            content_preview += "..."
                        
                        print(f"  [{j}] Child 片段")
                        print(f"      來源: {file_name}")
                        print(f"      長度: {len(doc.page_content)} 字元")
                        print(f"      內容: {content_preview}")
                        print()
            except Exception as e:
                print(f"⚠️  Child 檢索失敗: {e}\n")
        
        except Exception as e:
            print(f"❌ 檢索時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 80)


def main():
    """主函數"""
    # 定義測試查詢
    test_queries = [
        "退選規定",
        "學分數要求",
        "畢業資格",
        "請假程序",
        "獎學金申請"
    ]
    
    print("=" * 80)
    print("Parent-Child 檢索測試腳本")
    print("=" * 80)
    print("\n說明：")
    print("- Parent-Child 檢索會先找到相關的 Child 片段（小塊），")
    print("  然後返回對應的 Parent 文檔（大塊），提供更完整的上下文")
    print("- 這與普通檢索只返回小片段不同，可以提供更全面的資訊\n")
    
    # 執行測試
    test_parent_child_retrieval(test_queries, top_k=3)
    
    print("\n" + "=" * 80)
    print("✅ 預設測試完成")
    print("=" * 80)
    
    # 互動模式
    print("\n進入互動模式（輸入 'exit' 或 'quit' 退出）")
    print("-" * 80)
    
    try:
        retriever, vectorstore, store = initialize_parent_child_retriever()
        
        while True:
            query = input("\n請輸入查詢問題: ").strip()
            
            if query.lower() in ['exit', 'quit', '退出']:
                print("\n再見！")
                break
            
            if not query:
                print("請輸入有效的查詢")
                continue
            
            print()
            try:
                print(f"查詢: {query}")
                print("=" * 80)
                
                # Parent-Child 檢索
                print("【Parent-Child 檢索結果】")
                docs = retriever.invoke(query)
                
                if docs:
                    print(f"找到 {len(docs)} 個相關 Parent 文檔：\n")
                    for j, doc in enumerate(docs, 1):
                        source = doc.metadata.get("source", "未知來源")
                        file_name = Path(source).name if source != "未知來源" else "未知來源"
                        content_preview = doc.page_content[:500].replace('\n', ' ')
                        if len(doc.page_content) > 500:
                            content_preview += "..."
                        
                        print(f"  [{j}] 來源: {file_name}")
                        print(f"      長度: {len(doc.page_content)} 字元")
                        print(f"      內容: {content_preview}\n")
                else:
                    print("⚠️  未找到相關文檔\n")
                    
            except Exception as e:
                print(f"❌ 檢索失敗: {e}\n")
                import traceback
                traceback.print_exc()
            
    except KeyboardInterrupt:
        print("\n\n程式已中斷")
    except EOFError:
        print("\n\n程式已結束")


if __name__ == "__main__":
    main()
