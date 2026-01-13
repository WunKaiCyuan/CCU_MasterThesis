"""
檢索測試腳本
用於驗證向量資料庫建置效果和檢索品質
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


def initialize_components():
    """初始化所有組件"""
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
    print(f"正在連接到向量資料庫 ({Config.CHROMA_HOST}:{Config.CHROMA_PORT})...")
    try:
        vectorstore = Chroma(
            embedding_function=embeddings,
            host=Config.CHROMA_HOST,
            port=Config.CHROMA_PORT,
            collection_name=Config.CHROMA_COLLECTION_NAME
        )
        
        # 檢查集合中的文檔數量
        doc_count = vectorstore._collection.count()
        print(f"✅ 向量資料庫連接成功")
        print(f"   集合名稱: {Config.CHROMA_COLLECTION_NAME}")
        print(f"   文檔數量: {doc_count} 個片段")
        
        if doc_count == 0:
            print("\n⚠️  警告：向量資料庫中沒有文檔！")
            print("   請先執行 ingest.py 建置資料庫。")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ 連接向量資料庫失敗: {e}")
        print(f"   請確認 Chroma 服務是否在 {Config.CHROMA_HOST}:{Config.CHROMA_PORT} 運行")
        sys.exit(1)
    
    return vectorstore, embeddings


def test_retrieval(queries: List[str], top_k: int = 5, vectorstore=None, embeddings=None):
    """
    測試檢索功能
    
    Args:
        queries: 測試查詢列表
        top_k: 每個查詢返回的結果數量
        vectorstore: 向量資料庫實例（如果為 None 則自動初始化）
        embeddings: Embedding 模型實例（如果為 None 則自動初始化）
    """
    # 如果沒有提供，則初始化
    if vectorstore is None or embeddings is None:
        vectorstore, embeddings = initialize_components()
    
    # 建立檢索器（MMR）
    print(f"\n建立檢索器（MMR，返回前 {top_k} 個結果）...")
    retriever_mmr = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": 20}
    )
    print("✅ 檢索器建立成功\n")
    
    # 執行測試查詢
    for i, query in enumerate(queries, 1):
        print(f"\n【測試 {i}】")
        print(f"查詢: {query}")
        print("-" * 80)
        
        try:
            # 方式1: 使用 MMR 檢索（多樣化結果）
            print("【方式 1: MMR 檢索（多樣化結果）】")
            try:
                docs_mmr = retriever_mmr.invoke(query)
            except Exception:
                docs_mmr = []
            
            if docs_mmr:
                print(f"找到 {len(docs_mmr)} 個相關文檔片段：\n")
                for j, doc in enumerate(docs_mmr, 1):
                    source = doc.metadata.get("source", "未知來源")
                    file_name = Path(source).name if source != "未知來源" else "未知來源"
                    content_preview = doc.page_content[:300].replace('\n', ' ')
                    if len(doc.page_content) > 300:
                        content_preview += "..."
                    
                    print(f"  [{j}] 來源: {file_name}")
                    print(f"      長度: {len(doc.page_content)} 字元")
                    print(f"      內容: {content_preview}\n")
            else:
                print("⚠️  未找到相關文檔\n")
            
            # 方式2: 使用相似度檢索（帶分數）
            print("【方式 2: 相似度檢索（帶分數）】")
            try:
                docs_with_scores = vectorstore.similarity_search_with_score(query, k=top_k)
                if docs_with_scores:
                    print(f"找到 {len(docs_with_scores)} 個相關文檔片段：\n")
                    for j, (doc, score) in enumerate(docs_with_scores, 1):
                        source = doc.metadata.get("source", "未知來源")
                        file_name = Path(source).name if source != "未知來源" else "未知來源"
                        content_preview = doc.page_content[:300].replace('\n', ' ')
                        if len(doc.page_content) > 300:
                            content_preview += "..."
                        
                        # 分數越小表示越相似（cosine distance）
                        print(f"  [{j}] 來源: {file_name}")
                        print(f"      相似度分數: {score:.4f} (越小越相似)")
                        print(f"      長度: {len(doc.page_content)} 字元")
                        print(f"      內容: {content_preview}\n")
                else:
                    print("⚠️  未找到相關文檔\n")
            except Exception as e:
                print(f"⚠️  相似度檢索失敗: {e}\n")
        
        except Exception as e:
            print(f"❌ 檢索時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("測試完成")
    print("=" * 80)


def main():
    """主函數"""
    print("=" * 80)
    print("檢索測試腳本")
    print("=" * 80)
    
    # 初始化組件（只初始化一次）
    vectorstore, embeddings = initialize_components()
    
    # 定義測試查詢（您可以根據實際需求修改）
    test_queries = [
        "退選規定",
        "學分數要求",
        "畢業資格",
        "請假程序",
        "獎學金申請"
    ]
    
    print("\n" + "=" * 80)
    print("開始執行預設測試查詢")
    print("=" * 80)
    
    # 執行測試
    test_retrieval(test_queries, top_k=5, vectorstore=vectorstore, embeddings=embeddings)
    
    print("\n" + "=" * 80)
    print("✅ 預設測試完成")
    print("=" * 80)
    
    # 互動模式
    print("\n進入互動模式（輸入 'exit' 或 'quit' 退出）")
    print("-" * 80)
    
    try:
        while True:
            query = input("\n請輸入查詢問題: ").strip()
            
            if query.lower() in ['exit', 'quit', '退出']:
                print("\n再見！")
                break
            
            if not query:
                print("請輸入有效的查詢")
                continue
            
            print()
            test_retrieval([query], top_k=5, vectorstore=vectorstore, embeddings=embeddings)
            print()
            
    except KeyboardInterrupt:
        print("\n\n程式已中斷")
    except EOFError:
        print("\n\n程式已結束")


if __name__ == "__main__":
    main()
