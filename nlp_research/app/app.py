"""
RAG 系統 API 版本
使用 FastAPI 提供 REST API 介面
"""
import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional, List
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, field_validator
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from phoenix.otel import register
import chromadb
from langchain.retrievers import ParentDocumentRetriever
from core.config import Config
from core.serializable_mongodb_byte_store import SerializableMongoDBByteStore
from langchain.retrievers.multi_vector import SearchType


# 驗證設定
try:
    Config.validate()
except ValueError as e:
    print(f"❌ 設定檔錯誤: {e}")
    sys.exit(1)

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'app.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ==========================================
# 應用程式生命週期管理
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式生命週期管理（啟動和關閉）"""
    # 啟動時執行
    try:
        global rag_chain
        logger.info("開始初始化 RAG 組件...")
        rag_chain = initialize_components()
        logger.info("API 服務啟動成功")
    except Exception as e:
        logger.critical(f"服務啟動失敗: {e}", exc_info=True)
        raise
    
    yield
    
    # 關閉時執行（如果需要清理資源）
    logger.info("應用程式正在關閉...")


# 初始化 FastAPI 應用
app = FastAPI(
    title="CCU 校規 RAG 系統 API",
    description="基於微調 TAIDE 模型的校規問答系統 REST API",
    version="1.0.0",
    lifespan=lifespan
)

# 全域變數儲存初始化的組件
rag_chain = None
vectorstore = None
global_retriever = None

# 設定模板和靜態文件
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
if os.path.exists(templates_dir):
    templates = Jinja2Templates(directory=templates_dir)
else:
    templates = None

# 如果 static 目錄存在，則掛載靜態文件
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 掛載文檔目錄
if os.path.exists(Config.DATA_DIR):
    app.mount("/documents", StaticFiles(directory=Config.DATA_DIR), name="documents")


# ==========================================
# 請求/回應模型定義
# ==========================================
class QueryRequest(BaseModel):
    """查詢請求模型"""
    question: str = Field(..., description="要詢問的問題", min_length=1, max_length=500)
    skip_llm: bool = Field(False, description="是否跳過 LLM 生成，僅執行檢索")
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError("問題不能為空")
        return v.strip()


class ExplainRequest(BaseModel):
    """解釋請求模型"""
    question: str = Field(..., description="原始問題")
    file_name: str = Field(..., description="檔案名稱")
    content: str = Field(..., description="文獻內容片段")


class Source(BaseModel):
    """來源文獻模型"""
    file_name: str = Field(..., description="檔案名稱")
    content: str = Field(..., description="文獻內容片段")
    download_url: Optional[str] = Field(None, description="下載連結")


class QueryResponse(BaseModel):
    """查詢回應模型"""
    answer: str = Field(..., description="系統回答")
    processing_time: float = Field(..., description="處理時間（秒）")
    timestamp: str = Field(..., description="查詢時間戳")
    sources: List[Source] = Field([], description="參考來源文獻")


class HealthResponse(BaseModel):
    """健康檢查回應模型"""
    status: str = Field(..., description="服務狀態")
    vector_db_connected: bool = Field(..., description="向量資料庫連接狀態")
    vector_db_count: Optional[int] = Field(None, description="向量資料庫文檔數量")
    llm_model: str = Field(..., description="LLM 模型名稱")


# ==========================================
# 初始化函數
# ==========================================
def initialize_components():
    """初始化所有 RAG 組件"""
    global rag_chain, vectorstore, llm, global_retriever
    
    logger.info("開始初始化 RAG 組件...")
    
    # 1. 啟動全地端監控 (Arize Phoenix)
    logger.info("初始化 Phoenix 監控...")
    tracer_provider = register(
        project_name=Config.PHOENIX_PROJECT_NAME,
        auto_instrument=True,
        set_global_tracer_provider=False,
        batch=True,
        endpoint=Config.PHOENIX_ENDPOINT
    )
    
    # 2. 載入 Embedding 模型和向量資料庫
    logger.info("載入 Embedding 模型和向量資料庫...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=Config.MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
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
        collection_count = vectorstore._collection.count()
        logger.info(f"向量資料庫連接成功，共 {collection_count} 個文檔")
    except Exception as e:
        logger.error(f"載入向量資料庫失敗: {e}", exc_info=True)
        raise

    try:
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
    
    # 3. 設定檢索器
    # retriever = vectorstore.as_retriever(
    #     search_type="mmr",
    #     search_kwargs={"k": Config.RETRIEVER_K, "fetch_k": Config.RETRIEVER_FETCH_K}
    # )
    
    # 建立切分器（必須與建置時相同）
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHILD_CHUNK_SIZE,
        chunk_overlap=Config.CHILD_CHUNK_OVERLAP,
        separators=[
            "\n?第[一二三四五六七八九十百]+條",
            "\n?[一二三四五六七八九十百]+、",
            "\n\n",
            "\n",
            "。",
            " ",
            ""
        ],
        is_separator_regex=True
    )

    parent_document_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        search_kwargs={"k": Config.RETRIEVER_K},  # 檢索前 K 個 child 片段，然後返回對應的 parent
        search_type=SearchType.similarity
    )
    
    # 將 retriever 存為全域變數，供 skip_llm 模式使用
    global global_retriever
    global_retriever = parent_document_retriever
    
    # 4. 初始化 LLM 模型
    logger.info(f"初始化 LLM 模型 ({Config.LLM_MODEL_NAME})...")
    try:
        llm = ChatOllama(
            model=Config.LLM_MODEL_NAME,
            temperature=Config.LLM_TEMPERATURE,
            num_thread=Config.LLM_THREADS
        )
        logger.info("LLM 模型初始化成功")
    except Exception as e:
        logger.error(f"初始化 LLM 模型失敗: {e}", exc_info=True)
        raise
    
    # 5. 設計 RAG 流程
    template = """你是一位專業的國立中正大學校規諮詢助手。你的任務是根據提供的文獻內容，以專業、準確、友善的方式回答使用者關於校規的問題。

## 角色定位
- 你是一位熟悉中正大學所有校規、法規、行政規章的專業助手
- 你的回答必須完全基於提供的文獻內容，不能編造或推測
- 保持專業、友善、有禮貌的語氣

## 回答原則
1. **準確性優先**：所有回答必須嚴格依據提供的文獻內容，不得自行推測或補充未提及的資訊
2. **完整性**：如果文獻中有相關資訊，請提供完整且具體的回答，包含相關的條文、規定、程序等
3. **明確性**：如果文獻中沒有相關資訊或資訊不足，必須明確告知「根據提供的文獻內容，未找到相關資訊」或「文獻中關於此問題的資訊不足」
4. **引用來源**：回答時必須明確指出參考的檔案名稱（如果文獻內容中有提及）
5. **結構化回答**：對於複雜的問題，請以條列式或分段落的方式組織回答，讓使用者易於理解

## 回答格式
- 使用繁體中文回答
- 語氣專業但友善
- 重要資訊可以強調（如條文號碼、關鍵日期、重要程序等）
- 如果涉及多個相關條文，請分別說明

## 特別注意事項
- **絕對禁止**編造任何資訊，即使是看似合理的推測也不允許
- 如果使用者問的問題在文獻中找不到相關內容，直接說明「未找到相關資訊」，不要試圖用一般知識回答
- 如果文獻內容有矛盾或不清楚的地方，請指出這一點

## 文獻內容
{context}

## 使用者問題
{question}

## 請根據上述原則回答："""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        """格式化檢索到的文檔"""
        if not docs:
            return "未找到相關文獻內容。"
        return ("\n\n" + "-"*20 + "\n\n").join(doc.page_content for doc in docs)
    
    rag_chain = (
        {
            # "context": retriever | format_docs,
            "context": parent_document_retriever,
            "question": RunnablePassthrough()
        }
        | RunnablePassthrough.assign(
            formatted_context=lambda x: format_docs(x["context"])
        )
        | RunnablePassthrough.assign(
            answer=(
                lambda x: ChatPromptTemplate.from_template(template).format(context=x["formatted_context"], question=x["question"])
            ) | llm | StrOutputParser()
        )
    )
    
    logger.info("所有組件初始化完成")
    return rag_chain


# ==========================================
# API 端點
# ==========================================
@app.get("/", response_class=HTMLResponse, tags=["系統"])
async def root(request: Request):
    """根路徑，返回網頁介面"""
    if templates is None:
        raise HTTPException(status_code=500, detail="模板目錄不存在")
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api", tags=["系統"])
async def api_info():
    """API 資訊端點"""
    return {
        "name": "CCU 校規 RAG 系統 API",
        "version": "1.0.0",
        "description": "基於微調 TAIDE 模型的校規問答系統",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["系統"])
async def health_check():
    """健康檢查端點"""
    try:
        vector_db_connected = False
        vector_db_count = None
        
        if vectorstore:
            try:
                vector_db_count = vectorstore._collection.count()
                vector_db_connected = True
            except Exception as e:
                logger.warning(f"向量資料庫連接檢查失敗: {e}")
        
        return HealthResponse(
            status="healthy" if vector_db_connected and rag_chain else "degraded",
            vector_db_connected=vector_db_connected,
            vector_db_count=vector_db_count,
            llm_model=Config.LLM_MODEL_NAME
        )
    except Exception as e:
        logger.error(f"健康檢查失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"健康檢查失敗: {str(e)}")


@app.post("/query", response_model=QueryResponse, tags=["查詢"])
async def query(request: QueryRequest):
    """
    查詢端點
    
    - **question**: 要詢問的問題（1-500 字元）
    
    返回系統的回答和處理時間
    """
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG 系統尚未初始化完成")
    
    start_time = time.time()
    logger.info(f"收到查詢: {request.question[:100]}...")
    
    try:
        # 執行 RAG 查詢
        if request.skip_llm:
            # 純檢索模式
            logger.info("執行純檢索模式 (Skip LLM)")
            # 直接使用 retriever 檢索文檔
            # 注意：rag_chain 中的 retriever 是 parent_document_retriever
            # 我們需要存取 rag_chain 內部的 retriever，或者直接使用全域變數
            
            # 從全域變數 rag_chain 中提取 retriever 比較困難，因為它已經被封裝在 Runnable 中
            # 但我們在 initialize_components 中有定義 parent_document_retriever
            # 這裡簡單起見，我們重新建立一個 retriever 實例或者改用全域變數方式
            # 由於 initialize_components 是封閉的，我們需要修改它把 retriever 存到全域，
            # 或者我們可以利用 rag_chain 的第一步 invoke
            
            # 使用 Runnable 的 step 1
            # rag_chain = (step1 | step2 | ...)
            # 這裡我們稍微 hack 一下，直接使用 vectorstore 進行相似度搜尋
            # 但這樣會失去 ParentDocumentRetriever 的功能 (返回大區塊)
            
            # 最好的方式是修改 initialize_components 讓它返回 retriever，或者設為全域
            # 暫時解決方案：修改 initialize_components 將 retriever 存為 app.state.retriever
            # 但這需要重啟。
            
            # 替代方案：執行 chain 但攔截
            # 由於 chain 是固定的，我們無法輕易攔截中間結果而不執行 LLM
            
            # 讓我們修改 initialize_components，把 retriever 設為全域變數
            global global_retriever
            if 'global_retriever' in globals() and global_retriever:
                docs = global_retriever.invoke(request.question)
                answer = "已完成檢索 (LLM Skipped)"
            else:
                # Fallback: 如果沒有全域 retriever，只好執行完整 chain (這不符合需求)
                # 或者我們嘗試從 vectorstore 檢索 (僅 Child chunks)
                logger.warning("找不到全域 retriever，使用 vectorstore 進行簡易檢索")
                if vectorstore:
                     docs = vectorstore.similarity_search(request.question, k=Config.RETRIEVER_K)
                     answer = "已完成檢索 (Fallback: Vectorstore Only)"
                else:
                     raise HTTPException(status_code=500, detail="Retriever not available")
        else:
            # 完整 RAG 模式
            result = rag_chain.invoke(request.question)
            
            # 解析結果
            if isinstance(result, dict):
                 answer = result.get("answer", "")
                 docs = result.get("context", [])
            else:
                 # 相容舊版回傳
                 answer = str(result)
                 docs = []
             
        # 格式化來源
        sources = []
        for doc in docs:
            # 嘗試從 metadata 獲取檔名
            file_name = doc.metadata.get("file_name") or doc.metadata.get("filename") or "Unknown"
            
            # 產生下載連結
            # 假設 Config.DATA_DIR 中的檔案可以直接透過 /documents/<file_name> 訪問
            # 注意：這裡假設 file_name 就是檔名，但在 ingest_parent_docs.py 中 file_name 確實是 basename
            download_url = f"/documents/{file_name}" if file_name != "Unknown" else None
            
            # 如果是 Parent Document，可能我們已經在 page_content 中添加了 "[資料來源:xxx]"，這裡可以不做額外處理
            # 或者我們可以保留原始 page_content
            sources.append(Source(
                file_name=file_name,
                content=doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                download_url=download_url
            ))
        
        elapsed_time = time.time() - start_time
        logger.info(f"查詢完成，耗時 {elapsed_time:.2f} 秒")
        
        return QueryResponse(
            answer=answer,
            processing_time=round(elapsed_time, 2),
            timestamp=datetime.now().isoformat(),
            sources=sources
        )
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"處理查詢時發生錯誤: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"處理查詢時發生錯誤: {str(e)}"
        )


@app.get("/query", response_model=QueryResponse, tags=["查詢"])
async def query_get(
    question: str = Query(..., description="要詢問的問題", min_length=1, max_length=500)
):
    """
    GET 方式查詢端點（方便瀏覽器測試）
    
    - **question**: 要詢問的問題（1-500 字元）
    
    返回系統的回答和處理時間
    """
    # 驗證問題長度
    if len(question.strip()) > Config.MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"問題長度不能超過 {Config.MAX_QUERY_LENGTH} 個字元"
        )
    
    # 使用 POST 端點的邏輯
    request = QueryRequest(question=question.strip())
    return await query(request)


@app.post("/explain", tags=["說明"])
async def explain(request: ExplainRequest):
    """
    解釋端點
    
    請求 LLM 解釋特定文獻內容如何回答使用者的問題。
    """
    if not llm:
        raise HTTPException(status_code=503, detail="LLM 尚未初始化完成")
        
    try:
        # 建構完整檔案路徑
        file_path = os.path.join(Config.DATA_DIR, request.file_name)
        
        # 載入完整文件內容
        # 注意：這裡假設 ingestion 已經將 document_loader.py 移至 core 目錄
        # 如果尚未移動，請確保路徑正確
        from core.document_loader import load_single_document
        
        # 讀取完整文件 (啟用基本清理)
        try:
            full_doc = load_single_document(file_path, clean=True)
            full_content = full_doc.page_content
            # 截斷過長內容以避免超過 Context Window (例如最多 15000 字)
            if len(full_content) > 15000:
                 logger.warning(f"文件 {request.file_name} 內容過長 ({len(full_content)} 字)，進行截斷")
                 full_content = full_content[:15000] + "\n...(內容已截斷)..."
        except FileNotFoundError:
            # 如果找不到文件 (可能是測試環境或路徑問題)，降級使用 request.content
            logger.warning(f"找不到原始文件 {file_path}，降級使用片段內容")
            full_content = request.content
        except Exception as e:
            logger.error(f"讀取原始文件失敗: {e}，降級使用片段內容")
            full_content = request.content

        explanation_template = """你是一位專業的校規諮詢助教。使用者提出了一個問題，並提供了一份完整的參考文件內容。
你的任務是詳細分析這份文件，判斷它是否包含回答使用者問題的資訊。

## 使用者問題
{question}

## 參考文件全文 ({file_name})
{content}

## 分析要求
1. **關聯性判斷**：請先判斷這份文件是否與問題相關。
2. **具體引用**：如果相關，請務必指出是**第幾條**、**第幾項**或**哪一個章節**回答了這個問題。
3. **回答生成**：根據文件內容，簡潔扼要地回答問題。
4. **無關處理**：如果整份文件都與問題無關，請直接回答「此文件內容與問題無直接關聯」。

## 你的回答
請儘量控制在 300 字以內，並採用以下格式：
- **關聯性**：(高度相關/部分相關/無關)
- **依據條文**：(例如：第三條第二項、教務章程第五章等，若無則免填)
- **說明**：(你的分析與回答)
"""
        prompt = ChatPromptTemplate.from_template(explanation_template)
        chain = prompt | llm | StrOutputParser()
        
        result = chain.invoke({
            "question": request.question,
            "file_name": request.file_name,
            "content": full_content
        })
        
        return {"explanation": result}
        
    except Exception as e:
        logger.error(f"解釋生成失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"解釋生成失敗: {str(e)}")


# ==========================================
# 錯誤處理
# ==========================================
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """處理驗證錯誤"""
    logger.warning(f"驗證錯誤: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """處理一般異常"""
    logger.error(f"未預期的錯誤: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "伺服器內部錯誤，請稍後再試"}
    )


# 注意：請使用專案根目錄下的 run_app.py 來啟動應用程式
# 或使用: uvicorn app.app:app --host <host> --port <port>
