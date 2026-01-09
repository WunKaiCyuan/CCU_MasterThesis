"""
RAG 系統 API 版本
使用 FastAPI 提供 REST API 介面
"""
import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from phoenix.otel import register
# 導入配置模組
# 優先嘗試相對導入（作為模組時），失敗則嘗試絕對導入（直接運行時）
try:
    from .config import Config
except ImportError:
    # 直接運行時，確保當前目錄在路徑中
    import sys
    import os
    app_dir = os.path.dirname(os.path.abspath(__file__))
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    from config import Config

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


# ==========================================
# 請求/回應模型定義
# ==========================================
class QueryRequest(BaseModel):
    """查詢請求模型"""
    question: str = Field(..., description="要詢問的問題", min_length=1, max_length=500)
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError("問題不能為空")
        return v.strip()


class QueryResponse(BaseModel):
    """查詢回應模型"""
    answer: str = Field(..., description="系統回答")
    processing_time: float = Field(..., description="處理時間（秒）")
    timestamp: str = Field(..., description="查詢時間戳")


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
    global rag_chain, vectorstore, llm
    
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
        
        vectorstore = Chroma(
            embedding_function=embeddings,
            host=Config.CHROMA_HOST,
            port=Config.CHROMA_PORT,
            collection_name=Config.CHROMA_COLLECTION_NAME
        )
        collection_count = vectorstore._collection.count()
        logger.info(f"向量資料庫連接成功，共 {collection_count} 個文檔")
    except Exception as e:
        logger.error(f"載入向量資料庫失敗: {e}", exc_info=True)
        raise
    
    # 3. 設定檢索器
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": Config.RETRIEVER_K, "fetch_k": Config.RETRIEVER_FETCH_K}
    )
    
    # 4. 初始化 LLM 模型
    logger.info(f"初始化 LLM 模型 ({Config.LLM_MODEL_NAME})...")
    try:
        llm = ChatOllama(
            model=Config.LLM_MODEL_NAME,
            temperature=Config.LLM_TEMPERATURE,
        )
        logger.info("LLM 模型初始化成功")
    except Exception as e:
        logger.error(f"初始化 LLM 模型失敗: {e}", exc_info=True)
        raise
    
    # 5. 設計 RAG 流程
    template = """你是一位專業的國立中正大學校規與法規諮詢助手。你的任務是根據提供的文獻內容，以專業、準確、友善的方式回答使用者關於校規的問題。

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
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
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
        answer = rag_chain.invoke(request.question)
        
        elapsed_time = time.time() - start_time
        logger.info(f"查詢完成，耗時 {elapsed_time:.2f} 秒")
        
        return QueryResponse(
            answer=answer,
            processing_time=round(elapsed_time, 2),
            timestamp=datetime.now().isoformat()
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
