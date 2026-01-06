import phoenix as px
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from opentelemetry import trace as otel_trace
from phoenix.otel import register

MODEL_NAME = "intfloat/multilingual-e5-small"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "thesis_collection"

# ==========================================
# 1. 啟動全地端監控 (Arize Phoenix)
# ==========================================
# 啟動後可在瀏覽器開啟 http://localhost:6006 查看追蹤紀錄
session = px.launch_app()
# 手動建立追蹤導向
tracer_provider = register(
    project_name="my-thesis-app", # 為您的論文專案命名
    auto_instrument=True          # 自動偵測並掛載環境中的 OI 套件
)

print(f"✅ Phoenix 監控介面已啟動: {session.url}")
print(f"目前追蹤器狀態: {otel_trace.get_tracer_provider()}")


# ==========================================
# 2. 載入在地端資源 (Embedding & Vector DB)
# ==========================================
print("倒入向量資料庫中...")
# 使用與第一支程式相同的 Embedding 模型
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 載入已存在的 Chroma 資料庫
persistent_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

vectorstore = Chroma(
    client=persistent_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)

# 設定檢索器 (找最相關的 3 個片段)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ==========================================
# 3. 初始化微調後的 TAIDE 模型 (Ollama)
# ==========================================
llm = ChatOllama(
    model="cwchang/llama3-taide-lx-8b-chat-alpha1:latest", # 確保您已執行 ollama pull taide
    temperature=0.3,
)

# ==========================================
# 4. 設計 RAG 流程 (LangChain LCEL)
# ==========================================
# 針對論文研究設計的 Prompt
template = """你是一位熟讀中正大學校規的助手。請禮貌的根據以下提供的文獻內容，
以繁體中文回答問題。若內容中沒有相關資訊，請誠實回答不知道，不要編造事實。

文獻內容：
{context}

問題：{question}

專業回答："""

prompt = ChatPromptTemplate.from_template(template)

# 定義 RAG 鏈
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": (lambda x: f"query: {x}") | retriever | format_docs, 
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ==========================================
# 5. 執行查詢
# ==========================================
print("\n--- 系統準備就緒 ---")
try:
    while True:
        user_query = input("\n請輸入您的論文相關問題 (輸入 'exit' 離開): ")
        if user_query.lower() == 'exit':
            break
        
        print("\n正在檢索並生成回答...")
        # 執行 RAG
        response = rag_chain.invoke(user_query)
        
        print(f"\n[TAIDE 回覆]:\n{response}")
        print("\n💡 提示：您可以到 Phoenix 介面查看檢索到的原文片段。")

except KeyboardInterrupt:
    print("\n程式已結束")

# 保持 Phoenix 運作直到手動關閉
input("\n按下 Enter 鍵結束並關閉監控伺服器...")
