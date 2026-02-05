import os
import gradio as gr
import torch
import chromadb

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import core modules
# Ensure your project structure allows this import. 
# If 'core' is a sibling directory, this should work when running from the root.
from core.serializable_mongodb_byte_store import SerializableMongoDBByteStore

# --- Configuration (Adapted from 04_retrieval_parent_document.py) ---
EXP_CONFIG = {
    # MongoDB Connection
    "MONGODB_URI": "mongodb://admin:UTWi1dCo6jFxNlS0@localhost:27017",
    "MONGODB_DB": "ccu_school_rules",
    "MONGODB_COLLECTION": "ccu_rules",
    
    # ChromaDB Connection
    "CHROMA_HOST": "localhost",
    "CHROMA_PORT": 8000,
    "CHROMA_COLLECTION": "ccu_rules",
    
    # Embedding Model
    "EMBED_MODEL_NAME": "intfloat/multilingual-e5-small",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    
    # Text Splitting Parameters (4-0 Fixed)
    "CHILD_CHUNK_SIZE": 200,
    "CHILD_CHUNK_OVERLAP": 20,
    "PARENT_CHUNK_SIZE": 1000,
    "PARENT_CHUNK_OVERLAP": 0,
    
    "K": 5 # Number of documents to retrieve
}

# --- Initialization ---

# 1. MongoDB Store
print(f"🔗 Connecting to MongoDB: {EXP_CONFIG['MONGODB_DB']}...")
mongo_store = SerializableMongoDBByteStore(
    connection_string=EXP_CONFIG["MONGODB_URI"],
    db_name=EXP_CONFIG["MONGODB_DB"],
    collection_name=EXP_CONFIG["MONGODB_COLLECTION"]
)

# 2. Embedding Model
print(f"🧬 Loading Embedding Model: {EXP_CONFIG['EMBED_MODEL_NAME']}...")
embeddings = HuggingFaceEmbeddings(
    model_name=EXP_CONFIG["EMBED_MODEL_NAME"],
    model_kwargs={'device': EXP_CONFIG["DEVICE"]},
    encode_kwargs={'normalize_embeddings': True}
)

# 3. ChromaDB
print(f"📡 Connecting to ChromaDB Server at {EXP_CONFIG['CHROMA_HOST']}:{EXP_CONFIG['CHROMA_PORT']}...")
client = chromadb.HttpClient(
    host=EXP_CONFIG["CHROMA_HOST"],
    port=EXP_CONFIG["CHROMA_PORT"]
)
vectorstore = Chroma(
    client=client,
    collection_name=EXP_CONFIG["CHROMA_COLLECTION"],
    embedding_function=embeddings
)

# 4. Parent Document Retriever
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=EXP_CONFIG["CHILD_CHUNK_SIZE"], 
    chunk_overlap=EXP_CONFIG["CHILD_CHUNK_OVERLAP"]
)
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=EXP_CONFIG["PARENT_CHUNK_SIZE"], 
    chunk_overlap=EXP_CONFIG["PARENT_CHUNK_OVERLAP"]
)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=mongo_store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": EXP_CONFIG["K"]}
)

# --- Gradio App Logic ---

def search_and_chat(query, history):
    # e5 model needs "query: " prefix for better performance
    query_text = f"query: {query}"
    
    try:
        # Perform retrieval
        # invoke returns a list of Documents (the parent chunks)
        results = retriever.invoke(query_text)
        
        # Deduplication and Formatting
        search_results_markdown = ""
        seen_files = set()
        download_files = []
        
        # Limit the displayed/downloadable unique documents if needed, 
        # but here we just process all retrieved (K=5)
        
        display_count = 0
        for doc in results:
            source = doc.metadata.get('file_name', '未知來源')
            
            # Deduplicate by source filename
            if source in seen_files:
                continue
            seen_files.add(source)
            display_count += 1
            
            # 1. Truncate content for display
            content = doc.page_content
            if len(content) > 300:
                display_content = content[:300] + "...(略)..."
            else:
                display_content = content

            search_results_markdown += f"**文檔 {display_count}** (來源: {source})\n\n{display_content}\n\n---\n\n"
            
            # 2. Prepare file path for download
            # Assuming files are in the 'data' directory relative to app.py
            file_path = os.path.join("data", source)
            if os.path.exists(file_path):
                download_files.append(file_path)
        
        if not seen_files:
            search_results_markdown = "未找到相關文檔。"

        # Placeholder for LLM generation
        answer = f"已為您檢索到 {display_count} 篇相關的父文檔。請查看右側的「相關文檔內容」面板。"
        
    except Exception as e:
        error_msg = f"檢索過程中發生錯誤: {str(e)}"
        print(error_msg)
        search_results_markdown = error_msg
        answer = "抱歉，搜尋時發生錯誤。"
        download_files = []

    # Update history with "messages" format
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
    
    return history, search_results_markdown, download_files

with gr.Blocks(title="RAG Evaluation - Parent Document Retrieval") as demo:
    gr.Markdown("# 📚 CCU 校規檢索系統")
    gr.Markdown("使用 **Parent Document Retrieval** (索引: 4-0 Fixed)")
    
    with gr.Row():
        # Left: Chat Interface
        with gr.Column(scale=2):
            # Specify type="messages" to be explicit, though it seems to be the default causing issues with tuples
            chatbot = gr.Chatbot(label="對話歷史")
            msg = gr.Textbox(label="請輸入關於校規的問題", placeholder="例如：畢業門檻是什麼？")
            clear = gr.Button("清除對話")

        # Right: Document Display
        with gr.Column(scale=2):
            doc_display = gr.Markdown(label="相關父文檔內容 (預覽)", value="檢索到的父文檔將顯示在此處...")
            # File download component
            file_download = gr.File(label="下載相關檔案", interactive=False, file_count="multiple")

    # Interaction
    msg.submit(search_and_chat, [msg, chatbot], [chatbot, doc_display, file_download])
    msg.submit(lambda: "", None, msg) # Clear input after submit
    
    # Reset chatbot history to empty list instead of None
    clear.click(lambda: [], None, chatbot, queue=False)
    clear.click(lambda: "檢索到的父文檔將顯示在此處...", None, doc_display, queue=False)
    clear.click(lambda: None, None, file_download, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
