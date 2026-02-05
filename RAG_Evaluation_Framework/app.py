import os
import json
import gradio as gr
import torch
import chromadb
import google.generativeai as genai
from datetime import timedelta
import typing_extensions as typing

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_vector import SearchType
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from flashrank import Ranker

# Import core modules
from core.serializable_mongodb_byte_store import SerializableMongoDBByteStore

# --- 1. Fixed Strategy Configuration (4-0) ---
FIXED_CONFIG = {
    "MONGODB_DB": "ccu_school_rules",
    "MONGODB_COLLECTION": "ccu_rules",
    "CHROMA_COLLECTION": "ccu_rules",
    "CHILD_CHUNK_SIZE": 200,
    "CHILD_CHUNK_OVERLAP": 20,
    "PARENT_CHUNK_SIZE": 1000,
    "PARENT_CHUNK_OVERLAP": 0,
}

# --- 2. Semantic Strategy Configuration (4-1) ---
SEMANTIC_CONFIG = {
    "MONGODB_DB": "ccu_school_rules",
    "MONGODB_COLLECTION": "ccu_rules_semantic_parent",
    "CHROMA_COLLECTION": "ccu_rules_semantic_child",
}

# --- Shared Configuration ---
SHARED_CONFIG = {
    "MONGODB_URI": "mongodb://admin:UTWi1dCo6jFxNlS0@localhost:27017",
    "CHROMA_HOST": "localhost",
    "CHROMA_PORT": 8000,
    "EMBED_MODEL_NAME": "intfloat/multilingual-e5-small",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "K": 5,
    "SUMMARIES_PATH": "output/document_summaries.json",
    "MODEL_NAME": "models/gemini-2.5-flash-lite",
    "API_KEY": "API_KEY"
}

# --- Initialization ---

# 0. Load Document Summaries (for LLM Retrieval)
print(f"📂 Loading Summaries from {SHARED_CONFIG['SUMMARIES_PATH']}...")
DOCUMENT_SUMMARIES = []
ID_TO_METADATA = {} # Map doc_id/filename to metadata for quick lookup
if os.path.exists(SHARED_CONFIG["SUMMARIES_PATH"]):
    with open(SHARED_CONFIG["SUMMARIES_PATH"], "r", encoding="utf-8") as f:
        DOCUMENT_SUMMARIES = json.load(f)
        # Create a mapping for quick access to filename/doc_id
        for doc in DOCUMENT_SUMMARIES:
             # Ideally map 'doc_id' to filename, or filename to doc_id. 
             # The hybrid result generation uses filename.
             ID_TO_METADATA[doc['doc_id']] = doc
else:
    print("⚠️ Warning: Summaries file not found. LLM Retrieval will not work.")

# 1. Embedding Model
print(f"🧬 Loading Embedding Model: {SHARED_CONFIG['EMBED_MODEL_NAME']}...")
embeddings = HuggingFaceEmbeddings(
    model_name=SHARED_CONFIG["EMBED_MODEL_NAME"],
    model_kwargs={'device': SHARED_CONFIG["DEVICE"]},
    encode_kwargs={'normalize_embeddings': True}
)

# 2. FlashRank Ranker (Global)
print("🚀 Initializing FlashRank Ranker...")
flashrank_client = Ranker() 

# 3. Chroma Client (Shared)
print(f"📡 Connecting to ChromaDB Server at {SHARED_CONFIG['CHROMA_HOST']}:{SHARED_CONFIG['CHROMA_PORT']}...")
chroma_client = chromadb.HttpClient(
    host=SHARED_CONFIG["CHROMA_HOST"],
    port=SHARED_CONFIG["CHROMA_PORT"]
)

# --- Fixed Strategy Objects ---
print("⚙️ Setting up Fixed Strategy Store & Vectorstore...")
fixed_mongo_store = SerializableMongoDBByteStore(
    connection_string=SHARED_CONFIG["MONGODB_URI"],
    db_name=FIXED_CONFIG["MONGODB_DB"],
    collection_name=FIXED_CONFIG["MONGODB_COLLECTION"]
)
fixed_vectorstore = Chroma(
    client=chroma_client,
    collection_name=FIXED_CONFIG["CHROMA_COLLECTION"],
    embedding_function=embeddings
)
# Splitters for ParentDocumentRetriever
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=FIXED_CONFIG["CHILD_CHUNK_SIZE"], 
    chunk_overlap=FIXED_CONFIG["CHILD_CHUNK_OVERLAP"]
)
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=FIXED_CONFIG["PARENT_CHUNK_SIZE"], 
    chunk_overlap=FIXED_CONFIG["PARENT_CHUNK_OVERLAP"]
)

# --- Semantic Strategy Objects ---
print("🧠 Setting up Semantic Strategy Store & Vectorstore...")
semantic_mongo_store = SerializableMongoDBByteStore(
    connection_string=SHARED_CONFIG["MONGODB_URI"],
    db_name=SEMANTIC_CONFIG["MONGODB_DB"],
    collection_name=SEMANTIC_CONFIG["MONGODB_COLLECTION"]
)
semantic_vectorstore = Chroma(
    client=chroma_client,
    collection_name=SEMANTIC_CONFIG["CHROMA_COLLECTION"],
    embedding_function=embeddings
)

# --- Custom Retrievers ---

class SemanticRetriever:
    def __init__(self, vectorstore, docstore, strategy="base", k=5):
        self.vectorstore = vectorstore
        self.docstore = docstore
        self.strategy = strategy
        self.k = k
        if strategy == "rerank":
            self.compressor = FlashrankRerank(
                client=flashrank_client,
                top_n=k,
                score_threshold=0.1,
                model="ms-marco-TinyBERT-L-2-v2"
            )

    def invoke(self, query):
        if self.strategy == "base":
            children = self.vectorstore.similarity_search(query, k=self.k)
        elif self.strategy == "mmr":
            children = self.vectorstore.max_marginal_relevance_search(query, k=self.k)
        elif self.strategy == "rerank":
            base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 20})
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor, 
                base_retriever=base_retriever
            )
            children = compression_retriever.invoke(query)
        else:
            children = []

        parent_docs = []
        for child in children:
            parent_id = child.metadata.get("parent_id")
            if parent_id:
                retrieved_docs = self.docstore.mget([parent_id])
                if retrieved_docs and retrieved_docs[0]:
                    p_doc = retrieved_docs[0]
                    if isinstance(p_doc, Document):
                        parent_docs.append(p_doc)
                    else:
                        parent_docs.append(p_doc)
        return parent_docs

class LLMRetriever:
    """Uses Google Gemini to scan document summaries and return relevant ones."""
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        genai.configure(api_key=api_key)
        
        # Build Context (Cache locally in memory for now, or use GenAI caching if supported/needed)
        # Note: For efficiency in a live app, we pass the context in the prompt or use caching.
        self.summary_context = "\n".join([
            f"【ID: {s['doc_id']} | 標題: {s['doc_name']}】\n摘要：{s['summary']}\n關鍵字：{', '.join(s.get('keywords', []))}\n---"
            for s in DOCUMENT_SUMMARIES
        ])

    def invoke(self, query):
        class RelevantDoc(typing.TypedDict):
            doc_id: str
            confidence_score: float

        class RetrievalResponse(typing.TypedDict):
            relevant_docs: list[RelevantDoc]

        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": 0,
                "response_mime_type": "application/json",
                "response_schema": RetrievalResponse,
            }
        )
        
        prompt = f"""
        Document Summaries:
        {self.summary_context}
        
        User Question: {query}
        
        Task: Identify the top 10 most relevant documents from the provided summaries to answer the user's question.
        Return the 'doc_id' and a 'confidence_score' (0-100) for each.
        """
        
        try:
            response = model.generate_content(prompt)
            res_json = json.loads(response.text)
            
            # Convert to Document objects (Mocking content with summary for now or fetching real doc?)
            # Hybrid fusion needs 'file_name' to match.
            results = []
            for item in res_json.get("relevant_docs", []):
                d_id = item['doc_id']
                # Look up metadata
                meta = ID_TO_METADATA.get(d_id)
                if meta:
                    # Construct a Document
                    # We use the summary as page_content for now, or we could fetch the full doc if we had the file path mapping handy.
                    # Ideally, since this is for fusion, we just need the metadata 'file_name' to match vector results.
                    doc = Document(
                        page_content=f"Summary: {meta['summary']}",
                        metadata={"file_name": meta['doc_name'], "doc_id": d_id}
                    )
                    results.append(doc)
            return results
            
        except Exception as e:
            print(f"LLM Retrieval Error: {e}")
            return []

class HybridRetriever:
    """Combines Vector and LLM retrieval results using Rank Fusion."""
    def __init__(self, vector_retriever, llm_retriever):
        self.vector_retriever = vector_retriever
        self.llm_retriever = llm_retriever

    def invoke(self, query):
        # 1. Run in parallel (sequential for simplicity here)
        print("🔍 Running Vector Search...")
        vec_docs = self.vector_retriever.invoke(query) # Top K (e.g. 5 or 8)
        
        print("🤖 Running LLM Scan...")
        llm_docs = self.llm_retriever.invoke(query) # Top 10
        
        # 2. Rank Fusion (Weighted Score)
        # Score = Sum (1 * (8 - rank)) for top 8 consideration
        # NOTE: The original script considered top 8 from each.
        
        score_dict = {} # Key: file_name, Value: score
        
        # Helper to process results
        def process_ranked_docs(docs, weight=1.0):
            seen_in_source = set()
            for i, doc in enumerate(docs):
                fname = doc.metadata.get("file_name")
                if not fname or fname in seen_in_source: 
                    continue
                seen_in_source.add(fname)
                
                # Scoring: Max score 8 for rank 0, down to 1 for rank 7.
                # If rank > 7, score is 0 in original script? 
                # Original: 1 * (8 - index). So index 0 -> 8, index 7 -> 1.
                rank_score = max(0, 8 - i)
                if rank_score > 0:
                    current_score = score_dict.get(fname, {"score": 0, "doc": doc})
                    current_score["score"] += weight * rank_score
                    score_dict[fname] = current_score

        process_ranked_docs(vec_docs, weight=1.0)
        process_ranked_docs(llm_docs, weight=1.0) # Equal weight for now
        
        # 3. Sort by Score
        sorted_items = sorted(score_dict.values(), key=lambda x: x["score"], reverse=True)
        
        # Return Top K Documents
        final_docs = [item["doc"] for item in sorted_items[:8]] # Return Top 8
        return final_docs


# --- Retriever Factory ---

def get_retriever(strategy_full_name):
    # strategy_full_name format: "Category - Type"
    
    parts = strategy_full_name.split(" - ")
    category = parts[0]
    algo = parts[1] if len(parts) > 1 else "Base"
    
    k = SHARED_CONFIG["K"]
    
    # Base Vector Retriever (needed for Hybrid or Standalone)
    vector_retriever = None
    
    if category == "Fixed" or category == "Hybrid (Fixed + LLM)":
        if algo == "Base":
            vector_retriever = ParentDocumentRetriever(
                vectorstore=fixed_vectorstore,
                docstore=fixed_mongo_store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
                search_kwargs={"k": k}
            )
        elif algo == "MMR":
            vector_retriever = ParentDocumentRetriever(
                vectorstore=fixed_vectorstore,
                docstore=fixed_mongo_store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
                search_type=SearchType.mmr,
                search_kwargs={"k": k}
            )
        elif algo == "Rerank":
            candidate_retriever = ParentDocumentRetriever(
                vectorstore=fixed_vectorstore,
                docstore=fixed_mongo_store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
                search_kwargs={"k": k * 3}
            )
            compressor = FlashrankRerank(
                client=flashrank_client,
                top_n=k,
                score_threshold=0.1,
                model="ms-marco-TinyBERT-L-2-v2"
            )
            vector_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=candidate_retriever
            )
            
    elif category == "Semantic" or category == "Hybrid (Semantic + LLM)":
        algo_key = "base"
        if algo == "MMR": algo_key = "mmr"
        if algo == "Rerank": algo_key = "rerank"
        
        vector_retriever = SemanticRetriever(
            vectorstore=semantic_vectorstore,
            docstore=semantic_mongo_store,
            strategy=algo_key,
            k=k
        )
    
    # If Hybrid, combine with LLM Retriever
    if "Hybrid" in category:
        llm_retriever = LLMRetriever(model_name=SHARED_CONFIG["MODEL_NAME"], api_key=SHARED_CONFIG["API_KEY"])
        return HybridRetriever(vector_retriever, llm_retriever)
        
    return vector_retriever

# --- Gradio App Logic ---

def search_and_chat(query, strategy, history):
    query_text = f"query: {query}"
    
    try:
        # Get appropriate retriever
        retriever = get_retriever(strategy)
        
        # Perform retrieval
        results = retriever.invoke(query_text)
        
        # Deduplication and Formatting
        search_results_markdown = ""
        seen_files = set()
        download_files = []
        
        display_count = 0
        for doc in results:
            if not doc: continue 
            
            source = doc.metadata.get('file_name', '未知來源')
            
            if source in seen_files:
                continue
            seen_files.add(source)
            display_count += 1
            
            content = doc.page_content
            if len(content) > 300:
                display_content = content[:300] + "...(略)..."
            else:
                display_content = content

            search_results_markdown += f"**文檔 {display_count}** (來源: {source})\n\n{display_content}\n\n---\n\n"
            
            file_path = os.path.join("data", source)
            if os.path.exists(file_path):
                download_files.append(file_path)
        
        if not seen_files:
            search_results_markdown = "未找到相關文檔。"

        answer = f"[{strategy}] 已為您檢索到 {display_count} 篇相關文檔。"
        
    except Exception as e:
        error_msg = f"檢索過程中發生錯誤: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        search_results_markdown = error_msg
        answer = "抱歉，搜尋時發生錯誤。"
        download_files = []

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
    
    return history, search_results_markdown, download_files

with gr.Blocks(title="RAG Evaluation - Hybrid Retrieval") as demo:
    gr.Markdown("# 📚 CCU 校規檢索系統 (Hybrid Enhanced)")
    
    with gr.Row():
        # Left: Chat Interface
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="對話歷史")
            
            with gr.Accordion("⚙️ 設定 (Settings)", open=True):
                strategy_dropdown = gr.Dropdown(
                    choices=[
                        "Fixed - Base", "Fixed - MMR", "Fixed - Rerank",
                        "Semantic - Base", "Semantic - MMR", "Semantic - Rerank",
                        "Hybrid (Fixed + LLM) - Base", 
                        "Hybrid (Semantic + LLM) - Base"
                    ],
                    value="Fixed - Base",
                    label="檢索策略 (Retrieval Strategy)",
                )
            
            msg = gr.Textbox(label="請輸入關於校規的問題", placeholder="例如：畢業門檻是什麼？")
            clear = gr.Button("清除對話")

        # Right: Document Display
        with gr.Column(scale=2):
            file_download = gr.File(label="下載相關檔案", interactive=False, file_count="multiple")
            doc_display = gr.Markdown(label="相關文檔內容 (預覽)", value="檢索到的文檔將顯示在此處...")

    # Interaction
    msg.submit(search_and_chat, [msg, strategy_dropdown, chatbot], [chatbot, doc_display, file_download])
    msg.submit(lambda: "", None, msg) 
    
    clear.click(lambda: [], None, chatbot, queue=False)
    clear.click(lambda: "檢索到的文檔將顯示在此處...", None, doc_display, queue=False)
    clear.click(lambda: None, None, file_download, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
