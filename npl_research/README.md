# 論文研究專案：基於微調 TAIDE 模型之全地端 RAG 系統

本專案旨在建立一個完全在地化（On-premise）的學術論文助理系統。透過 **LangChain** 框架整合微調後的 **TAIDE (Llama 3.1)** 模型與 **RAG (檢索增強生成)** 技術，並實施全地端監控評估。

## 🚀 系統架構
本系統由三個核心組件組成：
1. **大腦 (LLM)**：`Llama-3.1-TAIDE-LX-8B-Chat` (經 LoRA 微調並合併導出為 GGUF 格式)。
2. **執行引擎 (Inference)**：**Ollama** (針對 Intel Mac 進行 CPU 運算優化)。
3. **監控與評估 (Observability)**：**Arize Phoenix** (全地端追蹤與語意空間分析)。

---

## 🛠 軟體需求與環境設定

### 1. 系統環境
* **作業系統**: macOS (Intel x86_64)
* **Python 版本**: 3.12 (建議使用 [Miniconda](docs.anaconda.com) 進行管理)
* **硬體建議**: 至少 16GB RAM，並確保硬碟剩餘空間大於 10GB。

### 2. 虛擬環境建立
```bash
# 建立環境
conda create -n nlp_research python=3.12 -y 
conda activate nlp_research

# 安裝必要套件
pip install -r requirements.txt
```

### 3. 在地端模型準備 (Ollama)
下載並安裝 Ollama for Mac。
**匯入微調後的 GGUF 模型（假設名為 my-thesis-model）**：

```bash
ollama create my-thesis-model -f Modelfile
```

## 📂 程式開發說明

本專案由三支核心程式組成，請依照以下研發流程執行：

本專案之技術運作流程如下：

#### **A. 準備階段 (知識與能力建構)**
1. **知識庫建立**：**LangChain** 執行語意切分並透過 BGE-M3 模型建立 **ChromaDB 向量資料庫**。
2. **模型微調**：使用 Unsloth 框架對 TAIDE 進行 LoRA 微調，強化學術領域理解。
3. **模型部署**：產出 **GGUF 檔案** 並匯入 **Ollama**，建立具備微調能力的本地服務。

#### **B. 執行階段 (即時問答流程)**
1. **撰寫指令**：**LangChain** 結合檢索到的文獻片段，寫好完整的 **Prompt**。
2. **傳輸指令**：**LangChain** 透過本地網路傳送給 **Ollama**。
3. **核心運算**：**Ollama** 讓 **微調後的 TAIDE 模型** 進行運算。
4. **結果回傳**：**Ollama** 將答案回傳給 **LangChain**。
5. **監控評估**：**Arize Phoenix** 紀錄全程軌跡，確保研究數據在全地端環境下進行分析。

### 程式一：`ingest.py` (NLP 語意向量資料庫建立)
本程式負責將原始 PDF 文獻轉化為可檢索的知識庫。
*   **技術核心**：使用 **SemanticChunker** 進行語意塊級（Chunk Level）切分，而非傳統的固定字數切分。這確保了論文中的學術概念不會被硬生生切斷。
*   **NLP 工具**：採用 **intfloat/multilingual-e5-small** 作為 Embedding 模型，其具備多語言語意對齊能力，能精確捕捉繁體中文學術術語。
*   **資料庫**：使用 **ChromaDB** 進行地端持久化儲存，數據存放在 `./db_taide` 目錄下。

### 程式二：`finetune.py` (LoRA 微調與 GGUF 封裝)
本程式建議在雲端 GPU 環境（如 RunPod 或 A100）執行，產出專屬的權重檔案。
*   **技術核心**：利用 **Unsloth** 框架對 `cwchang/llama3-taide-lx-8b-chat-alpha1:latest` 進行 LoRA 微調，針對特定學術領域進行指令微調（Instruction Tuning）。
*   **關鍵產出**：
    1.  **LoRA Adapters**：微調後的增量權重。
    2.  **GGUF 檔案**：將微調權重與原始模型合併（Merge）並進行 **4-bit 量化**，導出為單一的 `.gguf` 模型檔，以便讓 Intel Mac 的 CPU 順暢運行。

### 程式三：`app.py` (主應用程式與全地端監控)
這是系統的執行核心，將 RAG 邏輯與微調後的模型進行整合。
*   **模型呼叫**：透過 **Ollama** 載入程式二產出的 GGUF 模型，並藉由 **LangChain** 建立對話鏈（Chain）。
*   **監控機制**：整合 **Arize Phoenix**。當程式執行時，會自動在本地啟動 `http://localhost:6006` 介面。
*   **功能**：即時顯示檢索到的原文片段、模型思考軌跡及生成結果。所有軌跡數據皆存於本地，確保論文資料不外流。

---

### 📊 論文實驗紀錄指標 (Evaluation)

開發者應透過 **Arize Phoenix** 收集以下數據以撰寫論文實驗章節：
* **Hit Rate**：前 K 個檢索結果是否包含目標文獻內容。
* **Faithfulness (忠實度)**：評估 TAIDE 的回答是否嚴格基於檢索到的 PDF 片段。
* **Latency (延遲分析)**：記錄 Intel CPU 在執行 8B 模型時的推理速度與壓力測試。

---

### 🔒 數據隱私與資安聲明
本專案所有組件（包含模型推理、向量搜尋、文本切分及監控紀錄）均在 **本地端 (localhost) 執行**。數據不會傳輸至任何第三方雲端 API (如 OpenAI 或 LangSmith)，完全符合高度機密研究資料之保護需求。

---

### 🔗 資源連結
* [LangChain 官方文件](python.langchain.com)
* [TAIDE (Hugging Face)](huggingface.co)
* [Arize Phoenix 監控工具](docs.arize.com)