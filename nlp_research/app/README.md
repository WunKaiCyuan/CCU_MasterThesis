# CCU 校規 RAG 系統 - 應用程式模組

本模組是整個 RAG 系統的核心應用程式，提供基於 FastAPI 的 REST API 服務和網頁介面，用於查詢中正大學校規相關問題。

## 📋 目錄結構

```
app/
├── __init__.py          # 模組初始化文件
├── app.py               # FastAPI 應用程式主文件
├── config.py            # 配置管理模組
├── config.ini           # 配置文件（INI 格式）
├── templates/           # HTML 模板目錄
│   └── index.html       # 網頁前端介面
├── app.log              # 應用程式日誌文件
└── README.md           # 本文件
```

## ✨ 功能特色

### 1. **REST API 服務**
- 提供標準的 RESTful API 介面
- 支援 POST 和 GET 兩種查詢方式
- 自動生成 API 文檔（Swagger UI）
- 完整的錯誤處理和驗證機制

### 2. **網頁使用者介面**
- 現代化的響應式設計
- 即時字元計數和輸入驗證
- 載入動畫和狀態提示
- 友好的錯誤訊息顯示

### 3. **RAG 系統整合**
- 整合 LangChain RAG 流程
- 支援 MMR（最大邊際相關性）檢索
- 基於微調 TAIDE 模型的回答生成
- 完整的 Phoenix 監控追蹤

### 4. **配置管理**
- 基於 INI 格式的配置檔案
- 支援多環境配置
- 自動配置驗證
- 清晰的錯誤提示

## 🚀 快速開始

### 前置需求

1. **Python 環境**
   ```bash
   Python >= 3.12
   ```

2. **依賴套件**
   ```bash
   pip install -r ../requirements.txt
   ```

3. **服務依賴**
   - Chroma 向量資料庫（預設運行在 `localhost:8000`）
   - Ollama 服務（需安裝並載入微調後的 TAIDE 模型）
   - Phoenix 監控服務（可選，預設運行在 `localhost:4317`）

### 配置設定

編輯 `config.ini` 檔案以設定系統參數：

```ini
[Embedding]
MODEL_NAME = intfloat/multilingual-e5-small

[Chroma]
HOST = localhost
PORT = 8000
COLLECTION_NAME = ccu_rules

[LLM]
MODEL_NAME = KCWen/taide
TEMPERATURE = 0.3

[Phoenix]
ENDPOINT = http://localhost:4317
PROJECT_NAME = CCU-School-Rules-Assistant

[Retriever]
K = 5
FETCH_K = 20

[Query]
MAX_LENGTH = 500

[API]
HOST = 0.0.0.0
PORT = 8001
```

### 啟動應用程式

#### 方式一：使用啟動腳本（推薦）

從專案根目錄執行：

```bash
python run_app.py
```

#### 方式二：使用 uvicorn 直接啟動

```bash
uvicorn app.app:app --host 0.0.0.0 --port 8001
```

#### 方式三：從 app 目錄啟動

```bash
cd app
python -m uvicorn app:app --host 0.0.0.0 --port 8001
```

### 訪問服務

啟動成功後，可以通過以下方式訪問：

- **網頁介面**: http://localhost:8001/
- **API 文檔 (Swagger)**: http://localhost:8001/docs
- **API 文檔 (ReDoc)**: http://localhost:8001/redoc
- **健康檢查**: http://localhost:8001/health
- **API 資訊**: http://localhost:8001/api

## 📡 API 端點說明

### 1. 根路徑

**GET** `/`

返回網頁使用者介面。

### 2. API 資訊

**GET** `/api`

返回 API 基本資訊。

**回應範例：**
```json
{
  "name": "CCU 校規 RAG 系統 API",
  "version": "1.0.0",
  "description": "基於微調 TAIDE 模型的校規問答系統",
  "docs": "/docs",
  "health": "/health"
}
```

### 3. 健康檢查

**GET** `/health`

檢查系統健康狀態。

**回應範例：**
```json
{
  "status": "healthy",
  "vector_db_connected": true,
  "vector_db_count": 1234,
  "llm_model": "KCWen/taide"
}
```

### 4. 查詢端點

#### POST `/query`

使用 POST 方式提交查詢。

**請求格式：**
```json
{
  "question": "請說明中正大學的退選規定"
}
```

**回應格式：**
```json
{
  "answer": "根據中正大學校規...",
  "processing_time": 2.5,
  "timestamp": "2024-01-09T16:30:00"
}
```

#### GET `/query?question=您的問題`

使用 GET 方式查詢（方便瀏覽器測試）。

**範例：**
```
GET /query?question=請說明中正大學的退選規定
```

## 💻 使用範例

### Python 客戶端

```python
import requests

# 查詢問題
response = requests.post(
    "http://localhost:8001/query",
    json={"question": "請說明中正大學的退選規定"}
)

result = response.json()
print(f"回答：{result['answer']}")
print(f"處理時間：{result['processing_time']} 秒")
```

### cURL 命令

```bash
# POST 方式
curl -X POST "http://localhost:8001/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "請說明中正大學的退選規定"}'

# GET 方式
curl "http://localhost:8001/query?question=請說明中正大學的退選規定"
```

### JavaScript (Fetch API)

```javascript
async function query(question) {
  const response = await fetch('http://localhost:8001/query', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question: question })
  });
  
  const data = await response.json();
  console.log('回答：', data.answer);
  console.log('處理時間：', data.processing_time, '秒');
  return data;
}

// 使用範例
query("請說明中正大學的退選規定");
```

## ⚙️ 配置說明

### Embedding 設定

- `MODEL_NAME`: Embedding 模型名稱，用於將文檔轉換為向量

### Chroma 設定

- `HOST`: Chroma 向量資料庫主機位址
- `PORT`: Chroma 向量資料庫連接埠
- `COLLECTION_NAME`: 要使用的集合名稱

### LLM 設定

- `MODEL_NAME`: Ollama 中的模型名稱
- `TEMPERATURE`: 生成溫度（0-2），數值越高回答越有創意

### Phoenix 設定

- `ENDPOINT`: Phoenix 監控服務端點
- `PROJECT_NAME`: Phoenix 專案名稱（顯示在監控介面）

### Retriever 設定

- `K`: 返回的最相關文檔數量
- `FETCH_K`: 初步檢索的文檔數量（用於 MMR 演算法）

### Query 設定

- `MAX_LENGTH`: 查詢問題的最大長度（字元數）

### API 設定

- `HOST`: API 服務綁定的主機位址（`0.0.0.0` 表示所有介面）
- `PORT`: API 服務監聽的連接埠

## 🔍 故障排除

### 1. 無法連接到 Chroma 資料庫

**錯誤訊息：**
```
❌ 錯誤：無法連接到向量資料庫
```

**解決方法：**
- 確認 Chroma 服務是否正在運行
- 檢查 `config.ini` 中的 `CHROMA_HOST` 和 `CHROMA_PORT` 設定
- 確認集合名稱 `COLLECTION_NAME` 是否正確

### 2. 無法連接到 Ollama 服務

**錯誤訊息：**
```
❌ 錯誤：無法初始化 LLM 模型
```

**解決方法：**
- 確認 Ollama 服務是否正在運行：`ollama list`
- 檢查模型是否已安裝：`ollama list`
- 確認 `config.ini` 中的 `LLM_MODEL_NAME` 與 Ollama 中的模型名稱一致

### 3. 找不到設定檔

**錯誤訊息：**
```
找不到設定檔 config.ini
```

**解決方法：**
- 確認 `config.ini` 檔案存在於 `app/` 目錄或專案根目錄
- 檢查檔案權限

### 4. 端口已被占用

**錯誤訊息：**
```
Address already in use
```

**解決方法：**
- 修改 `config.ini` 中的 `API_PORT` 為其他端口
- 或關閉占用該端口的其他程式

### 5. 模板目錄不存在

**錯誤訊息：**
```
模板目錄不存在
```

**解決方法：**
- 確認 `templates/` 目錄存在於 `app/` 目錄下
- 確認 `index.html` 檔案存在

## 📊 監控與日誌

### 日誌文件

應用程式日誌會寫入 `app/app.log` 檔案，包含：
- 系統啟動資訊
- 查詢記錄
- 錯誤和警告訊息

### Phoenix 監控

如果 Phoenix 服務正在運行，可以訪問：
- **Phoenix 監控介面**: http://localhost:6006

監控內容包括：
- 檢索到的文檔片段
- 模型生成過程
- 查詢延遲分析
- 系統效能指標

## 🔒 安全注意事項

1. **生產環境部署**
   - 建議使用反向代理（如 Nginx）
   - 啟用 HTTPS
   - 設定適當的 CORS 政策
   - 限制 API 訪問頻率

2. **資料隱私**
   - 所有資料處理均在本地端執行
   - 不會將資料傳輸至第三方服務
   - 日誌文件可能包含敏感資訊，請妥善保管

## 📝 開發說明

### 修改 Prompt

編輯 `app.py` 中的 `template` 變數（約第 156 行）來調整系統的提示詞。

### 添加新的 API 端點

在 `app.py` 中添加新的路由函數：

```python
@app.get("/new-endpoint", tags=["新功能"])
async def new_endpoint():
    return {"message": "新功能"}
```

### 自訂錯誤處理

在 `app.py` 的錯誤處理區段添加自訂的異常處理器。

## 🔗 相關資源

- [FastAPI 官方文檔](https://fastapi.tiangolo.com/)
- [LangChain 官方文檔](https://python.langchain.com/)
- [Arize Phoenix 文檔](https://docs.arize.com/phoenix)
- [專案主 README](../README.md)

## 📄 授權

本專案為學術研究用途，請參考專案根目錄的授權文件。

---

**最後更新**: 2024-01-09
