# 模型訓練模組

本模組提供 LoRA 微調功能，使用 Unsloth 框架對 TAIDE 模型進行微調，並導出為 GGUF 格式供 Ollama 使用。

## 📋 需求

### 硬體需求
- **建議**：NVIDIA GPU（如 RTX 4090, A100）
- **記憶體**：至少 16GB VRAM（使用 4-bit 量化）
- **儲存空間**：至少 20GB 可用空間

### 軟體需求
```bash
# 安裝必要的 Python 套件
pip install unsloth trl transformers datasets
```

## ⚙️ 配置說明

### 配置文件：`config.ini`

所有訓練參數都在 `config.ini` 中設定，包含以下區段：

#### `[Model]` - 模型設定
- `BASE_MODEL_NAME`: 原始 TAIDE 模型名稱（預設：`taide/Llama-3.1-TAIDE-LX-8B-Chat`）
- `NEW_MODEL_NAME`: 微調後的新模型名稱（預設：`KCWen/taide`）
- `MAX_SEQ_LENGTH`: 最大序列長度（預設：`4096`）

#### `[LoRA]` - LoRA 參數設定
- `R`: LoRA Rank（建議值：8, 16, 32，預設：`16`）
- `LORA_ALPHA`: LoRA Alpha（預設：`16`）
- `LORA_DROPOUT`: LoRA Dropout（範圍：0-1，預設：`0.0`）
- `BIAS`: Bias 設定（`none`, `all`, `lora_only`，預設：`none`）
- `TARGET_MODULES`: 目標模組（用逗號分隔，預設：`q_proj,k_proj,v_proj,o_proj`）

#### `[Dataset]` - 數據集設定
- `DATASET_FILE`: 訓練資料集檔案路徑（JSONL 格式，預設：`train_data.jsonl`）

#### `[Training]` - 訓練參數
- `PER_DEVICE_TRAIN_BATCH_SIZE`: 每個裝置的訓練批次大小（預設：`2`）
- `GRADIENT_ACCUMULATION_STEPS`: 梯度累積步數（預設：`4`）
- `LEARNING_RATE`: 學習率（預設：`2e-4`）
- `MAX_STEPS`: 最大訓練步數（預設：`60`）
- `WARMUP_STEPS`: Warmup 步數（預設：`5`）
- `OUTPUT_DIR`: 輸出目錄（預設：`outputs`）
- `LOGGING_STEPS`: 日誌步數（預設：`1`）

#### `[Quantization]` - 量化設定
- `LOAD_IN_4BIT`: 載入模型時是否使用 4-bit 量化（`true`/`false`，預設：`true`）
- `QUANTIZATION_METHOD`: GGUF 量化方法（預設：`q4_k_m`）

#### `[Hardware]` - 硬體設定
- `AUTO_FP16_BF16`: 自動選擇 FP16/BF16（`true`/`false`，預設：`true`）

## 📊 數據集格式

訓練資料集應為 JSONL 格式，每行一個 JSON 物件，包含以下欄位：

```json
{"instruction": "問題或指令", "input": "可選的上下文來源", "output": "期望的回答"}
```

範例：
```json
{"instruction": "什麼是 RAG？", "input": "檢索到的文檔片段", "output": "RAG 是 Retrieval-Augmented Generation 的縮寫..."}
{"instruction": "解釋 LoRA 的優點", "input": "", "output": "LoRA 的優點包括..."}
```

**注意**：`input` 欄位是可選的，如果沒有上下文來源可以留空字串。

## 🚀 使用方法

### 1. 準備訓練數據集

確保您已經準備好符合格式的 `train_data.jsonl` 檔案。

### 2. 配置參數

編輯 `config.ini`，根據您的需求和硬體資源調整參數。

### 3. 執行訓練

```bash
# 確保您在訓練環境中（GPU 伺服器）
python training/finetune.py
```

### 4. 查看日誌

訓練過程中的所有日誌都會記錄在 `training/finetune.log` 檔案中。

### 5. 獲取模型

訓練完成後，GGUF 模型檔案會保存在 `outputs` 目錄中。您可以將其下載到本地 Mac 環境使用。

## 📝 輸出說明

訓練腳本會輸出以下內容：

1. **環境檢查**：CUDA 可用性、GPU 資訊、精度支援
2. **配置摘要**：所有訓練參數的總覽
3. **訓練進度**：每個步驟的詳細資訊和損失值
4. **訓練統計**：最終訓練損失和訓練時間
5. **模型匯出**：GGUF 檔案的位置和格式

## 🔧 故障排除

### 記憶體不足

如果遇到記憶體不足的問題，可以嘗試：

1. 降低批次大小：減少 `PER_DEVICE_TRAIN_BATCH_SIZE`
2. 增加梯度累積：增加 `GRADIENT_ACCUMULATION_STEPS`
3. 確保 `LOAD_IN_4BIT` 設為 `true`
4. 降低 `MAX_SEQ_LENGTH`

### 訓練速度慢

1. 確認 CUDA 可用：檢查 GPU 是否被正確識別
2. 使用 BF16：如果 GPU 支援，確保 `AUTO_FP16_BF16` 為 `true`
3. 減少 `MAX_STEPS`：如果是測試階段

### 配置錯誤

如果遇到配置驗證錯誤，檢查：

1. `config.ini` 檔案是否存在且格式正確
2. 所有必要的欄位都已填寫
3. 數值範圍是否合理（如 dropout 在 0-1 之間）
4. 數據集檔案是否存在

## 📚 相關文件

- [Unsloth 官方文檔](https://github.com/unslothai/unsloth)
- [TRL 訓練庫](https://github.com/huggingface/trl)
- [Transformers 文檔](https://huggingface.co/docs/transformers)
