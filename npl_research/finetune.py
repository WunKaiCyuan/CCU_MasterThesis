# 注意：此腳本建議在具備 NVIDIA GPU (如 RTX 4090, A100) 的環境執行
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. 設定區
model_name = "taide/Llama-3.1-TAIDE-LX-8B-Chat" # 原始 TAIDE 模型
max_seq_length = 4096 # LX 版本支援長文本
dataset_file = "train_data.jsonl" # 您準備的論文 Q&A 資料集

# 2. 載入模型與分詞器 (使用 Unsloth 優化版)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True, # 使用 4-bit 量化節省記憶體
)

# 3. 加入 LoRA 參數設定
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # LoRA Rank，論文實驗可調整此參數 (8, 16, 32)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)

# 4. 準備數據集 (格式化為 TAIDE 對話風格)
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # 配合 TAIDE 的 Prompt 格式
        text = f"<s>意圖: {instruction}\n來源: {input}\n回答: {output}</s>"
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("json", data_files=dataset_file, split="train")
dataset = dataset.map(formatting_prompts_func, batched = True)

# 5. 設定訓練參數
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # 根據資料量調整，論文實驗建議跑 1-3 個 Epoch
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
    ),
)

# 6. 開始訓練
print("正在開始微調 TAIDE 模型...")
trainer.train()

# 7. 儲存微調權重與匯出 GGUF (關鍵步驟)
# 這會自動合併 LoRA 權重並轉換為 Ollama 支援的 GGUF 格式
print("正在合併權重並轉換為 GGUF 格式...")
model.save_pretrained_gguf(
    "my-finetuned-taide", 
    tokenizer, 
    quantization_method = "q4_k_m" # 導出為 4-bit 量化 GGUF，適合 Intel Mac
)

print("✅ 微調與 GGUF 導出完成！請將 'my-finetuned-taide.gguf' 下載至您的 Mac 使用。")
