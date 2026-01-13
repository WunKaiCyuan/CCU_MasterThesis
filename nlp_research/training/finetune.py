"""
LoRA 微調腳本
使用 Unsloth 框架對 TAIDE 模型進行 LoRA 微調，並導出為 GGUF 格式

注意：此腳本建議在具備 NVIDIA GPU (如 RTX 4090, A100) 的環境執行
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from training.config import Config

# 設定日誌
_log_file = Path(__file__).parent / "finetune.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(_log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def check_environment():
    """檢查執行環境"""
    logger.info("=" * 60)
    logger.info("環境檢查")
    logger.info("=" * 60)
    
    # 檢查 CUDA 可用性
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA 可用")
        logger.info(f"   GPU 裝置數量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"   記憶體: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        logger.warning("⚠️  CUDA 不可用，將使用 CPU（訓練會非常慢）")
    
    # 檢查 BF16 支援
    if torch.cuda.is_bf16_supported():
        logger.info("✅ 支援 BF16 精度")
    else:
        logger.info("ℹ️  不支援 BF16，將使用 FP16")
    
    logger.info("")


def format_prompts(examples):
    """格式化提示為 TAIDE 對話風格"""
    instructions = examples["instruction"]
    inputs = examples.get("input", [""] * len(instructions))
    outputs = examples["output"]
    
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        # 配合 TAIDE 的 Prompt 格式
        if input_text:
            text = f"<s>意圖: {instruction}\n來源: {input_text}\n回答: {output}</s>"
        else:
            text = f"<s>意圖: {instruction}\n回答: {output}</s>"
        texts.append(text)
    
    return {"text": texts}


def main():
    """主訓練流程"""
    logger.info("=" * 60)
    logger.info("TAIDE 模型 LoRA 微調")
    logger.info("=" * 60)
    logger.info(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    try:
        # 驗證配置
        logger.info("正在驗證配置...")
        Config.validate()
        logger.info("✅ 配置驗證通過")
        logger.info("")
        
        # 顯示配置摘要
        logger.info("=" * 60)
        logger.info("配置摘要")
        logger.info("=" * 60)
        logger.info(f"基礎模型: {Config.BASE_MODEL_NAME}")
        logger.info(f"新模型名稱: {Config.NEW_MODEL_NAME}")
        logger.info(f"最大序列長度: {Config.MAX_SEQ_LENGTH}")
        logger.info(f"訓練資料集: {Config.DATASET_FILE}")
        logger.info(f"")
        logger.info(f"LoRA 設定:")
        logger.info(f"  Rank (r): {Config.LORA_R}")
        logger.info(f"  Alpha: {Config.LORA_ALPHA}")
        logger.info(f"  Dropout: {Config.LORA_DROPOUT}")
        logger.info(f"  Bias: {Config.LORA_BIAS}")
        logger.info(f"  目標模組: {', '.join(Config.LORA_TARGET_MODULES)}")
        logger.info(f"")
        logger.info(f"訓練設定:")
        logger.info(f"  批次大小: {Config.PER_DEVICE_TRAIN_BATCH_SIZE}")
        logger.info(f"  梯度累積步數: {Config.GRADIENT_ACCUMULATION_STEPS}")
        logger.info(f"  有效批次大小: {Config.PER_DEVICE_TRAIN_BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS}")
        logger.info(f"  學習率: {Config.LEARNING_RATE}")
        logger.info(f"  最大步數: {Config.MAX_STEPS}")
        logger.info(f"  Warmup 步數: {Config.WARMUP_STEPS}")
        logger.info(f"  輸出目錄: {Config.OUTPUT_DIR}")
        logger.info(f"  4-bit 量化載入: {Config.LOAD_IN_4BIT}")
        logger.info("")
        
        # 檢查環境
        check_environment()
        
        # 1. 載入模型與分詞器
        logger.info("=" * 60)
        logger.info("步驟 1/5: 載入模型與分詞器")
        logger.info("=" * 60)
        logger.info(f"正在載入模型: {Config.BASE_MODEL_NAME}")
        logger.info(f"使用 Unsloth 優化版本...")
        logger.info("（這可能需要幾分鐘，請稍候）")
        
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=Config.BASE_MODEL_NAME,
                max_seq_length=Config.MAX_SEQ_LENGTH,
                load_in_4bit=Config.LOAD_IN_4BIT,
            )
            logger.info("✅ 模型與分詞器載入成功")
        except Exception as e:
            logger.error(f"❌ 模型載入失敗: {e}", exc_info=True)
            raise
        
        # 2. 設定 LoRA 參數
        logger.info("")
        logger.info("=" * 60)
        logger.info("步驟 2/5: 設定 LoRA 參數")
        logger.info("=" * 60)
        logger.info("正在為模型加入 LoRA 適配器...")
        
        try:
            model = FastLanguageModel.get_peft_model(
                model,
                r=Config.LORA_R,
                target_modules=Config.LORA_TARGET_MODULES,
                lora_alpha=Config.LORA_ALPHA,
                lora_dropout=Config.LORA_DROPOUT,
                bias=Config.LORA_BIAS,
            )
            logger.info("✅ LoRA 適配器設定完成")
            
            # 顯示可訓練參數統計
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"可訓練參數: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
            logger.info(f"總參數: {total_params:,}")
        except Exception as e:
            logger.error(f"❌ LoRA 設定失敗: {e}", exc_info=True)
            raise
        
        # 3. 準備數據集
        logger.info("")
        logger.info("=" * 60)
        logger.info("步驟 3/5: 準備訓練數據集")
        logger.info("=" * 60)
        logger.info(f"正在載入數據集: {Config.DATASET_FILE}")
        
        try:
            dataset = load_dataset("json", data_files=Config.DATASET_FILE, split="train")
            logger.info(f"✅ 數據集載入成功")
            logger.info(f"   樣本數量: {len(dataset):,}")
            
            # 顯示數據集結構
            if len(dataset) > 0:
                logger.info(f"   數據集欄位: {list(dataset[0].keys())}")
            
            logger.info("正在格式化提示...")
            dataset = dataset.map(format_prompts, batched=True)
            logger.info("✅ 提示格式化完成")
        except Exception as e:
            logger.error(f"❌ 數據集載入失敗: {e}", exc_info=True)
            raise
        
        # 4. 設定訓練參數
        logger.info("")
        logger.info("=" * 60)
        logger.info("步驟 4/5: 設定訓練參數")
        logger.info("=" * 60)
        
        # 自動選擇精度
        if Config.AUTO_FP16_BF16:
            use_bf16 = torch.cuda.is_bf16_supported()
            use_fp16 = not use_bf16
        else:
            use_fp16 = False
            use_bf16 = False
        
        logger.info(f"使用 FP16: {use_fp16}")
        logger.info(f"使用 BF16: {use_bf16}")
        
        try:
            training_args = TrainingArguments(
                per_device_train_batch_size=Config.PER_DEVICE_TRAIN_BATCH_SIZE,
                gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
                warmup_steps=Config.WARMUP_STEPS,
                max_steps=Config.MAX_STEPS,
                learning_rate=Config.LEARNING_RATE,
                fp16=use_fp16,
                bf16=use_bf16,
                logging_steps=Config.LOGGING_STEPS,
                output_dir=Config.OUTPUT_DIR,
                save_strategy="steps",
                save_steps=Config.MAX_STEPS // 2,  # 在訓練中間保存一次
                save_total_limit=2,
                report_to="none",  # 不使用 wandb/tensorboard
            )
            
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=Config.MAX_SEQ_LENGTH,
                args=training_args,
            )
            logger.info("✅ 訓練器設定完成")
        except Exception as e:
            logger.error(f"❌ 訓練器設定失敗: {e}", exc_info=True)
            raise
        
        # 5. 開始訓練
        logger.info("")
        logger.info("=" * 60)
        logger.info("步驟 5/5: 開始訓練")
        logger.info("=" * 60)
        logger.info("正在開始微調 TAIDE 模型...")
        logger.info(f"預計訓練步數: {Config.MAX_STEPS}")
        logger.info("（這可能需要較長時間，請耐心等候）")
        logger.info("")
        
        try:
            train_result = trainer.train()
            logger.info("")
            logger.info("✅ 訓練完成！")
            logger.info(f"訓練損失: {train_result.training_loss:.4f}")
            logger.info(f"訓練時間: {train_result.metrics.get('train_runtime', 0):.2f} 秒")
        except Exception as e:
            logger.error(f"❌ 訓練失敗: {e}", exc_info=True)
            raise
        
        # 6. 儲存微調權重與匯出 GGUF
        logger.info("")
        logger.info("=" * 60)
        logger.info("模型匯出")
        logger.info("=" * 60)
        logger.info("正在合併 LoRA 權重並轉換為 GGUF 格式...")
        logger.info(f"量化方法: {Config.QUANTIZATION_METHOD}")
        logger.info("（這可能需要較長時間，請耐心等候）")
        
        try:
            # 確保輸出目錄存在
            output_path = Path(Config.OUTPUT_DIR)
            output_path.mkdir(parents=True, exist_ok=True)
            
            model.save_pretrained_gguf(
                Config.NEW_MODEL_NAME,
                tokenizer,
                quantization_method=Config.QUANTIZATION_METHOD
            )
            logger.info("✅ GGUF 匯出完成！")
        except Exception as e:
            logger.error(f"❌ GGUF 匯出失敗: {e}", exc_info=True)
            raise
        
        # 完成總結
        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ 微調與匯出完成！")
        logger.info("=" * 60)
        logger.info(f"模型名稱: {Config.NEW_MODEL_NAME}")
        logger.info(f"匯出格式: GGUF ({Config.QUANTIZATION_METHOD})")
        logger.info(f"輸出目錄: {Config.OUTPUT_DIR}")
        logger.info("")
        logger.info("請將生成的模型檔案下載至您的 Mac 使用。")
        logger.info(f"結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except ValueError as e:
        logger.error(f"配置錯誤: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程式執行失敗: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
