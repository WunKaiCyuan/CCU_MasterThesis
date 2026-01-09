"""
模型訓練模組配置
從 config.ini 檔案讀取設定
"""
import configparser
from pathlib import Path
import os

# 設定檔案路徑
# 先嘗試在同目錄下尋找，如果找不到則嘗試專案根目錄
_config_file_local = Path(__file__).parent / "config.ini"
_config_file_root = Path(__file__).parent.parent / "config.ini"

if _config_file_local.exists():
    CONFIG_FILE = _config_file_local
elif _config_file_root.exists():
    CONFIG_FILE = _config_file_root
else:
    raise FileNotFoundError(
        f"找不到設定檔 config.ini\n"
        f"已嘗試搜尋位置：\n"
        f"  - {_config_file_local}\n"
        f"  - {_config_file_root}\n"
        "請確認 config.ini 檔案存在於上述其中一個位置"
    )

# 讀取設定檔
_config = configparser.ConfigParser()
_config.read(CONFIG_FILE, encoding='utf-8')


def _get(section, key, value_type=str):
    """從設定檔讀取值並轉換類型"""
    if not _config.has_section(section):
        raise ValueError(f"設定檔中找不到區段: [{section}]")
    
    try:
        value = _config.get(section, key)
        if value_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif value_type == list:
            # 處理逗號分隔的列表
            return [item.strip() for item in value.split(',') if item.strip()]
        elif value_type == float:
            # 處理科學記號（如 2e-4）
            try:
                return float(value)
            except ValueError:
                # 嘗試解析科學記號
                import ast
                return float(ast.literal_eval(value))
        return value_type(value)
    except configparser.NoOptionError:
        raise ValueError(f"設定檔中找不到設定項: [{section}]{key}")
    except ValueError as e:
        raise ValueError(f"設定項 [{section}]{key} 的值類型錯誤: {e}")


class Config:
    """模型訓練配置類別"""
    
    # Model 設定
    BASE_MODEL_NAME = _get("Model", "BASE_MODEL_NAME")
    NEW_MODEL_NAME = _get("Model", "NEW_MODEL_NAME")
    MAX_SEQ_LENGTH = _get("Model", "MAX_SEQ_LENGTH", int)
    
    # LoRA 設定
    LORA_R = _get("LoRA", "R", int)
    LORA_ALPHA = _get("LoRA", "LORA_ALPHA", int)
    LORA_DROPOUT = _get("LoRA", "LORA_DROPOUT", float)
    LORA_BIAS = _get("LoRA", "BIAS")
    LORA_TARGET_MODULES = _get("LoRA", "TARGET_MODULES", list)
    
    # Dataset 設定
    DATASET_FILE = _get("Dataset", "DATASET_FILE")
    
    # Training 設定
    PER_DEVICE_TRAIN_BATCH_SIZE = _get("Training", "PER_DEVICE_TRAIN_BATCH_SIZE", int)
    GRADIENT_ACCUMULATION_STEPS = _get("Training", "GRADIENT_ACCUMULATION_STEPS", int)
    LEARNING_RATE = _get("Training", "LEARNING_RATE", float)
    MAX_STEPS = _get("Training", "MAX_STEPS", int)
    WARMUP_STEPS = _get("Training", "WARMUP_STEPS", int)
    OUTPUT_DIR = _get("Training", "OUTPUT_DIR")
    LOGGING_STEPS = _get("Training", "LOGGING_STEPS", int)
    
    # Quantization 設定
    LOAD_IN_4BIT = _get("Quantization", "LOAD_IN_4BIT", bool)
    QUANTIZATION_METHOD = _get("Quantization", "QUANTIZATION_METHOD")
    
    # Hardware 設定
    AUTO_FP16_BF16 = _get("Hardware", "AUTO_FP16_BF16", bool)
    
    @classmethod
    def validate(cls):
        """驗證配置是否正確"""
        errors = []
        
        # 檢查必要檔案是否存在
        dataset_path = Path(cls.DATASET_FILE)
        if not dataset_path.exists():
            errors.append(f"訓練資料集檔案不存在: {cls.DATASET_FILE}")
        
        # 驗證數值範圍
        if cls.LORA_R <= 0:
            errors.append(f"LoRA R 必須大於 0，目前為: {cls.LORA_R}")
        
        if cls.LORA_ALPHA <= 0:
            errors.append(f"LoRA Alpha 必須大於 0，目前為: {cls.LORA_ALPHA}")
        
        if not 0 <= cls.LORA_DROPOUT <= 1:
            errors.append(f"LoRA Dropout 必須在 0-1 之間，目前為: {cls.LORA_DROPOUT}")
        
        if cls.MAX_SEQ_LENGTH <= 0:
            errors.append(f"最大序列長度必須大於 0，目前為: {cls.MAX_SEQ_LENGTH}")
        
        if cls.PER_DEVICE_TRAIN_BATCH_SIZE <= 0:
            errors.append(f"批次大小必須大於 0，目前為: {cls.PER_DEVICE_TRAIN_BATCH_SIZE}")
        
        if cls.LEARNING_RATE <= 0:
            errors.append(f"學習率必須大於 0，目前為: {cls.LEARNING_RATE}")
        
        if cls.MAX_STEPS <= 0:
            errors.append(f"最大訓練步數必須大於 0，目前為: {cls.MAX_STEPS}")
        
        if cls.BIAS not in ['none', 'all', 'lora_only']:
            errors.append(f"Bias 設定必須是 none, all 或 lora_only，目前為: {cls.BIAS}")
        
        if errors:
            raise ValueError("配置驗證失敗：\n" + "\n".join(f"  - {error}" for error in errors))
        
        return True
