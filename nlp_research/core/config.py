"""
應用程式設定檔
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
        return value_type(value)
    except configparser.NoOptionError:
        raise ValueError(f"設定檔中找不到設定項: [{section}]{key}")
    except ValueError as e:
        raise ValueError(f"設定項 [{section}]{key} 的值類型錯誤: {e}")


class Config:
    """應用程式配置類別"""
    
    # Embedding 模型設定
    MODEL_NAME = _get("Embedding", "MODEL_NAME", str)
    
    # Chroma 向量資料庫設定
    CHROMA_HOST = _get("Chroma", "HOST", str)
    CHROMA_PORT = _get("Chroma", "PORT", int)
    CHROMA_COLLECTION_NAME = _get("Chroma", "COLLECTION_NAME", str)
    
    # LLM 模型設定
    LLM_MODEL_NAME = _get("LLM", "MODEL_NAME", str)
    LLM_TEMPERATURE = _get("LLM", "TEMPERATURE", float)
    LLM_THREADS = _get("LLM", "THREADS", int)
    
    # Phoenix 監控設定
    PHOENIX_ENDPOINT = _get("Phoenix", "ENDPOINT", str)
    PHOENIX_PROJECT_NAME = _get("Phoenix", "PROJECT_NAME", str)
    
    # 檢索器設定
    RETRIEVER_K = _get("Retriever", "K", int)
    RETRIEVER_FETCH_K = _get("Retriever", "FETCH_K", int)
    
    # 查詢設定
    MAX_QUERY_LENGTH = _get("Query", "MAX_LENGTH", int)
    
    # API 設定
    API_HOST = _get("API", "HOST", str)
    API_PORT = _get("API", "PORT", int)
    
    # MongoDB 設定
    MONGODB_CONNECTION_STRING = _get("MongoDB", "CONNECTION_STRING", str)
    MONGODB_DB_NAME = _get("MongoDB", "DB_NAME", str)
    MONGODB_COLLECTION_NAME = _get("MongoDB", "COLLECTION_NAME", str)
    
    # Parent-Child 設定
    PARENT_CHUNK_SIZE = _get("ParentChild", "PARENT_CHUNK_SIZE", int)
    PARENT_CHUNK_OVERLAP = _get("ParentChild", "PARENT_CHUNK_OVERLAP", int)
    CHILD_CHUNK_SIZE = _get("ParentChild", "CHILD_CHUNK_SIZE", int)
    CHILD_CHUNK_OVERLAP = _get("ParentChild", "CHILD_CHUNK_OVERLAP", int)
    
    # Ingestion 設定
    DATA_DIR = _get("Ingestion", "DATA_DIR", str)
    CHUNK_SIZE = _get("Ingestion", "CHUNK_SIZE", int)
    CHUNK_OVERLAP = _get("Ingestion", "CHUNK_OVERLAP", int)
    
    @classmethod
    def validate(cls):
        """驗證設定值的有效性"""
        if cls.CHROMA_PORT <= 0 or cls.CHROMA_PORT > 65535:
            raise ValueError(f"CHROMA_PORT 必須在 1-65535 之間，目前為 {cls.CHROMA_PORT}")
        
        if cls.RETRIEVER_K <= 0:
            raise ValueError(f"RETRIEVER_K 必須大於 0，目前為 {cls.RETRIEVER_K}")
        
        if cls.RETRIEVER_FETCH_K < cls.RETRIEVER_K:
            raise ValueError(f"RETRIEVER_FETCH_K ({cls.RETRIEVER_FETCH_K}) 必須大於等於 RETRIEVER_K ({cls.RETRIEVER_K})")
        
        if not (0 <= cls.LLM_TEMPERATURE <= 2):
            raise ValueError(f"LLM_TEMPERATURE 必須在 0-2 之間，目前為 {cls.LLM_TEMPERATURE}")
        
        if cls.API_PORT <= 0 or cls.API_PORT > 65535:
            raise ValueError(f"API_PORT 必須在 1-65535 之間，目前為 {cls.API_PORT}")
        
        if cls.CHUNK_SIZE <= 0:
            raise ValueError(f"CHUNK_SIZE 必須大於 0，目前為 {cls.CHUNK_SIZE}")
        
        if cls.CHUNK_OVERLAP < 0:
            raise ValueError(f"CHUNK_OVERLAP 必須大於等於 0，目前為 {cls.CHUNK_OVERLAP}")
        
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            raise ValueError(f"CHUNK_OVERLAP ({cls.CHUNK_OVERLAP}) 必須小於 CHUNK_SIZE ({cls.CHUNK_SIZE})")
        
        if not os.path.exists(cls.DATA_DIR):
            raise FileNotFoundError(f"資料目錄不存在: {cls.DATA_DIR}")
        
        # Parent-Child 設定驗證
        if cls.PARENT_CHUNK_SIZE <= 0:
            raise ValueError(f"PARENT_CHUNK_SIZE 必須大於 0，目前為 {cls.PARENT_CHUNK_SIZE}")
        
        if cls.CHILD_CHUNK_SIZE <= 0:
            raise ValueError(f"CHILD_CHUNK_SIZE 必須大於 0，目前為 {cls.CHILD_CHUNK_SIZE}")
        
        if cls.PARENT_CHUNK_SIZE <= cls.CHILD_CHUNK_SIZE:
            raise ValueError(f"PARENT_CHUNK_SIZE ({cls.PARENT_CHUNK_SIZE}) 必須大於 CHILD_CHUNK_SIZE ({cls.CHILD_CHUNK_SIZE})")
        
        return True
