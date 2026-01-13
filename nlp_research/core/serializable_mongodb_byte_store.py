import os
import sys
import logging
from langchain_core.documents import Document
from langchain_community.storage import MongoDBByteStore
import pickle
from core.config import Config

# 驗證設定
try:
    Config.validate()
except ValueError as e:
    print(f"❌ 設定檔錯誤: {e}")
    sys.exit(1)

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'app.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SerializableMongoDBByteStore(MongoDBByteStore):
    """簡單包裝 MongoDBByteStore，確保 Document 物件被正確序列化和反序列化"""
    
    def mset(self, key_value_pairs):
        """序列化 Document 物件後再寫入"""
        serialized_pairs = []
        for key, value in key_value_pairs:
            if isinstance(value, Document):
                value = pickle.dumps(value)
            serialized_pairs.append((key, value))
        return super().mset(serialized_pairs)
    
    def mget(self, keys):
        """反序列化 bytes 為 Document 物件"""
        # 從父類獲取序列化的 bytes
        byte_results = super().mget(keys)
        
        # 反序列化為 Document 物件
        deserialized_results = []
        for byte_value in byte_results:
            if byte_value is None:
                deserialized_results.append(None)
            elif isinstance(byte_value, bytes):
                try:
                    # 嘗試反序列化為 Document
                    doc = pickle.loads(byte_value)
                    # 確保返回的是 Document 物件
                    if isinstance(doc, Document):
                        deserialized_results.append(doc)
                    else:
                        logger.warning(f"反序列化結果不是 Document，而是 {type(doc)}")
                        deserialized_results.append(byte_value)
                except Exception as e:
                    logger.error(f"反序列化失敗: {e}")
                    deserialized_results.append(None)
            else:
                # 如果已經不是 bytes，可能是 Document（不應該發生，但為了安全）
                deserialized_results.append(byte_value)
        
        return deserialized_results