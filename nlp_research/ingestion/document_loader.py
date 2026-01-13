"""
文檔載入工具模組
提供統一的 PDF/DOCX 文件載入功能
"""
import os
import re
import logging
from typing import List
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    清理文本內容
    - 將多個連續換行合併為單一換行
    - 移除多餘的空格（但保留必要的空格）
    
    Args:
        text: 原始文本
        
    Returns:
        清理後的文本
    """
    # 將多個連續換行 (\n\n\n...) 取代為單一換行 (\n)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 移除行首行尾的多餘空格，但保留行內必要的空格
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines]
    text = '\n'.join(cleaned_lines)
    
    # 移除多餘的連續空格（但保留單一空格）
    text = re.sub(r' {2,}', ' ', text)
    
    return text


def load_documents(
    data_dir: str, 
    clean: bool = False,
    logger_instance: logging.Logger = None
) -> List[Document]:
    """
    載入目錄下的所有文件（PDF, DOCX）
    確保一檔一文，將多頁文件合併為單一文檔
    
    Args:
        data_dir: 資料目錄路徑
        clean: 是否清理文本內容（預設：False）
        logger_instance: 日誌記錄器實例（如果為 None，使用模組級別的 logger）
        
    Returns:
        文件列表（每個文件一個 Document 物件）
        
    Raises:
        FileNotFoundError: 如果資料目錄不存在
        ValueError: 如果未找到任何可讀取的文件
    """
    if logger_instance is None:
        logger_instance = logger
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"資料目錄不存在: {data_dir}")
    
    logger_instance.info(f"正在讀取資料夾 {data_dir} 中的所有文件...")
    
    # 定義副檔名與對應的解析器
    loaders_info = {
        ".pdf": PyPDFLoader,
        ".docx": UnstructuredWordDocumentLoader
    }
    
    raw_documents: List[Document] = []
    
    # 遍歷資料夾，手動處理以確保「一檔一文」
    for filename in os.listdir(data_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in loaders_info:
            continue
            
        file_path = os.path.join(data_dir, filename)
        try:
            loader_cls = loaders_info[ext]
            
            # 讀取檔案
            loaded_pages = loader_cls(file_path).load()
            
            if not loaded_pages:
                continue
            
            # 強制合併邏輯：將該檔案所有頁面內容串接，變成一個單一的 Document 物件
            full_content = "\n".join([p.page_content for p in loaded_pages])
            
            # 清理文本內容（如果需要）
            if clean:
                full_content = clean_text(full_content)
            
            # 保持 metadata 一致，這對後續 ParentDocumentRetriever 很重要
            merged_doc = Document(
                page_content=full_content,
                metadata={"source": file_path, "filename": filename}
            )
            
            raw_documents.append(merged_doc)
            logger_instance.info(f"  ✅ 成功載入檔案: {filename} (總長度: {len(full_content)} 字)")
            
        except Exception as e:
            logger_instance.warning(f"  ❌ 載入檔案 {filename} 時發生錯誤: {e}")
            continue
    
    if not raw_documents:
        raise ValueError(f"在 {data_dir} 中未找到任何可讀取的文件")
    
    logger_instance.info(f"文件讀取完成，共載入 {len(raw_documents)} 個完整檔案物件")
    return raw_documents
