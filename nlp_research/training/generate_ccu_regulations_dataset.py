"""
CCU 校規文本數據集生成器
讀取 CCU 校規文本並生成 100 條 LoRA 訓練數據
"""
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_text_file(file_path: Path) -> str:
    """讀取文本文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # 嘗試其他編碼
        with open(file_path, 'r', encoding='big5') as f:
            return f.read()


def read_pdf_file(file_path: Path) -> str:
    """讀取 PDF 文件（優先使用 pypdf，也支持 pdfplumber）"""
    # 優先使用 pypdf（已在 requirements.txt 中）
    try:
        from pypdf import PdfReader
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except ImportError:
        # 嘗試使用 pdfplumber
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text
        except ImportError:
            raise ImportError("請安裝 pypdf 或 pdfplumber 來讀取 PDF 文件：pip install pypdf")


def read_file(file_path: Path) -> str:
    """根據文件擴展名讀取文件"""
    suffix = file_path.suffix.lower()
    
    if suffix == '.txt':
        return read_text_file(file_path)
    elif suffix == '.pdf':
        return read_pdf_file(file_path)
    else:
        # 預設嘗試作為文本文件讀取
        logger.warning(f"未知的文件格式 {suffix}，嘗試作為文本文件讀取...")
        return read_text_file(file_path)


def split_text_into_chunks(text: str, min_chunk_size: int = 100, max_chunk_size: int = 500) -> List[str]:
    """將文本分割成適當大小的塊"""
    # 先按段落分割
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # 如果當前塊加上新段落不會超過最大長度，則合併
        if len(current_chunk) + len(para) + 1 <= max_chunk_size:
            if current_chunk:
                current_chunk += "\n" + para
            else:
                current_chunk = para
        else:
            # 保存當前塊（如果足夠大）
            if len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk)
            # 開始新塊
            current_chunk = para
    
    # 添加最後一塊
    if len(current_chunk) >= min_chunk_size:
        chunks.append(current_chunk)
    
    return chunks


def generate_qa_pairs(chunks: List[str], num_pairs: int = 100) -> List[Dict[str, str]]:
    """從文本塊生成問答對"""
    qa_pairs = []
    
    # 生成不同類型的問題模板
    question_templates = [
        "什麼是{keyword}？",
        "請說明{keyword}的規定",
        "{keyword}的相關規定是什麼？",
        "關於{keyword}，校規是如何規定的？",
        "請解釋{keyword}",
        "{keyword}的內容是什麼？",
        "校規中關於{keyword}的條文是什麼？",
        "請詳細說明{keyword}",
        "{keyword}有哪些規定？",
        "根據校規，{keyword}應該如何處理？",
    ]
    
    # 從文本中提取關鍵詞
    def extract_keywords(text: str, max_keywords: int = 3) -> List[str]:
        # 提取可能的關鍵詞（名詞、重要概念）
        # 簡單方法：提取較長的詞組
        words = re.findall(r'[\u4e00-\u9fff]{2,}', text)
        # 過濾常見詞
        common_words = {'規定', '條文', '學生', '學校', '應', '不得', '必須', '可以', '應該'}
        keywords = [w for w in words if w not in common_words and len(w) >= 2]
        # 返回前幾個最長的詞
        return sorted(set(keywords), key=len, reverse=True)[:max_keywords]
    
    # 為每個塊生成多個問答對
    chunk_idx = 0
    template_idx = 0
    
    while len(qa_pairs) < num_pairs and chunks:
        chunk = chunks[chunk_idx % len(chunks)]
        keywords = extract_keywords(chunk)
        
        if keywords:
            keyword = keywords[0]  # 使用第一個關鍵詞
            template = question_templates[template_idx % len(question_templates)]
            question = template.format(keyword=keyword)
            
            # 生成答案（使用文本塊作為答案，但可能需要簡化）
            answer = chunk.strip()
            
            # 如果答案太長，截取前500字
            if len(answer) > 500:
                answer = answer[:500] + "..."
            
            qa_pairs.append({
                "instruction": question,
                "input": "",  # 校規文本不需要額外的來源
                "output": answer
            })
            
            template_idx += 1
            if template_idx % len(question_templates) == 0:
                chunk_idx += 1
        else:
            chunk_idx += 1
            if chunk_idx >= len(chunks):
                break
    
    # 如果還不夠，生成一些通用問題
    while len(qa_pairs) < num_pairs:
        chunk = chunks[len(qa_pairs) % len(chunks)]
        # 生成更通用的問題
        generic_questions = [
            "請說明這段校規的內容",
            "這段校規規定了什麼？",
            "請解釋這段校規條文",
            "這段校規的意義是什麼？",
        ]
        question = generic_questions[len(qa_pairs) % len(generic_questions)]
        answer = chunk.strip()[:500] if len(chunk) > 500 else chunk.strip()
        
        qa_pairs.append({
            "instruction": question,
            "input": "",
            "output": answer
        })
    
    return qa_pairs[:num_pairs]


def generate_instruction_output_pairs(chunks: List[str], num_pairs: int = 100) -> List[Dict[str, str]]:
    """生成更自然的指令-輸出對（不使用問題模板）"""
    pairs = []
    
    for i in range(num_pairs):
        chunk = chunks[i % len(chunks)]
        chunk = chunk.strip()
        
        # 從文本中提取標題或第一句話作為指令
        lines = chunk.split('\n')
        first_line = lines[0].strip() if lines else ""
        
        # 如果第一行看起來像標題（較短），使用它作為指令
        if len(first_line) < 50 and first_line:
            instruction = f"請說明：{first_line}"
            output = '\n'.join(lines[1:]) if len(lines) > 1 else chunk
        else:
            # 否則生成通用指令
            # 提取關鍵詞
            keywords = re.findall(r'[\u4e00-\u9fff]{2,}', chunk[:100])
            if keywords:
                keyword = keywords[0]
                instruction = f"請說明{keyword}的相關規定"
            else:
                instruction = "請說明這段校規的內容"
            output = chunk
        
        # 限制輸出長度
        if len(output) > 500:
            output = output[:500] + "..."
        
        pairs.append({
            "instruction": instruction,
            "input": "",
            "output": output
        })
    
    return pairs


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='從 CCU 校規文本生成 LoRA 訓練數據集')
    parser.add_argument('input_file', type=str, help='輸入的校規文本文件路徑（支持 .txt, .pdf）')
    parser.add_argument('-o', '--output', type=str, default='train_data.jsonl', 
                       help='輸出的 JSONL 文件路徑（預設：train_data.jsonl）')
    parser.add_argument('-n', '--num', type=int, default=100, 
                       help='生成的訓練數據條數（預設：100）')
    parser.add_argument('--min-chunk', type=int, default=100, 
                       help='文本塊最小長度（預設：100）')
    parser.add_argument('--max-chunk', type=int, default=500, 
                       help='文本塊最大長度（預設：500）')
    parser.add_argument('--method', type=str, choices=['qa', 'instruction'], default='instruction',
                       help='生成方法：qa（問答對）或 instruction（指令-輸出對，預設）')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    output_path = Path(args.output)
    
    # 檢查輸入文件
    if not input_path.exists():
        logger.error(f"輸入文件不存在：{input_path}")
        return
    
    logger.info(f"正在讀取文件：{input_path}")
    
    # 讀取文件
    try:
        text = read_file(input_path)
        logger.info(f"成功讀取文件，文本長度：{len(text)} 字元")
    except Exception as e:
        logger.error(f"讀取文件失敗：{e}")
        return
    
    # 分割文本
    logger.info("正在分割文本...")
    chunks = split_text_into_chunks(text, args.min_chunk, args.max_chunk)
    logger.info(f"分割成 {len(chunks)} 個文本塊")
    
    if not chunks:
        logger.error("無法從文本中提取有效的文本塊，請檢查輸入文件")
        return
    
    # 生成訓練數據
    logger.info(f"正在生成 {args.num} 條訓練數據（方法：{args.method}）...")
    if args.method == 'qa':
        pairs = generate_qa_pairs(chunks, args.num)
    else:
        pairs = generate_instruction_output_pairs(chunks, args.num)
    
    logger.info(f"成功生成 {len(pairs)} 條訓練數據")
    
    # 保存為 JSONL 格式
    logger.info(f"正在保存到：{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    logger.info("✅ 完成！")
    logger.info(f"輸出文件：{output_path}")
    logger.info(f"數據條數：{len(pairs)}")
    
    # 顯示前3條示例
    logger.info("\n前3條數據示例：")
    for i, pair in enumerate(pairs[:3], 1):
        logger.info(f"\n示例 {i}:")
        logger.info(f"  指令: {pair['instruction']}")
        logger.info(f"  輸入: {pair['input'] if pair['input'] else '(空)'}")
        logger.info(f"  輸出: {pair['output'][:100]}...")


if __name__ == "__main__":
    main()
