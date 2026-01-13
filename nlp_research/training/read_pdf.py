"""
簡單的 PDF 讀取工具
讀取 PDF 文件並顯示或保存其文本內容
"""
import argparse
import logging
from pathlib import Path

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_pdf(file_path: Path) -> str:
    """讀取 PDF 文件內容"""
    try:
        from pypdf import PdfReader
        
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PdfReader(f)
            total_pages = len(pdf_reader.pages)
            logger.info(f"PDF 總頁數：{total_pages}")
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                text += f"\n--- 第 {page_num} 頁 ---\n"
                text += page_text + "\n"
                logger.info(f"已讀取第 {page_num}/{total_pages} 頁")
        
        return text.strip()
    
    except ImportError:
        logger.error("pypdf 未安裝，請執行：pip install pypdf")
        raise
    except Exception as e:
        logger.error(f"讀取 PDF 失敗：{e}")
        raise


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='讀取 PDF 文件內容')
    parser.add_argument('pdf_file', type=str, help='PDF 文件路徑')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='輸出的文本文件路徑（可選，不指定則顯示在終端）')
    parser.add_argument('--encoding', type=str, default='utf-8',
                       help='輸出文本文件的編碼（預設：utf-8）')
    
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf_file)
    
    # 檢查文件是否存在
    if not pdf_path.exists():
        logger.error(f"文件不存在：{pdf_path}")
        return
    
    if not pdf_path.suffix.lower() == '.pdf':
        logger.warning(f"文件擴展名不是 .pdf：{pdf_path.suffix}")
    
    logger.info(f"正在讀取 PDF：{pdf_path}")
    
    # 讀取 PDF
    try:
        text = read_pdf(pdf_path)
        logger.info(f"✅ 成功讀取，文本長度：{len(text)} 字元")
    except Exception as e:
        logger.error(f"❌ 讀取失敗：{e}")
        return
    
    # 輸出結果
    if args.output:
        # 保存到文件
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding=args.encoding) as f:
            f.write(text)
        
        logger.info(f"✅ 文本已保存到：{output_path}")
    else:
        # 顯示在終端
        print("\n" + "=" * 80)
        print("PDF 內容：")
        print("=" * 80)
        print(text)
        print("=" * 80)


if __name__ == "__main__":
    main()
