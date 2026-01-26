import os
import json
import hashlib
from datetime import datetime

def calculate_file_hash(file_path):
    """計算檔案的 SHA-256，用於追蹤文檔版本 (論文信度要求)"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def generate_document_inventory(data_folder, output_file):
    """掃描目錄並生成文檔清單"""
    
    # 支援的文檔格式
    valid_extensions = ('.pdf', '.docx', '.txt')
    inventory = []
    
    if not os.path.exists(data_folder):
        print(f"錯誤：找不到資料夾 {data_folder}")
        return

    # 取得檔案清單並排序 (確保 doc_id 穩定)
    files = sorted([f for f in os.listdir(data_folder) if f.lower().endswith(valid_extensions)])
    
    print(f"開始掃描資料夾: {data_folder}")
    print(f"找到 {len(files)} 份符合格式的文檔。\n")

    for index, filename in enumerate(files, start=1):
        file_path = os.path.join(data_folder, filename)
        
        # 建立文檔 metadata
        doc_info = {
            "doc_id": f"D{index:02d}",  # 生成如 D01, D02 的 ID
            "file_name": filename,
            "file_size_kb": round(os.path.getsize(file_path) / 1024, 2),
            "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d'),
            "sha256_hash": calculate_file_hash(file_path) # 確保實驗具備可追溯性
        }
        inventory.append(doc_info)
        print(f"[{doc_info['doc_id']}] 處理中: {filename}")

    # 儲存為 JSON 格式
    output_data = {
        "project": "CCU_Regulations_RAG_Study",
        "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "total_documents": len(inventory),
        "documents": inventory
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"\n✅ 掃描完成！文檔清單已儲存至: {output_file}")

if __name__ == "__main__":
    # --- 請根據你的環境修改路徑 ---
    DATA_PATH = "./data"          # 你的 70 份 PDF 存放路徑
    OUTPUT_PATH = "./output/document_index.json"
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    generate_document_inventory(DATA_PATH, OUTPUT_PATH)
