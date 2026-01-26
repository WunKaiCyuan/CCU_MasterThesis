import re

class TextCleaner:
    def __init__(self, text):
        self.text = text

    def remove_extra_spaces(self):
        """移除多餘空格與首尾空格"""
        # 將多個空格轉為單個，並修剪邊界
        self.text = re.sub(r' +', ' ', self.text)
        self.text = self.text.strip()
        return self

    def fix_line_breaks(self):
        """修復損壞的斷句 (PDF 常見問題)"""
        # 1. 處理行末連字號 (e.g., "inter-\nnet" -> "internet")
        self.text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', self.text)
        # 2. 將單個換行符號轉為空格（代表同一段落），保留兩個以上的換行（代表新段落）
        self.text = re.sub(r'(?<!\n)\n(?!\n)', ' ', self.text)
        return self

    def remove_special_characters(self):
        """移除不必要的特殊符號，僅保留常用標點與中英文數字"""
        # 保留：中文字、英文字母、數字、常見中英文標點
        pattern = r'[^\u4e00-\u9fa5a-zA-Z0-9\s,.，、。！？：；（）()\-]'
        self.text = re.sub(pattern, '', self.text)
        return self

    def normalize_newlines(self):
        """統一換行符號，避免過多空行"""
        # 將三個以上的換行縮減為兩個
        self.text = re.sub(r'\n{3,}', '\n\n', self.text)
        return self

    def get_result(self):
        """回傳清洗後的結果"""
        return self.text
