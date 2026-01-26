import http.server
import socketserver
import webbrowser

# 設定連接埠
PORT = 5001
# 你的 HTML 檔名
HTML_FILE = "rag_analysis.html" 

class MyHandler(http.server.SimpleHTTPRequestHandler):
    # 增加快取控制，確保你修改 JSON 後刷新網頁能立即看到結果
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

def start_server():
    try:
        with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
            print(f"🚀 Server 已啟動於: http://localhost:{PORT}")
            print(f"📖 正在讀取: {HTML_FILE}")
            
            # 自動開啟預設瀏覽器
            webbrowser.open(f"http://localhost:{PORT}/{HTML_FILE}")
            
            print("按 Ctrl+C 可以停止 Server")
            httpd.serve_forever()
    except OSError:
        print(f"❌ 連接埠 {PORT} 已被佔用，請更換 PORT 變數或關閉佔用的程式。")

if __name__ == "__main__":
    start_server()