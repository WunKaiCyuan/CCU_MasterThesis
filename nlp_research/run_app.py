#!/usr/bin/env python3
"""
應用程式啟動腳本
從專案根目錄啟動應用程式
"""
import sys
import os

# 將專案根目錄加入 Python 路徑
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    import uvicorn
    from core.config import Config
    
    # 驗證設定
    try:
        Config.validate()
    except ValueError as e:
        print(f"❌ 設定檔錯誤: {e}")
        sys.exit(1)
    
    host = Config.API_HOST
    port = Config.API_PORT
    
    print(f"啟動 API 服務於 {host}:{port}")
    print(f"訪問網頁介面: http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
    print(f"API 文檔: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs")
    
    uvicorn.run(
        "app.app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
