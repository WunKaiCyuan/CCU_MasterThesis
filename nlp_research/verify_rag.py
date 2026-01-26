
import requests
import json
import sys

def test_query(question="學士班修業年限"):
    url = "http://localhost:8001/query"
    payload = {"question": question}
    headers = {"Content-Type": "application/json"}
    
    print(f"Testing query: {question}")
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            print("\nResponse Received:")
            print(f"Answer: {data.get('answer')}")
            print(f"Time: {data.get('processing_time')}s")
            
            sources = data.get('sources', [])
            print(f"\nSources ({len(sources)}):")
            for i, source in enumerate(sources, 1):
                print(f"{i}. File: {source.get('file_name')}")
                print(f"   Download URL: {source.get('download_url')}")
                print(f"   Content: {source.get('content')[:100]}...")
            
            if sources:
                print("\n✅ Verification SUCCESS: Sources received.")
            else:
                print("\n⚠️ Verification WARNING: No sources received (but API worked).")
        else:
            print(f"\n❌ Verification FAILED: Status {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"\n❌ Verification ERROR: {e}")
        print("Please ensure the app is running: python run_app.py")

if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "學士班修業年限"
    test_query(question)
