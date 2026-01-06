import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "intfloat/multilingual-e5-small"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "thesis_collection"

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': 'cpu'}
)

persistent_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
vectorstore = Chroma(
    client=persistent_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)

user_query = input('請輸入想問的問題: ')
test_docs = vectorstore.similarity_search(user_query, k=5)
for i, doc in enumerate(test_docs):
    print(f"\n[檢索片段 {i+1}]:\n{doc.page_content[:150]}...")