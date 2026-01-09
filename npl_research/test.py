from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "intfloat/multilingual-e5-small"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
CHROMA_COLLECTION_NAME = "ccu_rules"

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': 'cpu'}
)

vectorstore = Chroma(
    embedding_function=embeddings,
    host=CHROMA_HOST,
    port=CHROMA_PORT,
    collection_name=CHROMA_COLLECTION_NAME
)

user_query = input('請輸入想問的問題: ')
test_docs = vectorstore.similarity_search(user_query, k=5)
for i, doc in enumerate(test_docs):
    print(f"\n[檢索片段 {i+1}]:\n{doc.page_content[:300]}...")