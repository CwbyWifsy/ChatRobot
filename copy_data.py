from app.services.vector_store import MilvusVectorStore


store = MilvusVectorStore()
collections = store.list_collections()
for collection in collections:
    print(f"collection:{collection} has book :{store.list_books(collection)}")

store.copy_collection(
    src_collection="novels",
    dst_collection="DouPoCangQiong"
)
