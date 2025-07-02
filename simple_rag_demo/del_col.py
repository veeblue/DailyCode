from pymilvus import MilvusClient, exceptions

uri = "http://localhost:19530"
token = "root:Milvus"

client = MilvusClient(
        uri=uri,
        token=token
    )

client.drop_collection(collection_name="rag_demo_collection")