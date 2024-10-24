from elasticsearch import Elasticsearch

# 创建Elasticsearch客户端
es = Elasticsearch(
    ["https://10.100.158.12:9200"],
    verify_certs=False,
    http_auth=("elastic", "*qdSlBZ7AmkaHhyf0VLN"),
)

# 索引名称
index_name = "doc"

# 删除索引中的所有文档
es.delete_by_query(index=index_name, body={"query": {"match_all": {}}})

# 确认删除操作完成
es.indices.refresh(index=index_name)

search_result = es.search(index=index_name, body={"query": {"match_all": {}}})

# 获取搜索结果的文档数量
num_docs = search_result["hits"]["total"]["value"]

if num_docs == 0:
    print("索引中没有文档。清除操作成功。")
else:
    print(f"索引中仍有 {num_docs} 个文档。清除操作未成功。")
