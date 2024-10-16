import json
from datetime import datetime
from elasticsearch import Elasticsearch

es = Elasticsearch(
    ["https://10.100.158.12:9200"],
    verify_certs=False,
    http_auth=("elastic", "*qdSlBZ7AmkaHhyf0VLN"),
)


contents = json.load(open("./bang_api.json", "r"))
for content in contents:
    sentence = content["content"]
    es.index(
        index="doc",
        body={"content": sentence, "publish_date": datetime.now()},
    )
