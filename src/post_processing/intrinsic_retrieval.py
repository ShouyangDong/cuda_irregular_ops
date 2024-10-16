from .src.bm25 import BM25

def retrieve_documentation(query, doc_path):
    bm25 = BM25(doc=doc_path)
    result = bm25.cal_similarity_rank(query)
    return result
