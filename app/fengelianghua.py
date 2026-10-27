"""分割文档"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from FlagEmbedding import BGEM3FlagModel

chunk = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ",", "。", "：", "！"],
    is_separator_regex=False,
)


def split_text(text_list):
    print("开始分割文档...")
    """根据文本列表切分文档"""
    all_chunks = []
    for text in text_list:
        if text and len(text.strip()) > 0:
            chunk_1 = chunk.split_text(text)
            all_chunks.extend(chunk_1)
    print("分割完成")

    return all_chunks


_model = None


def _get_model():
    global _model
    if _model is None:
        _model = BGEM3FlagModel(r"../bge-m3")
    return _model


def get_embedding(all_chunks):
    print("向量化数据...")
    """根据块列表向量化数据"""
    model = _get_model()
    all_embeddings = model.encode(
        all_chunks, return_dense=True, return_sparse=True, return_colbert_vecs=False
    )
    dense_vecs = all_embeddings["dense_vecs"].tolist()
    sparse_vecs = all_embeddings["lexical_weights"]
    # 清洗系数限量
    clean_sparse_vecs = []
    for vec in sparse_vecs:
        if vec is not None:
            clean_sparse_vecs.append(vec)
    print("向量化完成")
    return dense_vecs, clean_sparse_vecs
