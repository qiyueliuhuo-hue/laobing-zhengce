from ruku import load_document
from fengelianghua import split_text, get_embedding
from xieru import insert_to_milvus


def run_vector_pipeline(
    file_path,
    collection_name="tenant_default_knowledge",
    region="全国",
):
    print(f"开始加载文档...（region={region}）")
    document_list = load_document(file_path)
    text_list = [doc.page_content for doc in document_list]
    chunks = split_text(text_list)
    dense_vecs, sparse_vecs = get_embedding(chunks)
    insert_to_milvus(
        dense_vecs,
        sparse_vecs,
        chunks,
        collection_name=collection_name,
        source_name=file_path,
        region=region,
    )
    return len(chunks)


if __name__ == "__main__":
    # 用法示例：全国性政策 region=全国；省级政策 region=河北 等
    run_vector_pipeline(r"../河北省2025专升本通知.docx", region="河北")
