from pymilvus import MilvusClient, DataType

# 连接Milvus数据库
client = MilvusClient(host="http://localhost:19530", token="")

# schema 版本：加 region 后 = 2；旧库（无 region）= 1
SCHEMA_VERSION = 2


def _collection_fields(collection_name: str) -> set:
    """返回集合现有字段名集合（用于检测旧 schema）"""
    try:
        desc = client.describe_collection(collection_name)
        return {f["name"] for f in desc.get("fields", [])}
    except Exception:
        return set()


def _ensure_collection(collection_name: str, region: str = "全国"):
    """创建（或按需重建）集合，保证含 region 字段。

    若集合已存在但没有 region 字段（旧 schema），会打印提示并
    drop 重建 —— 调用方需已准备好重新入库的数据，否则可能丢数据。
    """
    if client.has_collection(collection_name):
        fields = _collection_fields(collection_name)
        if "region" not in fields:
            print(f"[警告] 集合 {collection_name} 是旧 schema（无 region 字段），将删除并重建！")
            client.drop_collection(collection_name)
            print(f"[警告] 已删除旧集合 {collection_name}，如需保留数据请勿在此后继续入库")
    if not client.has_collection(collection_name):
        # 创建集合
        schema = client.create_schema(auto_id=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("dense_vecs", DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field("sparse_vecs", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("chunk_id", DataType.INT64)
        schema.add_field("region", DataType.VARCHAR, max_length=64)
        client.create_collection(collection_name, schema=schema)


def insert_to_milvus(
    dense_vecs,
    sparse_vecs,
    texts,
    collection_name="tuixizcai",
    source_name="unknown",
    region="全国",
):
    """判断文件是否存在"""
    if client.has_collection(collection_name):
        res = client.query(
            collection_name=collection_name,
            filter=f"source == '{source_name}'",
            output_fields=["id"],
        )
        if res:
            ids = [item["id"] for item in res]
            client.delete(collection_name=collection_name, ids=ids)
            print(f"已删除旧版本 {len(ids)} 条")

    """将向量插入Milvus数据库"""
    _ensure_collection(collection_name)
    # 创建索引
    index_param = client.prepare_index_params()
    index_param.add_index(
        field_name="dense_vecs",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    index_param.add_index(
        field_name="sparse_vecs",
        # 倒排索引
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
    )

    data = [
        {
            "text": texts[i],
            "dense_vecs": dense_vecs[i],
            "sparse_vecs": sparse_vecs[i],
            "source": source_name,
            "chunk_id": i,
            "region": region,
        }
        for i in range(len(texts))
    ]

    client.insert(collection_name=collection_name, data=data)
    client.create_index(collection_name=collection_name, index_params=index_param)
    client.load_collection(collection_name=collection_name)
    print(f"集合创建成功：{collection_name}（region={region}）")

    print("存储完成")
