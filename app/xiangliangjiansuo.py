"""向量检索（上线版）
- schema 预检测：先查集合字段，避免"带 region filter 打旧库"触发异常刷屏
- 降级检索链：region 检索 -> 全库检索 -> 返回空（带原因）
- 两级缓存：检索结果按 (region, query) 缓存，重复问题跳过 embedding+rerank
"""
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from FlagEmbedding import BGEM3FlagModel, FlagReranker
import hashlib
import threading

from cache import cache

# milvus 连接配置
milvus_client = MilvusClient(host="http://localhost:19530", token="")

# 模型懒加载
_embedding_model = None
_reranker_model = None
_model_lock = threading.Lock()

# 集合 schema 缓存：{collection_name: {"has_region": bool, "checked": bool}}
_schema_cache = {}
_schema_lock = threading.Lock()


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        with _model_lock:
            if _embedding_model is None:
                _embedding_model = BGEM3FlagModel(r"../bge-m3")
    return _embedding_model


def _get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        with _model_lock:
            if _reranker_model is None:
                _reranker_model = FlagReranker(r"../bge-reranker-v2-m3")
    return _reranker_model


def _collection_has_region(collection_name: str) -> bool:
    """查询集合是否含 region 字段（结果缓存，避免每次打 schema）"""
    with _schema_lock:
        info = _schema_cache.get(collection_name)
        if info and info.get("checked"):
            return info["has_region"]
    has_region = False
    try:
        desc = milvus_client.describe_collection(collection_name)
        fields = {f["name"] for f in desc.get("fields", [])}
        has_region = "region" in fields
    except Exception as e:
        # 集合可能不存在或 Milvus 未启动；此时检索本身会失败，由上层处理
        print(f"[检索][warn] describe_collection 失败({collection_name}): {e}")
    with _schema_lock:
        _schema_cache[collection_name] = {"checked": True, "has_region": has_region}
    return has_region


def clear_schema_cache():
    """集合重建后调用，刷新 schema 缓存"""
    with _schema_lock:
        _schema_cache.clear()


def _build_region_filter(region: str = "") -> str:
    """构造 region 过滤表达式：空/全国 -> 不过滤；否则 region in ['XX','全国']"""
    if not region or region == "全国":
        return ""
    return f"region in ['{region}', '全国']"


def _cache_key(query_text: str, collection_name: str, region: str, top_n: int) -> str:
    raw = f"{collection_name}|{region}|{top_n}|{query_text.strip()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def hybrid_search(
    query_text,
    collection_name="tenant_default_knowledge",
    top_k=10,
    region="",
):
    """混合检索:稠密向量+稀疏向量 -> RRF融合。
    若集合支持 region 字段则带地区过滤；否则跳过过滤（兼容旧库，无异常刷屏）。
    """
    model = _get_embedding_model()
    emb = model.encode(
        [query_text], return_dense=True, return_sparse=True, return_colbert_vecs=False
    )
    dense_vecs = emb["dense_vecs"][0].tolist()
    sparse_vecs = emb["lexical_weights"][0]

    filter_expr = ""
    has_region = _collection_has_region(collection_name)
    if has_region:
        filter_expr = _build_region_filter(region)

    dense_search = AnnSearchRequest(
        [dense_vecs],
        "dense_vecs",
        {"metric_type": "COSINE", "params": {"ef": 100}},
        limit=top_k,
    )
    sparse_search = AnnSearchRequest(
        [sparse_vecs], "sparse_vecs", {"metric_type": "IP"}, limit=top_k
    )
    kw = {}
    out_fields = ["text", "source", "chunk_id"]
    if has_region:
        kw["filter"] = filter_expr
        out_fields.append("region")

    results = milvus_client.hybrid_search(
        collection_name=collection_name,
        reqs=[dense_search, sparse_search],
        ranker=RRFRanker(k=60),
        limit=top_k,
        output_fields=out_fields,
        **kw,
    )
    hits = []
    for hit in results[0]:
        hits.append(
            {
                "text": hit.entity.get("text"),
                "source": hit.entity.get("source"),
                "chunk_id": hit.entity.get("chunk_id"),
                "region": hit.entity.get("region", "全国"),
                "score": hit.score,
            }
        )
    return hits


def rerank(query_text, candidates, top_n=3):
    """精排（BGE-reranker）；失败时降级为按召回分排序"""
    if not candidates:
        return []
    try:
        rerank_model = _get_reranker_model()
        pairs = [(query_text, cand["text"]) for cand in candidates]
        scores = rerank_model.compute_score(pairs, normalize=True)
        for i, cand in enumerate(candidates):
            cand["rerank_score"] = float(scores[i])
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    except Exception as e:
        print(f"[检索][warn] rerank 失败，按召回分排序降级: {e}")
        for cand in candidates:
            cand["rerank_score"] = float(cand.get("score", 0))
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_n]


def rag_search(
    query_text,
    collection_name="tenant_default_knowledge",
    top_n=3,
    region="",
    use_cache=True,
):
    """完整检索（上线版）：
    1) 缓存命中直接返回（同 region 同问题跳过 embedding+rerank）
    2) region 检索 -> 精排
    3) 结果不足 -> 降级全库检索（不丢全国/其他省政策）
    4) 仍无 -> 返回 []（调用方负责友好提示）
    """
    print(f"检索：query: {query_text[:60]}, region: {region or '全国'}, cache={'on' if use_cache else 'off'}")

    ck = _cache_key(query_text, collection_name, region, top_n)
    if use_cache:
        hit = cache.get(ck)
        if hit is not None:
            print("[检索] 缓存命中")
            return hit

    def _search(region_eff):
        return hybrid_search(query_text, collection_name, top_k=10, region=region_eff)

    def _finish(cands):
        return rerank(query_text, cands, top_n=top_n)

    results = []
    try:
        # 第一级：按地区检索
        candidates = _search(region)
        print(f"region 检索召回：{len(candidates)}条")
        if candidates:
            results = _finish(candidates)
        # 第二级：地区检索结果不足时，降级全库（全国政策兜底）
        if len(results) < top_n and region and region != "全国":
            print(f"[降级] region 结果不足({len(results)}<{top_n})，补全库检索")
            cands_all = _search("")
            # 合并去重（按 chunk_id+source）
            seen = {f"{r.get('source')}|{r.get('chunk_id')}" for r in results}
            extra = [c for c in cands_all if f"{c.get('source')}|{c.get('chunk_id')}" not in seen]
            if extra:
                results = _finish(results + extra)
    except Exception as e:
        # 第三级：Milvus/模型故障 -> 记录并返回空（带原因给上层友好提示）
        print(f"[检索][error] 检索失败: {e}")
        return {"error": f"检索服务暂时不可用：{e}"}

    if use_cache and results and not isinstance(results, dict):
        cache.set(ck, results)

    if isinstance(results, dict):
        return results
    print(f"最终返回：{len(results)}条")
    return results


def clear_kb_cache():
    """政策库数据变化后：清空检索缓存 + schema 缓存"""
    cache.clear()
    clear_schema_cache()


if __name__ == "__main__":
    # 测试
    res = rag_search("退役安置政策有哪些？", region="河北")
    if isinstance(res, dict):
        print("ERROR:", res)
    else:
        for i, r in enumerate(res):
            print(f"\n--- Top {i+1} (score: {r['rerank_score']:.4f}) ---")
            print(f"来源: {r['source']}, region: {r.get('region', '')}")
            print(f"内容: {r['text'][:200]}...")
    print("cache stats:", cache.stats())
