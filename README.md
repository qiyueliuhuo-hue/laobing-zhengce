# 老兵政策助手（laobing-zhengce）

面向退役军人及其家属的政策帮扶智能问答助手：以退役安置、就业创业、抚恤优待、教育培训等政策法规为知识库，结合 RAG 检索、路线规划与出行建议，回答"报到要带什么材料、流程怎么走、去办事地点怎么去"这类实际问题。

## 功能特性

- **政策问答（RAG）**：Milvus 稠密+稀疏向量混合检索（RRF 融合）+ BGE-Reranker-v2-m3 精排，回答引用政策条文原文并标注出处。
- **地区维度**：政策分块带 `region` 字段（全国/省级），按地区过滤检索，地区结果不足时自动降级全库兜底。
- **办事路线规划**：集成高德开放平台——机构名 POI 定位（如"武装部"）、IP 定位出发地、驾车/步行/公交三种路线。
- **出行子图**：跨城办事时结合天气给出穿衣建议与沿途景点推荐。
- **对话记忆**：LangGraph + SQLite checkpointer 持久化，按 `thread_id` 隔离多用户会话，进程重启不丢。
- **两级缓存**：检索结果缓存 + 首问回答级缓存（语义判重），减少重复的 embedding/rerank/LLM 调用；政策库更新后可一键清空。
- **网页界面**：内置 HTTP 服务 + 单页前端，提供问答、清除记忆、缓存统计等接口。

## 技术栈

| 组件 | 选型 |
| --- | --- |
| Agent 框架 | LangChain / LangGraph（create_agent + SqliteSaver） |
| LLM | DeepSeek（OpenAI 兼容接口） |
| 向量库 | Milvus（standalone，本地 19530） |
| Embedding | BGE-M3（dense + sparse + colbert） |
| 精排 | BGE-Reranker-v2-m3 |
| 地图能力 | 高德开放平台 Web 服务 API |
| 前端 | 原生单页 HTML + Python http.server |
| 依赖管理 | uv（`pyproject.toml` + `uv.lock`，Python ≥ 3.14） |

## 目录结构

```
.
├── app/
│   ├── aiagent.py            # Agent 定义：政策助手 system prompt + 3 个工具
│   ├── xiangliangjiansuo.py  # 混合检索 + 精排 + region 降级 + 结果缓存
│   ├── amap_route.py         # 高德路线/天气/出行子图
│   ├── webapp.py             # HTTP 服务（/api/chat、/api/clear_cache 等）
│   ├── index.html            # 网页前端
│   ├── pipeline.py           # 单文档入库流水线（加载→切分→向量化→写入）
│   ├── ruku.py               # 多格式文档加载器（PDF/DOCX/MD/XLSX/TXT）
│   ├── fengelianghua.py      # 文本切分与 BGE-M3 向量化
│   ├── xieru.py              # Milvus 集合 schema 管理与写入
│   ├── cache.py              # 两级缓存实现
│   └── ceshi.py              # 测试脚本
├── *.docx / *.pdf            # 政策语料（国家及河北省退役军人政策文件）
├── bge-m3/                   # 模型权重（不入库，见下方下载指引）
├── bge-reranker-v2-m3/       # 模型权重（不入库）
├── pyproject.toml / uv.lock
└── .env.example
```

## 快速开始

### 1. 安装依赖

```bash
uv sync          # 或 pip install -r <(uv pip compile pyproject.toml)
```

### 2. 准备模型文件

两个 embedding/精排模型合计约 6GB，未纳入版本管理，下载到项目根目录（代码按 `../bge-m3` 相对路径加载）：

```bash
huggingface-cli download BAAI/bge-m3              --local-dir bge-m3
huggingface-cli download BAAI/bge-reranker-v2-m3  --local-dir bge-reranker-v2-m3
```

国内网络可用镜像：`set HF_ENDPOINT=https://hf-mirror.com`。

### 3. 配置密钥

```bash
copy .env.example .env
```

填入你的 DeepSeek API Key 与高德 Web 服务 Key（`.env` 已被 gitignore 排除，切勿提交）。

### 4. 启动 Milvus

本地需要可用的 Milvus（代码默认连接 `http://localhost:19530`），官方 standalone 部署参考：
<https://milvus.io/docs/install_standalone-docker-compose.md>

### 5. 政策语料入库

编辑 `app/pipeline.py` 底部的调用示例（文件路径 + region），逐个文件执行：

```bash
python app/pipeline.py    # 例：run_vector_pipeline(r"../河北省退役士兵安置办法.docx", region="河北")
```

### 6. 启动问答服务

```bash
python app/webapp.py      # 默认 http://127.0.0.1:8000
```

浏览器打开即可问答；命令行冒烟测试：`python app/aiagent.py`。

## 政策语料清单

项目内附带约 1.6MB 公开政策文件（国家及河北省），用于知识库复现：

- 退役军人就业创业促进条例
- 退役士兵教育培训政策（23 条）
- 河北省退役军人保障条例 / 河北省退役士兵安置办法
- 河北省自主就业退役士兵一次性经济补助发放管理暂行办法
- 退役士兵报到流程 / 退役安置
- 魏县退役军人事务局权责事项办事指南和流程图
- 等 15 份（见根目录 `.docx`/`.pdf` 文件）

均为政府公开发文；如需替换为你所在省份的政策文件，删除根目录文档并重新入库即可。

## 安全说明

- 对话记忆库（`mem.sqlite`）、缓存库（`l2_cache.sqlite`）、访问日志（`webapp.log`）含真实用户数据，已通过 `.gitignore` 排除，不会进入仓库。
- `.env` 含 API 密钥，不入库；克隆项目后请按上文自行配置。

## License

个人学习项目，政策文本版权归发布机关所有。
