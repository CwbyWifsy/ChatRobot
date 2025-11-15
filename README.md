# 小说 RAG 服务

该项目实现了一个基于 Milvus 的小说检索增强生成（RAG）系统，支持本地 Qwen3-0.6B 向量化模型、OpenAI 兼容对话模型以及小说分章节上传流程。

## 功能特性

- 📚 **小说管理**：通过异步 Python 脚本批量读取 UTF-8 TXT 小说，自动检测章节并按重叠窗口切分。
- 🧾 **哈希去重**：基于文件名 + 内容 + 用户确认的书名生成哈希，避免重复入库，并支持手动覆盖上传。
- 🧠 **本地嵌入**：加载本地 Qwen3-0.6B 嵌入模型生成向量，并将分片写入 Milvus。
- 🗂️ **Milvus 向量库**：自动创建数据库与集合，保存书名、章节、来源路径等元数据，方便追溯，并支持会话级集合记忆。
- 💬 **FastAPI 聊天接口**：提供带上下文的聊天接口，返回引用来源，记录用户与模型的每次对话。
- 🛠️ **可配置化**：所有关键参数（模型、API、Milvus、日志等）均通过 `.env` 配置。

## 快速开始

### 1. 准备环境

1. 安装 Python 3.12。
2. 安装项目依赖（可使用 Poetry 或直接 `pip install -r`，此处以 Poetry 为例）：

   ```bash
   pip install poetry
   poetry install
   ```

3. 准备本地 Qwen3-0.6B 嵌入模型，将模型目录路径配置在 `.env` 中的 `EMBEDDING_MODEL_PATH`。
4. 启动 Milvus 实例，并记录访问地址、认证信息。

### 2. 配置 `.env`

复制 `.env.example` 为 `.env`，根据实际情况修改：

```bash
cp .env.example .env
```

关键字段说明：

- `MILVUS_URI`：Milvus 服务地址，例如 `http://localhost:19530`。
- `EMBEDDING_MODEL_PATH` 与 `EMBEDDING_DIM`：本地嵌入模型路径与向量维度。
- `LLM_BASE_URL` / `LLM_MODEL_NAME` / `LLM_API_KEY`：OpenAI 兼容模型的接入信息。
- `LOG_DIRECTORY`：保存对话日志的目录。

### 3. 上传小说至 Milvus

使用 `scripts/upload_novels.py` 将指定文件夹中的 TXT 小说写入向量数据库：

```bash
python scripts/upload_novels.py /path/to/novels --collection fantasy-cn
```

脚本流程：

1. 遍历目录中的 `.txt` 文件。
2. 询问书名（可回车使用文件名）。
3. 计算文件哈希并检测 Milvus 中是否已存在。
4. 自动分章 + 重叠切分，生成嵌入并写入集合。

如需强制重传，可添加 `--force`。当未显式传入 `--collection` 参数时，脚本会列出当前所有集合及其包含的小说，便于选择目标集合；直接回车则沿用默认集合名称。

### 3.1 快速体验示例

以下示例演示了如何在本地创建一本文本小说并完成上传与对话：

```bash
# 1) 生成示例小说（或使用你自己的 TXT 文件）
python scripts/create_sample_novel.py examples

# 2) 将小说上传到 Milvus（会自动询问书名，可回车使用默认值）
python scripts/upload_novels.py ./examples --collection demo --force

# 3) 启动 FastAPI 服务
uvicorn app.main:app --reload

# 4) 使用 curl 体验聊天（需在另一个终端执行）
curl -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "demo", "collection": "demo", "query": "主角是如何遇到伙伴的？"}'
```

执行完成后，终端会返回模型回答以及引用的章节信息。你也可以在 `logs/` 目录中查看对应会话的日志记录。若想快速试验，只需运行一次 `create_sample_novel.py`，无需自己准备文本。

### 4. 启动 FastAPI 服务

```bash
uvicorn app.main:app --reload
```

可用接口：

- `POST /api/chat`：提交 `session_id`、用户问题，可选地指定 `collection`；服务会记住会话最近使用的集合，返回回答与引用来源。
- `GET /api/collections`：列出当前可用集合及其包含的小说。

### 5. 日志记录

所有对话历史会写入 `logs/interactions.log`，包含时间戳、会话 ID、用户问题与模型回复摘要。

## 项目结构

```
app/
  api/
    routes.py           # FastAPI 路由
  models/
    api.py              # Pydantic 数据模型
  services/
    chat_history.py     # 会话历史管理
    embedding.py        # 嵌入向量生成
    hashing.py          # 文件哈希工具
    rag.py              # RAG 流程封装
    text_splitter.py    # 章节 + 窗口切分
    vector_store.py     # Milvus 操作封装
  config.py             # 全局配置
  logger.py             # 日志配置
  main.py               # FastAPI 入口
scripts/
  upload_novels.py      # 小说上传脚本
.env.example            # 配置模板
pyproject.toml          # 依赖与元数据
README.md
```

## 其他说明

- FastAPI 服务和上传脚本均依赖 `.env` 配置。
- 若需要多集合管理，可在上传时使用 `--collection` 指定集合，并通过 `/api/collections` 查看全局概况。
- 上传脚本和 API 当前依赖内存中的会话历史，如需持久化可在此基础上扩展。
- 嵌入模型与对话模型均可替换为其他兼容方案，只需调整对应配置即可。

欢迎根据业务需求进一步扩展，如加入 Web 前端、鉴权、任务队列等功能。
