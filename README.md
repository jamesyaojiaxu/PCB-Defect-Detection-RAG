# PCB 多板型异常检测与知识问答系统

本项目面向多种 PCB 板型，支持为不同板型分别训练独立的异常检测模型，并提供集成 RAG 知识库的 Web 管理与智能问答界面。

系统主要包含以下功能：

* 多板型 PCB 数据管理；
* PatchCore / EfficientAD 异常检测模型训练；
* 单 GPU 串行训练任务队列；
* 模型管理与权重下载；
* RAG 知识库构建与检索；
* 基于 Web 页面的训练管理与知识问答。

---

## 一、快速启动

### 1. 环境示例

```text
Python: 3.10.20
Anomalib: 1.2.0
PyTorch: 2.4.1+cu124
CUDA runtime: 12.4
```

### 2. 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 3. 准备 PCB 数据集

将 PCB 数据整理为以下目录：

```text
data/pcb_categories/
├── C1/
│   ├── OK/
│   └── NG/
├── C2/
│   ├── OK/
│   └── NG/
└── ...
```

如果使用 Kaggle 原始 PCB 数据集，可参考第二部分进行自动整理。

### 4. 添加 RAG 知识资料

将用于知识问答的 PDF、Word、PPT、TXT 或 Markdown 文件放入：

```text
knowledge/
```

例如：

```text
knowledge/
├── AOI操作说明.pdf
├── PCB缺陷判定标准.pptx
├── 检验员培训资料.docx
└── 生产规范.txt
```

### 5. 构建 RAG 知识库

```bash
python build_rag_index.py
```

首次执行时会自动下载中文 Embedding 模型：

```text
BAAI/bge-small-zh-v1.5
```

如果暂时只使用模型训练和管理功能，可以跳过此步骤。此时 Web 页面中的聊天框会提示知识库尚未建立。

### 6. 启动 Web 服务

推荐使用以下方式启动系统，不自动创建训练任务：

```bash
python main.py \
  --data-root data/pcb_categories \
  --no-auto-train \
  --host 0.0.0.0 \
  --port 5000
```

浏览器访问：

```text
http://服务器IP:5000
```

进入页面后即可选择 PCB 板型、创建训练任务、查看训练状态以及使用 RAG 知识问答。

---

## 二、准备 PCB 数据集

系统支持直接使用已有 OK / NG 数据，也可以自动整理 Kaggle 原始 PCB 数据集。

### 方案 A：使用已有 OK / NG 数据

标准目录结构如下：

```text
data/pcb_categories/
├── C1/
│   ├── OK/
│   │   ├── image_001.jpg
│   │   └── image_002.jpg
│   └── NG/
│       └── image_101.jpg
├── C2/
│   ├── OK/
│   └── NG/
└── 任意板型名称/
    ├── OK/
    └── NG/
```

数据要求：

* 每个板型目录下必须直接包含 `OK/` 和 `NG/`；
* `OK` 至少包含 2 张图片；
* `NG` 至少包含 1 张图片；
* 图片直接存放在 `OK/` 和 `NG/` 中，不再嵌套子目录。

系统会在训练任务执行时，根据配置自动完成训练集和测试集划分。

### 方案 B：使用 Kaggle 原始数据集

下载 PCB 数据集：

```text
https://www.kaggle.com/datasets/nguyenkhanhai/pcb-dataset
```

假设原始数据存放于：

```text
/data/pcb-dataset
```

执行：

```bash
python prepare_pcb_dataset.py \
  --source /data/pcb-dataset \
  --output data/pcb_categories
```

脚本会将不同缺陷目录统一整理为标准的 `OK / NG` 目录格式。

需要注意：

* `prepare_pcb_dataset.py` 只负责整理数据目录；
* 不在该步骤生成固定的训练集和测试集；
* 实际训练时由系统根据配置自动完成数据划分。

---

## 三、启动模型训练

系统支持通过 Web 页面或命令行创建训练任务。

### 方式 1：通过 Web 页面选择板型

推荐使用该方式。

启动 Web 服务：

```bash
python main.py \
  --data-root data/pcb_categories \
  --no-auto-train \
  --host 0.0.0.0 \
  --port 5000
```

进入网页后：

1. 选择 `PatchCore` 或 `EfficientAD`；
2. 勾选需要训练的 PCB 板型；
3. 点击“为所选类别创建独立模型任务”；
4. 在训练队列中查看任务状态；
5. 训练完成后在模型仓库查看并下载模型权重。

每个 PCB 板型会分别创建和训练独立模型。

### 方式 2：启动时训练全部板型

```bash
python main.py \
  --data-root data/pcb_categories \
  --model patchcore \
  --host 0.0.0.0 \
  --port 5000
```

系统启动后会自动发现数据目录中的所有 PCB 板型，并依次创建训练任务。

### 方式 3：只训练指定板型

例如只训练 `C1` 和 `C2`：

```bash
python main.py \
  --data-root data/pcb_categories \
  --categories C1 C2 \
  --model patchcore \
  --host 0.0.0.0 \
  --port 5000
```

指定 GPU：

```bash
CUDA_VISIBLE_DEVICES=6 python main.py \
  --data-root data/pcb_categories \
  --categories C1 C2 \
  --model patchcore \
  --host 0.0.0.0 \
  --port 5000
```

---

## 四、模型选择

目前支持 `PatchCore` 和 `EfficientAD` 两种异常检测模型。

### 1. PatchCore

启动参数：

```bash
--model patchcore
```

特点：

* 推荐优先使用；
* 训练速度较快；
* 仅使用 OK 图片学习正常样本特征；
* 当前配置采用 `ResNet18` 作为特征提取网络；
* 使用约 10% 的特征构建核心集。

适合快速建立不同 PCB 板型的异常检测基线模型。

### 2. EfficientAD

启动参数：

```bash
--model efficient_ad
```

特点：

* 训练时间相对较长；
* 首次运行需要下载教师网络权重及相关辅助数据；
* 当前训练上限为 `100 epoch` 或 `10000 step`，以先达到者为准。

### 3. 修改训练参数

训练比例、批量大小以及模型相关参数可在以下文件中修改：

```text
configs/training.yaml
```

---

## 五、RAG 知识问答

系统集成了面向 PCB / AOI 场景的 RAG 知识库，可用于查询操作规范、缺陷标准、培训资料及项目说明等内容。

### 1. 添加知识资料

将资料放入：

```text
knowledge/
```

支持的资料示例：

```text
knowledge/
├── AOI操作说明.pdf
├── PCB缺陷判定标准.pptx
├── 检验员培训资料.docx
├── 生产规范.txt
└── 补充说明.md
```

项目根目录中的：

```text
README.md
```

默认也会加入知识库。

### 2. 构建知识库索引

执行：

```bash
python build_rag_index.py
```

当 `knowledge/` 中的资料发生变化后，可以再次执行该命令。

系统会根据文件指纹判断资料是否发生变化，并决定是否需要重新构建索引。

### 3. 配置问答模式


复制环境变量模板：

```bash
cp .env.example .env
nano .env
```

OpenAI 兼容接口示例：

```dotenv
RAG_LLM_BASE_URL=https://api.openai.com/v1
RAG_LLM_MODEL=填写实际模型名称
RAG_LLM_API_KEY=填写实际密钥
RAG_LLM_TIMEOUT=60
```

Ollama 示例：

```dotenv
RAG_LLM_BASE_URL=http://127.0.0.1:11434/v1
RAG_LLM_MODEL=qwen2.5:7b
RAG_LLM_API_KEY=
RAG_LLM_TIMEOUT=120
```

修改 `.env` 后需要重新启动：

```bash
python main.py ...
```

`.env` 已加入 Git 忽略列表，请勿将真实 API Key 提交到代码仓库。

### 4. 使用聊天框

启动 Web 页面后，点击右下角 `AI` 按钮即可进行知识问答。

例如：

```text
系统如何划分 OK 和 NG？
```

```text
C1 和 C2 是否共用模型？
```

```text
PatchCore 使用哪些图片进行训练？
```

```text
某种 AOI 缺陷的判定标准是什么？
```

具体回答效果取决于 `knowledge/` 中是否包含对应资料。

### 5. RAG 配置

Embedding 模型、文本分块大小、检索数量等参数可在以下文件中修改：

```text
configs/rag.yaml
```

---

## 六、常用 API

| 方法   | 地址                 | 作用               |
| ---- | ------------------ | ---------------- |
| GET  | `/api/health`      | 查看服务健康状态         |
| GET  | `/api/dataset`     | 查看板型及 OK / NG 数量 |
| POST | `/api/train`       | 创建训练任务           |
| GET  | `/api/tasks`       | 查看训练任务           |
| GET  | `/api/queue`       | 查看训练队列           |
| GET  | `/api/models`      | 查看已训练模型          |
| GET  | `/api/chat/status` | 查看 RAG 状态        |
| POST | `/api/chat`        | RAG 知识问答         |

### 创建训练任务

```bash
curl -X POST http://127.0.0.1:5000/api/train \
  -H 'Content-Type: application/json' \
  -d '{"name":"pcb","model_type":"patchcore","categories":["C1","C2"]}'
```

### 知识问答

```bash
curl -X POST http://127.0.0.1:5000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"系统如何划分OK和NG？","history":[]}'
```

---

## 七、项目结构

```text
prepare_pcb_dataset.py   Kaggle 原始数据整理
dataset_layout.py        板型发现、OK/NG 验证与数据划分
main.py                  命令行入口与 Web 服务启动
pcb_training_system.py   单 GPU 串行任务队列与 Anomalib 训练
web_interface.py         Flask Web 接口
templates/index.html     训练管理页面与聊天界面

build_rag_index.py       RAG 知识库构建入口
rag/loader.py            PDF/DOCX/PPTX/Markdown/TXT 读取与分块
rag/indexer.py           BGE 向量化与 Chroma 持久化
rag/retriever.py         相似度检索与来源整理
rag/assistant.py         证据回答与 OpenAI 兼容模型调用

configs/training.yaml    模型训练配置
configs/rag.yaml         RAG 检索与模型配置
```

其中：

```text
configs/training.yaml
```

用于配置数据划分、训练参数以及 PatchCore / EfficientAD 相关参数。

```text
configs/rag.yaml
```

用于配置 Embedding 模型、文本分块方式、检索数量以及 RAG 相关参数。
