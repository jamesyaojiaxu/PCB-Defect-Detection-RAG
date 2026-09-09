# PCB 多板型异常检测训练系统

这是一个通用的多类别PCB异常检测训练系统。系统不知道C1、C2等具体类别名，只识别统一的数据约定：每个类别目录下直接包含 `OK/` 和 `NG/`，文件夹中直接放图片。

```text
datasets/
├── 任意类别A/
│   ├── OK/
│   └── NG/
├── 任意类别B/
│   ├── OK/
│   └── NG/
└── 其他任意名称/
    ├── OK/
    └── NG/
```

系统会动态发现所有类别，为每个类别自动划分训练集并训练独立模型。

## 1. 核心逻辑

职责已经分开：

```text
prepare_pcb_dataset.py
    原始Kaggle数据 → 类别/OK、类别/NG
    不划分train/test

main.py + pcb_training_system.py
    动态发现任意类别
    训练开始时自动划分OK
    每个类别训练一个独立模型
```

默认自动划分规则：

- 每个类别的80% OK图片用于训练；
- 剩余20% OK图片用于正常测试；
- 该类别的全部NG图片用于异常测试；
- 随机种子固定为42，重复运行得到相同划分；
- 输入 `OK/NG` 不会被修改，任务目录中使用硬链接生成临时训练结构。

例如一个类别有493张OK和509张NG，训练时自动得到：

```text
training_results/<task_id>/dataset_split/
├── train/OK/    394张
├── test/OK/      99张
└── test/NG/     509张
```

比例可在 `configs/training.yaml` 中调整：

```yaml
data:
  normal_train_ratio: 0.8
  split_seed: 42
```

## 2. 项目结构

```text
PCB_Defect_Detection-main/
├── prepare_pcb_dataset.py      # 可单独复制使用的数据整理脚本
├── dataset_layout.py           # 通用类别发现、OK/NG验证、自动划分
├── main.py                     # 系统入口
├── pcb_training_system.py      # 训练队列和Anomalib调用
├── web_interface.py            # Flask API
├── build_rag_index.py          # 构建问答知识索引
├── rag/                        # 文档解析、Chroma检索和问答服务
├── templates/index.html        # 管理页面
├── configs/training.yaml       # 划分和模型参数
├── configs/rag.yaml            # RAG知识库与检索参数
├── knowledge/                  # PDF/PPTX/DOCX/Markdown/TXT知识资料
├── tests/                      # 单元测试
├── environment.yml             # Conda环境
├── requirements.txt            # Python依赖
├── data/pcb_categories/        # 独立脚本整理结果
├── training_results/           # 任务、自动划分和训练输出
├── trained_models/             # 最终模型权重
└── logs/                       # 日志
```

## 3. 上传到服务器的文件

推荐先在本地把数据整理为 `类别/OK、类别/NG`，服务器只上传下面这些内容：

```text
PCB_Defect_Detection-main/
├── main.py
├── dataset_layout.py
├── pcb_training_system.py
├── web_interface.py
├── build_rag_index.py
├── rag/
├── prepare_pcb_dataset.py      # 建议保留；仅运行系统时不会自动调用
├── configs/training.yaml
├── configs/rag.yaml
├── templates/index.html
├── knowledge/                  # 需要问答系统检索的资料
├── .env.example                # 大模型接口配置示例，不含密钥
├── requirements.txt
├── environment.yml
└── data/pcb_categories/        # 或换成自己的数据目录
    └── 任意板型/OK + NG
```

以下内容不是首次训练服务器的必需文件：

- `tests/`：只用于测试，建议第一次部署时上传，确认后可删除；
- 原始 `pcb-dataset/`：数据已经整理好时不必上传；
- `training_results/`、`logs/`：运行后自动创建；
- `trained_models/`：训练后自动创建；迁移已有模型时才上传；
- `data/rag_index/`：可在服务器重新构建；希望免去首次建库时也可以一起上传；
- `output/`、截图、编辑器配置、缓存和旧TXT标注：不需要上传。

如果准备在服务器上整理Kaggle原始数据，则还要上传原始 `pcb-dataset/`，再运行 `prepare_pcb_dataset.py`。该脚本与系统解耦，系统本身只读取整理后的OK/NG目录。

## 4. 独立数据整理脚本

[prepare_pcb_dataset.py](prepare_pcb_dataset.py) 是一个自包含脚本，不导入项目其他Python文件，可以单独复制到别处运行。

它只负责把Kaggle原始结构：

```text
pcb-dataset/
├── OK/Sxxxx/*.jpg
└── NG/缺陷代码/Sxxxx/*.jpg
```

整理成：

```text
data/pcb_categories/
├── C1/OK + NG
├── C2/OK + NG
├── C3/OK + NG
├── C4/OK + NG
└── C5/OK + NG
```

C1–C5只是在这个具体数据集上从文件名解析出的类别样例，不是系统内置概念。将这些目录改名为 `board_a`、`board_b`，系统仍可正常发现。

只分析：

```bash
python prepare_pcb_dataset.py --analyze-only
```

执行整理：

```bash
python prepare_pcb_dataset.py
```

指定路径：

```bash
python prepare_pcb_dataset.py \
  --source /data/pcb-dataset \
  --output /data/pcb_categories
```

重新生成或强制复制：

```bash
python prepare_pcb_dataset.py --force
python prepare_pcb_dataset.py --copy-files
```

脚本会：

- 解析文件名中的类别样例；
- 将JPG/PNG同名文件去重并优先保留JPG；
- 合并HS/QS/YW/ZW为NG，但在 `manifest.csv` 保留原缺陷代码；
- 默认使用硬链接，失败时自动复制；
- 生成 `manifest.csv` 和 `dataset_metadata.json`；
- 不创建任何 `train/` 或 `test/` 目录。

## 5. 系统直接使用自己的OK/NG数据

不使用Kaggle数据时，不需要运行整理脚本。直接准备：

```text
/data/my_boards/
├── phone_board/
│   ├── OK/*.jpg
│   └── NG/*.jpg
├── power_board/
│   ├── OK/*.jpg
│   └── NG/*.jpg
└── controller_board/
    ├── OK/*.jpg
    └── NG/*.jpg
```

验证：

```bash
python main.py --data-root /data/my_boards --validate-only
```

启动页面但不自动训练：

```bash
python main.py --data-root /data/my_boards --no-auto-train
```

为所有发现的类别创建训练任务：

```bash
python main.py --data-root /data/my_boards --model patchcore
```

只训练指定类别：

```bash
python main.py \
  --data-root /data/my_boards \
  --categories phone_board power_board
```

系统不限制类别数量，也不要求类别名使用C开头。

## 6. 训练与模型

训练任务使用单GPU串行队列，避免多个类别同时训练导致显存不足。

```text
pending → running → completed
                  └→ failed
pending → cancelled
```

每个类别生成独立权重：

```text
trained_models/
├── pcb_dataset_phone_board_patchcore.ckpt
├── pcb_dataset_power_board_patchcore.ckpt
└── pcb_dataset_controller_board_patchcore.ckpt
```

支持：

- PatchCore；
- EfficientAD；
- 任务状态持久化；
- 自动恢复等待任务；
- Web查看队列；
- 独立模型下载。

## 7. PCB知识助手与RAG

网站右下角增加了一个 `AI` 小聊天入口。点击后展开聊天界面，问题经过以下流程：

```text
用户问题 → BGE中文向量 → Chroma相似度检索 → 带来源的证据
                                           ↓
                         OpenAI兼容模型生成回答（可选）
```

实现参考了 `jamesyaojiaxu/AOI-agent` 中提交 `1dfef42 add AOI inspection RAG assistant` 的设计：文档带页码解析、本地中文Embedding、Chroma持久化、结构化证据和强制引用。当前项目使用更轻量的Flask实现，以兼容Python 3.10.20和现有Anomalib环境。

### 准备知识文档

把可信资料放入：

```text
knowledge/
├── AOI操作说明.pdf
├── PCB缺陷判定标准.pptx
├── 产线培训.docx
└── 补充说明.md
```

支持：`.md`、`.txt`、`.pdf`、`.docx`、`.pptx`。根目录 `README.md` 默认也会被索引。

构建知识索引：

```bash
python build_rag_index.py
```

首次运行会下载 `BAAI/bge-small-zh-v1.5`。Embedding固定在CPU上运行，不占用Anomalib训练GPU。资料发生变化后再次执行会自动重建；强制重建使用：

```bash
python build_rag_index.py --force
```

### 选择问答模式

不配置大模型时，聊天框仍能工作，但只返回最相关的证据摘录和来源。

需要生成式回答时：

```bash
cp .env.example .env
nano .env
```

填写OpenAI兼容接口：

```dotenv
RAG_LLM_BASE_URL=https://api.openai.com/v1
RAG_LLM_MODEL=你的模型名称
RAG_LLM_API_KEY=你的密钥
RAG_LLM_TIMEOUT=60
```

也支持本机Ollama、vLLM等兼容 `/v1/chat/completions` 的服务。例如Ollama：

```dotenv
RAG_LLM_BASE_URL=http://127.0.0.1:11434/v1
RAG_LLM_MODEL=qwen2.5:7b
RAG_LLM_API_KEY=
```

`.env` 已被Git忽略，不能把真实密钥上传到代码仓库。

## 8. Ubuntu 22.04 + Conda + CUDA 12.4

### 安装系统依赖

```bash
sudo apt update
sudo apt install -y git curl build-essential libgl1 libglib2.0-0 libgomp1
nvidia-smi
```

CUDA 12.x的Linux驱动至少应为525.60.13，建议使用550系列或更新驱动。PyTorch Wheel包含CUDA运行库，通常不必安装完整CUDA Toolkit。

### 安装Miniconda

```bash
cd /tmp
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### 创建环境

```bash
cd ~/PCB_Defect_Detection-main
conda env create -f environment.yml
conda activate pcb-anomaly
python --version
```

这里必须显示 `Python 3.10.20`。`environment.yml` 已固定该补丁版本。

### 安装CUDA 12.4版PyTorch

```bash
python -m pip install \
  torch==2.4.1 \
  torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install -r requirements.txt
python -m pip check
```

项目固定使用 `anomalib==1.2.0`。`requirements.txt` 写成 `anomalib[core]==1.2.0`，其中 `[core]` 只是同时安装训练所需组件，不会改变Anomalib版本。

验证GPU：

```bash
python - <<'PY'
import torch
import anomalib
import sys
print("Python:", sys.version.split()[0])
print("Anomalib:", anomalib.__version__)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "不可用")
PY
```

正确结果应包含：Python `3.10.20`、Anomalib `1.2.0`、PyTorch `2.4.1+cu124`，并且 `CUDA available` 为 `True`。

### 整理、建库、测试、运行

```bash
# 仅当当前还是Kaggle原始数据时执行；已有类别/OK/NG时跳过
python prepare_pcb_dataset.py --source /data/pcb-dataset --output data/pcb_categories

# 把知识资料放入knowledge/后执行；至少会索引README.md
python build_rag_index.py

python -m unittest discover -s tests -v
python main.py --data-root data/pcb_categories --validate-only
python main.py --data-root data/pcb_categories --no-auto-train --host 0.0.0.0 --port 5000
```

访问：

```text
http://服务器IP:5000
```

在浏览器页面选择板型和模型后点击创建任务。也可以直接启动并自动为全部板型创建PatchCore训练任务：

```bash
python main.py --data-root data/pcb_categories --model patchcore --host 0.0.0.0 --port 5000
```

如果Ubuntu防火墙已启用，还需放行端口：

```bash
sudo ufw allow 5000/tcp
```

## 9. Web API

| 方法 | 地址 | 作用 |
|---|---|---|
| GET | `/api/dataset` | 动态类别和OK/NG数量 |
| POST | `/api/train` | 为所选类别创建任务 |
| GET | `/api/tasks` | 任务列表 |
| GET | `/api/queue` | 当前队列 |
| GET | `/api/models` | 已训练模型 |
| GET | `/api/models/<name>/<category>/<model>/download` | 下载类别模型 |
| GET | `/api/chat/status` | 知识索引和问答模式状态 |
| POST | `/api/chat` | 检索知识库并回答问题 |

请求示例：

```bash
curl -X POST http://127.0.0.1:5000/api/train \
  -H 'Content-Type: application/json' \
  -d '{"name":"pcb","model_type":"patchcore","categories":["phone_board","power_board"]}'
```

问答请求示例：

```bash
curl -X POST http://127.0.0.1:5000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"系统如何划分OK和NG？","history":[]}'
```

## 10. 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖：独立脚本分类整理、格式去重、无预设名称的类别发现、训练时自动划分、缓存复用、任务创建和取消、RAG文档分块、证据回答和聊天API。

## 11. 官方参考

- [Conda Linux安装](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
- [PyTorch CUDA 12.4安装](https://docs.pytorch.org/get-started/previous-versions/)
- [Anomalib](https://github.com/open-edge-platform/anomalib)
- [NVIDIA CUDA兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [AOI-agent 新增AOI检验RAG提交](https://github.com/jamesyaojiaxu/AOI-agent/commit/1dfef42e76af3b9ef45a01497e0ccb33c45f2635)
- [Chroma](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
