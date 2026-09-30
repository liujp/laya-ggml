# laya-ggml

用 C 实现 [laya](https://github.com/NandhaKishorM/laya) 决策模型的推理引擎，外加一个可直接上线的 HTTP 服务。

laya 不是生成式 LLM，而是把「结构化决策」当分类来做：一次前向回答一批问题，
返回答案、**校准过的概率**和置信度。它支持三类决策 —— `choice`（多选一）、
`score`（有序量表）、`noul`（是否），并提供了 MCP server、pydantic 结构化输出等外围能力。

一个典型用法是从一句话里同时问出一组判断（意图、紧急程度、是否要退款……），
但引擎本身不绑定任何具体任务：问什么由请求时的 questions 决定。

原生实现依赖 PyTorch + transformers，冷启动几秒、部署要带 CUDA 运行时。
这里用 ggml 重写后：

- **不需要 PyTorch、不需要 transformers、也不需要装 laya** —— 引擎是独立实现，不是包装
- 运行时依赖只有 ggml（自动获取并编译）和 `tokenizers`（约 5 MB）
- 一次回答 12 个问题 **12.7 ms**，比 PyTorch 的批处理还快 1.6×

---

## 性能

测试条件：NVIDIA L20，f16 权重，一次请求 12 个问题，每个问题的序列 54–191 token。
延迟为稳态中位数（去掉首次的模型加载与 CUDA graph 捕获）。

引擎单次调用（不含分词与 HTTP）：

| | 延迟 | 说明 |
|---|---|---|
| PyTorch（GPU，批处理） | 20.9 ms | 基线 |
| **本引擎（打包一次前向）** | **12.7 ms** | **快 1.6×** |
| 本引擎（每个问题各跑一次前向） | 43.3 ms | 不做打包的话 |

端到端（含分词 + HTTP）：

| 方式 | 延迟 | 吞吐 |
|---|---|---|
| 每次起一个进程 | 1728 ms | 0.58 QPS |
| **HTTP 服务** | **24.5 ms** | **40.7 QPS** |

模型约 2.5 GB 显存；转成 f16 后磁盘减半，内存不变（加载时展开为 f32）。

与 PyTorch 的 **choice 一致率 97.5%**（80 个判断），f16 与 f32 结果逐位相同。

延迟主要随**一次请求问多少个问题**走（打包后一次前向，问题越多序列越长）：
上面是 12 个；用 `examples/triage.json`（5 个）实测为 8.0 ms / 124 QPS。

复现：`python3 bench_http.py --requests 60 --concurrency 1 4 8`
（模型路径可用 `LAYA_MODEL` / `LAYA_TOKENIZER` / `LAYA_QUESTIONS` 覆盖）。

---

## 特性

- **一键构建**：ggml 源码自动获取（版本锁定）并随之编译，不用手动准备任何依赖
- **CPU / CUDA**：自动选择，`LAYA_CPU=1` 强制 CPU
- **多问题打包**：一次请求里的 12 个问题合成一次前向，而不是跑 12 次（3.4×）
- **跨 checkpoint**：架构参数从 GGUF 读，22 层 768d 和 28 层 1024d 的 checkpoint 都实测可用
- **任意问题定义**：请求时内联传入，或启动时用 `--questions` 预置
- **协议兼容 laya-serve**：`/v1/systemone` 用同一套 wire 格式，已有客户端改个 baseUrl 即可
- **生产加固**：worker 自愈、请求超时、输入校验、`/metrics`、`/health`、Bearer 鉴权

---

## 快速开始

需要：CMake ≥ 3.14、C 编译器、Python 3 + `tokenizers`。CUDA 可选。

```bash
# 1. 构建 —— ggml 会自动获取并随之编译，无需手动准备依赖
cmake -B build -S . && cmake --build build -j

#    纯 CPU：  -DLAYA_CUDA=OFF
#    用本地已有的 llama.cpp（省一次下载）：
#             -DGGML_SOURCE_DIR=/path/to/llama.cpp

# 2. 转换模型（从 safetensors checkpoint）
#    checkpoint 目录需含 model.safetensors / rl_agent_config.json / tokenizer/
python3 convert_laya_gguf.py --model-dir /path/to/checkpoint --out model.gguf --quantize f16

# 3. tokenizer 给 Python 侧做分词用（引擎本身只吃 token id）
cp /path/to/checkpoint/tokenizer/tokenizer.json .
```

关于 ggml：它是独立的库（[ggml-org/ggml](https://github.com/ggml-org/ggml)），
llama.cpp 只是内嵌了一份副本 —— 本项目直接依赖 ggml 本身，不需要 llama.cpp。
本仓库不含 ggml 源码，配置时按以下顺序找：

1. `-DGGML_SOURCE_DIR=<path>` 指定的位置
2. 相邻的 `../ggml` 或 `../llama.cpp/ggml`
3. 都没有则**浅克隆 ggml 并锁定到 commit `353b63b4`**（0.25.3）

版本是钉死的：引擎是针对 ggml 0.25.3 写的，它的 API 会漂移。换版本用
`-DGGML_PINNED_TAG=<commit>`。

构建产物自带 ggml（`RPATH $ORIGIN`），运行时不需要设 `LD_LIBRARY_PATH`。

验证（这步直接喂 token id，不需要 tokenizer）：

```bash
echo '{"input_ids":[2,6241,1],"marker_pos":[1],"qtype":0}' | ./build/laya-infer model.gguf
# {"logits":[10.2746],"probs":[1.0000],"choice":0,"confidence":1.0000}
```

上面的数字对应 f16 转换的 sql_v3 checkpoint；换成你自己的模型结果自然不同，
这里只是确认引擎能跑通。

---

## 用法

### HTTP 服务（推荐）

```bash
pip install fastapi uvicorn
./laya_server.py --model model.gguf --tokenizer tokenizer.json --port 8100
```

请求体是 `{state, questions}` —— 和 `laya-serve` 一致。下面用的是 laya 官方的
`triage` preset（`examples/triage.json`，与 `laya.triage_questions()` 相同）：

```bash
curl -s localhost:8100/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"message": "I was charged twice for my subscription and I need it fixed today or I am cancelling"},
  "questions": {
    "intent": {"type":"choice", "instructions":"What does the customer want in `message`?",
               "criteria":{"refund":"money returned or a duplicate charge reversed",
                           "technical_help":"a bug, outage or integration problem",
                           "other":"none of the other options fits"}},
    "is_urgent": {"type":"noul", "instructions":"Does `message` communicate time pressure or a deadline?"},
    "frustration": {"type":"score", "instructions":"How frustrated does the customer sound in `message`?",
                    "criteria":["calm and neutral","concerned but civil","clearly annoyed"]}
  }
}'
```

响应是 laya 的 Jev 形状：

```jsonc
{
  "model": "model",
  "answers": {
    "intent":      {"type":"choice", "choice":"refund", "probabilities":{"refund":0.977, ...},
                    "confidence":0.92},
    "is_urgent":   {"type":"noul", "noul":0.0, "confidence":1.0},
    "frustration": {"type":"score", "score":2.19, "legend":{"0":"calm and neutral", ...},
                    "probabilities":{...}, "confidence":0.73}
  },
  "usage": {"input_tokens": 213, "output_tokens": 3}
}
```

要点：

- `state` 可以是字符串，也可以是 `{"字段名": "文本"}` —— 用字典时问题里的
  `instructions` 就能指名道姓地引用它（官方 preset 都这么做）
- `questions` 每次请求带，服务端不需要预置任何东西。想复用的话存成文件，
  启动时用 `--questions triage.json` 加载，请求里就不用带了
- 三种 `type`：`choice`（多选一，`criteria` 为字典）、`score`（有序量表，`criteria` 为数组）、
  `noul`（是否，不用 `criteria`）

> 上面这个 triage 示例只演示 wire 格式。随项目转换出的示例模型是在**别的任务**
> 上训的，跑 triage 拿到的数值不可信 —— 换任务族请先微调（见「限制」）。

**与 laya-serve 的关系**：`/v1/systemone` 用的是同一套协议（请求体同为
`{state, questions}`，响应同为 `{model, answers, usage}`），所以**为 `laya-serve`
写过客户端的话，改个 baseUrl 就能直接用**。两者的服务对象不同 —— laya-serve 跑的是
PyTorch 版模型，这里跑的是 ggml 引擎。

差异有两点：本服务**不支持 `model` 字段的 checkpoint 路由**（一次只加载一个模型），
且不返回 `answer_confidence`/`action`。

鉴权：设 `LAYA_API_KEY` 后需要 `Authorization: Bearer <key>`
（旧的 `X-API-Key` 仍可用）。

| 路由 | 说明 |
|---|---|
| `POST /v1/systemone` | 主路由，Jev 响应格式 |
| `POST /v1/predict` | 别名，本服务早期的响应格式 |
| `GET /health` | 存活状态（含存活 worker 数），免鉴权 |
| `GET /metrics` | 请求数/错误/超时/worker 重启次数 |
| `GET /v1/questions` | 列出启动时加载的问题集及其问题名 |

### 命令行

用法与 laya 自己的 CLI 一致：问句作位置参数，`--questions` 指向定义文件。

```bash
./laya_predict.py --model model.gguf --tokenizer tokenizer.json \
    --questions examples/triage.json 'I was charged twice, please refund today'
```

`--state` / `--schema` 是等价的旧写法，仍然支持。单次调用会启动引擎并加载模型
（约 1.8 s），要反复调用请用上面的 HTTP 服务。

### 引擎协议

`laya-infer` 是引擎可执行文件，按行读写 JSON（一行输入、一行输出），
HTTP 服务就是在它外面套的一层。正常使用不需要直接接触它，但调试时用得上：

```bash
echo '{"input_ids":[...],"marker_pos":[...],"qtype":0}' | ./build/laya-infer model.gguf
```

设 `LAYA_DAEMON=1` 时它不退出，每行一个请求 —— 服务用的就是这个模式。
多条可以用 `{"items":[...]}` 逐个前向，或 `{"packed":{...}}` 合成一次前向（更快）。

---

## 多个问题是怎么合成一次前向的

一个问题跑一次前向，12 个问题就是 12 次图遍历。PyTorch 之所以快，是因为它把
多个问题批进了一次前向。

这里的做法是**拼接，而不是把模型改成带 batch 维**：把所有问题的 token 连成一条
序列，用**块对角 attention mask** 隔开。模型看到的仍是一条普通 2-D 序列，图结构
一行不用改。配套处理了三件事：

- 位置编码在每个问题内部从 0 重新编号
- type embedding 从「整条序列共用一个」改成按 token 查表
- 输出按问题分组做 softmax（温度随该问题的 qtype 变）

mask 要两张：全局层用纯块对角，滑动窗口层用「块对角 ∩ 窗口」。

---

## 限制

用之前请先看这段。

**它是 laya 决策模型的引擎，不是通用 transformer 推理引擎。** 这些是写死的：
encoder FFN 为 GeGLU、head FFN 为 GeLU、RoPE 为 neox 风格、qtype 固定 3 种、
单序列上限 1024 token。换 backbone 或 laya 改结构就得改代码。

**换任务族不等于换配置。** 模型是在特定任务上训出来的。你给它一套语义全新的
question，它照样给出答案和置信度，但那个置信度不可信 —— 官方数据里没训过的
任务族 ECE 是 0.204（训过的是 0.030）。**换业务前请先用几十条标注数据测一下**，
准确率对不上就该微调了。

**能力边界**：一次只加载一个 checkpoint，所以不支持 laya-serve 那种
`model` 字段的多 checkpoint 路由；也不返回 `answer_confidence` / `action`。

**其它**：
- tokenizer 在 Python 侧（`tokenizers` 库，约 5 MB），不是纯 C 部署
- 依赖 ggml 的接口，实测版本为 0.25.3（pinned commit `353b63b4`）
- 单 worker 是单线程，吞吐上限约 40 QPS；多 worker 每个额外占 2.5 GB 显存
- 权重加载时统一展开为 f32，f16 只省磁盘不省内存

---

## 授权与依赖

本项目：**Apache-2.0**（见 `LICENSE`）。

### 需要安装

| 项目 | 授权 | 用途 |
|---|---|---|
| [ggml](https://github.com/ggml-org/ggml) | MIT | 张量库与 CUDA kernel。**自动获取并随项目编译**，不用手动装 |
| [tokenizers](https://github.com/huggingface/tokenizers) | Apache-2.0 | BPE 分词（Python 侧，`pip install tokenizers`）|
| fastapi / uvicorn | MIT / BSD | 仅 HTTP 服务需要，CLI 不需要 |

注意：**不需要安装 laya，也不需要 PyTorch**。本项目是 laya 模型的独立实现，
不是它的包装。本仓库不含模型权重，checkpoint 用 `convert_laya_gguf.py` 转换。
`rl_agent_config.json` / `slot_schema.json` / `tokenizer.json` 都是 checkpoint
派生的产物，已列在 `.gitignore`。
