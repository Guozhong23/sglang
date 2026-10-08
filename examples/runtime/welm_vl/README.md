# WeLM-v4.5-VL：950PR 推理与优化对照

本分支在此前 `welmv4@728b4b63b` 的 VL 适配基础上，完整合并了
[LinyuanLi0046/sglang 的 welmv4-exp 分支](https://github.com/LinyuanLi0046/sglang/tree/welmv4-exp)
至 `94dc9c8ed1b26af858d2ddb9e2e354f3c10d1564`。
新增文本 prefill 的 breakable 图、图内原生 FlashAttn、prefill/decode 混合分块，以及未覆盖捕获 bucket 时的完整 eager 回退；既有 MegaMoE、TP4 fused QKV、decode 图和缓存优化保留。
VL 当前使用 BF16、普通 attention TP，可选 DeepEP EP=TP。48 层主干的 KV mirror 与 OE 语义保留。
视觉编码、base/OE embedding 融合及图像行替换在图外完成，捕获的文本主干不会再次融合 OE；纯文本和图文请求使用一致的图输入边界。
图像缓存键同时包含像素和网格形状。

## 环境与制品

- 950PR 环境需配置 CANN、TorchNPU、Ascend Triton、本仓库 NPU 依赖及 Host embedding 映射支持。先加载服务器自己的 CANN/HCCL 和自定义算子环境，再执行脚本；脚本不会代为安装或升级依赖。
- 必须使用完整 **VL checkpoint**：权重及索引、configuration Python 文件、tokenizer、processor/image processor Python 文件与配置，以及模型原生聊天模板。只有 `config.json` 无法加载。
- 自定义 HF processor 必须提供 `resolve_tokenized_multimodal_inputs` 和 `process_resolved_tokenized_multimodal_prompt`；不能用通用 Qwen processor 替代。
- 优化配置需要 `deep_ep.Buffer/Config`。启用 MegaMoE 需要 `npu_ops_transformer.ops.mega_moe`；启用原生 FlashAttn 需要 `cann_ops_transformer.flash_attn/flash_attn_metadata`；TP4 fused QKV 需要匹配的 `cannbotdsl` 和 950PR 编译环境。
- 脚本先检查制品、processor、优化扩展和可见设备，再以小张量验证视觉 attention 的 BF16、head_dim=72 和多图隔离，最后启动 worker。扩展导入成功不代表所有算子已在目标设备执行验证。
- `--enable-over-encoding` 将 base/OE embedding 放在 Host 映射内存，需要足够 Host 内存。其他参数和权重能否放入 NPU 由实际设备容量决定。

静态制品检查不加载 NPU 扩展：

```bash
python3 examples/runtime/welm_vl/check_model.py /path/to/full/welm-vl-checkpoint
```

## 启动

默认 `WELM_VL_PROFILE=optimized`。在仓库根目录运行；`PYTHON_BIN` 可指定已配置的 NPU Python：

```bash
MODEL_PATH=/data2/weights/release/Welm-V4.5-80B-A3B-VLM \
TP_SIZE=4 BASE_DEVICE=4 PORT=6699 SERVED_MODEL_NAME=welmv4-vl \
bash examples/runtime/welm_vl/run_950pr.sh 2>&1 | tee welmv45-vl-optimized.log
```

`BASE_DEVICE` 是当前可见设备列表的起始编号；上例要求至少有 8 个可见设备。按服务器情况改为 0。
脚本从自身路径设置 `PYTHONPATH`，确保加载本分支代码。模型原生模板位于目录外时设置 `CHAT_TEMPLATE=/path/to/chat_template.jinja`。

两种配置的默认值如下：

| 项目 | optimized（默认） | baseline |
| --- | --- | --- |
| TP / EP | TP=4，EP=TP，DeepEP auto；支持 TP2/4/8 | TP=4，EP=1，普通 MoE；支持 TP1/2/4/8 |
| MegaMoE / 原生 FlashAttn | 开启，可分别覆盖为 0 | 强制关闭 |
| fused QKV | TP4 默认开启，TP2/8 默认关闭 | 强制关闭 |
| shared expert 多流 / OProj RS 流水 | 开启 | 关闭 |
| KV mirror query pruning | 开启 | 关闭 pruning，保留模型所需 K/V 复用 |
| decode 执行图 / radix cache | 开启 | 关闭 |
| 分块预填充 | 16384 tokens | 关闭 |
| 最大并发 / context length | 32 / 32768 | 1 / 8192 |
| 文本 prefill 图 | breakable，需原生 FlashAttn | 关闭 |
| prefill/decode 混合分块 | 开启，需原生 FlashAttn | 关闭 |
| 视觉执行图 | 关闭，视觉/OE 在图外执行 | 关闭 |

优化配置尊重显式设置的开关。例如在 DeepEP 配置中逐项关闭自定义优化定位差异：

```bash
MODEL_PATH=/path/to/full/welm-vl-checkpoint \
WELM_NPU_USE_MEGAMOE=0 WELM_NPU_USE_FLASH_ATTN=0 \
SGLANG_NPU_WELMV4_FUSED_QKV=0 \
bash examples/runtime/welm_vl/run_950pr.sh --disable-cuda-graph
```

`CONTEXT_LENGTH`、`MAX_PREFILL_TOKENS`、`MAX_RUNNING_REQUESTS`、`CHUNKED_PREFILL_SIZE`、`CUDA_GRAPH_MAX_BS` 和 `MEM_FRACTION_STATIC` 可调整容量。
`WELM_VL_PREFILL_GRAPH=0` 和 `WELM_VL_MIXED_CHUNK=0` 可独立关闭本次新增优化，用于与此前的优化路径对照。
显式设置 `WELM_NPU_USE_FLASH_ATTN=0` 时，两者默认也关闭；在 Flash 关闭时显式要求开启它们会提前报错。
`WELM_NPU_MEGAMOE_PREFILL_TOKEN_THRESHOLD` 默认 0；设为正数时，仅 padded pre-scatter token 数严格大于该值的 prefill 使用 MegaMoE，其余走已有回退路径。
共享专家只允许 gate/up 与 TopK 重叠，后续计算在进入 MegaMoE 前完成，保留上游最新执行顺序。
DeepEP normal 默认 AllGather；EP prefill 图要求 AllGather=1、AllToAll=0。
若在 eager prefill 模式改为 AllToAll，需同时设置 `WELM_VL_PREFILL_GRAPH=0` 和 `SGLANG_DEEPEP_NORMAL_USE_ALLGATHER=0`，两种通信策略不能同时开启。

自定义 TopK 的实际开关是 `SGLANG_NPU_MOE_GATING_TOPK_SIGMOID_NO_RENORM`，默认 0；只有确认算子支持模型所需的未归一化 sigmoid 权重时才设置为 1。
旧的 `SGLANG_NPU_WELMV4_USE_FUSED_TOPK` 没有代码消费者。
首次图文验证无需 reasoning/tool parser。`../welm_env.sh` 的频率惩罚排除列表来自文本模型部署，需核对 VL tokenizer 的 token ID 后再使用。

## 新增 prefill 优化与已有服务脚本

新图分为按 token 容量捕获的 Prompt[T] 和按请求数捕获的 Mirror[B]，捕获数量是 T bucket 数加 B bucket 数。
默认 `WELM_VL_PREFILL_TOKEN_BUCKETS="256 512 1024 2048 4096 8192 16384"`（空格分隔），
`SGLANG_WELMV4_PREFILL_GRAPH_BATCH_SIZES="1,2,4,8"`（逗号分隔）。
混合分块开启时，请求数可以向上补齐到 B bucket，dummy 行通过 mask 排除；关闭混合分块时保持上游 exact-B 规则。
任一 token/request bucket 不覆盖当前请求，整段文本主干回退 eager，避免只回放部分图造成 mirror 状态错误。
默认最大并发为 32，但 B bucket 最大为 8；超过 8 个请求的 batch 回退 eager。可按显存容量将 B 集合扩为 `1,2,4,8,16,32`，代价是增加捕获时间和图内存。
若旧脚本使用 `CHUNKED_PREFILL_SIZE=16512`，超过默认最大 T bucket `16384` 的实际 forward 同样回退。可先将 chunk 调为 `16384`，或扩大 T bucket；mixed batch 还需考虑 decode token 加入后的总 token 数。

若沿用已有 `runbf16-vlm-mix.sh`，启用本次优化需要在启动服务前设置：

```bash
export WELM_NPU_USE_FLASH_ATTN=1
export SGLANG_WELMV4_PREFILL_GRAPH_BATCH_SIZES=1,2,4,8
```

删除原先的 `--disable-prefill-cuda-graph`，在 `python -m sglang.launch_server` 参数中加入：

```bash
--cuda-graph-backend-prefill breakable \
--cuda-graph-bs-prefill 256 512 1024 2048 4096 8192 16384 \
--enable-mixed-chunk
```

原先 `WELM_NPU_USE_FLASH_ATTN=0` 的配置不满足 prefill 图条件。
EP=1 和 DeepEP EP=TP 均有支持路径；EP 图还要求 NORMAL AllGather 及 mirror local-sort/AllReduce。
保留 `--enable-kv-mirror`、BF16 模型/KV，并使用 PP=CP=DCP=1；attention DP、LoRA、VL 量化和 MTP 不在此次 VL 适配范围。
使用本仓 `run_950pr.sh` 时，调试可设 `WELM_VL_PREFILL_GRAPH=0`；直接调用 `python -m sglang.launch_server` 的旧脚本应移除新增 prefill 图参数并恢复 `--disable-prefill-cuda-graph`。
`WELM_VL_PREFILL_GRAPH` / `WELM_VL_MIXED_CHUNK` 仅由本仓 launcher 读取；旧脚本关闭 mixed chunk 时需移除 `--enable-mixed-chunk`。不要将同步式 MegaMoE debug 或 MoE tensor dump 与图捕获混用。

上游 PD 地址组装减少临时对象、GC 诊断和网关改动也保留了原提交历史。
详细 GC 诊断需要对应开关，网关 early-decode stream 默认关闭；这些改动没有开放 VL 的 PD 分离部署。

使用本仓 launcher 验收时，先用同一图片和相同采样参数比较 `WELM_VL_PREFILL_GRAPH=0 WELM_VL_MIXED_CHUNK=0` 与默认优化配置，
再覆盖图文/纯文本混合、图片跨 chunk、prefix 命中、不同 T/B bucket，以及超出 bucket 的 eager 回退。
图捕获成功与性能提升必须在 950PR 实测；单请求不会形成 prefill/decode 混合 batch，验证 mixed chunk 需要并发请求。
新增底层算子的实机检查入口：

```bash
python -m pytest -q test/manual/ascend/test_welmv4_prefill_graph_npu.py \
  test/manual/ascend/test_welmv4_prefill_flash_graph_npu.py
```

这些小张量 NPU 测试用于检查负 slot KV 写入、Flash 动态长度和 padding，不代替真实 checkpoint 的图文精度验收。

## 单图验证与对照

等待服务就绪，指定一张小图（首次建议约 448×448）：

```bash
IMAGE_PATH=/path/to/test.jpg SERVED_MODEL_NAME=welmv4-vl \
SGLANG_URL=http://127.0.0.1:6699 MAX_TOKENS=2048 \
bash examples/runtime/welm_vl/curl_single_image.sh | tee optimized-response.txt
```

脚本发送一次标准 OpenAI 图文请求，将图片编码为 data URL，使用 `temperature=0`。
通过条件：HTTP 200、最终文本非空、未因 token 上限截断，并且回答正确描述图片。HTTP 成功本身不能证明视觉精度。

然后停止服务，用同一 checkpoint、同一张图片、同一上下文与输出上限运行基础对照：

```bash
MODEL_PATH=/data2/weights/release/Welm-V4.5-80B-A3B-VLM \
WELM_VL_PROFILE=baseline TP_SIZE=4 BASE_DEVICE=4 PORT=6699 \
SERVED_MODEL_NAME=welmv4-vl CONTEXT_LENGTH=32768 \
bash examples/runtime/welm_vl/run_950pr.sh 2>&1 | tee welmv45-vl-baseline.log
```

对 baseline 再执行上述 curl，将结果另存为 `baseline-response.txt`。两种执行顺序可能产生浮点差异；比较图像内容识别和答案正确性，发现明显偏差时在优化配置下逐项关闭 MegaMoE、FlashAttn、fused QKV、decode 图定位。
进一步验收应覆盖同图重复请求、不同长宽比、多图、纯文本，以及图片跨 chunk 和前缀命中的请求；正式精度结论应来自固定数据集评测，单 curl 仅作冒烟验证。

## 单条图文请求 profiling

`profile_single_image.sh` 可直接复制到服务器运行，只依赖 Bash、curl 和 Python 3 标准库。
对已启动的服务，它依次执行：`GET /v1/models` → `POST /flush_cache` → `POST /start_profile` → **一次** `POST /v1/chat/completions` → `POST /stop_profile`。
模型名自动从服务读取，可兼容 `welmv4-vl` / `welmv45-vl`；可用 `SERVED_MODEL_NAME` 显式选择。脚本不额外发送推理预热请求。

在有 `dog.png` 的目录执行（下面的脚本路径对应当前服务器仓库）：

```bash
IMAGE_PATH=./dog.png \
SGLANG_URL=http://127.0.0.1:7788 \
PROFILE_ROOT=/data2/hw_lly/profiling3 \
MAX_TOKENS=512 \
bash /data2/hw_sgz/06_code_vlm/sglang/examples/runtime/welm_vl/profile_single_image.sh
```

如果把脚本单独复制到了 `01_scripts`，最后一行改为 `bash profile_single_image.sh` 即可。
采集前确保服务没有其他请求，也没有其他 profiling 会话：这些控制接口作用于服务的所有 TP worker，不能按某个 request ID 独立隔离。
服务端保持 `SGLANG_PROFILE_V2=0`（本分支默认值），V2 的手动 start/stop 尚未实现。

**要包含 ViT，必须处理独立的视觉缓存。** `/flush_cache` 只清 KV/前缀等缓存，不清图像 embedding 缓存。
之前对 `dog.png` 的请求若已填充视觉缓存，再请求同图可能跳过 vision encoder/projector；`cached_tokens=0` 也不能证明执行了 ViT。
可改用服务从未处理过的图片，或者在服务启动脚本的 `python -m sglang.launch_server` **之前**添加以下环境变量，再重启服务：

```bash
export SGLANG_VLM_CACHE_SIZE_MB=0
```

该设置只在客户端执行没有作用。禁用视觉缓存后，可先手动完成预热；采集脚本随后清前缀缓存，再记录一条包含视觉计算的请求。
保持当前服务、不改缓存设置时，脚本仍可采集单条请求实际执行的文本 prefill/decode 和其他运算；是否包含 ViT 以 trace 为准。

输出分为两处：

- **服务器**：`${PROFILE_ROOT}/welm_vl_single_<时间>_<进程号>/`，这是 `/start_profile.output_dir`。TP4 的 worker 由 TorchNPU 分别输出采集目录，文件名取决于版本。递归查找 `trace_view.json` 和算子/kernel 报告，不要期待 CUDA 路径的 `TP-*.trace.json.gz`。
- **客户端**：默认 `./welm_vl_profile_client/<本次ID>/`，保存 `request.json`、`response.json`、`profile_request.json`、`metadata.json`、`summary.json`，以及控制接口原始响应和 HTTP 状态码。可通过 `CLIENT_OUTPUT_DIR` 指定独立目录。客户端目录与服务器 trace 目录可能位于不同机器。

NPU 的活动参数在本分支中仍写 `CPU,GPU`，代码会将 GPU 映射到 NPU。脚本显式关闭 stack/shape 记录，不设 `num_steps` 或分阶段自动停止；因此会记录这次请求的实际 prefill 和所有 decode forward，直到响应完成，再等待 stop/export。
采集覆盖 scheduler/model worker；tokenizer/HTTP 进程中的全部 CPU 图片预处理不在同一个 profiler 内。

可调整参数：`MAX_TOKENS=512`、`REQUEST_TIMEOUT=1800`、`PROFILE_TIMEOUT=1800`、`FLUSH_TIMEOUT=30`、`WITH_STACK=0`、`RECORD_SHAPES=0`。
想采集缓存命中路径时设置 `FLUSH_CACHE=0`。若要开启调用栈，需要客户端 `WITH_STACK=1` 且服务端启动时 `SGLANG_PROFILE_WITH_STACK=True`；当前服务脚本的 `False` 会优先生效。
配置鉴权时使用 `SGLANG_API_KEY` 和可选的 `SGLANG_ADMIN_API_KEY`。

成功 start 后，请求失败或脚本收到中断会尽力停止本次 profiling；start/推理/stop 都不会自动重试。
若 start 超时，可能已有部分 worker 开始采集，需要查看服务日志并确认状态后再操作。stop 可能耗时数分钟；HTTP 成功后仍需检查服务器实际生成的 trace，才能确认 NPU events 和 ViT 覆盖。

## 当前范围

本次接通的是 BF16 单体 VL 服务。视频、MTP/speculative decoding、attention DP、视觉 DP、量化 VL、PP、PD/编码器分离部署尚未适配；上游 MXFP8 文本优化已同步，但不等同于 VL 量化支持。
图片请求暂不接受外部 `mm_hashes`，由 processor 根据像素与网格生成，保证缓存一致性。

CPU 回归覆盖共享专家执行顺序、MegaMoE 选择与绑定、VL 图元数据、视觉数学、OE 历史、实际 chunk/cache 切片与逐出重算、网格缓存键及启动脚本契约。
这些测试不能替代完整权重加载、950PR 算子/图捕获、Host 映射、输出精度和吞吐验收。实机失败时保留从首次异常开始的完整日志、实际启动参数、TorchNPU/CANN/算子版本及模型路径。

本次 `welmv4-exp@94dc9c8ed` 合入的本地验证结果：18 个 VL/模型/启动/图与 mixed-chunk 测试文件联合 **308 passed、34 subtests passed**；另行执行 4 个 PD/GC/传输地址测试文件 **43 passed、37 subtests passed**，共 **351 passed、71 个子用例通过**。
这些 CPU 测试执行真实 wrapper/OE/cache/runner 方法及小张量计算，以 mock 替代设备图捕获和大型 transformer；没有验证 NPU kernel 数值。Python 语法、shell 语法及差异空白检查通过。
完整 SGLang 包导入仍受本地依赖缺失限制，通用 runner 测试未通过收集；本机缺少 `cargo`，随上游合入的 Rust 网关测试未执行。

此前 `welmv4@728b4b63b` 适配的本地验证结果：VL/优化/启动相关 193 项、工具调用与频率惩罚 213 项、RoPE 与 mirror 数学 10 项，共 **416 passed，另 10 个子用例通过**。
工具测试中 15 项依赖未提供的 DeepSeek tokenizer，已明确排除；完整包的 DeepEP layout/ngram manager/serving chat 测试因缺少 `sentencepiece` 阻断，fused TopK 的 4 项在缺少 `sgl_kernel_npu` 时导入失败，不能算通过或数值回归结论。
本地只有 VL 配置文件，没有完整 checkpoint 或可用 NPU，因此尚未进行实机服务、curl、精度或性能验收。
