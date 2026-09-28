# WeLM-v4.5-VL：950PR 推理与优化对照

本分支将 [LinyuanLi0046/sglang 的 welmv4 分支](https://github.com/LinyuanLi0046/sglang/tree/welmv4)
合并至 `728b4b63b6f19c26b0f52c966b45c3c4483cc432`，在其文本主干上适配 VL。
同步了 MegaMoE 共享专家 gate/up 与 TopK 重叠、小 token 数的执行路径、MXFP8 文本权重处理、工具调用和频率惩罚修复。
VL 当前使用 BF16：普通 attention TP，可选 DeepEP EP=TP、prefill MegaMoE、原生 FlashAttn、TP4 fused QKV、decode 执行图、分块预填充和 radix cache。
模型原有 OE 与 48 层主干的 KV mirror 保留；图像 embedding 在 OE 融合后替换，图像缓存键同时包含像素和网格形状。

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
| 视觉 / prefill 执行图 | 关闭 | 关闭 |

优化配置尊重显式设置的开关。例如在 DeepEP 配置中逐项关闭自定义优化定位差异：

```bash
MODEL_PATH=/path/to/full/welm-vl-checkpoint \
WELM_NPU_USE_MEGAMOE=0 WELM_NPU_USE_FLASH_ATTN=0 \
SGLANG_NPU_WELMV4_FUSED_QKV=0 \
bash examples/runtime/welm_vl/run_950pr.sh --disable-cuda-graph
```

`CONTEXT_LENGTH`、`MAX_PREFILL_TOKENS`、`MAX_RUNNING_REQUESTS`、`CHUNKED_PREFILL_SIZE`、`CUDA_GRAPH_MAX_BS` 和 `MEM_FRACTION_STATIC` 可调整容量。
`WELM_NPU_MEGAMOE_PREFILL_TOKEN_THRESHOLD` 默认 0；设为正数时，仅 padded pre-scatter token 数严格大于该值的 prefill 使用 MegaMoE，其余走已有回退路径。
共享专家只允许 gate/up 与 TopK 重叠，后续计算在进入 MegaMoE 前完成，保留上游最新执行顺序。
DeepEP normal 默认 AllGather；若改为 AllToAll，需同时设置 `SGLANG_DEEPEP_NORMAL_USE_ALLGATHER=0`，二者不能同时开启。

自定义 TopK 的实际开关是 `SGLANG_NPU_MOE_GATING_TOPK_SIGMOID_NO_RENORM`，默认 0；只有确认算子支持模型所需的未归一化 sigmoid 权重时才设置为 1。
旧的 `SGLANG_NPU_WELMV4_USE_FUSED_TOPK` 没有代码消费者。
首次图文验证无需 reasoning/tool parser。`../welm_env.sh` 的频率惩罚排除列表来自文本模型部署，需核对 VL tokenizer 的 token ID 后再使用。

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

## 当前范围

本次接通的是 BF16 单体 VL 服务。视频、MTP/speculative decoding、attention DP、视觉 DP、量化 VL、PP、PD/编码器分离部署尚未适配；上游 MXFP8 文本优化已同步，但不等同于 VL 量化支持。
图片请求暂不接受外部 `mm_hashes`，由 processor 根据像素与网格生成，保证缓存一致性。

CPU 回归覆盖共享专家执行顺序、MegaMoE 选择与绑定、VL 图元数据、视觉数学、OE 历史、实际 chunk/cache 切片与逐出重算、网格缓存键及启动脚本契约。
这些测试不能替代完整权重加载、950PR 算子/图捕获、Host 映射、输出精度和吞吐验收。实机失败时保留从首次异常开始的完整日志、实际启动参数、TorchNPU/CANN/算子版本及模型路径。

本次本地验证结果：VL/优化/启动相关 193 项、工具调用与频率惩罚 213 项、RoPE 与 mirror 数学 10 项，共 **416 passed，另 10 个子用例通过**。
工具测试中 15 项依赖未提供的 DeepSeek tokenizer，已明确排除；完整包的 DeepEP layout/ngram manager/serving chat 测试因缺少 `sentencepiece` 阻断，fused TopK 的 4 项在缺少 `sgl_kernel_npu` 时导入失败，不能算通过或数值回归结论。
本地只有 VL 配置文件，没有完整 checkpoint 或可用 NPU，因此尚未进行实机服务、curl、精度或性能验收。
