# WeLM-v4.5-VL: 950PR 基础推理

此目录提供 BF16 单体服务的首次验证入口：普通 TP、EP=1、单请求、纯文本/图像输入。
当前关闭 MTP、执行图、分块预填充及 radix cache；不支持视频、attention DP、视觉 DP、MegaMoE、PD/编码器分离部署。模型原有 OE 和 48 层主干所需的 KV mirror 保留。

## 前置条件

- 使用已配置 CANN、TorchNPU、Ascend Triton 和本仓库 NPU 依赖的 950PR 环境。沿用该仓原有 WeLM 文本模型可运行的环境；无需为了本模型覆盖整个 `welmv4.py` 或更换 Transformers 版本。
- 完整的 **VL checkpoint**：权重及索引、configuration Python 文件、tokenizer、processor/image processor Python 文件与配置、模型原生聊天模板。只有 `config.json` 无法加载。
- 自定义 HF processor 必须提供 `resolve_tokenized_multimodal_inputs`、`process_resolved_tokenized_multimodal_prompt`。这些方法来自模型制品，不能用通用 Qwen processor 替代。
- 启动默认使用 4 张可见 NPU，可设置 TP=8；TP=1/2 仅表示维度可切分，是否装得下由设备/Host 内存决定。
- `--enable-over-encoding` 使用目标仓现有的 Host/NPU 映射 embedding，需要相应 CANN/TorchNPU 支持和充足 Host 内存。启动会执行现有映射检查。

## 启动

在仓库根目录运行。先确保 `python3` 对应已配置的 NPU Python 环境，也可以设置 `PYTHON_BIN`。

```bash
MODEL_PATH=/path/to/full/welm-vl-checkpoint \
TP_SIZE=4 BASE_DEVICE=0 PORT=6677 \
bash examples/runtime/welm_vl/run_950pr.sh 2>&1 | tee welmv45-vl-950pr.log
```

`BASE_DEVICE=4` 表示从当前可见设备列表的第 4 个设备开始使用。脚本从自身路径设置 `PYTHONPATH`，确保加载本分支代码。
如果模型原生模板位于目录外，可设置 `CHAT_TEMPLATE=/path/to/chat_template.jinja`。首次验证不配置 reasoning/tool parser，直接检查模型原始文本输出。

启动脚本先检查制品、processor 和可见 NPU，再用小张量检查 BF16、head_dim=72 的视觉 attention 及两张图之间的隔离，最后启动 worker。它会将旧文本启动脚本常用的 MegaMoE、原生 FlashAttn、自定义 TopK、fused QKV 和视觉执行图开关关闭，使用普通 TP 路径。
可在不启动服务的机器上做静态制品检查：

```bash
python3 examples/runtime/welm_vl/check_model.py /path/to/full/welm-vl-checkpoint
```

## 单次 curl 验证

等待服务日志显示就绪，在另一个终端指定一张小图（首次建议约 448×448 像素），执行：

```bash
IMAGE_PATH=/path/to/test.jpg \
SGLANG_URL=http://127.0.0.1:6677 \
bash examples/runtime/welm_vl/curl_single_image.sh
```

脚本将本地图片编码为 data URL，生成标准 OpenAI 图文请求，并且只发送 **一次** `curl` 推理请求。无需让服务端访问客户端图片路径，也不会手工拼接模型特殊 token。

通过条件：HTTP 200、返回非空 `choices[0].message.content`、没有因 token 上限截断、回答正确描述输入图片。HTTP 成功本身不证明视觉精度。
若模型思考较长，可设置 `MAX_TOKENS=2048`；不要把仅有 reasoning 或被截断的输出当作最终答案。

## 验证边界

CPU 测试用于检查视觉数学、OE 历史 token 处理和接口契约，不能替代实机验收。完整权重加载、950PR 上 `BF16 + head_dim=72` 的视觉 attention、Host embedding 映射和最终图文输出，都需要在目标环境运行上述流程确认。
如果拉起失败，保留 `welmv45-vl-950pr.log` 中从首次异常开始的完整堆栈，并记录 TorchNPU/CANN 版本及实际模型制品路径。
