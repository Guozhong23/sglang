# WeLM MegaMoE BF16 共享专家 prefill 融合

更新日期：2026-10-14。代码位于个人仓库 `welmv4-exp` 分支，源自 `team/welmv4-exp`，已拉取的基线为 `db87a1b14335b012acc15b28b6f94d0110e037bd`。推送目标为个人远端 `origin`（`Guozhong23/sglang`）的同名分支。仓库级提交人为 `sunguozhong <sunguozhong@huawei.com>`；未修改全局 Git 用户。

## 1. 当前行为：一个主开关

仅由 `WELM_NPU_USE_MEGAMOE` 控制 MegaMoE。删除此前方案中的独立共享融合开关 `WELM_NPU_MEGAMOE_FUSE_SHARED_EXPERTS`；不再提供服务内的 routed-only MegaMoE 分支。每一次实际 MegaMoE 调用都融合共享专家。

| 主开关 | 批次条件 | MoE 执行 | 共享专家 |
| --- | --- | --- | --- |
| 0 | 任意 | 原小算子/backend，不读取或检查 MegaMoE 阈值 | 原共享 FFN，沿用原多流策略 |
| 1 | 纯 prefill，超过阈值，布局和容量满足 | BF16 MegaMoE | 算子内融合，不启动外部共享 FFN/共享流 |
| 1 | 未超过阈值、容量或布局不满足 | 原小算子/backend | 原共享 FFN，沿用原多流策略 |
| 1 | MIXED、decode、verify、NextN/draft、mirror consumer | 原小算子/backend | 原共享 FFN，沿用原执行策略 |

原共享多流仍受 `SGLANG_NPU_USE_MULTI_STREAM`、DeepEP NORMAL AllGather、批次和布局条件控制。设为 1 时，符合原有条件的回退路径恢复 overlap；设为 0 时仍可串行运行。融合调用本身不受该开关影响，不再执行外部共享分支。

### 阈值的准确含义

`WELM_NPU_MEGAMOE_PREFILL_TOKEN_THRESHOLD` 仅在主开关开启时读取。正阈值使用严格大于比较：512 对应 `<=512` 回退、`>512` 才有资格进入 MegaMoE；值 `<=0` 表示不设长度下限。环境变量定义默认 0，本文启动脚本默认 512，可覆盖。

比较对象是 **attention TP scatter 前、padding 后的批次 token 行数**，不是单条请求的上下文长度，也不是单卡 MoE 行数。入图时按选中的图桶容量决定路径，例如实际 513 行补齐到 1024 桶时按 1024 判断。非 DP 用本卡物理行数乘 attention TP size；DP 用所有参与 rank 一致的 padded slot，不能根据本 rank 有效行数独立选择。

容量另行按每卡物理行数检查，当前上限为 16384。全局 16512 桶、TP4 对应每卡 4128 行，可满足容量条件。超过容量必须在发起 collective 之前一致回退；回退不强制共享专家串行。

## 2. BF16 接入及语义

接口依据：[MegaMoE 共享专家接入指导](../../02_ops/megamoe/mega_moe共享专家接入指导.md)，算子源码基线 `64a85ec3ccd2fec57c9cffa5f75805c3ed70195d`。需要配套共享版本的 wheel 与 CANN 算子包；仅凭 wheel 的版本名不能确认 ABI 匹配。

融合语义：

```text
y = sum(topk_weight[j] * routed_FFN[j](x)) + shared_FFN(x)
```

初始化时缓存完整共享 BF16 权重的 ND 视图：W1 从 `[2*Is,H]` 扩为 `[1,2*Is,H]`，W2 从 `[H,Is]` 扩为 `[1,H,Is]`，复用原存储。调用传入 `l1_weights_shared`、`l2_weights_shared` 两个 tensor 列表，以及值为 `None` 的两个 shared scale 参数。路由、TopK 和专家偏置保留原语义。

初始化检查：

- NPU WeLMV4 target，BF16 activation/routed/shared 权重，EP>1、MoE TP=1、DeepEP backend。
- 共享 MLP 为完整副本，TP rank 0/size 1，down 不做 reduce；使用 SiLU(gate) * up，无 bias、独立 shared gate 或有效 clamp。
- routed 权重为连续 EP 分片，允许正常 identity 专家映射，拒绝动态 EPLB、重映射及冗余专家布局。
- 校验实际 shape、dtype、连续性、同设备和 NPU ND 格式，以及 wrapper 的四个 shared 参数。
- 所有候选层的 routed 维度一致；不存在候选层或显式开启但权重/语义不支持时明确报错。

本次仅实现 BF16。开启主开关时不支持 MXFP8 MegaMoE 共享融合；关闭时使用已有小算子/backend 量化路径。NextN/draft 和 mirror consumer 不绑定融合 runtime。

每个 EP 通信域复用一个 SymmBuffer，在加载和后处理权重之后、KV cache 定容及图捕获之前创建。提前加载通信扩展并初始化 HCCL，按文档 barrier/sync 后创建窗口。不在 forward 中加载、转换权重或新建通信上下文。未向异常退出路径增加 barrier，保留原 `close()` 及通用销毁钩子。

模型以 `use_welm_prefill_megamoe` 的**实际调用选择**决定是否跳过外部 shared。融合结果直接作为完整 MLP 输出，不重复相加或 all-reduce；末层返回 `(combined, None, None)` 并清理分量缓存。已删除原强制共享串行参数及 MegaMoE 专用 shared gate/up overlap 分支，阈值/容量回退可进入原完整共享多流路径。

### 候选模型

[04_model/config.json](../../04_model/config.json) 为 BF16、H=2048、routed/shared I=512、512 个 routed experts、topK=10、1 个 shared expert、无 shared gate/clamp。EP4 对应 routed W1/W2 `[128,1024,2048]` / `[128,2048,512]`，shared W1/W2 `[1,1024,2048]` / `[1,2048,512]`。

当前 `server.sh` 指向另一台机器的 `/data2/hw_sgz/welm/00_weights/WeLM-v4.5-80B-A3B-GZ-RELEASE/1/`。部署时仍需核对实际权重配置；不能仅以本地候选配置代替运行时校验。

## 3. Prefill 图与 padding

只允许纯 `EXTEND` 使用 MegaMoE。图缓存 key 不区分 EXTEND/MIXED，因此开启主开关时在 graph replay 入口拒绝 MIXED，防止 MIXED 复用已捕获融合的 EXTEND 图；关闭主开关时保留原 MIXED 图行为。

沿用 `torch.npu.graph(..., auto_dispatch_capture=True)` 的 breakable prefill 路径，没有增加 eager graph break。当前 adapter 要求 BF16 compute/KV、关闭 DP attention、PP=CP=DCP=1；EP 图还要求 EP=TP、DeepEP NORMAL AllGather，排除 AllToAll。首轮设备验收为 TP4/EP4/DP1；DP eager 的统一参与策略保留，DP attention 入图不属于现有能力。

图中有效行 mask 继续读取设备侧 `num_token_non_padded`，不会把 capture dummy 的有效长度固化进 replay。调用前清理无效 hidden 行，MegaMoE TopK IDs 保持合法，padding 路由权重清零；调用后对包含共享分支的完整输出去 padding。仅清零路由权重不足以抑制共享输出。DP idle rank 只要物理 slot 非空就继续参与，最终在 transport 边界清理 padding。

CPU 测试能检查选择、mask 和调用契约，不能证明 CANN 算子可捕获。配套 wheel/CANN 的实际 stream、workspace/通信窗口生命周期、capture/replay 和整模型数值仍需设备验证。不能将 eager 可运行视为完成入图验收。

## 4. 代码位置

| 文件 | 实现 |
| --- | --- |
| [environ.py](python/sglang/srt/environ.py) | 单主开关及阈值语义 |
| [model_runner.py](python/sglang/srt/model_executor/model_runner.py) | 主开关关闭/draft 提前返回，开启后的资格检查 |
| [welmv4_megamoe.py](python/sglang/srt/hardware_backend/npu/moe/welmv4_megamoe.py) | BF16 shared 权重验证、缓存、通信初始化及融合调用 |
| [welmv4.py](python/sglang/srt/models/welmv4.py) | 纯 prefill 选择；实际融合跳过外部 shared，回退保留多流 |
| [welmv4_dp_attention.py](python/sglang/srt/models/welmv4_dp_attention.py) | 统一 slot/global prefill 选择，删除回退强制串行 |
| [welm_prefill_graph.py](python/sglang/srt/model_executor/runner/welm_prefill_graph.py) | 主开关开启时隔离 MIXED replay |

## 5. 启动配置

```bash
export WELM_NPU_USE_MEGAMOE=1
export WELM_NPU_MEGAMOE_PREFILL_TOKEN_THRESHOLD=512
export SGLANG_NPU_USE_MULTI_STREAM=1
```

改成 `WELM_NPU_USE_MEGAMOE=0` 即整体使用小算子和原共享多流，忽略 MegaMoE 阈值；无需设置独立共享开关。上述变量在启动时固定，不能在已有图捕获后动态切换。

[server.sh](../../01_scripts/server.sh) 已补充默认 512 阈值，并允许从外部覆盖主开关和阈值；原多流设置 1 保留。该脚本当前指定 `--disable-prefill-cuda-graph`，因此直接运行它不会验证 prefill 入图。其远端 `PYTHONPATH` 和模型/vendor 路径保留，部署前应确保该路径中的代码已同步本次修改。

新增 [BF16 prefill 图启动脚本](scripts/npu/run_welm_bf16_megamoe_shared_prefill.sh)，默认主开关 1、阈值 512、多流 1，三者均可通过环境覆盖。基于 1014 P 配置，显式 BF16、TP4/EP4、breakable 图和 DeepEP NORMAL AllGather。先提供 `MODEL_PATH`、`MEGAMOE_VENDOR_DIR`、`ASCEND_MF_STORE_URL`、`ASCEND_MF_HCOM_URL`，并准备匹配的运行环境，再执行：

```bash
bash scripts/npu/run_welm_bf16_megamoe_shared_prefill.sh --help
bash scripts/npu/run_welm_bf16_megamoe_shared_prefill.sh
```

脚本使用当前个人仓的 Python 代码，只启动 prefill worker；decode/router 沿用部署流程。本次没有启动实际服务或安装设备依赖。

## 6. 验证

本地测试命令：

```bash
python -m unittest discover -s test/manual/ascend -p 'test_welmv4_megamoe_*.py' -v
python -m unittest discover -s test/manual/ascend -p 'test_welmv4_prefill_*.py' -v
```

测试覆盖主开关关闭时不读阈值/不导入 runtime 依赖、开启时默认融合、阈值 508/512/516、单卡 16385 超容量回退共享多流、shared 只计算一次、末层分量、动态 padding、DP idle、decode/verify/NextN，以及 MIXED 图隔离。CPU 测试使用真实 BF16 Tensor 和实际生产方法，通信、设备流和 MegaMoE 算子使用替身。

本次统一复测：MegaMoE runtime/model 19 项通过；prefill 回归 67 项通过、8 项设备测试因缺少 NPU 跳过，合计 **86 项通过、8 项跳过**。修改的 Python 语法、两个启动脚本 `bash -n`、新启动脚本和算子验收脚本 `--help`、`git diff --check` 均通过。

此前 registered `test_welmv4_deepep_layout.py` 在导入 SGLang 时因缺少 `orjson` 而收集失败，未执行该套测试。当前环境没有 `torch_npu`、共享版 `npu_ops_transformer` 和 Ascend 运行时，尚无设备精度、图捕获或性能结论。

### Ascend 验收入口

[算子验收脚本](test/manual/ascend/test_welmv4_megamoe_shared_npu.py)：

```bash
torchrun --standalone --nproc-per-node=4 \
    test/manual/ascend/test_welmv4_megamoe_shared_npu.py \
    --hidden 2048 --routed-intermediate 512 --shared-intermediate 512 \
    --experts-per-rank 128 --topk 10 --rows 32,128,520,4128
```

脚本用 routed-only 算子加独立共享 FFN 作为**测试参考**对照融合输出，该参考不是服务模式。检查专家计数、padding、动态输入、同桶多次 replay，并确认 replay 不重新执行 Python wrapper。默认 `rtol=atol=0.03` 仅为可配置 smoke test 起点，不代替项目精度标准。不同桶顺序执行；多个活图交替 replay、完整模型 logits/生成和性能仍需另行验收。`--eager-only` 只用于诊断。

设备上应比较两组：主开关 0 的小算子+共享多流基线，以及主开关 1 的阈值选择+共享融合；保持模型、拓扑、图桶、请求和多流设置一致。重点验证阈值两侧、同桶有效行数变化、大/小/大桶切换、末层 residual/RMSNorm、长时间 replay，以及完整 MoE 延迟、prefill 耗时、TTFT、tokens/s 和显存。

后续设备验收需要实际共享版 wheel/CANN 包及其构建信息、可用 Ascend 环境、目标权重与项目精度基准。`02_ops/megamoe` 当前只有指导文档；邻近 `ops-transformer_3987` 的通用 `cann_ops_transformer` 接口不能视作配套 `npu_ops_transformer` ABI。
