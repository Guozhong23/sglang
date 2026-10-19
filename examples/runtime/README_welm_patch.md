# WeLM 推理补丁迁移

本仓库已合入原 `welm_patch` 的三个功能，不需要再执行外部覆盖脚本。提交 `98105d14dc` 的新增行为及后续 `db87a1b143` 的免罚表复用统一由 **`SGLANG_ENABLE_WELM_SERVICE_PATCH`** 控制，默认 `0`（关闭）。启动前设为 `1` 才启用，修改后须重启服务，不支持运行中切换。

开启总开关后：

- `SGLANG_FORWARD_UNKNOWN_TOOLS=1` 时，基础工具解析器允许流式返回未在请求 `tools` 中注册的工具名，与非流式行为保持一致。该子开关默认仍关闭。
- `--tool-call-parser qwen25` 的非流式 `<tool_call>` JSON 解析失败时，使用 `json-repair` 修复后再次解析。流式仍使用原有增量 JSON 解析器。
- 频率惩罚支持指定 token 免罚和按生成长度增加倍率。张量在当前 batch 的设备上创建，去除原补丁在 import 时枚举所有 CUDA 设备的逻辑；生成长度按请求维护并随 batch 合并、过滤。

迁移以仓库提交 `0504090947f447d1dc609c52d15614bc808aea7a` 为基线，保留了现有 `StructuralTag`、`get_model_structural_tag`、原生结构约束和 `parses_required_natively` 等接口。原补丁整文件覆盖会删除这些接口，导致 GLM47 等解析器在服务启动时导入失败。

## 关闭时的兼容行为

未设置总开关或设置为 `0` 时，行为对应 `98105d14dc` 合入前：

- Qwen25 非流式坏 JSON 不修复，且不导入 `json_repair`；合法 JSON 与流式增量 JSON 解析保持原样。
- 基础流式解析器不透传未知工具，即使 `SGLANG_FORWARD_UNKNOWN_TOOLS=1`。非流式未知工具透传早于该提交，仍按这个原有子开关工作。
- 只使用原始 frequency penalty。忽略全部 `FREQUENCY_PENALTY_*` 扩展配置（包括非法值），不分配生成长度张量，不读取免罚列表，不初始化、上传或使用共享 mask。
- `--reasoning-parser qwen3 --tool-call-parser qwen25` 无需改变。总开关不修改 reasoning 解析顺序、工具结构约束或模型权重。

## 部署

若开启总开关，在实际启动服务的 Python 环境安装修复依赖（标准、NPU、CPU、XPU 和 other 安装清单均已声明）。关闭时该功能不依赖此模块的导入：

```bash
python -m pip install json-repair
```

将合入后的仓库同步到部署机，并确保 `PYTHONPATH` 或 editable install 指向该仓库。例如用户日志中的部署目录是：

```bash
export PYTHONPATH=/data2/hw_sgz/02_code/sglang/python${PYTHONPATH:+:$PYTHONPATH}
python -c 'import sglang; print(sglang.__file__)'
python -c 'from sglang.srt.function_call.function_call_parser import FunctionCallParser; from sglang.srt.function_call.base_format_detector import StructuralTag; print("parser imports OK")'
```

原 WeLM 环境变量已保存在同目录的 `welm_env.sh`。从仓库根目录启动时可使用：

```bash
source examples/runtime/welm_env.sh
export SGLANG_ENABLE_WELM_SERVICE_PATCH=1
# 在现有 launch_server 命令中加上：
# --tool-call-parser qwen25
```

`welm_env.sh` 中的 token ID 列表来自原模型部署，须与使用的 tokenizer 对应。该文件仅设置子开关和免罚/倍率参数，不自动打开总开关。它设置 `SGLANG_FORWARD_UNKNOWN_TOOLS=1`；如果希望只允许请求声明的工具，可在 source 后设置 `SGLANG_FORWARD_UNKNOWN_TOOLS=0`。总开关关闭时，该子开关仍按提交前规则影响非流式解析。

频率惩罚扩展要求总开关开启且请求中 `frequency_penalty` 非零。相关环境变量：

| 变量 | 含义 |
| --- | --- |
| `SGLANG_ENABLE_WELM_SERVICE_PATCH` | 总开关，默认 `0`；`1` 开启本页的新增行为和免罚表复用 |
| `FREQUENCY_PENALTY_EXCLUDE_TOKENS` | 非负整数的 JSON 列表；未设置或 `[]` 表示不排除 |
| `FREQUENCY_PENALTY_FACTOR_START` | 非负整数，倍率开始增长前的长度；原配置为 `0` |
| `FREQUENCY_PENALTY_FACTOR_OFFSET` | 正整数，增长间隔；原配置为 `2048` |

START 和 OFFSET 同时设置时，第 `n` 次累计输出 token 的增量倍率为 `1 + max(n - START, 0) // OFFSET`，与原补丁单请求公式一致。只放大本次新增惩罚，不追溯放大已累计惩罚；免罚 token 仍计入长度。缺少任一倍率变量时不缩放。非法配置会在初始化实际使用的频率惩罚器时给出包含变量名的错误。

### 免罚表复用

总开关开启时，每个 worker 固定使用启动时的模型、设备和免罚配置。首次需要频率惩罚时读取并校验免罚配置，在 CPU 上构造免罚表，再上传当前设备；之后所有批次复用这一份只读表，避免重复执行 NPU `index_fill_` 及其可能产生的索引读回。没有 LRU、缓存 key 或淘汰机制；修改免罚配置需要重启服务。请求的累计惩罚和生成长度仍独立保存。

首次上传采用阻塞拷贝，保证表发布后可安全读取；后续复用不再读取免罚配置、构造或上传表。建议预热包含一条非零 `frequency_penalty` 的请求，使真实请求到来前完成初始化。未设置免罚 token 或设置为 `[]` 时只记录初始化完成、不构造表；全部请求 `frequency_penalty=0` 时不触发初始化。总开关关闭时完全跳过这条初始化和复用路径。

## 回归测试

在完整 SGLang 开发环境，从仓库根目录运行：

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/function_call/test_welm_patch.py \
  test/registered/unit/function_call/test_unknown_tool_name.py \
  test/registered/unit/function_call/test_function_call_parser.py \
  test/registered/unit/sampling/test_welm_frequency_penalty.py
```

测试覆盖总开关缺省/关闭/开启、与未知工具子开关的组合、关闭时不导入 JSON 修复库和忽略非法惩罚配置、Qwen3+Qwen25 流式/非流式组合、混合多工具调用、结构约束接口、免罚 token、倍率边界、mask 一次初始化和 batch 合并/过滤。CPU 测试验证逻辑；NPU 上的模型加载、算子执行和实际流式请求需要在部署环境验收。

2026-10-19 总开关改动验证：以上 `test_welm_patch.py`、`test_unknown_tool_name.py`、`test_welm_frequency_penalty.py` 共 43 项通过；另用原 patch 逆向还原并核验 Git blob 的合入前源码作对照，195 项差分检查通过（关闭与历史版本比较，开启与本次修改前版本比较），合计 238 项。使用真实 PyTorch、解析器、协议和 JSON 库，仅绕过无关的顶层包初始化；没有启动完整服务或执行 NPU 测试。

原补丁迁移时的历史验证结果（不代表本次开关改动验收）：213 项通过。由于宿主机未安装完整推理依赖，测试使用临时包路径绕过 `sglang` 顶层及 penaltylib 的启动导入，实际解析器、协议、XGrammar 和 PyTorch 均使用真实实现；没有模拟被测逻辑。另有 15 项需要 DeepSeek tokenizer 的既有测试因缺少 `sentencepiece` 等环境依赖而排除。此结果不代表完整 `launch_server` 或 NPU 模型启动验证。外部安装包也已在临时目录验证正常安装、重复安装、修复旧补丁、拒绝未知文件版本和导入失败后的完整回滚。
