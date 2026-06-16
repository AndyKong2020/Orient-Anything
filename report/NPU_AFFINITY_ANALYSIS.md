# Orient Anything NPU 亲和性分析

生成日期：2026-06-16

## 分析口径

本报告按“对象绑定、主导路径拆解、算子级拆解、理论平衡点初判、NPU 特有亲和项检查、等价重写建议、待测项收敛”的流程分析，结论绑定如下口径：

| 项 | 口径 |
| --- | --- |
| target_platform | Ascend 950PR |
| NPU 架构版本 | 3510 |
| 运行栈 | PyTorch + torch-npu |
| 部署 dtype | PyTorch float32 权重与激活，未启用 FP16/BF16 autocast |
| 阶段 | 单图整网推理，不区分 prefill/decode |
| 输入形态 | RGB 图像，CPU 侧 resize、center crop、normalize 后进入 NPU |
| NPU 计算主体 | DINOv2-Large 视觉 backbone + MLP 角度预测头 |
| 非 NPU 主体 | 背景移除、图像渲染、WebUI、PIL/NumPy 后处理 |

结论边界：本报告判断的是当前部署流程在 Ascend 950PR 上的单卡推理亲和性。没有 profile 的理论判断均标注为初判或待测，不能替代上板 microbench 或端到端性能 profiling。

## 分析结论

Orient Anything V1 的 NPU 亲和性集中在模型前向主体。DINOv2-Large 的 patch embedding、24 层 transformer block、attention QKV/score/value、MLP、LayerNorm，以及项目 MLP 预测头，主要落在 `Conv2d + Linear + MatMul + Softmax + GELU + LayerNorm + BatchNorm + elementwise add/mul + reshape/transpose/view` 这类标准视觉 transformer 推理算子上。完整模型 smoke 已在 NPU 上通过，官方 V1 Demo 对比中核心角度输出对齐，因此模型主路径具备明确 NPU 可用性。

但该项目不是全链路 NPU 原生应用。图像预处理、背景移除、坐标轴渲染、WebUI 服务和图像叠加仍是 CPU、ONNXRuntime 或 Python 路径。对单图交互式 Demo 来说这不阻塞功能；对性能交付来说，需要把模型前向、预处理、后处理、服务头开销分开统计。

五路径主导判断如下：Cube 路径是模型主体的主导路径；MTE/FixPipe 与 host/head 对单图延迟有中等影响；Vector 路径主要承载 norm、activation、softmax 和 elementwise，压力中等；communication 在当前单卡路径中不参与。

## 主导路径拆解

| 子段 | 主导路径 | 压力 | 证据 | 亲和判定 | 待测 |
| --- | --- | --- | --- | --- | --- |
| 图像预处理到 NPU 输入 | MTE/FixPipe + host/head | 低到中 | CPU 侧完成 resize、crop、normalize，再把 `pixel_values` 迁移到 NPU。单图输入约 3x224x224，H2D 字节量小。 | 中 | 批量输入时 H2D、processor 吞吐和 pinned memory 策略待测。 |
| Patch embedding | Cube + MTE | 中 | 14x14 patch 的 `Conv2d` 后接 flatten/transpose，输入 crop 224 时产生 256 个 patch token。 | 好 | 首层 layout 和 batch 放大后的 tile 利用率待测。 |
| Token/位置嵌入 | Vector + MTE | 低 | CLS token expand、cat、位置编码 add，主要是小规模 elementwise 和 shape 操作。 | 中 | shape 操作是否引入额外 contiguous 拷贝待 profile。 |
| Attention QKV 与输出投影 | Cube | 高 | 每层 3 个 QKV Linear 和 1 个 output Linear，hidden size 1024，24 层。 | 好 | float32 在 torch-npu 上的实际 Cube dtype 档位和效率待 profile。 |
| Attention score/value | Cube + Vector | 中 | 16 heads，257 tokens，`matmul -> div -> softmax -> matmul`。token 数固定且不长，matmul 规模中等，softmax 落 Vector。 | 好到中 | softmax、transpose、contiguous 对单图延迟占比待测。 |
| Transformer MLP | Cube + Vector | 高 | 每层 `Linear -> GELU -> Linear`，MLP ratio 4，hidden 1024，intermediate 4096。FLOPs 占主体。 | 好 | GELU 和 Linear 间是否存在可融合机会待 profile。 |
| LayerNorm、LayerScale、残差 | Vector | 中 | 每层两次 LayerNorm，加 layer scale 逐元素乘法和 residual add。 | 中 | Vector repeat/mask 利用率和小 kernel launch 占比待测。 |
| 项目 MLP 预测头 | Cube + Vector + host/head | 低到中 | CLS token 后接两层 Linear/BatchNorm/ReLU，输出 902 维。矩阵较小，单图 batch=1。 | 中 | 小矩阵 tile 利用率与 launch/head 占比待测。 |
| 输出解码 | host/head + Vector(CPU) | 低 | 当前把 902 维 logits 从 NPU 转 CPU，再做 slice、argmax、softmax 和标量化。D2H 字节量小。 | 中 | 大 batch 时是否保留 NPU 解码待测。 |
| 背景移除 | host/head | 中 | 由 ONNXRuntime U2Net 在 CPU 侧执行，不属于 NPU 算子路径。 | 非 NPU | 如果追求端到端性能，需单独测 CPU 预处理耗时。 |
| 坐标轴渲染与图片叠加 | host/head | 中 | Python/NumPy/PIL 完成坐标轴渲染、旋转和叠加。 | 非 NPU | 批量服务可考虑关闭渲染，仅返回数值。 |
| 单卡通信 | communication | 无 | 当前单卡推理无 HCCL、AllReduce 或跨卡 dispatch/combine。 | 不适用 | 多卡并发或分布式推理未验证。 |

## 算子级拆解

当前部署的 DINOv2-Large 配置为 24 层，hidden size 1024，16 个 attention heads，patch size 14，MLP ratio 4。CPU processor 输出 224x224 图像，因此 patch token 为 16x16，再加 CLS token 得到 257 tokens。

| 阶段 | 主要算子 | 路径 | 当前状态 | 说明 |
| --- | --- | --- | --- | --- |
| 输入迁移 | `Tensor.to("npu")`、dtype cast | MTE/FixPipe | 通过 | 输入 tensor 小，功能无阻塞。 |
| Patch embedding | `Conv2d`、`flatten`、`transpose` | Cube/MTE | 通过 | 标准 ViT patch projection。 |
| Embedding 拼接 | `expand`、`cat`、broadcast `add`、dropout(eval identity) | Vector/MTE | 通过 | 主要是小张量 shape 与 elementwise。 |
| QKV 投影 | `Linear` x3、`view`、`transpose` | Cube/MTE | 通过 | attention 主体计算之一。 |
| Attention score | `matmul`、`transpose`、标量 `div` | Cube/Vector | 通过 | `Q @ K^T / sqrt(d)`。 |
| Attention prob | `softmax`、dropout(eval identity) | Vector | 通过 | token 长度 257，softmax 规模不大。 |
| Attention value | `matmul`、`permute`、`contiguous`、`view` | Cube/MTE | 通过 | `P @ V` 后恢复 hidden layout。 |
| Attention output | `Linear` | Cube | 通过 | 输出投影。 |
| Norm/residual | `LayerNorm`、elementwise `mul`、`add` | Vector | 通过 | DINOv2 pre-norm 和 LayerScale。 |
| FFN | `Linear`、`GELU`、`Linear` | Cube/Vector | 通过 | FLOPs 主体之一。 |
| CLS 提取 | slice/index | MTE/head | 通过 | 只取 CLS token 进入预测头。 |
| 预测头 | `Linear`、`BatchNorm1d`、`ReLU`、`Linear`、`BatchNorm1d` | Cube/Vector/head | 通过 | 输出 902 维 logits。 |
| 解码 | D2H `cpu`、slice、`argmax`、`softmax`、scalar cast | host/head | 通过 | 故意放 CPU，避免 NPU tensor 和 CPU/PIL 混用。 |
| TTA 聚合 | `quantile`、mask indexing、`mean`、`cos`、`sin`、`sqrt`、`atan2` | Vector/host | 待测 | 默认关闭；随机 crop 未纳入主验证。 |

## 理论量化与平衡点初判

基于当前 DINOv2-Large 配置，对单张 224x224 输入做粗略 FLOPs 估算：

| 项 | 估算值 |
| --- | ---: |
| patch embedding FLOPs | 约 0.31 GFLOPs |
| 单层 transformer FLOPs | 约 6.74 GFLOPs |
| 24 层 backbone FLOPs | 约 162.0 GFLOPs |
| 项目预测头 FLOPs | 约 0.0035 GFLOPs |
| 总模型 FLOPs | 约 162.0 GFLOPs |
| 粗略参数量 | 约 304M |

按 roofline 与分路径拆解口径，单段时间可粗略看作：

```text
T_segment ≈ T_head + max(T_GM, T_compute主导)
```

当前部署未启用 FP16/BF16 autocast，模型配置为 float32。若按权重至少从 GM 读一次估算，304M 参数对应约 1.2 GB FP32 权重流量，单图算术密度约 133 FLOP/Byte。Ascend 950PR 的 TF32 Cube 平衡点可粗略落在百级 FLOP/Byte，FP16/BF16 Cube 平衡点约 270 FLOP/Byte。因此当前 float32 单图路径大致位于 TF32/FP32 口径平衡点附近，不能简单写成纯 compute-bound；它可能同时受 Cube 计算、权重搬运和 framework head 影响。

如果后续启用 FP16/BF16 autocast，权重流量可降到约 0.6 GB，算术密度可粗略升至约 266 FLOP/Byte，接近 950PR FP16/BF16 Cube 平衡点。是否实际更快取决于 torch-npu 对 DINOv2 Linear/MatMul/LayerNorm/GELU 的 dtype dispatch、转换开销和精度验收，当前标为待测。

Vector 路径不应用 Cube 平衡点判断。LayerNorm、GELU、softmax、residual add、BatchNorm1d 等属于 Vector 或 framework 小算子组合，单图下很可能受 repeat/mask 利用率、kernel 数和 head 开销影响。MTE/FixPipe 路径中，输入和输出 tensor 字节量很小，主要压力来自权重和中间激活搬运，以及 transpose/contiguous 是否产生真实拷贝。

## NPU 特有亲和项检查

| 亲和项 | 判断 | 证据与说明 | 待测 |
| --- | --- | --- | --- |
| tile 驻留与 double buffer | 中到好 | 主体是规则 Linear/MatMul/Conv2d，hidden 1024、intermediate 4096 适合 Cube 路径。预测头 batch=1、小矩阵 902 维，tile 利用率可能偏低。 | 需要 profiler 确认大 Linear 与小 head 的 tile 利用率。 |
| 32B/512B 对齐与小包 | 中 | hidden 1024 对 FP32 为 4096B/token，天然对齐；token 数 257 在序列维有尾块风险。950 使用 512B cacheline 与 4x128B sector，不应套 A2 GM 512B 规则。 | transpose/contiguous 和小 kernel 是否产生碎片化搬运待测。 |
| repeat/mask 密度 | 中 | GELU、LayerNorm、softmax、BatchNorm、elementwise add/mul 规模中等；单图 batch 低，部分 Vector kernel 可能 mask 利用率不足。 | `aiv_vec_time` 或等价 profile 字段待采集。 |
| layout 折叠 | 中 | flatten/transpose/view 多数是 metadata 或 stride 变换，但 attention 里的 `contiguous` 可能产生真实搬运。 | 需要 profile 或 graph trace 判断真实 GM bytes，避免双计 view/slice。 |
| reduce/state layout | 中 | LayerNorm 和 softmax 是主要 reduce；没有大型状态更新。TTA 聚合涉及 quantile 和 mask indexing，但默认关闭。 | TTA 打开后的 quantile/mask/circular mean 路径待测。 |
| 同步边界 | 中 | 当前是 PyTorch eager/Gradio 单图服务，模型由多个 framework kernel 组成，kernel 间同步和 launch/head 对单图延迟有影响。 | graph capture、batch 推理或编译优化收益待测。 |
| communication | 不参与 | 单卡推理没有 HCCL 或 collective。 | 多卡服务、任务级并发和 HCCL 路径待测。 |
| CV Fusion/NDDMA 机会 | 中 | 模型主体有 `matmul + GELU/LayerNorm/elementwise` 组合，但当前由框架分算子执行。950 具备 CV fusion 与 NDDMA 机会，是否能利用取决于后端图优化。 | 是否有可落地融合图和 dtype/layout conversion 优化待测。 |

## 数学等价重写建议

当前原流程已能在 NPU 上跑通，未发现阻塞单卡功能复现的算子级问题；但如果目标从“功能复现”转为“性能交付”，建议尝试以下等价或近似重写，并重新走五路径分析：

| 重写方向 | 预期收益 | 精度约束 | 状态 |
| --- | --- | --- | --- |
| FP16/BF16 autocast 推理 | 降低权重和激活 GM bytes，提高 Cube 利用率 | 与 float32 输出角度和 confidence 做固定样例 parity，必要时扩展到评测集角度误差 | 待测 |
| 批量推理入口 | 摊薄权重读取、kernel launch 和 host/head 开销，提升大 Linear tile 利用率 | 输出逐样本与单图模式对齐 | 待测 |
| 数值输出与渲染解耦 | benchmark 时只返回角度和 confidence，不生成坐标轴图 | 不影响模型数值输出 | 建议采用 |
| 背景移除可选前处理缓存 | 对固定图片或批量数据预先缓存背景移除结果 | 缓存图像必须与在线 rembg 输出一致 | 待测 |
| NPU 侧批量解码 | 大 batch 下在 NPU 上完成 argmax/softmax，只回传少量标量 | 与 CPU 解码结果一致 | 待测 |
| 图模式或编译优化 | 减少 eager kernel head 和同步开销 | 输出与 eager 模式对齐 | 待测 |

这些重写不改变项目语义，但会改变性能瓶颈：batch 和低精度会把瓶颈更明显地推向 Cube 和 MTE；关闭渲染会让报告更准确地衡量 NPU 模型前向，而不是 WebUI 后处理。

## 待测清单

| 待测项 | 为什么需要 | 建议方法 |
| --- | --- | --- |
| torch-npu float32 Linear/MatMul 实际 dispatch | 当前 dtype 是 float32，但是否使用 TF32 或其他内部路径会影响平衡点判断。 | 用 profiler 记录 matmul/linear kernel、dtype 和耗时。 |
| Cube/Vector/MTE 时间拆解 | 现在有功能 smoke 和理论估算，但没有各路径耗时占比。 | 用 Ascend profiling 采集 Cube、Vector、MTE、kernel launch 统计。 |
| batch size 扫描 | 单图交互模式不代表批量吞吐。 | 测 batch=1/2/4/8 的模型前向，不含 rembg 和渲染。 |
| FP16/BF16 autocast | 可能提升算术密度和吞吐，但需确认精度。 | 固定样例和小评测集做角度 parity 与耗时对比。 |
| TTA 路径 | TTA 默认关闭，包含随机 crop、quantile、三角函数和 mask indexing。 | 固定随机种子后测 CPU/NPU parity 和耗时。 |
| `contiguous` 与 layout 拷贝 | view/slice 不能按真实搬运双计，但 contiguous 可能产生拷贝。 | 通过 profiler 或 graph trace 统计实际 GM bytes。 |
| 多卡/通信 | 当前单卡无 communication 路径。 | 如果需要多卡服务，单独验证 HCCL 或任务级并发。 |
| 端到端 WebUI 耗时拆分 | WebUI 包含 CPU 预处理、NPU 前向、CPU 后处理和网络服务。 | 拆分计时，分别报告 preprocess/model/postprocess/server。 |

## 结论表

| 路径 | 压力 | 证据 | 结论 | 待测 |
| --- | --- | --- | --- | --- |
| Cube | 高 | 约 162 GFLOPs 主体来自 DINOv2 Linear/MatMul/MLP；完整模型 smoke 通过。 | 基于原流程，亲和性好。 | dtype dispatch、tile 利用率和 batch 扫描。 |
| Vector | 中 | LayerNorm、GELU、softmax、BatchNorm、residual add/mul 均在模型前向中覆盖。 | 基于原流程，亲和性中到好。 | repeat/mask 利用率和小 kernel 占比。 |
| MTE/FixPipe | 中 | H2D/D2H 小，但权重和中间激活搬运不可忽略；shape/layout 路径存在潜在拷贝。 | 基于原流程，亲和性中。 | 真实 GM bytes、contiguous 拷贝、L2 命中。 |
| communication | 无 | 单卡推理无 collective。 | 当前不参与。 | 多卡服务另测。 |
| host/head | 中到高 | 单图 WebUI 包含 CPU processor、rembg、渲染和 Gradio；模型 eager 也有多 kernel head。 | 基于原流程，是端到端延迟的重要非 NPU 因素。 | 服务拆时、graph/batch 优化。 |

最终判断：Orient Anything V1 的 NPU 亲和性应写为“模型主体亲和性好，单图端到端为混合 CPU/NPU 路径”。当前证据足以支持单卡 NPU 功能复现和官方 Demo 数值对齐；若要写性能结论，需要补齐 profile、dtype、batch 和 TTA 的待测项。
