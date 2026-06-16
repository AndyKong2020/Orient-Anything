# Orient Anything — 物体朝向估计 NPU 部署及亲和性报告

| 项 | 内容 |
|---|---|
| 任务编号 | Orient-Anything |
| 任务用途 | 单图物体三维朝向估计,输出 azimuth / polar / rotation / confidence,并生成坐标轴叠加图 |
| 仓库 | https://github.com/SpatialVision/Orient-Anything |
| 版本 / commit | V1 / main,NPU 适配版 |
| 报告人 | N/A |
| 日期 | 2026-06-16 |
| 硬件 | Ascend 950PR ×8 / CANN 9.0.0 |
| 软件 | torch 2.10.0+cpu / torch_npu 2.10.0 / Python 3.11.6 / transformers 4.38.0 / gradio 5.9.0 |

---

## 1. 技术栈梳理
- 主语言 Python / PyTorch。模型为 DINOv2-Large 视觉主干 + MLP 角度预测头。
- 任务方法:CPU 侧图像预处理,可选 rembg 背景移除;NPU 侧执行 DINOv2 与 MLP 前向;CPU 侧解码角度并用 PIL/NumPy 渲染三维坐标轴叠加图。
- ML 框架:PyTorch + transformers,Ascend 侧由 torch_npu 接管标准 PyTorch 算子。
- 训练口径:README 说明模型训练依赖 2M 渲染标注图,但本仓未提供正式训练脚本、数据集配置、loss 封装或分布式训练入口。现有公开代码可构造 DINOv2_MLP 的训练态 forward/backward 最小闭环。
- CUDA 依赖:无必需 CUDA 路径;原项目按 PyTorch device 运行,NPU 适配后保留 CUDA/CPU 回退。
- 自定义核:无 .cu / C++ 扩展;没有需要重写的项目自定义 CUDA kernel。
- 第三方库:transformers、huggingface_hub、gradio、rembg、onnxruntime、Pillow、NumPy、matplotlib、scikit-image。
- 模型权重 / 来源:官方 V1 Orient-Anything 权重 + facebook/dinov2-large 处理器与主干配置;本次验证口径为官方 V1 Demo 同款 902 维预测头。

## 2. 部署步骤
- [x] 依赖安装:使用独立 Python 环境安装 torch_npu、transformers、gradio、rembg、onnxruntime、Pillow、NumPy 等运行依赖。
- [x] 编译 / 构建:无项目级编译步骤,无 CUDA 扩展构建。
- [x] 权重获取:使用官方 V1 权重与 DINOv2-Large 配置,缓存目录放在数据盘;报告不展开下载过程。
- [x] NPU 适配改动(device、torch_npu、禁用 CUDA 核等):入口导入 torch_npu;优先选择 `npu:0`;输入 `pixel_values` 显式迁移到 NPU;模型输出先转 CPU 后做 `argmax`、`softmax` 与标量化;模型缓存、Gradio 监听地址和端口改为环境变量配置。
- [x] 功能保留:原始 Gradio WebUI、背景移除、测试时增强开关、角度文本输出和结果图片输出均保留。
- [x] 训练补测:在 NPU 上构造 batch=2 随机输入和合成分类标签,分别验证 frozen backbone + head 训练、全模型训练的单步反传和 SGD step。

命令:
```bash
export ASCEND_RT_VISIBLE_DEVICES=<last-four-card-id>
export ORIENT_CACHE_DIR=<mounted-data-cache>
export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=<port>
python app.py

python tests/test_inference_decode.py
```

部署边界:模型前向在 NPU,图像预处理、rembg 背景移除、PIL/NumPy 坐标轴渲染和 Gradio 服务仍在 CPU/host 侧。这是当前项目形态下的 NPU+CPU 混合部署,不是全链路 NPU 原生化。

## 3. 验证用例
- 输入数据:项目示例图 `assets/demo.png`。
- 训练输入:随机 RGB tensor `[2,3,224,224]` + 合成 azimuth / polar / rotation / confidence 分类标签。batch=2 是因为预测头含 `BatchNorm1d`,训练态 batch=1 不合法。
- 验证功能:Ascend NPU 基础 tensor、预测头解码、直接模型推理、Gradio WebUI/API、官方 V1 Demo 对齐、训练态反传最小闭环。
- 运行口径:单卡 `npu:0`,关闭测试时增强;分别验证背景移除关闭和开启两条路径。

| 用例 | 运行方式 | 实测输出 | 结果 |
|---|---|---|---|
| NPU tensor smoke | 导入 torch_npu,创建 NPU tensor 并求和 | NPU 可用,设备数 1,tensor sum = 2.0 | 通过 |
| 预测头解码 | unittest 构造 902 维和 722 维 logits | 2 个测试通过,用时 2.390s | 通过 |
| 直接推理,不移除背景 | NPU 前向 + CPU 解码 | azimuth 3.0,polar 0.0,rotation 1.0,confidence 0.9965,耗时 0.077s | 通过 |
| 直接推理,移除背景 | CPU rembg + NPU 前向 + CPU 解码 | azimuth 355.0,polar 0.0,rotation 2.0,confidence 0.2945,耗时 0.553s | 通过 |
| WebUI/API,不移除背景 | Gradio `/predict` | 返回叠加图;文本为 3.0 / 0.0 / 1.0 / 1.0,耗时 0.611s | 通过 |
| WebUI/API,移除背景 | Gradio `/predict` | 返回背景处理后的叠加图;文本为 355.0 / 0.0 / 2.0 / 0.29,耗时 1.808s | 通过 |
| 官方 V1 Demo 对齐 | 同一示例图对比官方 Demo 与本地 NPU WebUI | 不移除背景:两边均为 3.0 / 0.0 / 1.0 / 1.0;移除背景:角度均为 355.0 / 0.0 / 2.0,confidence 官方 0.34、本地 0.29 | 角度对齐 |
| 训练 smoke,frozen backbone | DINOv2-Large NPU 前向,冻结主干,MLP head 训练态 backward + SGD step | out_shape `(2,902)`,loss `18.931299`,耗时 0.523s | 通过 |
| 训练 smoke,全模型 | DINOv2-Large 与 MLP head 全部训练态 backward + SGD step | out_shape `(2,902)`,loss `21.358852`,耗时 0.509s | 通过 |

期望输出:能够返回物体朝向角度、方向置信度和坐标轴叠加图。  
实测输出:单图两种背景开关均可返回图像和数值结果;核心角度与官方 V1 Demo 对齐。  
与 CPU/GPU 基准对比:本轮使用官方 V1 在线 Demo 做推理行为基准;不移除背景时四项完全一致,移除背景时三项角度一致、confidence 有小幅差异。训练 smoke 使用合成数据,只验证 NPU 反传算子链路和 optimizer step,没有 CPU/GPU 训练基准,也不代表完整数据集训练复现。

## 4. NPU 亲和性

| 指标 | 数值 |
|---|---|
| 能否在 NPU 跑通 | 是,DINOv2-Large + MLP head 单卡 NPU 前向通过 |
| NPU 利用率 (npu-smi) | 未形成稳定采样指标;本报告不写利用率结论 |
| HBM 占用 | 未采集峰值;模型约 304M 参数,FP32 权重约 1.2GB,不含激活与运行时开销 |
| 关键算子是否回退 CPU | 模型主体未观察到功能性 CPU 兜底;CPU 路径为预处理、rembg、渲染和 WebUI |
| 性能(吞吐/时延) | 单图直接前向链路 0.077s(无背景移除) / 0.553s(含背景移除);WebUI/API 0.611s / 1.808s;训练 smoke 为单步验证,不写吞吐结论 |

口径:Ascend 950PR、PyTorch eager、float32 权重与激活、单图推理。当前未启用 FP16/BF16 autocast,未采集 profiler,因此以下计算分布为算子结构与粗略理论判断,不是实测 PipeUtilization。

训练口径:公开仓库没有训练入口,因此本轮只做最小训练闭环。全模型 smoke 覆盖 DINOv2-Large 主干和 MLP head 的 forward、loss、backward、SGD step;head smoke 覆盖冻结主干微调场景。torch_npu 在反传中给出 internal format 回退到 base format 的 warning,但没有中断训练步骤。该 warning 需要在正式训练前结合 profiler 和多步 loss 稳定性继续观察。

**模型主体算子明细**:

| 阶段 | 主要算子 | 路径 | 判定 |
|---|---|---|---|
| 输入迁移 | `Tensor.to("npu")`,dtype cast | MTE/FixPipe | 跑通 |
| Patch embedding | `Conv2d`,`flatten`,`transpose` | Cube/MTE | 跑通 |
| Token/位置嵌入 | `expand`,`cat`,broadcast `add` | Vector/MTE | 跑通 |
| QKV 投影 | `Linear` x3,`view`,`transpose` | Cube/MTE | 跑通 |
| Attention score | `matmul`,`transpose`,`div` | Cube/Vector | 跑通 |
| Attention prob | `softmax` | Vector | 跑通 |
| Attention value | `matmul`,`permute`,`contiguous`,`view` | Cube/MTE | 跑通 |
| FFN | `Linear`,`GELU`,`Linear` | Cube/Vector | 跑通 |
| Norm/residual | `LayerNorm`,elementwise `mul/add` | Vector | 跑通 |
| 预测头 | `Linear`,`BatchNorm1d`,`ReLU`,`Linear`,`BatchNorm1d` | Cube/Vector/head | 跑通 |
| 解码 | D2H `cpu`,slice,`argmax`,`softmax`,scalar cast | host/head | 跑通 |
| TTA 聚合 | `quantile`,mask indexing,`mean`,`cos`,`sin`,`sqrt`,`atan2` | Vector/host | 默认关闭,待测 |
| 训练反传 | Linear/MatMul/Conv/LayerNorm/GELU/Softmax/BatchNorm 的 backward,SGD step | Cube/Vector/MTE/head | 单步 smoke 跑通 |

**各计算单元逐条判定**:

| 单元 | 压力 | 判定 | 证据 |
|---|---|---|---|
| 算力(Cube,矩阵卷积) | 高 | 亲和 | DINOv2-Large 24 层 transformer,hidden 1024,16 heads;主路径为 Conv2d、Linear、MatMul、FFN |
| 向量(Vector,归一激活) | 中 | 亲和但需 profiler | LayerNorm、GELU、Softmax、BatchNorm、elementwise add/mul 均在前向中覆盖 |
| 搬运(MTE/FixPipe) | 中 | 可跑通,存在优化空间 | H2D/D2H 小;权重、激活、transpose/contiguous 可能贡献真实搬运 |
| 通信(communication) | 无 | 不参与 | 当前单卡推理,无 HCCL、AllReduce 或模型并行 |
| 调度(host/head) | 中到高 | 端到端重要因素 | PyTorch eager 多 kernel + Gradio 单图服务;预处理、rembg、渲染均在 host 侧 |

**理论平衡点初判**:

| 项 | 估算值 |
|---|---:|
| patch embedding FLOPs | 约 0.31 GFLOPs |
| 单层 transformer FLOPs | 约 6.74 GFLOPs |
| 24 层 backbone FLOPs | 约 162.0 GFLOPs |
| 项目预测头 FLOPs | 约 0.0035 GFLOPs |
| 粗略参数量 | 约 304M |

单图 FP32 权重流量按至少读取一次估算约 1.2GB,算术密度约 133 FLOP/Byte。950PR FP16/BF16 平衡点约 270 FLOP/Byte,当前 float32 eager 路径不能直接写成纯 compute-bound,更准确的判断是 Cube 计算、权重/激活搬运和 framework head 共同影响。若后续启用 FP16/BF16 autocast,权重流量约降到 0.6GB,算术密度约 266 FLOP/Byte,接近 FP16/BF16 平衡点,但需要重新做精度和 profiler 验证。

算子回退清单:
- 已知非 NPU 路径:AutoImageProcessor resize/crop/normalize、rembg/ONNXRuntime 背景移除、PIL/NumPy 坐标轴渲染、Gradio 服务。
- 未观察到模型主体功能性 CPU 兜底;但尚未用 profiler 证明每个 kernel 的实际后端和耗时占比。

profiler 摘要:
- 本轮未采集 Ascend profiler,不提供算子耗时排名、PipeUtilization、HBM 峰值或 NPU 利用率结论。
- 已有证据为 NPU smoke、模型前向 smoke、WebUI/API smoke、官方 Demo 数值对齐和解码 unittest。

## 5. 阻塞项

| 阻塞点 | 原因 | 是否硬阻塞 | CANN/AscendC 替代方案 | 兜底 |
|---|---|---|---|---|
| 仓内无 benchmark/eval 脚本 | 原项目主要提供单图 Demo 和 Python 调用,无标注评测集入口 | 否 | 不需要 AscendC;应补批量评测脚本和指标定义 | 用官方 Demo parity + 单图 smoke 验证功能 |
| 仓内无正式训练脚本 | 缺少数据集读取、loss 汇总、训练参数、checkpoint 保存、分布式启动和评测闭环 | 是,阻塞完整训练复现 | 不需要 AscendC;应先补 PyTorch 训练工程,再考虑 NPU profiler 优化 | 当前只能证明模型代码的 NPU 单步反传可用 |
| rembg 与图像渲染不在 NPU | 背景移除走 ONNXRuntime CPU,结果可视化走 PIL/NumPy | 否 | 可单独替换为 NPU/CV 加速,但不影响模型前向 | 保持 CPU 后处理,性能统计与模型前向分开 |
| TTA 路径未纳入主验证 | TTA 含随机 crop、quantile、mask indexing 与三角函数聚合 | 否 | 可固定随机种子后做 NPU/CPU parity | 默认关闭 TTA,先保证原 WebUI 主路径 |
| 缺少 profiler 指标 | 当前只有功能 smoke 和理论估算 | 否 | 用 Ascend profiler 采集 Cube/Vector/MTE/head 统计 | 报告中不写利用率和热点耗时结论 |
| 多卡/通信未验证 | 当前模型单图可单卡装下,无模型并行需求 | 否 | 数据并行多实例更合适,不建议优先 TP/HCCL | 单卡交互式服务先交付 |

## 6. 结论
- 运行方案:NPU+CPU 混合部署。DINOv2-Large backbone 与 MLP 预测头在 NPU 上跑通;预处理、背景移除、输出解码后的图像渲染和 WebUI 仍在 CPU/host 侧。
- 功能结论:单图朝向估计、可选背景移除、角度文本输出、confidence 输出、坐标轴叠加图和 Gradio WebUI/API 均验证通过。
- 训练结论:公开模型代码的训练态最小闭环在 NPU 上通过,包括 frozen backbone 微调和全模型单步 backward + SGD step;但仓库缺少正式训练脚本与数据集配置,不能宣称完成 2M 渲染数据训练复现。
- 对齐结论:关闭背景移除时本地 NPU 与官方 V1 Demo 四项输出一致;开启背景移除时三项角度一致,confidence 小幅不同。
- 亲和性结论:模型主体由标准视觉 transformer 算子组成,Cube 路径压力高且亲和;Vector 与 MTE 压力中等;communication 不参与;端到端延迟受 host/head、CPU 预处理和渲染影响明显。
- 待办 / 风险:补正式训练入口、真实数据集 loader、训练 loss/metric、checkpoint 保存恢复、多步 loss 稳定性、Ascend profiler、HBM 峰值、batch size 扫描、FP16/BF16 autocast 精度对齐、TTA 固定随机验证和批量 benchmark。当前不应宣称全链路 NPU 原生化、完整训练复现或论文级指标复现。
