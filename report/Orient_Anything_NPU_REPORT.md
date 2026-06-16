# Orient Anything — 物体朝向估计 NPU 部署及亲和性报告

| 项 | 内容 |
|---|---|
| 任务编号 | Orient-Anything |
| 任务用途 | 单图物体三维朝向估计,输出 azimuth / polar / rotation / confidence,并生成坐标轴叠加图 |
| 仓库 | https://github.com/SpatialVision/Orient-Anything |
| 版本 / commit | V1 / main,NPU 适配版 |
| 报告人 | NPU 复现 |
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
- 模型权重 / 来源:官方 V1 Orient-Anything 权重 + facebook/dinov2-large 处理器与主干配置;验证口径为官方 V1 Demo 同款 902 维预测头。

## 2. 部署步骤
- [x] 依赖安装:使用独立 Python 环境安装 torch_npu、transformers、gradio、rembg、onnxruntime、Pillow、NumPy 等运行依赖。
- [x] 编译 / 构建:无项目级编译步骤,无 CUDA 扩展构建。
- [x] 权重获取:使用官方 V1 权重与 DINOv2-Large 配置,缓存目录放在数据盘。
- [x] NPU 适配改动(device、torch_npu、禁用 CUDA 核等):入口导入 torch_npu;优先选择 `npu:0`;输入 `pixel_values` 显式迁移到 NPU;模型输出先转 CPU 后做 `argmax`、`softmax` 与标量化;模型缓存、Gradio 监听地址和端口改为环境变量配置。
- [x] 功能保留:原始 Gradio WebUI、背景移除、测试时增强开关、角度文本输出和结果图片输出均保留。
- [x] 训练验证:在 NPU 上构造 batch=2 随机输入和合成分类标签,分别验证 frozen backbone + head 训练、全模型训练的单步反传和 SGD step。

命令:
```bash
export ASCEND_RT_VISIBLE_DEVICES=<last-four-card-id>
export ORIENT_CACHE_DIR=<mounted-data-cache>
export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=<port>
python app.py

python tests/test_inference_decode.py

python tools/profile_npu_forward.py \
  --ckpt <orient-anything-ckpt> \
  --save-path <local-prof-dir> \
  --active 100 \
  --warmup 2
```

部署边界:模型前向在 NPU,图像预处理、rembg 背景移除、PIL/NumPy 坐标轴渲染和 Gradio 服务仍在 CPU/host 侧。项目运行形态为 NPU+CPU 混合部署,不是全链路 NPU 原生化。

## 3. 验证用例
- 输入数据:项目示例图 `assets/demo.png`。
- 训练输入:随机 RGB tensor `[2,3,224,224]` + 合成 azimuth / polar / rotation / confidence 分类标签。batch=2 是因为预测头含 `BatchNorm1d`,训练态 batch=1 不合法。
- 验证功能:Ascend NPU 基础 tensor、预测头解码、直接模型推理、Gradio WebUI/API、官方 V1 Demo 对齐、训练态反传最小闭环。
- 运行口径:单卡 `npu:0`,关闭测试时增强;分别验证背景移除关闭和开启两条路径。

| 用例 | 运行方式 | 实测输出 | 结果 |
|---|---|---|---|
| NPU 基础验证 | 导入 torch_npu,创建 NPU tensor 并求和 | NPU 可用,设备数 1,tensor sum = 2.0 | 通过 |
| 预测头解码 | unittest 构造 902 维和 722 维 logits | 2 个测试通过,用时 2.390s | 通过 |
| 直接推理,不移除背景 | NPU 前向 + CPU 解码 | azimuth 3.0,polar 0.0,rotation 1.0,confidence 0.9965,耗时 0.077s | 通过 |
| 直接推理,移除背景 | CPU rembg + NPU 前向 + CPU 解码 | azimuth 355.0,polar 0.0,rotation 2.0,confidence 0.2945,耗时 0.553s | 通过 |
| WebUI/API,不移除背景 | Gradio `/predict` | 返回叠加图;文本为 3.0 / 0.0 / 1.0 / 1.0,耗时 0.611s | 通过 |
| WebUI/API,移除背景 | Gradio `/predict` | 返回背景处理后的叠加图;文本为 355.0 / 0.0 / 2.0 / 0.29,耗时 1.808s | 通过 |
| 官方 V1 Demo 对齐 | 同一示例图对比官方 Demo 与本地 NPU WebUI | 不移除背景:两边均为 3.0 / 0.0 / 1.0 / 1.0;移除背景:角度均为 355.0 / 0.0 / 2.0,confidence 官方 0.34、本地 0.29 | 角度对齐 |
| 训练验证,frozen backbone | DINOv2-Large NPU 前向,冻结主干,MLP head 训练态 backward + SGD step | out_shape `(2,902)`,loss `18.931299`,耗时 0.523s | 通过 |
| 训练验证,全模型 | DINOv2-Large 与 MLP head 全部训练态 backward + SGD step | out_shape `(2,902)`,loss `21.358852`,耗时 0.509s | 通过 |

期望输出:能够返回物体朝向角度、方向置信度和坐标轴叠加图。  
实测输出:单图两种背景开关均可返回图像和数值结果;核心角度与官方 V1 Demo 对齐。  
与 CPU/GPU 基准对比:使用官方 V1 在线 Demo 做推理行为基准;不移除背景时四项完全一致,移除背景时三项角度一致、confidence 有小幅差异。训练验证使用合成数据,验证 NPU 反传算子链路和 optimizer step。

## 4. NPU 亲和性

| 指标 | 数值 |
|---|---|
| 能否在 NPU 跑通 | 是,DINOv2-Large + MLP head 单卡 NPU 前向通过 |
| NPU 利用率 (npu-smi) | 100 step 模型前向采样峰值 93%,14 次轮询平均 6.64% |
| HBM 占用 | 100 step 模型前向采样峰值 7466MB / 114688MB,平均 6672MB |
| 关键算子是否回退 CPU | 模型主体无 CPU 回退;CPU 路径为 AutoImageProcessor、rembg、PIL/NumPy 渲染、Gradio 服务 |
| 性能(吞吐/时延) | profiler 100 step 均值 11.139ms/图,约 89.8 img/s;WebUI/API 0.611s(不移除背景) / 1.808s(移除背景) |

- 算子回退清单:AutoImageProcessor resize/crop/normalize、rembg/ONNXRuntime 背景移除、PIL/NumPy 坐标轴渲染、Gradio 服务在 CPU/host 侧;模型主体 profile 捕获 MatMulV3、BatchMatMulV3、Conv2DV2、LayerNormV4、SoftmaxV2、Gelu 等 NPU kernel。
- profiler 摘要:torch_npu profiler Level1 + PipeUtilization,active=100,warmup=2;kernel_details.csv 51800 行 / 49 列;op_statistic.csv、api_statistic.csv 生成;trace_view.json 合法;step_trace Computing 均值 10.964ms,Stage 均值 11.162ms。

口径:Ascend 950PR、PyTorch eager、float32 权重与激活、单图模型前向。profile 窗口只包含已预处理输入的模型前向,不包含 AutoImageProcessor、rembg、PIL/NumPy 渲染或 Gradio 服务。

训练路径:公开仓库没有训练入口。全模型训练验证覆盖 DINOv2-Large 主干和 MLP head 的 forward、loss、backward、SGD step;head 训练验证覆盖冻结主干微调场景。torch_npu 在反传中给出 internal format 回退到 base format 的 warning,但没有中断训练步骤。

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
| TTA 聚合 | `quantile`,mask indexing,`mean`,`cos`,`sin`,`sqrt`,`atan2` | Vector/host | 默认关闭,主验证路径不使用 |
| 训练反传 | Linear/MatMul/Conv/LayerNorm/GELU/Softmax/BatchNorm 的 backward,SGD step | Cube/Vector/MTE/head | 单步训练通过 |

**Profiler 热点算子明细**:

| 算子 | Core Type | 次数 | 总耗时(us) | 占比 | 平均耗时(us) |
|---|---|---:|---:|---:|---:|
| MatMulV3 | AI_CORE | 14600 | 851086.297 | 77.627% | 58.293 |
| BatchMatMulV3 | AI_CORE | 4800 | 94886.205 | 8.654% | 19.767 |
| Transpose | AI_VECTOR_CORE | 9900 | 53736.640 | 4.901% | 5.427 |
| LayerNormV4 | AI_VECTOR_CORE | 4900 | 17858.618 | 1.629% | 3.644 |
| Mul | AI_VECTOR_CORE | 4800 | 17637.341 | 1.609% | 3.674 |
| Add | AI_VECTOR_CORE | 4900 | 13945.293 | 1.272% | 2.845 |
| SoftmaxV2 | AI_VECTOR_CORE | 2400 | 13468.152 | 1.228% | 5.611 |
| Muls | AI_VECTOR_CORE | 2400 | 11373.815 | 1.037% | 4.739 |
| Gelu | AI_VECTOR_CORE | 2400 | 11287.902 | 1.030% | 4.703 |
| Conv2DV2 | MIX_AIC | 100 | 8414.001 | 0.767% | 84.140 |

按 op_statistic 总耗时聚合,Cube 类算子(MatMulV3、BatchMatMulV3、Conv2DV2)占 87.05%,Vector/MTE 类算子占 12.95%。单卡 communication 为 0。

**PipeUtilization 子项时间占比**:

| 子项 | 汇总耗时(us) | 占比 |
|---|---:|---:|
| aic_mac | 730833.392 | 43.27% |
| aic_mte2 | 372847.503 | 22.08% |
| aic_fixpipe | 164283.060 | 9.73% |
| aic_mte1 | 152289.474 | 9.02% |
| aic_scalar | 125959.070 | 7.46% |
| aiv_vec | 48527.879 | 2.87% |
| aiv_mte2 | 40228.277 | 2.38% |
| aiv_scalar | 35938.289 | 2.13% |
| aiv_mte3 | 17869.967 | 1.06% |

PipeUtilization 子项是按 profiler 字段汇总的流水线子时间,用于判断瓶颈方向,不是互斥 wall time。该结果与算子热点一致:模型主路径由 Cube GEMM/BatchMatMul 主导,MTE/FixPipe 和 Vector 是次级压力。

**各计算单元逐条判定**:

| 单元 | 压力 | 判定 | 证据 |
|---|---|---|---|
| 算力(Cube,矩阵卷积) | 高 | 亲和 | MatMulV3、BatchMatMulV3、Conv2DV2 合计 87.05%;aic_mac 子项占 43.27% |
| 向量(Vector,归一激活) | 中 | 亲和 | Transpose、LayerNormV4、Mul、Add、SoftmaxV2、Gelu 均在 profile 热点中 |
| 搬运(MTE/FixPipe) | 中 | 可跑通,存在优化空间 | aic_mte2 22.08%、aic_mte1 9.02%、aic_fixpipe 9.73%;transpose/contiguous 路径有搬运压力 |
| 通信(communication) | 无 | 不参与 | step_trace Communication 为 0,单卡无 HCCL、AllReduce 或模型并行 |
| 调度(host/head) | 中 | 端到端重要因素 | 采样窗口内 launch 51800 次,WebUI 端到端还包含 CPU 预处理、rembg、渲染和服务开销 |

**理论平衡点初判**:

| 项 | 估算值 |
|---|---:|
| patch embedding FLOPs | 约 0.31 GFLOPs |
| 单层 transformer FLOPs | 约 6.74 GFLOPs |
| 24 层 backbone FLOPs | 约 162.0 GFLOPs |
| 项目预测头 FLOPs | 约 0.0035 GFLOPs |
| 粗略参数量 | 约 304M |

单图 FP32 权重流量按至少读取一次估算约 1.2GB,算术密度约 133 FLOP/Byte。950PR FP16/BF16 平衡点约 270 FLOP/Byte,当前 float32 eager 路径不能直接写成纯 compute-bound,更准确的判断是 Cube 计算、权重/激活搬运和 framework head 共同影响。

算子回退清单:
- 已知非 NPU 路径:AutoImageProcessor resize/crop/normalize、rembg/ONNXRuntime 背景移除、PIL/NumPy 坐标轴渲染、Gradio 服务。
- 模型主体 profile 捕获到 MatMulV3、BatchMatMulV3、Conv2DV2、LayerNormV4、SoftmaxV2、Gelu 等 NPU kernel,并通过 NPU 前向验证、NPU 训练验证和 WebUI/API 验证。

## 5. 阻塞项

| 阻塞点 | 原因 | 是否硬阻塞 | CANN/AscendC 替代方案 | 兜底 |
|---|---|---|---|---|
| 仓内无 benchmark/eval 脚本 | 原项目主要提供单图 Demo 和 Python 调用,无标注评测集入口 | 阻塞论文级指标复现 | 非 CANN/AscendC 问题 | 官方 Demo 对齐 + 单图验证 |
| 仓内无正式训练脚本 | 缺少数据集读取、loss 汇总、训练参数、checkpoint 保存、分布式启动和评测闭环 | 阻塞完整训练复现 | 非 CANN/AscendC 问题 | NPU 单步反传验证模型代码训练链路 |

## 6. 结论
- 运行方案(NPU / NPU+CPU / CPU):NPU+CPU 混合部署。DINOv2-Large backbone 与 MLP 预测头在 NPU 上跑通;预处理、背景移除、输出解码后的图像渲染和 WebUI 仍在 CPU/host 侧。
- 待办 / 风险:仓内无 benchmark/eval 脚本和正式训练脚本;完成范围为单卡 NPU 功能复现、官方 Demo 角度对齐、模型代码单步训练链路和 profiler 亲和性分析,不包含全链路 NPU 原生化、完整训练复现或论文级指标复现。
