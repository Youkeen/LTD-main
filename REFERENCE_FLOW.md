# 真实参考特征校正检测器

本分支实现讨论中的研究方案，尚未验证检测性能或创新性。默认 `train.py`
使用 `--method reference_flow`；`--method ltd` 可运行原始 LTD 基线。
旧 LTD 分类头不能加载到新模型；可通过 `--clip_checkpoint` 仅复用其冻结 CLIP 权重。

## 模型流程

训练图像 → 冻结 CLIP → 固定第 20 个 block 的 CLS 特征（零起始索引 19）。
不使用层选择、层间差分或原 LTD 的双分支 Transformer。

1. 训练开始前，用确定性的 CLIP 预处理提取训练集特征，默认每类最多 4096 张。
   库保存固定层特征、最终 CLIP 语义向量、标签、路径、文件哈希；不缓存可训练网络的隐藏特征。
2. 用最终 CLIP 语义向量检索真实库，取前 4 个邻居的相似度加权特征作为参考。
   排除同一路径、已知的同文件哈希和余弦相似度 ≥ 0.9999 的候选。
   哈希过滤仅在查询本身也被选入库时可用；语义阈值不能保证识别所有视觉近重复。
3. 用输入与真实参考之间的线性插值构造训练状态。时间 1 是输入，时间 0 是参考，
   监督速度为“输入减参考”。网络接收当前状态、原始输入特征和时间，**不接收参考**。
4. 同时从最相近的 16 个假样本中随机选一个干扰样本。真实、假输入使用同一选择规则。
   默认不要求不同生成器，支持单生成器训练；当前未实现按生成器标签排除。
5. 用当前网络、相同时间实时计算干扰样本的隐藏特征并停止梯度，随机混入校正网络隐藏层。
   默认每个样本 50% 概率启用，逐通道混合比例在 0～0.2 均匀采样。
   混合不会改变路径的真实参考、速度监督、原始输入条件或真假标签。
6. 正常分支和干扰分支都从输入特征实际运行两步反向时间 Euler 校正。
   分类头接收两步预测速度及总校正量，输出 fake 的 logit。

损失包括正常分支 BCE、速度 MSE，以及干扰分支 BCE、速度 MSE、弱预测一致性。
分类训练使用实际预测的 rollout，而不是直接把监督参考或插值终点喂给分类器。
混合 rollout 中干扰样本也沿自身无混合预测路径推进。

推理只有：**图像 → 冻结 CLIP → 两步校正 → 真假概率**。
不需要真实库、假库、标签或混合。由于参考不作为推理输入，模型学习的是训练配对关系的
近似校正，不能保证每张测试图最终到达某个具体真实锚点。
也不预设“假图一定需要更大校正”；真假关系由分类头学习。

## 训练

工作目录是包含 `train.py` 的 `LTD-main` 子目录，安装原 `requirements.txt` 中的依赖。
示例使用 `sd1_4` 数据布局：

```text
DATA_ROOT/
  train/0_real/...
  train/1_fake/...
  val/0_real/...
  val/1_fake/...
```

PowerShell 单行示例（替换数据路径）：

```powershell
python train.py --method reference_flow --arch CLIP:ViT-L/14 --data_mode sd1_4 --wang2020_data_path D:/YOUR_DATA --name reference_flow_v1 --checkpoints_dir ./checkpoints --batch_size 16 --num_threads 0 --niter 5 --lr 0.00005 --fix_backbone --clip_checkpoint ./checkpoints/DRCT_sdv1.4.pth
```

最后一个参数可省略，此时按原项目逻辑加载／下载官方 CLIP。
`--clip_checkpoint` 需要原始 LTD 格式 checkpoint，且其 CLIP 必须与 `--arch` 一致。
ProGAN 布局使用 `--data_mode wang2020`，目录仍为 `train/progan` 和 `test/progan`。
`ours` 模式继续使用分别存放 train.pickle / val.pickle 的真实和假图路径清单。

首次运行自动保存 `checkpoints/reference_flow_v1/training_bank.pt`。
复用库时传 `--bank_path`；会检查 backbone 名称、层索引、训练路径及标签。
**数据内容、CLIP 权重或特征预处理变化后必须重新建库**，这些变化不会被全部自动识别。
请使用独立训练／验证目录；不要把验证或测试图放入训练清单。
两类各至少需要两个可用且有差异的样本；没有合法邻居时明确报错，不回退到自身。

保存的 `best` / `model_epoch_N.pth` 包含完整 CLIP、校正网络、分类头与模型配置，
以及优化器状态。当前训练入口不提供断点恢复；checkpoint 可独立用于推理。

## 验证

无需传入特征库，模型配置从 checkpoint 恢复，也无需重新下载 CLIP。
测试目录下应分别包含 `0_real` 与 `1_fake`：

```powershell
python validate.py --ckpt ./checkpoints/reference_flow_v1/best --real_path D:/YOUR_TEST --fake_path D:/YOUR_TEST --data_mode wang2020 --batch_size 16 --device cuda --result_folder ./results/reference_flow
```

未提供测试路径时保留原 `dataset_paths.py` 的 DRCT 基准配置；需先修改其中的数据路径。
主要报告固定阈值 0.5 的 accuracy 和 AP。原验证器额外计算的最优测试集阈值是事后指标，
不应作为可部署或公平比较的阈值结果。
默认图像尺寸为 224；336px CLIP 模型需同步调整训练和评估预处理，本版本示例以 ViT-L/14 为准。

## 消融与检查

- `--mix_weight 0`：去掉干扰分支，检验其贡献。
- `--flow_steps 1` / `2` / `4`：各自重新训练，比较单步与多步。
- `--flow_weight 0`：去掉速度监督，仅保留分类及启用的混合约束。
- `--anchor_topk 1`：使用单个最近真实样本。
- `--method ltd`：原始 LTD 基线。

```powershell
python -m unittest discover -s tests -v
```

测试使用小型冻结编码器和合成样本，覆盖流方向、隐藏混合、梯度隔离、检索排除、
正常／干扰损失、训练更新、库复用和 checkpoint 一致性。不代表真实数据上的泛化结果。
之后仍需对比同参数单步回归、静态锚点差异和普通 Mixup；这些基线未在此版本实现。
