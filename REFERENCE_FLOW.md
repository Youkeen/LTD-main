# 真实参考监督的两阶段特征 Flow

本版本实现已确定的研究方案：先学习只依赖当前状态和时间的速度场，再冻结速度场，训练真假分类头。
推理不需要参考库。本实现尚未验证真实数据上的检测性能或跨生成器泛化能力。

## 模型与默认配置

- 冻结 CLIP ViT-L/14，取零起始索引 11、12、13、14、15 五个 block 的 CLS。
- 各层分别 LayerNorm 后按深度拼接，得到 5120 维状态。五层共用一张参考图像和一个速度网络。
- 速度网络输入只有当前状态 z_t 和标量时间 t，输出 5120 维速度；不输入原始端点 x 或真实参考 r。
- 真实库默认包含全部真实训练图像。每个输入从语义 Top-4 中均匀采样一个真实样本作为参考。
  参考是单张图像的五层联合特征，不使用候选特征的加权平均。
- 从 t=1 开始做四步反向 Euler，收集四个完整速度向量。线性分类头输入为 20480 维。
  不再重复拼接等于速度和的总校正量，不对每个样本的速度序列做 LayerNorm，以保留幅值信息。
- 默认第一阶段 5 个 epoch，第二阶段最多 5 个 epoch。第一版关闭隐藏混合、虚假库干扰与一致性损失。

`--feature_layers` 可覆盖层列表；`--feature_layer 19` 显式使用旧单层表示。
`--flow_steps 1` 可做单步对照。静态 CLIP 对照必须使用相同层、权重和预处理。

## 数据与准备

在包含 train.py 的 LTD-main 子目录运行。示例为 PowerShell 单行命令，也可在 Linux 中替换路径使用。
先在训练环境安装 requirements.txt 中的依赖。数据示例：

```text
DATA_ROOT/
  train/0_real/...
  train/1_fake/...
  val/0_real/...
  val/1_fake/...
```

验证集只用于源域分类模型选择；测试集使用未见生成器，不能混入训练集或参考库。
训练入口检查训练与验证路径不重叠，但仍需自行保证没有跨划分的复制图像或近重复内容。

```powershell
python prepare_feature_banks.py --arch CLIP:ViT-L/14 --feature_layers 11 12 13 14 15 --data_mode sd1_4 --wang2020_data_path D:/YOUR_DATA --real_bank_output ./feature_banks/real_layers11_15.pt --bank_size 0 --batch_size 16 --num_threads 0
```

`--bank_size 0` 表示全部真实训练图像；如需限制内存，可改为至少 2 的固定数量。
建库使用确定性的 CLIP 预处理。只提供 real_bank_output，不需要虚假库。
库中保存中间特征、归一化最终语义向量、路径、文件哈希和表示配置。
检索排除自身、字节相同的文件以及语义相似度达到 duplicate_threshold 的候选；
语义阈值不能保证排除所有视觉近重复。没有合法参考时明确报错。

默认使用官方 CLIP。如果复用原 LTD checkpoint 的冻结 CLIP，建库与训练命令都添加：

```text
--clip_checkpoint ./checkpoints/DRCT_sdv1.4.pth
```

这个文件必须是原 LTD 格式，包含 model.* CLIP 权重。不要仅在其中一个命令中指定。
模型权重、层配置、训练文件内容或特征预处理改变后应重新建库；元数据不能识别其中所有变化。
与当前 CLIP 权重及表示完全一致的旧真实库可以复用，但新旧速度网络权重不能互换训练。

## 一次命令完成两阶段训练

```powershell
python train.py --method reference_flow --arch CLIP:ViT-L/14 --feature_layers 11 12 13 14 15 --data_mode sd1_4 --wang2020_data_path D:/YOUR_DATA --real_bank_path ./feature_banks/real_layers11_15.pt --name reference_flow_two_stage --checkpoints_dir ./checkpoints --batch_size 16 --num_threads 0 --flow_epochs 5 --niter 5 --flow_steps 4 --lr 0.00005
```

batch size 16 只是示例，应按显存调整。训练输入保留原数据管线中的在线随机裁剪、翻转及所配置的图像增强。
不额外缓存训练图像特征、增强视图、语义邻居或速度响应。

### 第一阶段：只训练速度场

每批训练图像同时包含真实与生成样本，二者遵循同一参考选择规则。
对原始特征 x 和采样的真实参考 r，构造：

```text
z_t = (1 - t) * r + t * x
u   = x - r
L_FM = mean((v(z_t, t) - u)^2)
```

只更新速度网络；CLIP 和分类器不更新。每批只做一次速度预测，不执行四步分类 rollout。
默认 10% 的时间样本显式取 t=1，其他时间在 [0,1) 随机采样；通过 --start_time_prob 调整。
这让实际推理起点直接接受速度监督。flow_weight 仅缩放 FM 损失，必须为正。

第一阶段按固定 epoch 数结束，不根据未训练分类器的准确率选择速度场。
保存 flow_final.pth 后，释放参考库，冻结速度网络，并重新建立只含分类头参数的优化器。

### 第二阶段：只训练分类器

从输入 x 自身出发，运行实际预测轨迹：

```text
state = x
for t in [1.0, 0.75, 0.5, 0.25]:
    velocity = v(state, t)
    state = state - 0.25 * velocity
```

拼接四步预测速度，使用 BCE 训练线性分类头。
速度特征在 no_grad 下计算，分类损失不会回传到速度场。
不检索参考，不使用监督插值状态替代预测状态。
每个分类 epoch 后在源域验证集评估，以准确率选择 best，固定阈值为 0.5。

输出位于 checkpoints/reference_flow_two_stage/：

- flow_final.pth：第一阶段完成后的速度场和冻结 CLIP；分类器尚未训练，不能用于真假评估。
- model_epoch_N.pth：第二阶段各保存轮次的完整模型。
- best：源域验证准确率最优的第二阶段模型，用于最终测试。
- opt.txt、train/、val/：参数和按阶段分开的 TensorBoard 记录。

checkpoint 包含完整 CLIP、速度网络、分类头、结构版本、训练阶段及优化器状态。
当前 train.py 不提供中断续训入口；重新启动会重新进行两阶段训练。

## 推理与兼容性

```powershell
python validate.py --ckpt ./checkpoints/reference_flow_two_stage/best --real_path D:/YOUR_TEST --fake_path D:/YOUR_TEST --data_mode wang2020 --resize_short_side 256 --batch_size 16 --device cuda --result_folder ./results/reference_flow_two_stage
```

测试目录包含 0_real 和 1_fake。也可以使用现有的 --dataset genimage --dataset_root 路由。
推理仅为图像 → 冻结 CLIP → 四步预测校正 → 线性分类头，不读取特征库，也不下载 CLIP。
validate.py 会拒绝误用第一阶段 flow_final.pth，以免评估未训练的分类器。

新 checkpoint 的 architecture_version=2。未包含该字段的旧联合训练 checkpoint 走
models/reference_flow_legacy.py 中的原始结构加载路径，保留旧结果的推理兼容性。
原始 --method ltd 与 --method clip_linear 入口仍可使用。
新两阶段训练不接受非零 mix_prob/mix_weight/consistency_weight 或 fake_bank_path，
不再通过 flow_weight=0 做无 FM 联合分类消融。

主要报告固定阈值 0.5 的 ACC 和 AP。验证器另算的测试集最优阈值属于事后指标，不能据此选模型或调参。
新旧 CLIP 预处理应保持匹配；示例仅针对 224px ViT-L/14。

## 对照和实现检查

```powershell
python -B -m unittest test_multilayer_flow test_linear_ablation -v
```

测试使用实际的小型 CLIP 架构与合成数据，无需下载权重，覆盖五层表示、单真实参考、
起点监督、积分方向、两阶段参数更新、训练循环与阶段 checkpoint，以及旧结构恢复。
不代表真实数据上的检测或泛化结果。

已有对照脚本现比较 A_flow（多步两阶段）、B_linear（相同静态特征的线性头）、
C_one_step（单步两阶段），读取新训练生成的 opt.txt，沿用数据、参考库、CLIP 和训练设置：

```powershell
python run_flow_ablation.py --reference-opt ./checkpoints/reference_flow_two_stage/opt.txt --genimage-root D:/GenImage --prefix two_stage_v1 --dry-run
```

检查打印命令后，去掉 --dry-run 执行；已有输出目录会被拒绝覆盖。
该脚本不再执行旧版 C_no_flow_loss。静态参考差向量、同规模 MLP、联合训练等更完整研究对照尚未在此脚本实现。

路径是人为构造的特征代理路径；中间状态不保证对应有效图像，预测终点也不保证落在某个具体真实参考上。
速度场由真实与生成训练样本共同学习，不能描述成只用真实图像学习的场。
需要另行验证校正后的真实参考接近程度、特征是否塌缩，以及多步响应相对单步和静态差异是否有跨生成器收益。
