# SEED-OLF 公开数据基础验证报告

**项目：** MindScents

**分析日期：** 2026-08-10

**数据范围：** 公开 SEED-OLF，32 名参与者 × 3 次实验 × 24 个试次，共 2,304 个刺激试次及其 2,304 个恢复期配对记录

**总体判断：** **可用于内部研究决策，但必须附带限制条件（Share with caveats）**

## 1. 结论先行

本轮已经建立并执行了第一版可复现、分组无泄漏的公开数据基线。结果不支持“现有基础 EEG 特征已经在气味身份之外稳定解释了主观效价”这一主张。

最强的零样本 LOSO 基线不是 EEG，而是仅利用训练折估计的气味条件先验：log loss 为 **0.4411**，Brier score 为 **0.1385**，balanced accuracy 为 **0.8093**。加入刺激期频谱 EEG 后，AUROC 从 **0.8142** 上升至 **0.8579**，但主要概率指标反而变差：log loss 为 **0.4464**，Brier score 为 **0.1409**，balanced accuracy 为 **0.7840**。参与者分块 bootstrap 得到的 log-loss 改善为 **-0.0054，95% CI [-0.0165, 0.0079]**；区间跨越零，不能判定为可靠增益。

这不是“研究失败”。它揭示了真正值得做的公开数据研究问题：

> 在严格控制气味身份、参与者历史和跨日变化以后，哪些 EEG 表征仍能解释个体主观反应差异？

现阶段不应直接上更大的深度模型来追逐整体分类率。下一轮应先做通道级差分熵、气味控制的残差/分歧建模、跨日少样本个性化，再决定 EEGNet 或 TSception 是否值得训练。

## 2. 已完成的验证范围

### 数据与安全

- 使用受限 NumPy pickle 解析器，不允许任意 Python 全局对象反序列化。
- 验证了刺激与恢复记录的一一配对、标签一致性、形状和有限值。
- 生成一个试次一行的 canonical manifest；窗口或相位不会跨训练/测试折。
- 200 Hz 仅由“15 秒阶段、3,000 个采样点”推断，数据文件内没有采样率元数据，因此所有频带结果仍是工程性验证，不是最终生理解释。
- 官方范式将 clean breathing 放在刺激和自评之后；本报告称其为“恢复期”，不称其为“刺激前基线”。

### 无泄漏评估

- **LOSO：** 外层测试为一整名未见参与者；内层按参与者分组调参。
- **个体内跨日：** 留出一整次 session；其余 session 用于训练。
- 气味先验、标准化、特征筛选和模型参数均只在训练折拟合。
- 主要终点为 held-out log loss，并同时报告 Brier score、balanced accuracy、AUROC 和 PR-AUC。
- 模型增益使用参与者分块 bootstrap，而不是把 2,304 个相关试次当成独立样本。

### 已实现模型

1. 全局训练折先验；
2. 气味条件训练折先验；
3. Welch 频谱特征 + 正则化 logistic regression；
4. 气味先验 + 频谱 EEG；
5. 刺激/恢复配对频谱表示；
6. OAS 协方差 + Log-Euclidean Riemannian 表示；
7. 气味先验 + Riemannian EEG；
8. 刺激/恢复相对 Riemannian 表示。

实现只引用 SciPy、scikit-learn、pyRiemann，以及作者维护或成熟 EEG 库中的候选深度模型。详细来源见 [`docs/model-sources.md`](../docs/model-sources.md)。

## 3. 基础效果结果

### 3.1 刺激期与恢复期存在显著频谱差异

参与者级配对结果如下；绝对带功率为刺激减恢复的 dB 差值，显著性使用 Wilcoxon 检验并进行 BH-FDR 校正。

| 特征 | 频带 | 平均差 | 95% CI | FDR q |
|---|---:|---:|---:|---:|
| log band power | delta | +1.731 dB | [1.304, 2.189] | 2.0e-7 |
| log band power | theta | +0.889 dB | [0.609, 1.173] | 3.7e-6 |
| log band power | alpha | +0.536 dB | [0.165, 0.925] | 0.0164 |
| log band power | beta | +0.248 dB | [0.093, 0.396] | 0.0076 |
| log band power | gamma | +0.454 dB | [0.222, 0.700] | 0.0016 |

相对功率的主要变化为 delta 增加 **0.0524**，theta、beta、gamma 分别减少 **0.0096、0.0225、0.0076**；alpha 在 FDR 校正后未达到 0.05。

这些差异只能解释为“刺激阶段与刺激后恢复阶段不同”。由于两阶段还同时改变了呼吸、任务阶段、时间位置，并可能存在气味残留，不能写成“气味导致放松”或纯粹的嗅觉神经效应。

### 3.2 跨 session 稳定性整体偏弱且具有气味依赖

40 个“气味 × 特征族 × 频带”组合中，仅 **8/40** 的 ICC(2,1) 达到 0.50。较稳定的组合集中在：

- 气味 1：绝对 delta/theta、相对 delta/theta/beta；
- 气味 3：绝对 delta/theta、相对 beta。

最高值是气味 1 的相对 beta，ICC = **0.833**；气味 3 的相对 beta 为 **0.742**。其余大部分组合稳定性不足，说明单次 session 的 EEG 反应不宜被视为稳定的个体特征。

### 3.3 主观效价预测：气味先验仍是最强主基线

#### 未见参与者 LOSO，全部 2,304 个试次

| 模型 | Log loss ↓ | Brier ↓ | Balanced accuracy ↑ | AUROC ↑ |
|---|---:|---:|---:|---:|
| 气味先验 | **0.4411** | **0.1385** | **0.8093** | 0.8142 |
| 气味 + 刺激期频谱 EEG | 0.4464 | 0.1409 | 0.7840 | **0.8579** |
| 气味 + 配对频谱 EEG | 0.4550 | 0.1439 | 0.7861 | 0.8533 |
| 气味 + 刺激期 Riemannian EEG | 0.5670 | 0.1833 | 0.7365 | 0.7900 |
| 仅刺激期频谱 EEG | 0.6818 | 0.2443 | 0.5268 | 0.5395 |
| 仅刺激期 Riemannian EEG | 0.7394 | 0.2681 | 0.5206 | 0.5182 |

频谱 EEG 可能含有一定排序信息，因为 AUROC 提升；但它没有通过预先设定的增量价值门槛：log loss 的 bootstrap 区间跨零，Brier 和 balanced accuracy 也没有改善。Riemannian 表示在当前 15 秒全段、全通道协方差设计下明显不合适，不应继续作为主路线，除非先引入滤波器组、空间正则或更合理的相对协方差设计。

#### 个体内留一 session

气味先验的 log loss 为 **0.4135**；气味 + 刺激期频谱 EEG 为 **0.6109**。跨日个性化下 EEG 退化更明显，提示 session shift、短样本训练和信号质量是关键限制。未来采集不能只依赖单日校准。

## 4. 质量核验结果

自动校验全部通过：

- manifest：2,304 个完整试次；
- 预测：36,864 行，覆盖 8 个模型 × 2 个协议；
- 重新计算指标与汇总表的最大绝对差：**1.11e-16**；
- notebook：15 个单元，其中 6 个代码单元，错误输出为 0；
- 单元测试：6/6 通过；
- 原始 PKL 未修改，生成物独立存放。

核验不等于生理效度已经确认。采样率、通道顺序、参考方式、前置滤波和精确气味元数据仍需从权威资料补齐。

## 5. 对公开数据项目的可实施安排

### 项目 A：Beyond Odor Priors

**当前状态：** 第一阶段得到有信息量的阴性结果。

**下一实验：** 通道级差分熵和频谱特征；训练折内剔除气味可解释部分；预测 train-derived 气味先验的残差或分歧试次。

**成功判据：** EEG 增强模型的参与者分块 log-loss 改善 95% CI 排除零，且 Brier 不恶化，并至少在一个跨 session 协议中方向一致。

### 项目 B：How Much Calibration Is Enough?

**当前状态：** 零样本与个体内跨日基线已完成。

**下一实验：** 仅用目标参与者 session 1 的 4、8、16、24 个分层校准试次，在 session 2–3 上测试；每个校准量重复确定性抽样。

**输出：** 校准量—log-loss 曲线、参与者异质性和达到稳定增益所需的最小试次数。

**用途：** 直接决定 MindScents 首次使用时的校准时长，而不是凭经验选次数。

### 项目 C：Olfactory EEG Response Fingerprints

**当前状态：** 频带级 ICC 基线已完成，稳定性集中在少数组合。

**下一实验：** 加入通道/脑区表征、跨 session representational similarity 和检索；在获得官方 montage 后再做拓扑解释。

**用途：** 决定 64 通道中哪些区域需要保留，以及多 session 是否为必需。

## 6. 对未来 MindScents 采集的直接建议

1. 从第一批数据就保留 **EEG + PPG/原始脉搏波 + EDA + 自评**，但先把每个模态的增量价值分开验证，再做融合。
2. 同一试次必须同时包含真正的刺激前基线、刺激期、自评和刺激后恢复期。SEED-OLF 的恢复段不能替代刺激前基线。
3. Dolcos 64 通道系统、PPG 和 EDA 使用同一硬件触发或可审计的同步事件流；保存原始时间戳、漂移、丢包和触发延迟，不只保存对齐后的特征。
4. 至少跨多个 session 采集。公开数据的个体内跨日结果表明，单日效果不能代表可复现的个人反应。
5. 主要自评应是连续或有序的“放松程度”，同时收集 valence、arousal、pleasantness、intensity、familiarity 和 preference，避免让二元效价代替放松。
6. 先验注册增量比较顺序：气味/历史 → EEG → PPG/HRV → EDA → 多模态融合。每一步都必须证明超越前一步，而非只报告融合模型准确率。
7. 在少样本曲线完成前，不给出精确参与者数和重复次数。最终样本量应由参与者层级效应、跨日方差、预期最小增量和模拟功效共同决定。

## 7. 模型路线决定

当前最专业、最合适的路线不是立即训练最大的模型，而是按证据门槛推进：

1. **继续：** 正则化线性模型 + 通道级 differential entropy / band power，针对气味控制后的残差目标；
2. **继续：** few-shot 个体化和 session normalization；
3. **有条件进入：** EEGNet，使用作者实现或经论文核对的 Braindecode 实现，至少 5 个固定种子并使用内层 early stopping；
4. **有条件进入：** TSception，前提是官方 montage 可用且通道空间关系可被正确编码；
5. **暂缓：** DSEN，直到获得作者代码或足够完整的架构细节；
6. **暂缓：** EEG foundation model。当前误差的主因更像研究目标、气味捷径和跨日漂移，不是模型容量不足。

## 8. 复现入口

```bash
python -m pip install '.[test,notebook]'
pytest
export SEED_OLF_DATA_ROOT=/absolute/path/to/SEED-OLF
seed-olf-baselines --output-dir artifacts/baseline_v1
seed-olf-riemannian --output-dir artifacts/baseline_v1
python scripts/validate_outputs.py --output-dir artifacts/baseline_v1
python scripts/build_notebook.py --output-dir artifacts/baseline_v1
```

公开汇总位于 `results/baseline_v1/`；完整本地运行产物位于被 Git 忽略的 `artifacts/baseline_v1/`。执行版 notebook 为 `notebooks/baseline_effects.ipynb`。

## 9. 权威参考边界

- SEED-OLF：原始论文 DOI `10.1109/TAFFC.2026.3662364` 与上海交通大学官方范式介绍；
- 频谱估计：SciPy `signal.welch`；
- 线性概率模型与分组验证：scikit-learn；
- Riemannian 方法：pyRiemann 官方文档及原始方法论文；
- EEGNet：Lawhern 等原论文及作者 ARL 仓库；
- TSception：Ding 等原论文及作者仓库；
- 模型选择原则：参考 MOABB 大规模可复现基准，优先建立强经典基线，再评估深度模型。

任何无法追溯到论文作者、官方文档或成熟维护库的同名实现，都不会进入正式比较。
