# EBSD2026 Kikuchi Pattern Calibration

本仓库用于探索 EBSD 菊池花样读取、菊池带增强匹配、菊池花样到菊池球的空间校正，以及 pattern center / 投影半径偏差校正。

## 目前实现的方法

### 1. 数据读取

- 从 EDAX H5 文件中读取 EBSD map、pattern center 和 OHP 菊池带信息。
- 从 UP2 文件中读取对应的原始菊池花样。
- 当前 pattern center 使用 H5 中真实读取的 `X-Star / Y-Star / Z-Star`，不是虚拟值。
- OHP band 数据按 8 条 band 读取，每条包含 `rho_bin / theta_deg / width / intensity`。

主要代码：

- `h5_band_enhanced_match.py`
- `export_h5_band_examples.py`

### 2. 菊池带可视化与坐标约定检查

- 支持将 H5 中的 OHP 菊池带投影回原始菊池花样。
- 已修正 Hough rho 到像素坐标的换算问题。
- 当前较可靠的 H5 band 线条约定为 `normal_theta_rho+_yup`。
- 可输出原始 pattern 与对应 H5 菊池带叠加图，用于检查 band 是否与真实菊池线一致。

主要代码：

- `export_h5_band_examples.py`
- `h5_band_enhanced_match.py`

### 3. 菊池花样预处理

- 对原始菊池花样做圆形有效区域 mask，忽略外部背景。
- 支持背景扣除、局部衬度归一化和线状响应增强。
- 预处理结果保留 raw pattern 与 corrected / enhanced pattern 两套可视化，不对原图做不可逆替换。

主要代码：

- `h5_band_enhanced_match.py`
- `visualize_calibration_pipeline.py`

### 4. 菊池球匹配

- 从 kikuchipy / EMsoft master pattern H5 中读取高分辨率 nickel master sphere。
- 将 detector plane 上的 pattern 像素通过 pattern center 投影到球面。
- 当前默认推荐匹配目标是 weighted 混合增强：
  - image line response: `0.45`
  - corrected intensity: `0.15`
  - H5 OHP band response: `0.40`
- 匹配时默认自动搜索 detector convention，避免手工固定错误坐标约定。
- 纯 H5 band-only 匹配已经尝试过，但整体效果比 weighted 混合增强差，目前不作为默认方案。
- 新增连续几何精配准：
  - 以当前 H5-band-enhanced 匹配作为初始值。
  - 使用 `scipy.optimize.least_squares` 连续优化小角度旋转增量和 `pcz/radius_scale`。
  - loss 同时包含 master/pattern 的 band/intensity 残差，以及逐条 H5 OHP band 与同 HKL family 标准菊池带的角度残差。
  - 默认开启 match score guard：最终解的整体匹配分数不能比初始值下降超过 `0.02`，避免为了压低 band 几何误差而破坏整张图匹配。
  - 每个 pattern 会保存超参数、优化 trace、初始/最终逐条 band 角误差、中间曲线和最终可视化。
- 新增 DIP 参数化可微配准原型：
  - 固定随机噪声输入一个小型 PyTorch MLP，由网络输出小角度旋转增量、`radius_scale`，以及可选 `pcx/pcy` 偏移。
  - 不让网络直接生成图像，而是让网络生成配准参数；每一步用可微球面采样实时计算 image line、H5 band、intensity 和同 HKL family band geometry loss，再通过 AdamW 梯度下降。
  - 默认 PC 不参与优化，只做旋转和半径微调；需要测试 PC 时可加 `--optimize-pc`。
  - 保存 `hyperparameters.json`、DIP loss trace、候选 checkpoint 评估、逐条 band residual 对比、HKL label 对齐图和最终空间匹配图。

主要代码：

- `h5_band_enhanced_match.py`
- `batch_final_spatial_visualizations.py`
- `visualize_calibration_pipeline.py`
- `continuous_band_geometric_refinement.py`
- `dip_parametric_registration.py`

### 5. Pattern center 与投影半径偏差校正

- 在 H5 pattern center 基础上搜索 `pcx / pcy` 像素偏移。
- 同时搜索 `pcz` 的 radius scale，用于模拟投影半径或 detector distance 的标定偏差。
- 支持单张 pattern 校正和 10 组 batch 校正。
- CPU 版与 GPU 版都保留；GPU 版使用 PyTorch CUDA 加速局部 PC/radius 搜索。
- 新增基于 HKL label 一致性的半径精配准原型：
  - EDAX OHP 数据中每条 band 只有 `rho/theta/width/intensity`，当前文件没有逐条 band 的直接 HKL label。
  - 从 H5 phase header 读取 HKL families，例如 FCC phase 的 `(111)/(200)/(220)/(311)`。
  - 将检测到的 H5 OHP band 投影到 master sphere 后，根据最接近的 HKL family 推断 label。
  - 半径搜索的 loss 同时包含原来的 pattern/master 匹配分数，以及同一 HKL family 的 band 法向一致性。
  - 对 10 组 Area1 high pattern 的全局半径汇总显示，平均 composite score 最高在 `radius_scale=1.00`，说明当前 H5 `pcz` 不需要明显全局缩放；单张 pattern 的 radius 最优值变化更多是在补偿局部姿态或匹配误差。
- 当前推荐使用 GPU weighted 恢复版本：

```powershell
D:\anaconda3\envs\torch\python.exe .\batch_pc_radius_bias_correction_gpu.py `
  --map area1_high `
  --count 10 `
  --strategy linspace `
  --match-mode weighted `
  --force-convention auto `
  --pc-shifts-px=-18,-12,-6,0,6,12,18 `
  --radius-scales=0.86,0.90,0.94,0.98,1.00,1.02 `
  --local-steps-deg=1.5,0.5 `
  --sphere-lon-count 420 `
  --sphere-colat-count 210
```

主要代码：

- `pc_radius_bias_correction.py`
- `batch_pc_radius_bias_correction.py`
- `batch_pc_radius_bias_correction_gpu.py`
- `labeled_band_radius_refinement.py`

### 6. 可视化输出

- 输出 raw pattern / preprocessed pattern 在高分辨率 Kikuchi sphere 上的最终空间匹配结果。
- 输出 PC/radius 搜索的 score landscape。
- 输出 batch contact sheet，方便横向检查多组 pattern。
- 输出 H5 OHP band 的 inferred HKL label overlay，以及 transformed H5 band 与同 HKL 标准球大圆的对齐图。
- 最终空间匹配图中，pattern patch 只有很小的显示用 surface lift，避免视觉上误判为物理半径不同；真实投影张角由 `pcz/radius_scale` 决定。
- 可视化输出默认写入 `outputs/`，不上传 GitHub。

## 版本改动

### 2026-05-23

- 新增 `continuous_band_geometric_refinement.py`，实现逐条 Kikuchi band 几何残差的连续优化匹配。
- 连续优化变量包括旋转增量、`pcz/radius_scale`，可选小范围 `pcx/pcy`；默认只优化旋转和半径，PC 作为小范围可选项。
- 新脚本记录主要超参数：初始随机匹配参数、预处理参数、least-squares 迭代参数、旋转/半径/PC bounds、loss 权重、match score guard。
- 对 10 组 Area1 high pattern 测试时，9/10 组逐条 band 平均角误差下降；平均 band angle gain 约 `0.19 deg`，平均 rotation delta 约 `0.55 deg`，平均 `radius_scale=0.99975`。
- 新增 `dip_parametric_registration.py`，尝试 DIP / deep image prior 风格的参数化神经配准：小型 MLP 输出旋转、半径和可选 PC 偏移，PyTorch 实时计算可微 loss 并反传。
- DIP 默认超参数包括 `steps=350`、`lr=2e-3`、`rotation_bound_deg=6`、`radius_min=0.98`、`radius_max=1.02`、`residual_points=1600`，loss 权重为 image line `1.0`、H5 band `0.8`、intensity `0.15`、band geometry `0.6`。
- DIP 也使用 match score guard，默认最终 full match score 不能比初始值下降超过 `0.02`。
- 使用 GPU 跑 10 组 Area1 high pattern 测试时，9/10 组逐条 band 平均角误差下降；平均 band angle gain 约 `0.206 deg`，平均 match score change 约 `-0.0058`，无样本超过 `0.02` 的 score drop，平均 rotation delta 约 `0.717 deg`，平均 `radius_scale=1.00003`。
- 新增 `labeled_band_radius_refinement.py`，尝试用 HKL family label 一致性增强半径精配准。
- 将最终空间匹配可视化中的 pattern surface lift 从 `1.018` 调小到 `1.006`；这是显示层参数，不参与 loss，避免误判菊池球半径与 pattern 投影半径不一致。
- 使用 10 组 Area1 high pattern 测试 labeled radius refinement：per-pattern 局部搜索有小幅 score 提升，但全局半径汇总最优仍为 `radius_scale=1.00`。
- 移除全局 DIC 式 PC/radius refinement 原型；该模块对当前 Kikuchi sphere 匹配帮助不稳定，暂不作为本仓库方法保留。
- 新增 GPU 版 PC/radius batch 校正脚本 `batch_pc_radius_bias_correction_gpu.py`。
- GPU 版使用 PyTorch CUDA 加速局部候选参数搜索，环境为 `D:\anaconda3\envs\torch\python.exe`。
- 新增 `--match-mode` 参数：
  - `weighted`：默认推荐，混合 image line、intensity、H5 band。
  - `band_dominant`：提高 H5 band 权重的实验模式。
  - `band_only`：只使用 H5 band 几何响应的实验模式。
- 新增 `--force-convention` 参数，用于强制 detector convention 或保持 `auto`。
- 尝试过 `band_only + flip_y`，但整体匹配效果比 weighted 版本差，因此已将默认模式回退为 `weighted + auto`。
- 新增 README 维护规范：后续代码改动必须同步更新“目前实现的方法”和“版本改动”。
- 新增 `.gitignore`，后续只上传代码和文档，不上传数据、输出图、缓存、模型权重。
- 将已被 Git 跟踪的输出图、缓存、日志和示例图片从 Git 跟踪中移除；本地文件保留，GitHub 只保留代码和文档。

### 2026-05-22

- 新增 H5 OHP 菊池带读取与叠加可视化。
- 修正 OHP band 的 rho 像素换算问题，使 H5 菊池带和原始 pattern 对齐。
- 新增高分辨率 Kikuchi sphere 最终空间匹配可视化。
- 新增 PC/radius 单张偏差校正脚本。
- 新增 batch final spatial visualization 脚本。

## GitHub 上传规则

- 只提交代码和文档。
- 不提交 EBSD 原始数据、UP2/H5 数据、训练权重、TensorBoard 日志、可视化输出图、缓存文件。
- 每次修改算法或参数默认值时，同步更新本 README。
