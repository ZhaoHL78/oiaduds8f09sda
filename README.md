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
- 新增霍夫点匹配与三维空间还原：
  - H5 OHP band 本身已经是霍夫检测结果，脚本将每条 band 规范化为 detector Hough space 中的点 `(theta, rho)`。
  - 对候选旋转、`radius_scale` 和可选 PC 偏移，将同 HKL family 的标准 Kikuchi 球晶面法向反投影回 detector，得到预测 Hough 点。
  - loss 使用 observed/predicted Hough 点的 `theta/rho` 残差，加上同 HKL family band 法向角残差；先做多起点粗搜索，再做连续 least-squares。
  - 默认保留 match score guard，防止纯霍夫点几何跳到“点更近但整张图明显错”的匹配。
  - 使用最终参数沿每条 detector band 采样点，并投影回 master sphere，输出三维点坐标 CSV 与 3D 可视化。
- 新增球面霍夫点的空间膨胀/收缩迭代匹配：
  - 将 detector Hough 直线 `x cos(theta) + y sin(theta) = rho` 反投影为穿过电子束源点的三维平面。
  - 该平面与单位 Kikuchi 球相交为大圆；对应的球面霍夫点为该大圆的单位法向：
    `n_h = normalize([a, b, c / D_eff])`。
  - 空间膨胀系数定义为 `s`，有效 detector distance 为 `D_eff = D / s`，等价于 `PCz_eff = PCz / s`；`s > 1` 表示角张量膨胀，`s < 1` 表示收缩。
  - 已验证解析球面霍夫点与 endpoint cross-product 几何法得到的法向一致，验证图中角差接近 `0 deg`。
  - 迭代流程为 `rotation match -> expansion/contract -> lr decay -> repeat`，每轮先做球面点旋转匹配，再调膨胀系数，逐步减小搜索半径直到收敛。
  - loss 以同 HKL family 的球面霍夫点角距离为主，同时保留 full pattern match score guard，避免为了霍夫点更近而牺牲整张图的空间匹配。
  - 输出公式验证图、球面霍夫点匹配图、膨胀迭代 trace、逐条 band 三维还原点和最终空间匹配图。
- 新增独立的球面 Radon 峰图匹配 pipeline：
  - 该方向暂时不沿用前一版半径膨胀/霍夫点微调逻辑，而是把实验 pattern 和 master sphere 都转换为球面函数后，在球面 plane-normal Hough/Radon 空间中匹配峰图。
  - 实验 pattern 先按 H5 PC 反投影到单位球面，master sphere 使用 EMsoft/kikuchipy master pattern 的 band response。
  - 对实验球面和 master sphere 分别做多尺度大圆积分，得到 spherical Hough/Radon response。
  - 在 plane-normal Hough space 中用非极大值抑制提取峰点；每个峰保存 `normal direction / peak strength / bandwidth / band profile / asymmetry` 描述符。
  - 构建实验峰图和标准峰图，使用三角不变量生成候选 orientation。
  - 对候选 orientation 使用 partial optimal transport 做全局峰匹配；当前实现是带质量约束的线性规划 partial OT。
  - 在最佳候选基础上联合优化 orientation、`pcx/pcy` 偏移和 `pcz/radius_scale`，然后回到原始球面 pattern 做局部 refinement。
  - 最终输出 orientation、phase、校正 PC、partial OT 匹配峰和对应 `{hkl}` 解释表。

主要代码：

- `h5_band_enhanced_match.py`
- `batch_final_spatial_visualizations.py`
- `visualize_calibration_pipeline.py`
- `continuous_band_geometric_refinement.py`
- `dip_parametric_registration.py`
- `hough_point_spatial_reconstruction.py`
- `spherical_hough_expansion_refinement.py`
- `spherical_radon_graph_pipeline.py`

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

### 2026-05-25

- 新增 `spherical_radon_graph_pipeline.py`，实现新的球面 Radon 峰图匹配原型，不再以之前的空间膨胀/收缩版本作为主线。
- 新 pipeline 对实验 pattern 和标准 master sphere 分别执行多尺度 spherical Hough/Radon transform，在 plane-normal Hough space 中提取峰点。
- 峰描述符包括法向方向、峰强度、最佳带宽、横向 band profile 和 asymmetry。
- 使用峰图的三角不变量生成 orientation 候选，再使用 partial optimal transport 进行全局峰匹配。
- 当前 partial OT 使用 `scipy.optimize.linprog` 求解：实验峰和标准峰都有质量上限，只运输指定比例的总质量，因此允许缺峰、假峰和局部遮挡。
- 在 OT 选出的候选上，联合优化 orientation、`pcx/pcy` 和 `pcz/radius_scale`，再回到原始球面 pattern 上以 image/band score 做局部 refinement。
- 对 `area1_high idx=2661` 做了一次端到端尝试，输出目录为 `outputs/spherical_radon_graph_try_20260525_v2/area1_high/idx_02661/`。
- 该样本最终 phase 为 `Face Centered Cubic`，校正 PC 从 H5 的 `(0.52863, 0.59259, 0.61504)` 变为约 `(0.52697, 0.59524, 0.62151)`，最终 sphere score 约 `0.1971`，保留 6 个 partial OT 峰匹配和 `{hkl}` 解释。
- 当前限制：实验 pattern 只是球面上的局部 patch，Radon 峰在可见视场附近聚集，峰匹配数量仍偏少；下一步应增加 antipodal/晶体对称约束、把峰 profile 与 master 局部带强度一起用于更稳定的 OT cost。

### 2026-05-24

- 新增 `spherical_hough_expansion_refinement.py`，实现球面霍夫点的空间膨胀/收缩迭代匹配。
- 推导并实现 detector Hough line 到球面霍夫点的解析映射：`x cos(theta) + y sin(theta) = rho` 先反投影为三维平面，再由 `n_h = normalize([a, b, c / D_eff])` 得到单位球面上的霍夫点。
- 引入空间膨胀系数 `s`：`D_eff = D / s`，等价于 `PCz_eff = PCz / s`；该系数描述已知球面曲率下的角张量膨胀或收缩。
- 增加曲率验证可视化：令 `u = |rho_pc| / D`，球面角距 `beta = atan(u)`，弦/弧尺度修正可写为 `a_curv(u) = u / atan(u)`。
- 新脚本使用交替迭代策略：先在球面霍夫点空间优化旋转，再优化膨胀/收缩系数，每次迭代按 `lr_decay` 缩小搜索步长。
- 输出 `00_formula_verification.png`、`02_initial_spherical_hough_points.png`、`04_alternating_expansion_trace.png`、`06_final_spherical_hough_points.png`、`09_reconstructed_band_points_3d.png` 和 `10_final_spatial_after_expansion.png` 等可视化。
- 对 10 组 Area1 high pattern 测试时，10/10 组球面霍夫点角误差下降，平均下降约 `0.949 deg`；最终 `expansion` 平均约 `0.9903`，范围约 `0.9410-1.0245`。
- 同一批测试中 full pattern match score 平均变化约 `-0.0298`，没有超过 `max_match_score_drop=0.03` 的保护阈值；说明该方法确实增强了 band geometry 对齐，但当前仍会用掉几乎全部 score guard，需要后续继续平衡图像匹配和 band 几何匹配。
- 本次输出目录为 `outputs/spherical_hough_expansion_selected10_20260524/area1_high/`，输出图和 CSV 按规则不上传 GitHub。

### 2026-05-23

- 新增 `continuous_band_geometric_refinement.py`，实现逐条 Kikuchi band 几何残差的连续优化匹配。
- 连续优化变量包括旋转增量、`pcz/radius_scale`，可选小范围 `pcx/pcy`；默认只优化旋转和半径，PC 作为小范围可选项。
- 新脚本记录主要超参数：初始随机匹配参数、预处理参数、least-squares 迭代参数、旋转/半径/PC bounds、loss 权重、match score guard。
- 对 10 组 Area1 high pattern 测试时，9/10 组逐条 band 平均角误差下降；平均 band angle gain 约 `0.19 deg`，平均 rotation delta 约 `0.55 deg`，平均 `radius_scale=0.99975`。
- 新增 `dip_parametric_registration.py`，尝试 DIP / deep image prior 风格的参数化神经配准：小型 MLP 输出旋转、半径和可选 PC 偏移，PyTorch 实时计算可微 loss 并反传。
- DIP 默认超参数包括 `steps=350`、`lr=2e-3`、`rotation_bound_deg=6`、`radius_min=0.98`、`radius_max=1.02`、`residual_points=1600`，loss 权重为 image line `1.0`、H5 band `0.8`、intensity `0.15`、band geometry `0.6`。
- DIP 也使用 match score guard，默认最终 full match score 不能比初始值下降超过 `0.02`。
- 使用 GPU 跑 10 组 Area1 high pattern 测试时，9/10 组逐条 band 平均角误差下降；平均 band angle gain 约 `0.206 deg`，平均 match score change 约 `-0.0058`，无样本超过 `0.02` 的 score drop，平均 rotation delta 约 `0.717 deg`，平均 `radius_scale=1.00003`。
- 新增 `hough_point_spatial_reconstruction.py`，将 H5 OHP Kikuchi bands 转成霍夫点后进行几何匹配，并用匹配参数还原 band 的三维球面坐标。
- 霍夫点匹配默认超参数包括 `hough_random_starts=180`、`rotation_bound_deg=6`、`radius_min=0.98`、`radius_max=1.02`、`theta_scale_deg=2.5`、`rho_scale_fraction=0.025`、`band_angle_scale_deg=8`。
- 该脚本默认只接受没有超过 `max_match_score_drop=0.03` 的霍夫点解；如果没有更好的可接受解，会回退到受保护的初始匹配，避免错误三维还原。
- 使用 10 组 Area1 high pattern 测试霍夫点匹配：平均 `theta` 误差降低约 `0.379 deg`，平均 `rho` 误差降低约 `1.05 px`，同 HKL band 法向角误差降低约 `0.347 deg`，平均 match score change 约 `-0.0059`，没有样本超过 `0.03` 的 score drop。
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
