# EBSD2026 Kikuchi Pattern Calibration

本仓库用于探索 EBSD 菊池花样读取、菊池带增强匹配、菊池花样到菊池球的空间校正，以及 pattern center / 投影半径偏差校正。

## 目前实现的方法

### 1. 数据读取

- 从 EDAX H5 文件中读取 EBSD map、pattern center 和 OHP 菊池带信息。
- 从 UP2 文件中读取对应的原始菊池花样。
- 当前 pattern center 使用 H5 中真实读取的 `X-Star / Y-Star / Z-Star`，不是虚拟值。
- OHP band 数据按 8 条 band 读取，每条包含 `rho_bin / theta_deg / width / intensity`。
- `ANG/DATA` 同时保存每个 pattern 的软件 indexing 结果，包括 `Orientations(9)`、`Phase`、`IQ`、`CI` 和 `Fit`。
- 当前 EDAX H5 的 `OHP/DATA` 没有逐条 band 的直接 `{hkl}` label；若要直接使用软件 indexing 的晶面信息，需要用 `ANG/DATA/Orientations` 加 phase header 中的 HKL families 反推标准晶面在 detector / Kikuchi sphere 上的位置。

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
  - 实验侧默认不再把整张图的 line-response 像素场直接作为 Radon 输入，而是沿 H5/OHP 软件标定出的 Kikuchi lines 采样，将这些线按 H5 PC 校正回球面后再做 spherical transfer。
  - 软件线校正回球面后，本身可解析得到对应的 plane-normal Hough peak；当前默认峰源为 `software_lines_plus_radon`，即先使用软件线解析峰，再用球面 Radon 补充峰。
  - 对实验球面和 master sphere 分别做多尺度大圆积分，得到 spherical Hough/Radon response，用于验证软件线峰和提取补充峰。
  - 在 plane-normal Hough space 中提取峰点；每个峰保存 `normal direction / peak strength / bandwidth / band profile / asymmetry` 描述符。
  - 构建实验峰图和标准峰图，使用三角不变量生成候选 orientation。
  - 对候选 orientation 使用 partial optimal transport 做全局峰匹配；当前实现是带质量约束的线性规划 partial OT。
  - 在最佳候选基础上联合优化 orientation、`pcx/pcy` 偏移和 `pcz/radius_scale`；参与匹配的软件线峰会随 PC 更新重新计算球面法向。
  - 最终输出 orientation、phase、校正 PC、partial OT 匹配峰和对应 `{hkl}` 解释表。
- 新增固定 PC 的路线对比脚本：
  - `compare_matching_routes_fixed_pc.py` 同时运行旧 weighted image/band route 和新 spherical Radon peak-graph route。
  - 两条路线都严格使用 H5 中读取的 PC，不优化 `pcx/pcy/pcz`，只比较 orientation 和匹配策略差异。
  - 额外输出 H5/OHP 8 条软件 Kikuchi line 到 8 个球面 plane-normal Hough 点的解释图，并与 dense spherical Hough/Radon response 查询网格区分开。
- 新增直接软件 HKL 定位脚本：
  - `direct_hkl_sphere_localization.py` 不做 orientation 搜索，而是直接读取 H5 `ANG/DATA/Orientations` 和 phase HKL families。
  - 对每个 HKL family 生成立方对称等价晶面法向，例如 FCC 的 `(111)/(200)/(220)/(311)`。
  - 将晶面法向用软件 orientation 旋到 detector 坐标，再用当前 H5 PC 投影成 detector 上的预测 Kikuchi line，以及球面 plane-normal Hough 点。
  - 输出 OHP 实测线与软件 orientation+HKL 预测线的 detector 叠加图、detector/crystal 两套球面 normal 点图，以及按软件 orientation 直接放到 master sphere 坐标中的 raw / enhanced pattern。
  - 当前该路线的用途是检查“软件 indexing 给出的取向是否能解释 OHP band 的球面位置”，不是替代 full-pattern matching。
- 新增软件 orientation 直接投影脚本：
  - `software_orientation_sphere_projection.py` 专门实现 `PC -> 实验 detector sphere -> 软件 orientation -> 标准 Kikuchi master sphere` 这条路线。
  - 该脚本不做 orientation matching、不做 HKL label 匹配、不做 PC 优化；只使用 H5 读取的 `X-Star/Y-Star/Z-Star` 和 `ANG/DATA/Orientations`。
  - H5 中软件 orientation 以 `Orientations(9)` 旋转矩阵保存；脚本额外输出一个 `ZXZ` Euler angle 参考值，便于和欧拉角表述对应。
  - 默认使用前面由 OHP band 几何验证过的坐标约定 `orientation_op=G_T`、`detector_convention=flip_xy`，将 detector 上的球面向量放到 master/crystal 坐标中。
  - 输出四类图：PC 回到 detector sphere、软件 orientation 放到 master sphere 的 3D 图、master sphere 经纬展开图、软件 orientation 坐标框架。

主要代码：

- `h5_band_enhanced_match.py`
- `batch_final_spatial_visualizations.py`
- `visualize_calibration_pipeline.py`
- `continuous_band_geometric_refinement.py`
- `dip_parametric_registration.py`
- `hough_point_spatial_reconstruction.py`
- `spherical_hough_expansion_refinement.py`
- `spherical_radon_graph_pipeline.py`
- `compare_matching_routes_fixed_pc.py`
- `direct_hkl_sphere_localization.py`
- `software_orientation_sphere_projection.py`

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
- 修正实验 transfer 的输入：默认 `--experimental-transfer-source h5_lines`，即沿 H5/OHP 软件标定的 Kikuchi line 采样并校正回球面，而不是直接用整张图的 line-response 像素场。
- 新增 `--experimental-peak-source`，默认 `software_lines_plus_radon`：软件线解析峰作为主峰，球面 Radon 峰作为补充峰；也可切换为 `software_lines` 或 `radon` 做消融。
- 峰描述符包括法向方向、峰强度、最佳带宽、横向 band profile 和 asymmetry。
- 使用峰图的三角不变量生成 orientation 候选，再使用 partial optimal transport 进行全局峰匹配。
- 当前 partial OT 使用 `scipy.optimize.linprog` 求解：实验峰和标准峰都有质量上限，只运输指定比例的总质量，因此允许缺峰、假峰和局部遮挡。
- 参考球面 Radon / spherical convolution 文献后，新增可选 `--radon-kernel profiled`，用中心正响应加两侧负旁瓣近似带宽敏感 band profile；但对当前默认的稀疏 H5 软件线输入，消融显示该核会降低稳定性，所以默认仍使用 `--radon-kernel gaussian`。
- 新增 `--ot-edge-weight`，在 partial OT 候选评分中加入峰图边结构一致性损失，惩罚匹配后两两法向角距离不一致的候选。
- 在 OT 选出的候选上，联合优化 orientation、`pcx/pcy` 和 `pcz/radius_scale`；PC 更新时会重算软件线解析峰的球面法向，再回到原始球面 pattern 上以 image/band score 做局部 refinement。
- 对 `area1_high idx=2661` 做了一次端到端尝试，输出目录为 `outputs/spherical_radon_graph_pipeline_plusradon_20260525/area1_high/idx_02661/`。
- 该样本最终 phase 为 `Face Centered Cubic`，校正 PC 从 H5 的 `(0.52863, 0.59259, 0.61504)` 变为约 `(0.53207, 0.59219, 0.61307)`，最终 sphere score 约 `0.1974`，保留 13 个 partial OT 峰匹配和 `{hkl}` 解释。
- 当前限制：该结构化峰图方向已经跑通，但 final image score 仍低于原先 weighted image/band 匹配；下一步应加强 HKL family 一致性、晶体对称下的等价峰处理，以及 master 峰的高分辨率稳定提取。
- 新增 `compare_matching_routes_fixed_pc.py` 做固定 PC 的两路线对比。
- 对 `area1_high idx=2661` 的固定 PC 对比结果：weighted image/band route 的 score 约 `0.3024`，spherical Radon peak-graph route 的 score 约 `0.2317`，新路线保留 `21` 个 partial OT 峰匹配。
- 该对比同时输出 `01_hough_line_to_sphere_point_explanation.png`，说明 8 条 H5/OHP Kikuchi line 在球面 plane-normal Hough space 中对应 8 个真实峰点；而 `experimental spherical Hough/Radon response` 中的大量点只是 normal-grid 上的查询采样点，不是实际检测出的 band 数量。
- 固定 PC 对比输出目录为 `outputs/fixed_pc_route_comparison_20260525/area1_high/idx_02661/`。
- 参考 `EBSD球面霍夫.pdf` 对当前方法做对比：已有 pipeline 和图中方案都包含球面反投影、多尺度 spherical Hough/Radon、峰图、三角候选、partial OT 和局部 refinement；差异是当前实现使用直接采样网格和线性规划 OT，没有实现文献中的球谐/NFFT/Wigner-D 快速球面相关和连续梯度峰搜索。
- 对 `area1_high idx=2661` 的 Radon kernel 消融：`profiled` 负旁瓣核在稀疏 H5 软件线输入上将 graph route score 拉低到约 `0.0371`；`gaussian + OT edge loss` 保持约 `0.2317`，因此默认回到 `gaussian`，保留 `profiled` 作为将来对连续球面强度图的实验选项。
- 新增 `direct_hkl_sphere_localization.py`，直接使用 H5 `ANG/DATA/Orientations` 和 phase HKL families 判断 pattern 在 Kikuchi sphere 上的位置。
- 对 `area1_high idx=2661` 的直接软件 HKL 定位结果：选择 `orientation_op=G_T`、`detector_convention=flip_xy` 后，8 条 OHP band 全部可与软件 orientation 预测的 HKL line 匹配，weighted mean plane-normal angle 约 `5.66 deg`，最大约 `12.00 deg`。
- 直接 HKL 定位输出目录为 `outputs/direct_hkl_sphere_localization_20260525/area1_high/idx_02661/`，包含 detector 叠加图、球面 Hough 点图、pattern 直接投影到 master sphere 的可视化和逐条 band `{hkl}` 对应 CSV。
- 新增 `software_orientation_sphere_projection.py`，修正“软件 orientation 直接定位”的表达方式：先用 PC 将整张实验 pattern 反投影到 detector sphere，再用 H5 软件 orientation 将这块球面 patch 放到标准 Kikuchi master sphere。
- 对 `area1_high idx=2661` 的软件 orientation 直接投影结果：使用 H5 PC `(0.52863, 0.59259, 0.61504)` 和 H5 `Orientations(9)`，默认 `orientation_op=G_T`、`detector_convention=flip_xy`；参考 `ZXZ` Euler angle 约为 `(135.02, 21.15, -64.28) deg`。
- 软件 orientation 直接投影输出目录为 `outputs/software_orientation_sphere_projection_20260526/area1_high/idx_02661/`，包含 `01_pc_backprojection_to_detector_sphere.png`、`02_software_orientation_position_on_master_sphere_3d.png`、`03_software_orientation_position_on_master_sphere_map.png` 和 `04_software_orientation_frame_on_master_sphere.png`。

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
