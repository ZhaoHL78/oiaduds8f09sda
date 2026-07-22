# EBSD2026 EDAX Projection Tools

本仓库当前保留一条更干净的 EDAX/OIM 几何投影路线：从 EDAX H5 和 UP2 中读取原始 Kikuchi pattern、PC、实验几何和 OIM orientation，再把实验 pattern 放回标准 Kikuchi master sphere 上做可视化和诊断。

旧的球面匹配、Hough/Radon 图匹配、DIP、PC/radius refinement 和批量优化实验脚本已经删除。当前重点不是继续做自动匹配优化，而是先理解 GitHub 拉入的 EBSD 代码中每一步坐标变换的含义。

如果只想快速了解当前项目主线、核心脚本和推荐路线，先读 `PROJECT_CORE.md`。

## 目前实现的方法

### 1. EDAX 数据读取

- `project_edax_oim_to_sphere.py` 是核心读取和投影模块。
- 从 EDAX H5 中读取：
  - `Pattern Center Calibration/X-Star, Y-Star, Z-Star`
  - `ANG/DATA/DATA/Orientations`
  - `Sample/Sample Tilt`
  - `Camera/Elevation Angle`
  - `Camera/Azimuthal Angle`
  - `Column/SemKV`
- 从 UP2 文件中按 index 读取对应原始 Kikuchi pattern。
- 使用 `kikuchipy.detectors.EBSDDetector(..., convention="edax")` 将 EDAX PC 转换为内部几何，并用 kikuchipy 的 `_get_direction_cosines_from_detector()` 生成 detector/sample 几何方向。

### 2. Pattern 到 master sphere 的投影

- `project_edax_oim_to_sphere.py` 读取 EMsoft/kikuchipy master pattern 的 Lambert hemisphere 数据。
- 对原始 pattern 做背景扣除和强度归一化，仅用于显示和相关性诊断，不改变原始数据。
- 根据 OIM orientation 的不同矩阵解释生成候选：
  - `edax_g_inverse_row_major`
  - `edax_g_direct_row_major`
  - `edax_g_inverse_col_major`
  - `edax_g_direct_col_major`
- 通过实验 pattern 与 master sphere 采样强度的相关性，诊断当前 orientation 矩阵应该如何解释。

### 3. 坐标链诊断

- `diagnose_edax_transform_chain.py` 用于系统检查：
  - 原图方向：原图、上下翻转、左右翻转、转置等。
  - orientation 矩阵解释：`g`、`g.T`、`inv(g)`、`inv(g).T`。
  - sample tilt / camera elevation 候选组合。
- 输出 detector-only 球面 patch、top transform candidates、score CSV，帮助判断 EDAX 图像方向、PC convention 和 orientation convention 是否一致。

### 4. 多样本和三维可视化

- `visualize_edax_projection_sets.py` 将多张 pattern 以同一固定链路投影到 master sphere 的经纬展开图上。
- `visualize_edax_match_3d.py` 用 PyVista 输出标准 Kikuchi sphere 和实验 pattern patch 的 3D 叠加图，同时导出 glTF 和 HTML viewer。
- `visualize_scan_position_pc_correction.py` 和 `score_scan_position_pc_correction.py` 用 scan position 估计不同 pattern 在扫描视场中造成的 PC 偏移，并比较不同 PC correction 模型。
- 注意：`visualize_edax_match_3d.py` 需要当前 Python 环境安装 `pyvista`/VTK；未安装时，其余 2D 投影和诊断脚本仍可正常运行。

### 4.1 单张 Kikuchi 的预处理、球面标定和 PC finetune

- `single_kikuchi_pc_finetune.py` 用于先挑一张 Kikuchi pattern 跑通完整流程。
- 流程包括：
  - 读取 H5 中的 EDAX PC、OIM orientation、sample tilt 和 camera geometry。
  - 从 UP2 读取同 index 的 raw Kikuchi pattern。
  - 做背景扣除、强度归一化、CLAHE 对比增强和 band-enhanced 图。
  - 用 EDAX PC 生成 detector-frame spherical patch。
  - 用 orientation 把 patch 放到 crystal-frame master sphere 上。
  - 在局部范围内微调 PC，并用实验 pattern 与 master sphere 采样值的相关性作为目标函数。
- 默认测试样例为 Cu Area 1 HighR 的 `pattern-index=2661`。
- 默认 PC 搜索范围为 `PCx/PCy ±0.02`、`PCz ±0.04`，先粗网格再细网格。
- 输出：
  - `single_kikuchi_pc_finetune_overview.png`
  - `single_kikuchi_pc_finetune_summary.csv`
  - `orientation_scores.csv`
  - `pc_finetune_scores.csv`

### 4.2 Pt-3 四组 in-plane EBSD 的同一晶面球形标定

- `pt3_same_face_spherical_calibration.py` 用于处理 Pt-3 的四组 90° in-plane 旋转 EBSD：
  - `Area 3-360 / OIM Map 1`
  - `Area 3-90 / OIM Map 1`
  - `Area 3-180 / OIM Map 1`
  - `Area 3-270 / OIM Map 1`
- 先把四张 SEM 按已知 in-plane 角度旋回 `Area 3-360` 坐标系。
- 在旋回后的 SEM 中定义同一个晶面内部 ROI，并在每个 mapping 的 ROI 内选择 high-IQ/high-CI Kikuchi。
- 每张 Kikuchi 都强制使用保守圆形 detector mask；默认半径为 `0.40 * min(H, W)`，圆外区域不参与背景扣除、衬度增强、PC scoring 或球面投影。
- 预处理包括：
  - 保守圆形 mask；
  - mask 内强度归一化；
  - mask-aware Gaussian background removal；
  - CLAHE contrast enhancement；
  - band-enhanced response。
- orientation 处理不再直接覆盖软件给出的 orientation。流程先用每张 Kikuchi 自己的 H5 orientation 把 pattern 放到 crystal/master sphere 上。
- 然后利用 cubic master sphere 的 24 个 proper symmetry，在对称等价位置中选择最终落点，使四张 pattern 满足共同轴线闭包：`Q180 ~= Q90^2`、`Q270 ~= Q90^3`。
- 默认 `--orientation-mode reference_variant`，只用参考图确定一次 EDAX H5 orientation matrix convention，其余图仍使用各自 H5 orientation 数值。
- 默认 PC finetune 范围收紧为 `PCx/PCy ±0.01`、`PCz ±0.02`，只做单张 pattern 的小范围几何细调。
- 同时记录每张 Kikuchi 上 H5/EDAX 原始 PC 和 refined PC 的像素位置，并把这个 PC 方向通过同一条 orientation + cubic symmetry 落点链投到同一个 master sphere 上，用于判断四张 pattern 的晶体学锚点是否满足共同轴线关系。
- 输出：
  - `pt3_same_face_roi_selection.png`
  - `pt3_same_face_spherical_calibration_workflow.png`
  - `pt3_same_face_3d_kikuchi_sphere.png`
  - `pt3_pc_positions_on_patterns.png`
  - `pt3_pc_positions_on_same_sphere.png`
  - `pt3_pc_positions_same_sphere_axis_view.png`
  - `pt3_pc_positions_on_same_sphere_lon_colat.png`
  - `pt3_clear_final_spherical_kikuchi_maps.png`
  - `pt3_clear_3d_front_facing_kikuchi_spheres.png`
  - `pt3_same_sphere_axis_aligned_kikuchi_patterns.png`
  - `pt3_same_sphere_axis_view.png`
  - `pt3_same_sphere_reference_patch_view.png`
  - `pt3_same_sphere_oblique_axis_view.png`
  - `pt3_front_view_Area_3-360.png` / `pt3_front_view_Area_3-90.png` / `pt3_front_view_Area_3-180.png` / `pt3_front_view_Area_3-270.png`
  - `pt3_same_face_spherical_calibration_summary.csv`
  - `pt3_cubic_symmetry_axis_prior_summary.csv`

### 5. 固定 PC 的单晶 zone-axis 倾斜和 PC 漂移模拟

- `simulate_111_tilt_kikuchi_patterns.py` 用固定 EDAX PC 模拟指定 zone axis 的单晶 Kikuchi pattern。
- 默认从 `D:\project\EBSD2026\ebsd.edaxh5` 的 Area 1 HighR 读取 `X-Star/Y-Star/Z-Star` 作为固定 PC。
- 可用 `--zone H K L` 指定 0 deg 时 detector 中心方向对准的晶体方向，例如 `--zone 1 2 3`。
- 上下倾斜只绕 detector 水平轴改变一个角度变量。
- 默认倾斜角为 `-10, -5, 0, 5, 10` deg。
- 可用 `--pc 0.5 0.5 0.5` 指定固定 PC。
- 可用 `--pc-x-values 0.4 0.45 0.5 0.55` 模拟横向 beam shift / PCx 漂移。
- 可用 `--circular-transparent` 输出 EDAX 风格圆形 detector PNG，圆外 alpha=0。
- 默认 master pattern 为 kikuchipy 自带 Ni/FCC master pattern；如果有 Cu master sphere，可用 `--master` 替换。

### 6. 模拟 pattern 的 IPF 标注

- `annotate_simulated_ipf_points.py` 将模拟得到的表观方向投到标准立方晶体反极图三角区。
- 当前用于两组模拟数据：
  - `(1,2,3)` zone axis，`PC=(0.5,0.5,0.5)`，倾斜角 `0,-2,2,5`。
  - `(1,3,5)` zone axis，倾斜角 0 deg，统一标定 PC 为 `(0.5,0.5,0.5)`，模拟实际 `PCx=0.4,0.45,0.5,0.55` 的横向 beam shift。
- 输出两个透明背景 IPF，每张包含四个标注点。
- 同时输出 `simulated_ipf_indexed_points.csv`，记录每个点的表观方向和最近整数 Miller 指数。

### 7. (1,2,3) 探测器/样品倾斜分离模拟

- `simulate_123_detector_sample_tilt_cases.py` 专门生成 `(1,2,3)` 的四个几何工况：
  - 默认探测器和样品位置。
  - 探测器向上倾斜 5 deg。
  - 样品向上倾斜 2 deg。
  - 样品向下倾斜 2 deg。
- 所有 pattern 使用 `PC=(0.5,0.5,0.5)`，输出为 EDAX 圆形透明 PNG。
- 对应 IPF 图中，前两个工况使用黑色空心圆，后两个工况使用红色空心圆。
- metadata 中记录每个工况对应的表观方向和最近整数 Miller 指数。

### 8. Pt-1 EDAX 数据分类

- `classify_pt1_ebsd_data.py` 用于读取 `D:\EBSD-data\Pt-1` 中的 EDAX H5 和 UP2 文件元信息。
- 只读取 H5 map 元数据、ANG/OHP 统计和 UP2 header，不搬运原始大体积 pattern 数据。
- 输出：
  - `pt1_h5_maps.csv`：Pt-1 H5 中每个 OIM map 的尺寸、PC、phase、IQ/CI/Fit、OHP 信息。
  - `pt1_up2_files.csv`：每个 UP2 文件的分辨率、pattern 数量、文件大小。
  - `pt1_match_classification.csv`：H5 map 与 UP2 的一一匹配/未匹配分类。
  - `pt1_classification_summary.md`：便于直接阅读的分类总结。
- 当前匹配规则是：先按 pattern count 精确匹配，再在同一 count 内按采集时间顺序配对。

### 9. Pt-1 SEM 与对应 Kikuchi 导出

- `export_pt1_sem_kikuchi_pairs.py` 读取 Pt-1 分类后的可靠 H5-UP2 对应组。
- 每个 EBSD map 从 H5 读取对应的 `SEM-PRIAS Images/DATA/SEM`。
- 每个 map 选取 `ANG/DATA` 中 valid 且 IQ 最高的位置，从对应 UP2 直接读取同 index 的 raw Kikuchi pattern。
- 输出 SEM 标注图、raw Kikuchi 显示图、圆形透明 Kikuchi 和 SEM/Kikuchi 并排图。
- Kikuchi 只做显示灰度归一化，不做旋转、翻转或几何校正。

### 10. H5 全部 mapping 的 SEM 与 UP2 文件名对应

- `export_h5_mapping_sem_correspondence.py` 读取 `20251209Pt.edaxh5` 中所有含 `ANG/DATA` 和 `SEM-PRIAS Images/DATA/SEM` 的 OIM mapping。
- 导出每个 mapping 对应的 SEM 图，并生成总览 contact sheet。
- 自动扫描本机 UP2 文件，也会解析 Windows 回收站中的 `$I*.up2` 元数据，把 `$R*.up2` 恢复为原始 UP2 文件名。
- 当前对应规则是 `specimen + pattern count` 精确匹配，并在同一组内按采集/修改时间顺序配对。
- 输出 `h5_mapping_sem_correspondence.csv`，记录 H5 mapping、SEM 图路径、匹配到的原始 UP2 文件名和实际本机位置。

### 10.1 Pt-3 外部 EDAX IPF 与 SEM/BSE 固定对应

- `export_pt3_external_ipf_sem_mapping.py` 用于 Pt-3 UP2 `Area 4/5/7/9` 的最终 IPF/SEM 对应可视化。
- 这一步不再使用 H5 内置的 `SEM-PRIAS Images/DATA/SEM` 低分辨率 SEM，也不再用 H5 `Orientations` 近似重算 IPF-Z。
- 原因是 EDAX 软件导出的 IPF 图和外部 BSE tif 使用了软件自己的裁剪、坏点处理、颜色 convention 和 SEM 对应关系；直接把 H5 里的 `400 x 512` SEM 与重算 IPF 叠加，会造成 SEM/IPF 视场和角度关系错误。
- 固定对应关系：
  - UP2 `20251209_Pt-3_Area 4_OIM Map 1.up2` -> H5 `20251209/Pt-3/Area 3-90/OIM Map 1` -> IPF `90.bmp` -> SEM `2-90bse.tif`
  - UP2 `20251209_Pt-3_Area 5_OIM Map 1.up2` -> H5 `20251209/Pt-3/Area 3-180/OIM Map 1` -> IPF `180.bmp` -> SEM `2-180bse.tif`
  - UP2 `20251209_Pt-3_Area 7_OIM Map 1.up2` -> H5 `20251209/Pt-3/Area 3-270/OIM Map 1` -> IPF `270.bmp` -> SEM `2-270bse.tif`
  - UP2 `20251209_Pt-3_Area 9_OIM Map 1.up2` -> H5 `20251209/Pt-3/Area 3-360/OIM Map 1` -> IPF `0.bmp` -> SEM `2-360bse.tif`
- 输出：
  - `pt3_area4_5_7_9_corrected_ipf_bse_contact_sheet.png`
  - `pt3_area4_5_7_9_corrected_external_mapping.csv`
  - 每组单独的 EDAX IPF PNG、BSE full PNG 和裁底栏后的 BSE crop PNG。

### 10.2 IPF 颜色与 Kikuchi orientation 对应验证

- H5 中真实的一一对应关系是 `pattern_index = row * ncols + col`；同一个 index 同时索引 `ANG/DATA` 的 orientation/IQ/CI/phase、`OHP/DATA` 的 Kikuchi band，以及 UP2 中对应的 raw Kikuchi pattern。
- 对 Pt-3 `Area 3-90/180/270/360`，每组均满足 `259 x 291 = 75369 = ANG count = OHP count = UP2 pattern count`。
- 与 EDAX 导出的 IPF bmp 对比后，H5 orientation 的 IPF-Z 颜色 convention 确认为：
  - 将 `Orientations` 作为 row-major `G`；
  - 用 `G @ [0, 0, 1]` 得到样品 ND 在晶体坐标中的方向；
  - 通过 cubic symmetry 折叠到 `[001]-[101]-[111]` 反极图三角；
  - 使用 `[001]=red`、`[101]=green`、`[111]=blue` 的 cubic IPF 颜色键。
- EDAX/OIM 软件导出的 IPF-Z 是纯 orientation 颜色图，不把 CI/IQ 作为亮度权重，也不按 phase label 把像素清黑；CI/IQ/phase 只能作为单独质量图或诊断图。
- 与 EDAX 软件 BMP 直接比较时，H5 grid IPF 需要按照物理步长比例重采样到软件导出的显示尺寸，例如 Pt high-resolution 的 `806 x 720` grid 对应软件 BMP 的 `711 x 550`。
- 之前使用 `G.T @ ND` 或默认 CI-weighted IPF-Z 会导致颜色和 EDAX 导出图不一致，现已在 `export_h5_ipf_bse_maps.py` 中修正。
- 当前验证输出目录：`outputs/pt3_ipf_orientation_validation`，包含高分辨率 H5-grid IPF、EDAX/H5 IPF 对比、IPF 三角颜色键、以及选点 IPF -> 三角区 -> 同 index raw Kikuchi 的验证图。

### 10.3 Pt-3 90° Kikuchi finetune 后重新输出 IPF map

- `export_pt3_area90_finetuned_ipf_map.py` 用于把 Pt-3 90° 这组数据的 Kikuchi finetune 结果重新表达为 EBSD IPF map。
- 输入 mapping：
  - H5: `20251209/Pt-3/Area 3-90/OIM Map 1`
  - UP2: `20251209_Pt-3_Area 4_OIM Map 1.up2`
  - EDAX IPF reference: `E:\ZHL\ZHL-EDAX\20251209Pt\Pt-3\90.bmp`
- 先用 `stable_global_pc_orientation_calibration.py` 对 5 个 high-IQ/high-CI Kikuchi 重新做稳定 PC + orientation residual finetune。
- 本次 90° 结果：
  - stable global PC residual: `dPC=(-0.002, +0.002, -0.008)`
  - 5 个 Kikuchi 的 residual orientation 使用稳健中位数作为整张 map 的全局小角度校正：`dR=(-0.25, -0.35, -0.20) deg`
  - residual 应用方式与球面 finetune 一致：`G_refined = G @ delta.T`
- 输出 IPF 使用和 EDAX 原图一致的像素尺寸 `714 x 550`；H5 原始 grid 为 `259 x 291`，`Step X=0.2 um`、`Step Y=0.173205 um`，其物理宽高比与 `714/550` 一致。
- 输出：
  - `pt3_area90_finetuned_ipf_clean_714x550.png`
  - `pt3_area90_finetuned_ipf_edax_style_714x550.png`
  - `pt3_area90_edax_h5_finetuned_ipf_comparison_714x550_panels.png`
  - `pt3_area90_finetuned_ipf_metadata.json`
  - `pt3_area90_finetuned_ipf_parameters.csv`

### 10.4 Pt high-resolution 60° AFM/SEM/IPF 对应

- `align_pt_highres60_afm_same_fov.py` 是当前推荐的 Pt high-resolution 60° AFM 对齐流程。
- 修正后的物理约束是：AFM 和 H5 SEM 是近似同一视场，不能使用 SEM/IPF 图上的 scale bar 推导 AFM 覆盖大小。
- AFM `.ibw` 直接读出的数组方向与 AFM 软件截图不同；当前默认先做 `rot90`，得到与软件显示一致的 AFM 方向。
- AFM 配准目标是 H5 `SEM-PRIAS Images/DATA/SEM` 的 raw row order。H5 SEM 只有在进入 EDAX IPF-Z frame 时才做 `flipud`。
- 默认同视场配准参数为 `AFM rot90 -> resize to SEM -> center rotation +10 deg`；`stretch-x/stretch-y` 和平移参数显式保留，用于表示 SEM 在 70° tilt 后做简单 tilt correction 带来的单轴拉伸/残余畸变。
- 当前输出目录：`outputs/pt_highres60_afm_same_fov_alignment`，关键图包括：
  - `afm_sem_same_fov_orientation_check.png`
  - `afm_sem_ipf_same_fov_alignment_overview.png`
  - `afm_scharr_normalmap_rot90_with_colorbar.png`
  - `afm_height_same_fov_warped_to_raw_sem.png`
  - `afm_height_same_fov_warped_to_ipf_frame.png`

### 11. UP2 到 EBSD 与 Kikuchi 的逐文件对应

- `export_up2_ebsd_kikuchi_correspondence.py` 以 UP2 文件为主线读取 `E:\ZHL\EBSD-RAW\20251209Pt`。
- 对每个 UP2 判断是否能对应到 H5 中的 EBSD mapping。
- 匹配成功的 UP2：从 H5 中选取 valid 且 IQ 最高的 index，读取同 index 的 raw Kikuchi，并输出 SEM 标注点 + Kikuchi 并排图。
- 未匹配的 UP2：标记为无 H5 EBSD 对应；如果 UP2 有 pattern，则导出中心 index 的 Kikuchi 预览用于人工检查。
- 输出 `up2_ebsd_kikuchi_correspondence.csv` 和 matched/unmatched contact sheet。

### 12. Pt-1 四组 in-plane EBSD 的 SEM 对齐和公共圆 ROI

- `align_pt1_inplane_sem_common_circle.py` 读取 Pt-1 中同网格的四组 in-plane EBSD：
  - `Area 90degree / OIM Map 1`
  - `Area 180degree / OIM Map 1`
  - `Area 270degree / OIM Map 1`
  - `Area 360degree / OIM Map 1`
- 同时读取对应的 UP2 文件，验证 H5 pattern count 与 UP2 pattern count 一致。
- 以 `Area 90degree` 为参考，对其余 SEM 反向旋转 `-90/-180/-270` 度，并用 SEM 边缘图做保守平移微调。
- 由四张 transformed SEM 的有效 mask 交集计算最大内切圆，作为四组 EBSD 的公共 circular ROI。
- 输出 aligned SEM、overlay、公共 mask、最大圆、以及该圆反投影回原始 SEM 的图。

### 13. Pt high-resolution 十二组 30° EBSD 的 LightGlue 对齐和 PC 锚点可视化

- `pt_highres_30deg_lightglue_calibration.py` 读取 `E:\ZHL\EBSD-RAW\20251217Pt-high resolution` 中的 12 组高分辨率 EBSD：
  - H5 mapping: `Area 8-0, 8-30, ..., 8-330 / OIM Map 1`
  - UP2: `Area 3` 到 `Area 14`，按真实采集顺序对应 `30°, 60°, ..., 330°, 0°`。
- 注意：这批数据不能用自然角度顺序推断 `0° -> Area 3`。H5 时间戳显示采集顺序为 `30,60,...,330,0`，若映射错位会导致 Kikuchi pattern 与 H5/OHP band 明显不匹配。
- 使用 LightGlue + SuperPoint 对相邻角度 SEM 做配准：`30 -> 0`、`60 -> 30`、...、`330 -> 300`。
- 每一对配准先按已知 30° in-plane rotation 预旋转，再用 LightGlue/SuperPoint 匹配和 RANSAC affine refinement，最后把所有 SEM transform 串联到 0° 坐标系。
- 在共同有效区域内自动选择一个远离晶界、IQ/CI 综合较高的同一物理位置点，并反投影到每个 EBSD mapping 的 raw SEM/grid index。
- 对这 12 个 index 读取对应 UP2 Kikuchi pattern，沿用之前流程：
  - 圆形 detector mask；
  - 背景扣除、CLAHE 衬度增强；
  - H5/EDAX PC + H5 orientation 投影到 master sphere；
  - 小范围 PC finetune；
  - 用旧的 detector-space forward matching 逻辑验证：把 master sphere 按当前 PC + orientation 采样回 detector 像素，并与实验 pattern 做 NCC/残差对比；
  - 主球面匹配输出保持 H5 orientation + refined PC，不再被 cubic symmetry 后处理覆盖；
  - 另开 `symmetry_results` 分支，按 Pt-3 的 cubic symmetry axis placement 逻辑选择等价晶体分支，仅作为共同轴线/对称性诊断；
  - 用 4e9e21b 的高清 front-view 思路分别输出主匹配和 cubic 诊断的球面正视图；
  - 单独绘制 PC 在 pattern 上的位置和 PC 在 master sphere 上的晶体学锚点。
- 当前本机运行结果中，SEM 相邻配准全部使用 `lightglue_superpoint`，没有使用 fallback；各对 RANSAC inlier 约 48-81 个，RMSE 约 2.0-2.6 px。
- 当前固定流程来自 `21dce0a` 和 `4e9e21b` 的可视化/对称性思路，但球面匹配本身回到更早的 forward detector validation：最终匹配位置不由 cubic branch 强行改写。
- 当前 12 组运行的共同 SEM 参考点为 `(342.07, 275.83)`；修正 H5/UP2 对应关系后，detector-validated refined score 为约 `0.218-0.340`。
- 当前 cubic symmetry placement 诊断为 `score=9.21718`、`angle30=30.16°`、共同轴约为 `(0.385, -0.295, 0.874)`。
- 当前 forward detector validation 表明 cubic branch 不是所有 pattern 的最佳匹配：4/12 张 detector-space 分数低于 base/refined placement，其中 180° 从 `0.34024` 降到 `0.32439`。因此 cubic 输出只作为诊断，不作为最终 sphere matching 结果。
- 输出：
  - `pt_highres_sem_lightglue_alignment_overview.png`
  - `pt_highres_same_point_selection.png`
  - `pt_highres_selected_kikuchi_pc_patterns.png`
  - `pt_highres_spherical_calibration_workflow.png`
  - `pt_highres_same_sphere_lon_colat.png`
  - `pt_highres_same_sphere_3d.png`
  - `pt_highres_pc_anchor_lon_colat.png`
  - `pt_highres_pc_anchor_3d.png`
  - `pt_highres_sphere_matching_front_views.png`
  - `pt_highres_sphere_match_front_pattern_000.png` 到 `pt_highres_sphere_match_front_pattern_330.png`
  - `pt_highres_forward_detector_validation.png`
  - `pt_highres_forward_detector_validation_scores.csv`
  - `pt_highres_cubic_symmetry_same_sphere_lon_colat.png`
  - `pt_highres_cubic_symmetry_same_sphere_3d.png`
  - `pt_highres_cubic_symmetry_pc_anchor_lon_colat.png`
  - `pt_highres_cubic_symmetry_pc_anchor_3d.png`
  - `pt_highres_cubic_symmetry_front_views.png`
  - `pt_highres_cubic_front_pattern_000.png` 到 `pt_highres_cubic_front_pattern_330.png`
  - `pt_highres_pair_alignments.csv`
  - `pt_highres_30deg_spherical_calibration_summary.csv`
  - `pt_highres_30deg_cubic_symmetry_placement_summary.csv`
  - `pt_highres_30deg_cubic_symmetry_axis_prior_summary.csv`
  - `pt_highres_sem_transforms_raw_to_angle0.npz`

### 13.1 Pt high-resolution 数据总览与质检可视化

- `export_pt_highres_data_overview.py` 是 12 组 high-resolution 数据的轻量总览导出入口。
- 该脚本不重新运行 LightGlue、PC finetune 或球面匹配，只读取原始 H5/UP2 和 `pt_highres_30deg_lightglue_calibration.py` 已生成的 summary。
- 输出用于快速检查 H5 SEM、EDAX-style H5 IPF-Z、UP2 Kikuchi、OHP band、PC residual 和 score 是否一一对应。
- IPF-Z 输出默认使用 EDAX/OIM 约束：row-major `G @ ND`、cubic `[001]-[101]-[111]` fold、`[001]=red/[101]=green/[111]=blue`、sqrt saturation、无 CI/IQ dimming、无 phase mask，并按 EDAX BMP reference 的物理显示尺寸重采样。
- 默认读取 `E:\ZHL\20251209Pt-EBSD MAP\pt-high resolution` 中的 `0.bmp, 30.bmp, ..., 330.bmp` 作为软件 reference，对比 H5 重新生成的 IPF-Z。
- OHP band overlay 使用已修正的 `normal_theta_rho+_yup` 约定，与 `export_publication_h5_kikuchi_bands.py` 保持一致。
- 同时输出 `pt_highres_ohp_overlay_diagnostics.csv`，记录每张图的 OHP 线在 band-enhanced Kikuchi 上的响应，用作 H5/OHP/UP2 对应关系的 sanity check。
- 输出：
  - `pt_highres_sem_ipf_overview.png`
  - `pt_highres_edax_ipf_reference_comparison.png`
  - `pt_highres_edax_ipf_0_90_detailed_comparison.png`
  - `pt_highres_edax_ipf_reference_metrics.csv`
  - `pt_highres_kikuchi_ohp_overview.png`
  - `pt_highres_quality_pc_score_overview.png`
  - `pt_highres_existing_calibration_result_index.png`
  - `pt_highres_12map_visual_inventory.csv`
  - `pt_highres_raw_up2_inventory.csv`
  - `pt_highres_ohp_overlay_diagnostics.csv`
  - `per_angle/angle_***_*.png`

### 14. Pt Kikuchi 批量球面匹配校准固定流程

- `batch_pt_kikuchi_spherical_calibration.py` 是当前推荐的固定批处理入口，用于在 Pt EBSD 数据中自动挑选若干高质量 Kikuchi，并复用单张 pattern 已验证的球面匹配流程。
- 默认数据：
  - H5: `D:\EBSD-data\Pt-1\20251209Pt.edaxh5`
  - UP2 root: `D:\EBSD-data`
  - specimen filter: `Pt-3`
- 流程：
  - 用已有 `collect_h5_maps / collect_up2_candidates / match_h5_to_up2` 规则自动匹配 H5 mapping 和本地 UP2 文件。
  - 每个 matched mapping 默认挑选 1 张 high-IQ/high-CI Kikuchi；可用 `--patterns-per-map` 和 `--max-patterns` 扩展数量。
  - 对每张 Kikuchi 执行固定流程：完整圆形 Kikuchi disk mask、背景扣除、CLAHE 衬度增强、scan-position PC correction、残余 PC finetune、残余 orientation finetune。
  - 默认 mask 为 centered full disk，半径 `0.49 * min(H, W)`；对 470 x 470 的 Pt-3 pattern，对应圆心 `(234, 234)`、半径 `230`，用于覆盖完整 Kikuchi 圆形区域。
  - 可用 `--mask-mode estimated` 调用 Hough 圆检测，但 Pt-3 默认不用它，因为局部强 Kikuchi band 可能让 Hough 偏到错误圆心。
  - 默认 `--pc-initial scan_position`，先根据 EBSD 扫描位置估计电子束作用点造成的 PC 漂移，再以这个 PC 作为局部 finetune 初值。
  - PC residual finetune 后，固定 refined PC，只搜索一个高精度小角度 residual orientation，默认 `--orientation-bound-deg 0.5`、`--orientation-step-deg 0.05`，即每轴 `-0.5°` 到 `+0.5°`。
  - 同时输出顺序诊断：A 路线为 `scan-position PC -> PC residual -> orientation residual`，B 路线为 `scan-position PC -> orientation residual -> PC residual`，用于判断 PC 和 orientation 是否在互相代偿。
  - PC residual 和 orientation residual 采用顺序分解：PC residual 先解释 detector ray geometry / pattern center 导致的非刚性投影差异；orientation residual 再解释球面上的整体小角度刚性旋转残差。单张 pattern 中二者不是数学上唯一可分的，因此 summary 同时保存四个阶段 score 供人工判断。
  - 不做 cubic symmetry/axis placement；这是 detector-validated 的单张球面匹配校准流程。
- 当前默认运行在本机 Pt-3 数据上匹配到 5 个 mapping，并输出 5 张详细 stage-wise 校准图。5 张均选择 `edax_g_direct_row_major`，PC+orientation score 均高于 map-PC score。
- 输出：
  - `pt_kikuchi_spherical_calibration_contact_sheet.png`
  - `pt_kikuchi_spherical_calibration_summary.csv`
  - `pt_kikuchi_spherical_calibration_summary.md`
  - `per_pattern/<pattern_key>/<pattern_key>_spherical_calibration_overview.png`
  - `per_pattern/<pattern_key>/<pattern_key>_position_pc_orientation_finetune.png`
  - `per_pattern/<pattern_key>/<pattern_key>_pc_orientation_order_comparison.png`
  - `per_pattern/<pattern_key>/orientation_scores.csv`
  - `per_pattern/<pattern_key>/pc_finetune_scores.csv`
  - `per_pattern/<pattern_key>/orientation_finetune_trace.csv`
  - `per_pattern/<pattern_key>/orientation_before_pc_trace.csv`
  - `per_pattern/<pattern_key>/pc_after_orientation_finetune_scores.csv`
  - `per_pattern/<pattern_key>/single_kikuchi_pc_finetune_summary.csv`

### 15. PC/orientation residual 联合可解释诊断

- `joint_pc_orientation_explainability.py` 用于验证 PC residual 和 orientation residual 的互相代偿关系，并给出可解释性证据，而不是只追求单一 NCC 分数。
- 注意：该脚本包含 H5/OHP band center/profile/width 等额外诊断项，系统更复杂，当前不作为主匹配流程，只用于分析 PC/orientation 代偿关系。
- 默认读取同一组 Pt EBSD mapping：
  - H5: `D:\EBSD-data\Pt-1\20251209Pt.edaxh5`
  - specimen: `Pt-3`
  - area: `Area 3-90`
  - 默认选择 3 张 high-IQ/high-CI Kikuchi。
- 每张 Kikuchi 使用 H5 PC/orientation 作为先验，同时读取 H5/OHP 中的软件 Hough Kikuchi band：`rho/theta/width/intensity`。
- 优化变量为：
  - `delta PCx, delta PCy, delta PCz`
  - `delta orientation rx, ry, rz`
- 目标函数拆成多个可解释分量：
  - image NCC：整张 pattern 和 master sphere 的 detector-space 匹配。
  - H5/OHP band center：软件标出的 Kikuchi band 中心线是否落在 master band 响应峰上。
  - band profile NCC：沿 H5/OHP band 法向的实验 profile 与 master profile 是否一致。
  - band width consistency：profile 宽度残差是否下降，用于判断 PC-like 非刚性畸变是否真的被改善。
  - PC prior / orientation prior：限制 PC 和 orientation 不要互相过度代偿。
- 输出中同时给出 `PC only`、`orientation only` 和 `joint PC+orientation` 三类候选，并用证据标签解释残差来源：
  - `orientation_dominant_pc_not_supported_by_width`
  - `mixed_orientation_and_pc_supported`
  - `pc_dominant_width_distortion_reduced`
  - `ambiguous_or_already_aligned`
- 当前 Pt-3 Area 3-90 三张默认结果显示：joint objective 均提升，H5/OHP band center error 明显下降，但 band width error 没有同步下降，因此解释更偏向 orientation residual 主导，而不是 PC residual 主导。
- 输出：
  - `joint_pc_orientation_explainability_contact_sheet.png`
  - `joint_pc_orientation_explainability_summary.csv`
  - `per_pattern/<pattern_key>/<pattern_key>_joint_pc_orientation_explainability.png`
  - `per_pattern/<pattern_key>/joint_optimization_trace.csv`

### 16. 稳定全局 PC residual + orientation residual 主流程

- `stable_global_pc_orientation_calibration.py` 是当前更推荐的简化主流程，用于避免 band width/profile 目标让系统过度复杂。
- 默认读取同一组 Pt EBSD mapping：
  - H5: `D:\EBSD-data\Pt-1\20251209Pt.edaxh5`
  - specimen: `Pt-3`
  - area: `Area 3-90`
  - 默认选择 5 张 high-IQ/high-CI Kikuchi。
- 核心约束：
  - PC residual 是 mapping-level 的全局共享偏移，所有选中 Kikuchi 使用同一个 `delta PCx, delta PCy, delta PCz`。
  - 每张 Kikuchi 的 PC 仍从 scan-position PC 出发：`stable PC_i = scan_position_PC_i + global_delta_PC`。
  - PC 固定后，每张 Kikuchi 只做小范围 orientation residual，默认每轴 `-0.5°` 到 `+0.5°`、步长 `0.05°`。
- 目标函数只使用原来验证过的 detector-space image NCC：
  - preprocessed intensity NCC；
  - band-enhanced image NCC；
  - global PC prior。
- 不再使用 band width/profile loss，避免局部 band 特征把 PC/orientation 拟合带偏。
- 当前 Pt-3 Area 3-90 默认 5 张结果：
  - global PC residual: `(-0.002, +0.002, -0.008)`；
  - 5 张 pattern 均满足 `scan PC score < global PC score < PC+orientation score`；
  - 说明在保持 PC 全局稳定后，PC residual 和 orientation residual 都提供了稳定增益。
- 输出：
  - `stable_global_pc_diagnostic.png`
  - `stable_global_pc_orientation_contact_sheet.png`
  - `stable_global_pc_orientation_summary.csv`
  - `global_pc_residual_grid.csv`
  - `per_pattern/<pattern_key>/<pattern_key>_stable_global_pc_orientation.png`
  - `per_pattern/<pattern_key>/orientation_residual_trace.csv`

### 17. Pt AFM -> SEM -> EBSD/IPF 配准

- `align_pt_afm_sem_ipf.py` 用于把 `D:\EBSD project\3d数据\pt-afm\Pt-1.ibw` 中的 AFM 图像配准到对应的外部 SEM/BSE，再根据已确认的 SEM/IPF 关系放入 EBSD/IPF 坐标系。
- 默认对应关系沿用 Pt-3 90° 这组数据：
  - AFM: `D:\EBSD project\3d数据\pt-afm\Pt-1.ibw`
  - SEM/BSE: `E:\ZHL\20251209Pt-EBSD\2-90bse.tif`
  - EBSD/IPF: Pt-3 `Area 3-90 / OIM Map 1`
  - UP2: `20251209_Pt-3_Area 4_OIM Map 1.up2`
  - IPF: 优先使用 `outputs\pt3_area90_finetuned_ipf_map\pt3_area90_finetuned_ipf_clean_714x550.png`，不存在时回退到 EDAX `90.bmp`。
- AFM IBW 当前读出为 `1024 x 1024 x 4`，通道为 `HeightRetrace / AmplitudeRetrace / PhaseRetrace / ZSensorRetrace`，扫描尺寸为 `18 um`。
- 依赖：`igor2` 用于读取 `.ibw`，`lightglue`/`torch` 用于 SuperPoint 特征和匹配。
- 配准流程：
  - 自动裁掉 SEM/BSE 底部显微镜信息栏，只保留真实图像区域。
  - 对 AFM 多通道和 SEM 生成 normalized / high-pass / inverted / Sobel / Canny 特征图。
  - 使用 LightGlue + SuperPoint 提取匹配点，并用 RANSAC full affine 估计 AFM -> SEM 变换；full affine 用于保留 AFM/SEM 之间的非等比例尺度差。
  - 候选排序加入轻量物理尺度约束，避免少量局部特征把 AFM 放大到不合理尺度。
  - 将 AFM height/amplitude warp 到 SEM frame，再和 Pt-3 90° IPF frame 叠加。
- 当前本机结果：最佳候选为 `ZSensorRetrace_hp` 对 `sem_hp`，LightGlue/SuperPoint 为 `11/23` RANSAC inliers，RMSE `4.70 px`；三晶界交点、左上斜晶界和右上台阶边界均落在 SEM/IPF 的对应位置。
- 输出：
  - `afm_sem_ipf_alignment_overview.png`
  - `lightglue_afm_sem_inlier_matches.png`
  - `lightglue_afm_sem_candidates.csv`
  - `afm_channels_preview.png`
  - `candidate_sem_preview.png`
  - `sem_2_90_content_norm.png`
  - `ipf_resized_to_sem_frame.png`
  - `afm_amplitude_warped_to_sem.png`
  - `afm_height_warped_to_sem.png`
  - `afm_sem_ipf_alignment_metadata.json`

### 17.1 Pt high-resolution 60° AFM -> EBSD 对齐与 normalmap

- `align_pt_highres60_afm_same_fov.py` 用于把 `D:\EBSD project\3d数据\pt-afm\Pt-2high resolution.ibw` 配准到 Pt high-resolution 的 60° EBSD 数据：
  - H5: `E:\ZHL\EBSD-RAW\20251217Pt-high resolution\20251217.edaxh5`
  - H5 mapping: `20251217/Pt foil-high resolution/Area 8-60/OIM Map 1`
  - EDAX IPF reference: `E:\ZHL\20251209Pt-EBSD MAP\pt-high resolution\60.bmp`
- AFM 读出为 `1024 x 1024`，通道为 `HeightRetrace / AmplitudeRetrace / PhaseRetrace / ZSensorRetrace`，扫描尺寸 `70 um`；SEM/IPF 图上的 scale bar 不参与配准。
- AFM `.ibw` 需要先做 `rot90` 才与 AFM 软件截图方向一致。配准目标使用 H5 SEM raw row order；进入 EDAX IPF-Z frame 时，再对 SEM/AFM overlay 一起 `flipud`。
- 默认配准是同视场模型：`AFM rot90 -> resize 到 SEM 尺寸 -> 围绕中心逆时针 10°`。`stretch-x/stretch-y` 和平移保留为显式参数，用于表达 70° SEM tilt correction 的单轴拉伸和残余畸变，但不再让 scale bar 决定 AFM 覆盖大小。
- AFM normalmap 用 `HeightRetrace` 做平面扣除后，通过 Scharr 算子提取 `dz/dx`、`dz/dy`，构造 `[-dz/dx, -dz/dy, 1]`，再使用同视场 affine 的平面旋转部分把 normalmap 表达到 SEM/IPF frame。affine 的 scale/shear 不进入高度梯度。
- 当前输出目录：`outputs/pt_highres60_afm_same_fov_alignment`。
- 输出：
  - `afm_sem_same_fov_orientation_check.png`
  - `afm_sem_ipf_same_fov_alignment_overview.png`
  - `ebsd60_sem_h5_raw_afm_alignment_frame.png`
  - `ebsd60_sem_h5_flipud_ipf_frame.png`
  - `ebsd60_ipf_edax_reference.png`
  - `afm_height_rot90_software_orientation.png`
  - `afm_height_same_fov_warped_to_raw_sem.png`
  - `afm_height_same_fov_warped_to_ipf_frame.png`
  - `afm_scharr_normalmap_rot90.png`
  - `afm_scharr_normalmap_rot90_with_colorbar.png`
  - `afm_normalmap_same_fov_warped_to_raw_sem.png`
  - `afm_normalmap_same_fov_warped_to_ipf_frame.png`
  - `pt_highres60_afm_same_fov_alignment_data.npz`
  - `pt_highres60_afm_same_fov_alignment_metadata.json`

### 18. AFM 法向量与 EBSD 表面晶面指数图

- `afm_ebsd_surface_index.py` 用于把已经配准好的 AFM 高度场转换为表面法向量，并把这些法向量与 EBSD orientation 结合，得到样品表面法向在晶体坐标系中的 `{hkl}` / surface-index 数据。
- 这个流程参考 Brüning et al. 2023 的 AFM+EBSD facet type 思路：
  - 从 AFM `HeightRetrace` 输入原始 depthmap；
  - 默认不做 plane leveling 和 smoothing，直接用 Scharr 算子计算 `dz/dx`、`dz/dy`；
  - 在 AFM 原始像素坐标中构造 `[-dz/dx, -dz/dy, 1]` normalmap；
  - AFM->SEM affine 只取 polar decomposition 得到的平面旋转部分，用于把 normalmap 转到 EBSD/IPF top-view frame；affine 的 scale/shear 不再进入高度梯度；
  - 计算 sample-frame surface normal；
  - 根据 AFM->SEM->EBSD mapping 找到每个 AFM 像素对应的 EBSD orientation；
  - 用 Pt-3 90° finetuned orientation residual 后的 `G` 把 sample normal 转到 crystal frame；
  - 对 Pt/fcc cubic symmetry 做 `abs + sort` 折叠，得到 `{hkl}` 等价 surface-index direction。
- 法向量颜色：
  - sample-frame normal direction map 使用 HSV：azimuth 决定 hue，tilt 决定 saturation；
  - crystal-frame surface-index map 使用 `{100}=red`、`{110}=green`、`{111}=blue` 的 cubic facet colour key。
- 当前本机结果：
  - 默认 AFM 输入为用户新给出的 `C:\Users\WHJ\OneDrive\xwechat_files\wxid_udhlesdsllnu22_8cd9\msg\file\2026-07\Pt-1(1).ibw`；
  - AFM 有效配准区域中 `1,035,695` 个像素有 EBSD orientation，对应 AFM 总像素的 `0.988`；
  - 默认应用 Pt-3 90° finetuned orientation residual `(-0.25, -0.35, -0.20) deg`；
  - Scharr normalmap 输出保存在 `outputs/pt_afm_ebsd_scharr_surface_index/`；
  - 选取一个 AFM/EBSD 有效点，读取同 index 的 UP2 Kikuchi，并叠加 H5/OHP bands，用于检查 EBSD orientation 与 AFM normal 的晶体学耦合；OHP band 使用 `normal_theta_rho+_yup` 约定；
  - 输出完整 `.npz` 数据，同时导出 stride=4 的 CSV 和 PLY 点云，便于后续统计、3D 可视化或导入外部软件。
- 输出：
  - `afm_height_nm.png`
  - `afm_scharr_normalmap.png`
  - `afm_scharr_normalmap_with_colorbar.png`
  - `afm_scharr_dz_dcol_um_per_um.png`
  - `afm_scharr_dz_drow_um_per_um.png`
  - `afm_normal_tilt_deg.png`
  - `afm_normal_azimuth_deg.png`
  - `afm_crystal_surface_index_color.png`
  - `afm_normals_surface_index_overview.png`
  - `ebsd_afm_surface_index_top_view.png`
  - `ebsd_ipf_top_view_sem_frame.png`
  - `surface_index_top_view_ebsd_frame.png`
  - `afm_surface_normals_3d.png`
  - `afm_surface_index_3d.png`
  - `afm_surface_index_3d_interactive.html`
  - `kikuchi_ebsd_afm_surface_index_coupling.png`
  - `facet_type_color_key.png`
  - `afm_ebsd_surface_index_data.npz`
  - `afm_ebsd_surface_index_point_cloud_stride4.csv`
  - `afm_ebsd_surface_index_point_cloud_stride4.ply`
  - `nearest_hkl_counts.csv`
  - `afm_ebsd_surface_index_metadata.json`

主要代码：

- `project_edax_oim_to_sphere.py`
- `diagnose_edax_transform_chain.py`
- `single_kikuchi_pc_finetune.py`
- `batch_pt_kikuchi_spherical_calibration.py`
- `joint_pc_orientation_explainability.py`
- `stable_global_pc_orientation_calibration.py`
- `pt3_same_face_spherical_calibration.py`
- `pt_highres_30deg_lightglue_calibration.py`
- `align_pt_afm_sem_ipf.py`
- `align_pt_highres60_afm.py`
- `afm_ebsd_surface_index.py`
- `visualize_edax_projection_sets.py`
- `visualize_edax_match_3d.py`
- `visualize_scan_position_pc_correction.py`
- `score_scan_position_pc_correction.py`
- `simulate_111_tilt_kikuchi_patterns.py`
- `annotate_simulated_ipf_points.py`
- `simulate_123_detector_sample_tilt_cases.py`
- `classify_pt1_ebsd_data.py`
- `export_pt1_sem_kikuchi_pairs.py`
- `export_h5_mapping_sem_correspondence.py`
- `export_up2_ebsd_kikuchi_correspondence.py`
- `align_pt1_inplane_sem_common_circle.py`
- `preview_gltf_pyvista.py`

## 运行示例

本机当前数据路径示例：

```powershell
D:\anaconda3\envs\torch\python.exe .\project_edax_oim_to_sphere.py `
  --h5 D:\project\EBSD2026\ebsd.edaxh5 `
  --up2 "C:\Users\WHJ\Desktop\kikuchi-super resolution\20260512_Cu_Area 1_OIM Map 1.up2" `
  --map-group "/20260512/Cu/Area 1/OIM Map 1HighR" `
  --pattern-index 2661 `
  --master D:\anaconda3\envs\torch\Lib\site-packages\kikuchipy\data\emsoft_ebsd_master_pattern\ni_mc_mp_20kv_uint8_gzip_opts9.h5 `
  --output outputs\github_edax_visualizations\area1_high_idx2661_projection.png
```

```powershell
D:\anaconda3\envs\torch\python.exe .\diagnose_edax_transform_chain.py `
  --h5 D:\project\EBSD2026\ebsd.edaxh5 `
  --up2 "C:\Users\WHJ\Desktop\kikuchi-super resolution\20260512_Cu_Area 1_OIM Map 1.up2" `
  --map-group "/20260512/Cu/Area 1/OIM Map 1HighR" `
  --pattern-index 2661 `
  --master D:\anaconda3\envs\torch\Lib\site-packages\kikuchipy\data\emsoft_ebsd_master_pattern\ni_mc_mp_20kv_uint8_gzip_opts9.h5 `
  --output-dir outputs\github_edax_visualizations\diagnose_area1_high_idx2661
```

```powershell
D:\anaconda3\envs\torch\python.exe .\single_kikuchi_pc_finetune.py `
  --h5 D:\project\EBSD2026\ebsd.edaxh5 `
  --up2 "C:\Users\WHJ\Desktop\kikuchi-super resolution\20260512_Cu_Area 1_OIM Map 1.up2" `
  --map-group "/20260512/Cu/Area 1/OIM Map 1HighR" `
  --pattern-index 2661 `
  --output-dir outputs\single_kikuchi_pc_finetune
```

```powershell
D:\anaconda3\envs\torch\python.exe .\pt3_same_face_spherical_calibration.py `
  --h5 E:\ZHL\EBSD-RAW\20251209Pt\20251209Pt.edaxh5 `
  --up2-root E:\ZHL\EBSD-RAW\20251209Pt `
  --output-dir outputs\pt3_same_face_spherical_calibration `
  --mask-radius-fraction 0.40 `
  --orientation-mode reference_variant
```

```powershell
D:\anaconda3\envs\torch\python.exe .\simulate_111_tilt_kikuchi_patterns.py `
  --output-dir outputs\simulated_111_tilt_patterns `
  --height 480 `
  --width 480 `
  --tilts -10 -5 0 5 10 `
  --mode corrected
```

```powershell
D:\anaconda3\envs\torch\python.exe .\simulate_111_tilt_kikuchi_patterns.py `
  --output-dir outputs\simulated_111_tilt_pc050_circular_transparent `
  --height 1024 `
  --width 1024 `
  --pc 0.5 0.5 0.5 `
  --tilts -10 -5 0 5 10 `
  --mode corrected `
  --circular-transparent
```

```powershell
D:\anaconda3\envs\torch\python.exe .\simulate_111_tilt_kikuchi_patterns.py `
  --output-dir outputs\simulated_123_tilt_pc050_circular_transparent `
  --height 1024 `
  --width 1024 `
  --zone 1 2 3 `
  --pc 0.5 0.5 0.5 `
  --tilts 0 -2 2 5 `
  --mode corrected `
  --circular-transparent
```

```powershell
D:\anaconda3\envs\torch\python.exe .\simulate_111_tilt_kikuchi_patterns.py `
  --output-dir outputs\simulated_135_pcx_sweep_circular_transparent `
  --height 1024 `
  --width 1024 `
  --zone 1 3 5 `
  --pc 0.5 0.5 0.5 `
  --tilts 0 `
  --pc-x-values 0.4 0.45 0.5 0.55 `
  --mode corrected `
  --circular-transparent
```

```powershell
D:\anaconda3\envs\torch\python.exe .\annotate_simulated_ipf_points.py `
  --output-dir outputs\ipf_annotations
```

```powershell
D:\anaconda3\envs\torch\python.exe .\simulate_123_detector_sample_tilt_cases.py
```

```powershell
D:\anaconda3\envs\torch\python.exe .\classify_pt1_ebsd_data.py `
  --data-dir D:\EBSD-data\Pt-1 `
  --output-dir outputs\pt1_classification
```

```powershell
D:\anaconda3\envs\torch\python.exe .\export_pt1_sem_kikuchi_pairs.py `
  --h5 D:\EBSD-data\Pt-1\20251209Pt.edaxh5 `
  --classification-csv outputs\pt1_classification\pt1_match_classification.csv `
  --output-dir outputs\pt1_sem_kikuchi_pairs
```

```powershell
D:\anaconda3\envs\torch\python.exe .\export_h5_mapping_sem_correspondence.py `
  --h5 D:\EBSD-data\Pt-1\20251209Pt.edaxh5 `
  --output-dir outputs\h5_mapping_sem_correspondence
```

```powershell
D:\anaconda3\envs\torch\python.exe .\export_up2_ebsd_kikuchi_correspondence.py `
  --h5 E:\ZHL\EBSD-RAW\20251209Pt\20251209Pt.edaxh5 `
  --up2-root E:\ZHL\EBSD-RAW\20251209Pt `
  --output-dir outputs\up2_ebsd_kikuchi_correspondence_E_20251209Pt
```

```powershell
D:\anaconda3\envs\torch\python.exe .\align_pt1_inplane_sem_common_circle.py `
  --h5 E:\ZHL\EBSD-RAW\20251209Pt\20251209Pt.edaxh5 `
  --up2-root E:\ZHL\EBSD-RAW\20251209Pt `
  --output-dir outputs\pt1_inplane_sem_common_circle_E_20251209Pt
```

```powershell
D:\anaconda3\envs\torch\python.exe .\pt_highres_30deg_lightglue_calibration.py `
  --h5 "E:\ZHL\EBSD-RAW\20251217Pt-high resolution\20251217.edaxh5" `
  --up2-root "E:\ZHL\EBSD-RAW\20251217Pt-high resolution" `
  --output-dir outputs\pt_highres_30deg_lightglue_calibration
```

```powershell
D:\anaconda3\envs\torch\python.exe .\export_pt_highres_data_overview.py `
  --h5 "E:\ZHL\EBSD-RAW\20251217Pt-high resolution\20251217.edaxh5" `
  --up2-root "E:\ZHL\EBSD-RAW\20251217Pt-high resolution" `
  --calibration-dir outputs\pt_highres_30deg_lightglue_calibration `
  --output-dir outputs\pt_highres_data_overview
```

```powershell
D:\anaconda3\envs\torch\python.exe .\align_pt_highres60_afm.py `
  --afm "D:\EBSD project\3d数据\pt-afm\Pt-2high resolution.ibw" `
  --h5 "E:\ZHL\EBSD-RAW\20251217Pt-high resolution\20251217.edaxh5" `
  --angle 60 `
  --edax-ipf "E:\ZHL\20251209Pt-EBSD MAP\pt-high resolution\60.bmp" `
  --output-dir outputs\pt_highres60_afm_alignment
```

```powershell
D:\anaconda3\envs\torch\python.exe .\batch_pt_kikuchi_spherical_calibration.py `
  --h5 D:\EBSD-data\Pt-1\20251209Pt.edaxh5 `
  --up2-root D:\EBSD-data `
  --specimen Pt-3 `
  --patterns-per-map 1 `
  --output-dir outputs\pt_batch_kikuchi_spherical_calibration
```

```powershell
D:\anaconda3\envs\torch\python.exe .\stable_global_pc_orientation_calibration.py `
  --h5 D:\EBSD-data\Pt-1\20251209Pt.edaxh5 `
  --up2-root D:\EBSD-data `
  --specimen Pt-3 `
  --area "Area 3-90" `
  --pattern-count 5 `
  --output-dir outputs\pt_stable_global_pc_orientation
```

```powershell
D:\anaconda3\envs\torch\python.exe .\align_pt_afm_sem_ipf.py `
  --afm "D:\EBSD project\3d数据\pt-afm\Pt-1.ibw" `
  --sem E:\ZHL\20251209Pt-EBSD\2-90bse.tif `
  --ipf outputs\pt3_area90_finetuned_ipf_map\pt3_area90_finetuned_ipf_clean_714x550.png `
  --output-dir outputs\pt_afm_sem_ipf_alignment
```

```powershell
D:\anaconda3\envs\torch\python.exe .\afm_ebsd_surface_index.py `
  --afm "C:\Users\WHJ\OneDrive\xwechat_files\wxid_udhlesdsllnu22_8cd9\msg\file\2026-07\Pt-1(1).ibw" `
  --alignment-metadata outputs\pt_afm_sem_ipf_alignment\afm_sem_ipf_alignment_metadata.json `
  --finetuned-ipf-metadata outputs\pt3_area90_finetuned_ipf_map\pt3_area90_finetuned_ipf_metadata.json `
  --h5 "D:\EBSD project\EBSD-data\Pt-1\20251209Pt.edaxh5" `
  --h5-group "20251209/Pt-3/Area 3-90/OIM Map 1" `
  --up2 "D:\EBSD project\EBSD-data\Pt-1\20251209_Pt-3_Area 4_OIM Map 1.up2" `
  --output-dir outputs\pt_afm_ebsd_scharr_surface_index
```

## 版本改动

### 2026-07-22

- 新增 `export_pt_highres_data_overview.py`，用于把 Pt high-resolution 12 组 30° EBSD 数据的 H5/UP2/index/SEM/IPF/Kikuchi/OHP/PC/score 统一导出成总览图和索引表。
- 该入口只做可视化和质检，不重新运行 LightGlue 配准或球面匹配；high-res calibration 本身已在修正 UP2 映射后重新运行。
- 修正通用 IPF-Z 可视化约束：默认输出 EDAX/OIM-style orientation IPF，不再用 CI/IQ 权重压暗，也不按 phase label 清黑；Pt high-resolution 总览现在会读取软件导出的 `0.bmp, 30.bmp, ..., 330.bmp`，将 H5 orientation IPF 重采样到同一物理显示尺寸，并输出软件-vs-H5 对比图和 chromaticity error 表。
- 修正 Pt high-resolution 的 H5/UP2 对应关系：正式 12 组 UP2 为 `Area 3` 到 `Area 14`，每组 `470 x 470 x 580320`，但对应角度按真实采集顺序为 `30°, 60°, ..., 330°, 0°`，不是 `0°, 30°, ...`。
- `export_pt_highres_data_overview.py` 新增 `pt_highres_ohp_overlay_diagnostics.csv`，每次输出 OHP overlay 时同步记录 OHP 线在 band-enhanced Kikuchi 上的响应，避免 H5/OHP 与 UP2 pattern 错位时只靠肉眼发现。
- 新增 `align_pt_highres60_afm.py`，把 `Pt-2high resolution.ibw` 配准到 Pt high-resolution 60° EBSD 的 H5 SEM/IPF frame，并单独输出 AFM height、amplitude、Scharr normalmap、normalmap color key、LightGlue matches、候选 overlay 预览和 AFM->EBSD60 overlay。
- 修正 Pt high-resolution 60° AFM/SEM/IPF 对齐：旧版误用 SEM/IPF scale bar 和 EBSD 物理宽度，把 AFM 缩成小块。新版 `align_pt_highres60_afm_same_fov.py` 使用 AFM/SEM 同视场先验，先把 AFM `rot90` 到软件显示方向，再与 H5 SEM raw row order 做中心旋转约 10° 的同视场仿射；只有进入 EDAX IPF frame 时才 `flipud`。

### 2026-07-21

- 新增 `afm_ebsd_surface_index.py`，将 Pt AFM 高度场转换为 sample-frame surface normal，并结合 Pt-3 90° EBSD orientation 得到 crystal-frame surface-index / `{hkl}` 数据。
- 该流程参考 Brüning et al. 2023 的 AFM+EBSD facet type 方法：AFM 法向量经 AFM->SEM affine 映射到 EBSD top-view frame，再用 EBSD orientation 转到晶体坐标系，最后按 cubic symmetry 折叠并用 `{100}=red, {110}=green, {111}=blue` 编码。
- AFM+EBSD surface-index 当前推荐输出保存在 `outputs/pt_afm_ebsd_scharr_surface_index/`：有效 AFM+EBSD 像素为 `1,035,695`，占 AFM 总像素 `0.988`，并输出完整 `.npz`、stride=4 CSV/PLY 点云、Scharr normalmap、3D surface-index 图和 EBSD top-view overlay。
- 修正 AFM 法向量计算：新版 `afm_ebsd_surface_index.py` 默认读取 `Pt-1(1).ibw`，直接对 `HeightRetrace` depthmap 使用 Scharr 算子提取 normalmap；AFM->SEM affine 只提供平面旋转，不再把 affine scale/shear 混入高度梯度。
- Scharr normalmap 版本输出保存在 `outputs/pt_afm_ebsd_scharr_surface_index/`，新增 `afm_scharr_normalmap.png`，并在 `.npz` 中保存 `normals_afm`、`scharr_dz_dcol`、`scharr_dz_drow`。
- 扩展 AFM+EBSD surface-index 可视化：每个中间结果单独输出图片，normalmap 新增方向 color key，surface-index 新增可旋转 Plotly 3D HTML，并新增 `kikuchi_ebsd_afm_surface_index_coupling.png` 用同 index 的 UP2 Kikuchi + H5/OHP bands 解释 `EBSD orientation + AFM normal -> crystal surface index` 的耦合关系。
- 修正 `kikuchi_ebsd_afm_surface_index_coupling.png` 中的 OHP band 叠加：之前误把 `Maximum Rho Fraction` 再次乘入 `rho`，导致 Hough 线向中心收缩并与 Kikuchi band 明显不重合；现在与 `export_publication_h5_kikuchi_bands.py` 一致，使用 `rho_px=(rho_bin-circle_size/2)*detector_diameter/circle_size` 和 `normal_theta_rho+_yup`。
- 新增 `align_pt_afm_sem_ipf.py`，读取 Pt AFM `Pt-1.ibw`，用 LightGlue/SuperPoint + RANSAC full affine 配准到外部 SEM/BSE `2-90bse.tif`，再通过 Pt-3 90° 的固定 SEM/IPF 对应关系确认 AFM 与 EBSD/IPF map 的位置关系。
- AFM IBW 当前读出 `1024 x 1024 x 4`，扫描尺寸 `18 um`，通道为 `HeightRetrace / AmplitudeRetrace / PhaseRetrace / ZSensorRetrace`；脚本自动裁 SEM 底栏、生成多通道 high-pass/edge 特征并选择尺度合理的 affine 候选。
- 本次 AFM->SEM->IPF 输出保存在 `outputs/pt_afm_sem_ipf_alignment/`：最佳候选 `ZSensorRetrace_hp -> sem_hp`，`11/23` inliers，RMSE `4.70 px`，三晶界位置与 Pt-3 90° IPF/SEM 对齐。
- 新增 `batch_pt_kikuchi_spherical_calibration.py`，把当前效果最好的单张 Kikuchi 球面匹配校准方案固定成可复用批处理流程。
- 默认在 `D:\EBSD-data\Pt-1\20251209Pt.edaxh5` 和 `D:\EBSD-data` 中自动匹配 Pt-3 的 H5/UP2 数据，并从每个 matched mapping 中挑选 high-IQ/high-CI Kikuchi。
- 批处理流程复用 `single_kikuchi_pc_finetune.py` 的背景扣除、CLAHE、H5 orientation 投影和 PC scoring；批处理层新增完整 disk mask、scan-position PC 初值和 residual orientation finetune，不做 cubic symmetry 轴线摆放。
- 已在本机 Pt-3 数据上跑通 5 张 Kikuchi：`Area 3-0 idx=9879`、`Area 3-90 idx=6088`、`Area 3-180 idx=75009`、`Area 3-270 idx=18635`、`Area 3-360 idx=68376`。5 张均自动选择 `edax_g_direct_row_major`，PC+orientation score 均高于 map-PC score。
- 修正 Pt 批处理默认 mask：由偏小的保守圆改为完整 centered Kikuchi disk，默认半径 `0.49 * min(H, W)`，本机 470 x 470 Pt pattern 为 `(cx=234, cy=234, r=230)`。
- 更新 Pt 批处理 finetune 顺序：先用 scan position 对 PC 做确定性初值校正，再做残余 PC finetune，最后在 refined PC 固定后做小角度 residual orientation finetune。
- 将 residual orientation 默认搜索改为高精度小范围：每轴 `-0.5°` 到 `+0.5°`、步长 `0.05°`，避免 orientation 用过大的自由度代偿 PC 误差。
- 新增 PC/orientation 残差顺序诊断：同时比较 `scan PC -> PC residual -> orientation residual` 与 `scan PC -> orientation residual -> PC residual`，并输出 `*_pc_orientation_order_comparison.png`。
- 新增 `*_position_pc_orientation_finetune.png` 和 `orientation_finetune_trace.csv`，可视化 map PC、scan-position PC、residual PC、PC+orientation 四个阶段的球面位置和 score 变化。
- 新增 `joint_pc_orientation_explainability.py`，把 PC/orientation 残差放入同一个联合诊断框架中，并引入 H5/OHP Kikuchi band 的 center/profile/width 约束。
- 该脚本默认在 Pt-3 `Area 3-90 / OIM Map 1` 中选择 3 张 high-IQ/high-CI Kikuchi，输出 scan、PC-only、orientation-only、joint 四种候选，以及 PC-like / orientation-like evidence。
- 当前默认结果显示 3 张 joint objective 均提升，H5/OHP band center error 明显下降，但 band width error 没有同步下降，因此 residual interpretation 均为 `orientation_dominant_pc_not_supported_by_width`。
- 新增 `stable_global_pc_orientation_calibration.py`，作为当前更推荐的简化主流程：同一 mapping 多张 Kikuchi 共享一个 global PC residual，固定稳定 PC 后再做每张 pattern 的 orientation residual。
- 该脚本不使用 band width/profile loss，只使用原来验证过的 preprocessed intensity + band-enhanced detector-space NCC，并通过 global PC prior 保持 PC 稳定。
- 已在 Pt-3 `Area 3-90 / OIM Map 1` 默认 5 张 Kikuchi 上跑通：global PC residual 为 `(-0.002, +0.002, -0.008)`，5 张均满足 `scan PC score < global PC score < PC+orientation score`。
- 新增本地输出：
  - `outputs/pt_batch_kikuchi_spherical_calibration/pt_kikuchi_spherical_calibration_contact_sheet.png`
  - `outputs/pt_batch_kikuchi_spherical_calibration/pt_kikuchi_spherical_calibration_summary.csv`
  - `outputs/pt_batch_kikuchi_spherical_calibration/per_pattern/*/*_spherical_calibration_overview.png`
  - `outputs/pt_batch_kikuchi_spherical_calibration/per_pattern/*/*_position_pc_orientation_finetune.png`
  - `outputs/pt_batch_kikuchi_spherical_calibration/per_pattern/*/*_pc_orientation_order_comparison.png`
  - `outputs/pt_batch_kikuchi_spherical_calibration/per_pattern/*/orientation_finetune_trace.csv`
  - `outputs/pt_joint_pc_orientation_explainability/joint_pc_orientation_explainability_contact_sheet.png`
  - `outputs/pt_joint_pc_orientation_explainability/joint_pc_orientation_explainability_summary.csv`
  - `outputs/pt_joint_pc_orientation_explainability/per_pattern/*/*_joint_pc_orientation_explainability.png`
  - `outputs/pt_stable_global_pc_orientation/stable_global_pc_diagnostic.png`
  - `outputs/pt_stable_global_pc_orientation/stable_global_pc_orientation_contact_sheet.png`
  - `outputs/pt_stable_global_pc_orientation/stable_global_pc_orientation_summary.csv`
  - `outputs/pt_stable_global_pc_orientation/per_pattern/*/*_stable_global_pc_orientation.png`

### 2026-07-06

- 修正 `pt_highres_30deg_lightglue_calibration.py` 的球面匹配输出：以前的真正匹配判断是 detector-space forward validation，即把 master sphere 按 PC + orientation 采样回 detector 像素并与实验 pattern 计算 NCC/残差。
- 将 12 组 Pt high-resolution 流程拆成两条分支：主 `matching_results` 保持 H5 orientation + PC finetune，不再被 cubic symmetry 改写；`symmetry_results` 是克隆后的 cubic symmetry axis diagnostic。
- 新增 `pt_highres_forward_detector_validation.png` 和 `pt_highres_forward_detector_validation_scores.csv`，同时比较 base/refined placement 与 cubic diagnostic placement。当前结果显示 5/12 张 cubic placement 的 detector-space 分数变差，120° 下降最明显，所以 cubic branch 不能当作最终匹配输出。
- 新增主匹配高清输出 `pt_highres_sphere_matching_front_views.png` 和 `pt_highres_sphere_match_front_pattern_000.png` 到 `pt_highres_sphere_match_front_pattern_330.png`；保留 `pt_highres_cubic_symmetry_*` 作为单独诊断输出。
- `pt_highres_30deg_lightglue_calibration.py` 保留 Pt-3 的 cubic symmetry axis placement 思路，但现在只在克隆分支中作为诊断路线使用。
- 该流程保留 LightGlue/SuperPoint 的 12 组 SEM 对齐和同一物理点选择；每张 Kikuchi 先单独完成 EDAX PC + H5 orientation + PC finetune，再在诊断分支中用 cubic symmetry 选择等价晶体分支。
- 移除默认的 fixed-placement residual 路线和 match-preserving 连续旋转优化，避免和 Pt-3 固定流程混在一起。
- 新增主匹配和 cubic 诊断两套高清正视球面图，采用 4e9e21b 中的 front-view rasterization 方式。
- 已用 12 组 Pt high-resolution 数据跑通：共同参考点 `(342.07, 275.83)`，12 张图均有对应 raw EBSD index，当前 cubic symmetry placement 的 `angle30=22.82°`。
- 新增本地输出：
  - `outputs/pt_highres_30deg_lightglue_calibration/pt_highres_sphere_matching_front_views.png`
  - `outputs/pt_highres_30deg_lightglue_calibration/pt_highres_sphere_match_front_pattern_000.png` 到 `pt_highres_sphere_match_front_pattern_330.png`
  - `outputs/pt_highres_30deg_lightglue_calibration/pt_highres_forward_detector_validation.png`
  - `outputs/pt_highres_30deg_lightglue_calibration/pt_highres_forward_detector_validation_scores.csv`
  - `outputs/pt_highres_30deg_lightglue_calibration/pt_highres_cubic_symmetry_front_views.png`
  - `outputs/pt_highres_30deg_lightglue_calibration/pt_highres_cubic_front_pattern_000.png` 到 `pt_highres_cubic_front_pattern_330.png`

### 2026-07-04

- 新增 `pt_highres_30deg_lightglue_calibration.py`，用于 Pt high-resolution 十二组 30° in-plane EBSD 的 SEM 相邻配准、同一物理点选取、Kikuchi 球面标定和 PC 锚点可视化。
- 新脚本优先使用 LightGlue + SuperPoint；本机已安装 `lightglue`，并确认 `torch 2.12.0.dev20260408+cu128` 能使用 RTX 5090。
- 当前运行输出保存在 `outputs/pt_highres_30deg_lightglue_calibration/`，仅作为本地结果，不提交 GitHub。
- 当前诊断结论：SEM align 质量较好，但 H5 orientation 经过 cubic symmetry 选择后得到的最优 `Q30` 为约 `22.82°`，与物理 30° 存在差异，后续应重点检查 orientation/PC/坐标链在标准球面上的几何解释。

### 2026-07-02

- 更新 `pt3_same_face_spherical_calibration.py`，在 Pt-3 四组同晶面 Kikuchi 流程中显式标注每张 pattern 的 PC 位置。
- 新增两类 PC 可视化：
  - 在 raw/preprocessed Kikuchi pattern 上同时画出 H5/EDAX PC 和 refined PC。
  - 把 PC 点投影到 cubic-symmetry-corrected 的同一个 master sphere 上，和四张 Kikuchi patch 的最终落点一起显示。
- `pt3_same_face_spherical_calibration_summary.csv` 新增 PC 像素坐标和 PC 球面方向列，方便判断 PC 漂移是否对应四组 in-plane mapping 的晶体学锚点变化。
- 已生成本地输出：
  - `outputs/pt3_same_face_spherical_calibration/pt3_pc_positions_on_patterns.png`
  - `outputs/pt3_same_face_spherical_calibration/pt3_pc_positions_on_same_sphere.png`
  - `outputs/pt3_same_face_spherical_calibration/pt3_pc_positions_same_sphere_axis_view.png`
  - `outputs/pt3_same_face_spherical_calibration/pt3_pc_positions_on_same_sphere_lon_colat.png`

### 2026-06-30

- 新增 `single_kikuchi_pc_finetune.py`，用于单张 Kikuchi pattern 的端到端验证：图像预处理、EDAX PC 球面标定、orientation 投影到 master sphere、局部 PC finetune 和可视化。
- 默认样例使用 Cu Area 1 HighR 的 `pattern-index=2661`。
- 当前 PC finetune 目标函数为实验 pattern 与 master sphere 采样值的相关性，综合 corrected intensity 与 band-enhanced response。
- 已生成本地输出：
  - `outputs/single_kikuchi_pc_finetune/single_kikuchi_pc_finetune_overview.png`
  - `outputs/single_kikuchi_pc_finetune/single_kikuchi_pc_finetune_summary.csv`
  - `outputs/single_kikuchi_pc_finetune/orientation_scores.csv`
  - `outputs/single_kikuchi_pc_finetune/pc_finetune_scores.csv`
- 新增 `pt3_same_face_spherical_calibration.py`，用于 Pt-3 四组 in-plane EBSD 的同一晶面 Kikuchi 球形标定。
- Pt-3 脚本强制使用保守圆形 Kikuchi mask，并在 mask 内做背景扣除和衬度增强；默认 mask 半径为 `0.40 * min(H, W)`，用于避免边缘缺口进入评分和球面投影。
- Pt-3 脚本不再直接调整/覆盖 orientation；现在先用每张 H5 orientation 定位，再用 cubic symmetry 选择等价 master-sphere 落点，使四张 pattern 尽量满足同一球面轴线的旋转闭包。
- 已生成本地输出：
  - `outputs/pt3_same_face_spherical_calibration/pt3_same_face_roi_selection.png`
  - `outputs/pt3_same_face_spherical_calibration/pt3_same_face_spherical_calibration_workflow.png`
  - `outputs/pt3_same_face_spherical_calibration/pt3_same_face_3d_kikuchi_sphere.png`
  - `outputs/pt3_same_face_spherical_calibration/pt3_clear_final_spherical_kikuchi_maps.png`
  - `outputs/pt3_same_face_spherical_calibration/pt3_clear_3d_front_facing_kikuchi_spheres.png`
  - `outputs/pt3_same_face_spherical_calibration/pt3_same_sphere_axis_aligned_kikuchi_patterns.png`
  - `outputs/pt3_same_face_spherical_calibration/pt3_same_sphere_axis_view.png`
  - `outputs/pt3_same_face_spherical_calibration/pt3_same_sphere_reference_patch_view.png`
  - `outputs/pt3_same_face_spherical_calibration/pt3_same_sphere_oblique_axis_view.png`
  - `outputs/pt3_same_face_spherical_calibration/pt3_same_face_spherical_calibration_summary.csv`
  - `outputs/pt3_same_face_spherical_calibration/pt3_cubic_symmetry_axis_prior_summary.csv`

### 2026-06-29

- 从 GitHub `origin/main` 检查更新，本地代码已是最新版本。
- 新增 `classify_pt1_ebsd_data.py`，用于读取 `D:\EBSD-data\Pt-1` 的 EDAX H5/UP2 元数据并分类。
- 新增 `export_pt1_sem_kikuchi_pairs.py`，用于读取每个 Pt-1 EBSD map 对应的 SEM，并导出一张同 index 的 raw Kikuchi pattern。
- 新增 `export_h5_mapping_sem_correspondence.py`，用于导出 H5 中全部 OIM mapping 的 SEM，并和本机/回收站中可找到的 UP2 原始文件名做对应。
- 新增 `export_up2_ebsd_kikuchi_correspondence.py`，用于逐个 UP2 建立 EBSD mapping 与 Kikuchi 示例对应。
- 新增 `align_pt1_inplane_sem_common_circle.py`，用于对 Pt-1 四组 in-plane EBSD 的 SEM 做对齐，并计算最大公共圆形 ROI。
- 分类输出保存在 `outputs/pt1_classification/`，仅用于本地分析，不提交 GitHub。
- 当前 Pt-1 分类逻辑区分：
  - H5 与 UP2 能一一匹配的可靠分析组。
  - 只有 UP2 原始 pattern、未找到 Pt-1 H5 元数据对应关系的孤立组。

### 2026-06-23

- 新增 `PROJECT_CORE.md`，整理当前 EBSD 项目的核心目标、数据主线、核心脚本、推荐路线和 GitHub 提交规则。
- 新增 `export_h5_ipf_bse_maps.py`，用于从 EDAX H5 导出 IPF-Z、IQ/CI、SEM/BSE、FOV 和 montage，并改为命令行参数驱动，避免写死本机路径。
- 新增 `export_publication_h5_kikuchi_bands.py`，用于导出透明背景 Kikuchi pattern、H5/OHP band 和叠加图，便于论文图和报告图使用。
- 继续保持 GitHub 只提交代码和文档，不提交 H5/UP2 原始数据、输出图、缓存或模型权重。

### 2026-06-22

- 扩展 `simulate_111_tilt_kikuchi_patterns.py`，支持任意 zone axis：`--zone H K L`。
- 新增横向 PC 漂移模拟参数：`--pc-x-values`。
- 已生成 `(1,2,3)` zone axis 的倾斜序列，`PC=(0.5,0.5,0.5)`，倾斜角 `0,-2,2,5`：
  - `outputs/simulated_123_tilt_pc050_circular_transparent/individual/*.png`
- 已生成 `(1,3,5)` zone axis 的横向 beam shift / PCx 漂移序列，倾斜角 0 deg，`PCx=0.4,0.45,0.5,0.55`：
  - `outputs/simulated_135_pcx_sweep_circular_transparent/individual/*.png`
- 新增 `annotate_simulated_ipf_points.py`，用统一 `PC=(0.5,0.5,0.5)` 对两组模拟结果做 IPF 标注。
- 已生成两个透明背景反极图：
  - `outputs/ipf_annotations/ipf_zone_123_tilt_points_transparent.png`
  - `outputs/ipf_annotations/ipf_zone_135_pcx_points_transparent.png`
- 新增 `simulate_123_detector_sample_tilt_cases.py`，生成 `(1,2,3)` 的四个探测器/样品倾斜分离工况。
- 已生成四张新的透明 Kikuchi pattern：
  - `outputs/simulated_123_detector_sample_tilt_cases/individual/*.png`
- 已生成黑圈/红圈标注的透明 IPF：
  - `outputs/ipf_annotations/ipf_zone_123_detector_sample_tilt_points_transparent.png`

### 2026-06-19

- 新增 `simulate_111_tilt_kikuchi_patterns.py`。
- 支持固定 EDAX PC 下的 [111] 单晶上下倾斜模拟。
- 当前默认 PC 从 H5 中读取：`(0.528627, 0.592593, 0.615038)`。
- 已生成两组本地输出：
  - `outputs/simulated_111_tilt_patterns/simulated_111_tilt_contact_sheet.png`
  - `outputs/simulated_111_tilt_patterns_raw/simulated_111_tilt_contact_sheet.png`
- 新增圆形透明 PNG 输出选项 `--circular-transparent`。
- 已用 `PC=(0.5, 0.5, 0.5)` 生成本地 EDAX 圆形透明输出：
  - `outputs/simulated_111_tilt_pc050_circular_transparent/individual/*.png`
  - `outputs/simulated_111_tilt_pc050_circular_transparent_raw/individual/*.png`

### 2026-06-08

- 从 GitHub `origin/main` 拉取新的 EBSD 代码，新增：
  - `project_edax_oim_to_sphere.py`
  - `diagnose_edax_transform_chain.py`
  - `visualize_edax_match_3d.py`
  - `visualize_edax_projection_sets.py`
  - `visualize_scan_position_pc_correction.py`
  - `score_scan_position_pc_correction.py`
- 删除旧的球面匹配和优化实验代码，包括 weighted sphere matching、Radon/Hough peak graph、DIP、PC/radius refinement、closed-loop refinement、geometry-only 旧实验等脚本。
- 将新脚本中的默认数据路径改为本机当前路径：`D:\project\EBSD2026\ebsd.edaxh5`、桌面 UP2 数据目录和 `D:\anaconda3\envs\torch` 下的 master pattern。
- 已生成基础可视化：
  - `outputs/github_edax_visualizations/area1_high_idx2661_projection.png`
  - `outputs/github_edax_visualizations/area2_high_idx19802_projection.png`
  - `outputs/github_edax_visualizations/diagnose_area1_high_idx2661/top_transform_candidates.png`
  - `outputs/github_edax_visualizations/diagnose_area1_high_idx2661/detector_only_spherical_geometry.png`
  - `outputs/github_edax_visualizations/edax_projection_sets/area1_highr_fixed_chain_lambert_band.png`
  - `outputs/github_edax_visualizations/edax_projection_sets/area2_highr_fixed_chain_lambert_band.png`
- README 改为描述当前 GitHub EDAX 几何投影代码和可视化入口。

## GitHub 上传规则

- 只提交代码和文档。
- 不提交 EBSD 原始数据、UP2/H5 数据、训练权重、TensorBoard 日志、可视化输出图、缓存文件。
- 每次修改算法或参数默认值时，同步更新本 README。
