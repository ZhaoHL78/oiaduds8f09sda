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
  - UP2: `Area 3` 到 `Area 14`，对应 0° 到 330°。
- 使用 LightGlue + SuperPoint 对相邻角度 SEM 做配准：`30 -> 0`、`60 -> 30`、...、`330 -> 300`。
- 每一对配准先按已知 30° in-plane rotation 预旋转，再用 LightGlue/SuperPoint 匹配和 RANSAC affine refinement，最后把所有 SEM transform 串联到 0° 坐标系。
- 在共同有效区域内自动选择一个远离晶界、IQ/CI 综合较高的同一物理位置点，并反投影到每个 EBSD mapping 的 raw SEM/grid index。
- 对这 12 个 index 读取对应 UP2 Kikuchi pattern，沿用之前流程：
  - 圆形 detector mask；
  - 背景扣除、CLAHE 衬度增强；
  - H5/EDAX PC + H5 orientation 投影到 master sphere；
  - 小范围 PC finetune；
  - cubic symmetry 等价落点选择，使 12 张 pattern 尽量满足共同 30° 轴线关系；
  - 单独绘制 PC 在 pattern 上的位置和 PC 在 master sphere 上的晶体学锚点。
- 当前本机运行结果中，SEM 相邻配准全部使用 `lightglue_superpoint`，没有使用 fallback；各对 RANSAC inlier 约 48-81 个，RMSE 约 2.0-2.6 px。
- 一个重要诊断结果：在 H5 orientation + cubic symmetry 等价落点下，最优共同轴的 `Q30` 约为 `22.82°`，不是理想 30°。这说明 SEM 物理旋转成立，但软件 orientation / PC / 坐标链映射到标准球面后仍存在需要继续解释的几何偏差。
- 输出：
  - `pt_highres_sem_lightglue_alignment_overview.png`
  - `pt_highres_same_point_selection.png`
  - `pt_highres_selected_kikuchi_pc_patterns.png`
  - `pt_highres_spherical_calibration_workflow.png`
  - `pt_highres_same_sphere_lon_colat.png`
  - `pt_highres_same_sphere_3d.png`
  - `pt_highres_pc_anchor_lon_colat.png`
  - `pt_highres_pc_anchor_3d.png`
  - `pt_highres_pair_alignments.csv`
  - `pt_highres_30deg_spherical_calibration_summary.csv`
  - `pt_highres_30deg_cubic_symmetry_axis_prior_summary.csv`
  - `pt_highres_sem_transforms_raw_to_angle0.npz`

主要代码：

- `project_edax_oim_to_sphere.py`
- `diagnose_edax_transform_chain.py`
- `single_kikuchi_pc_finetune.py`
- `pt3_same_face_spherical_calibration.py`
- `pt_highres_30deg_lightglue_calibration.py`
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

## 版本改动

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
