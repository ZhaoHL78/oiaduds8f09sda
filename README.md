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

主要代码：

- `project_edax_oim_to_sphere.py`
- `diagnose_edax_transform_chain.py`
- `visualize_edax_projection_sets.py`
- `visualize_edax_match_3d.py`
- `visualize_scan_position_pc_correction.py`
- `score_scan_position_pc_correction.py`
- `simulate_111_tilt_kikuchi_patterns.py`
- `annotate_simulated_ipf_points.py`
- `simulate_123_detector_sample_tilt_cases.py`
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

## 版本改动

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
