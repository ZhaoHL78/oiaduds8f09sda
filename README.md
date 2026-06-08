# EBSD2026 EDAX Projection Tools

本仓库当前保留一条更干净的 EDAX/OIM 几何投影路线：从 EDAX H5 和 UP2 中读取原始 Kikuchi pattern、PC、实验几何和 OIM orientation，再把实验 pattern 放回标准 Kikuchi master sphere 上做可视化和诊断。

旧的球面匹配、Hough/Radon 图匹配、DIP、PC/radius refinement 和批量优化实验脚本已经删除。当前重点不是继续做自动匹配优化，而是先理解 GitHub 拉入的 EBSD 代码中每一步坐标变换的含义。

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

主要代码：

- `project_edax_oim_to_sphere.py`
- `diagnose_edax_transform_chain.py`
- `visualize_edax_projection_sets.py`
- `visualize_edax_match_3d.py`
- `visualize_scan_position_pc_correction.py`
- `score_scan_position_pc_correction.py`
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

## 版本改动

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
