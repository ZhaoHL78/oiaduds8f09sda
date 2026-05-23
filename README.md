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

主要代码：

- `h5_band_enhanced_match.py`
- `batch_final_spatial_visualizations.py`
- `visualize_calibration_pipeline.py`

### 5. Pattern center 与投影半径偏差校正

- 在 H5 pattern center 基础上搜索 `pcx / pcy` 像素偏移。
- 同时搜索 `pcz` 的 radius scale，用于模拟投影半径或 detector distance 的标定偏差。
- 支持单张 pattern 校正和 10 组 batch 校正。
- CPU 版与 GPU 版都保留；GPU 版使用 PyTorch CUDA 加速局部 PC/radius 搜索。
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

### 6. 可视化输出

- 输出 raw pattern / preprocessed pattern 在高分辨率 Kikuchi sphere 上的最终空间匹配结果。
- 输出 PC/radius 搜索的 score landscape。
- 输出 batch contact sheet，方便横向检查多组 pattern。
- 可视化输出默认写入 `outputs/`，不上传 GitHub。

## 版本改动

### 2026-05-23

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
