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

### 6. 全局 DIC 式 PC/radius 校正原型

- 参考论文 `Global DIC-based sample-detector geometry refinement for accurate EBSD indexing` 的思路，新增多张 pattern 的全局残差场校正原型。
- 先用当前 weighted + H5 band 增强方法获得每张 pattern 的初始空间匹配。
- 将 master sphere 渲染回 detector plane，与实验 pattern 的 band / contrast 特征做局部 DIC 光流匹配。
- 将多张 pattern 的局部位移残差平均，用有限差分估计 `pcx / pcy / pcz radius scale` 的灵敏度。
- 使用阻尼最小二乘求全局 PC/radius 更新，并用 line search 判断是否接受更新。
- 当前结论：
  - 可信子集上可以得到小幅 PC/radius 修正，并降低平均 DIC 残差。
  - 混入明显错误初始匹配后，平均残差场会被污染，line search 会倾向于拒绝更新。
  - 该方法暂时作为实验原型保留，后续需要加入初始匹配质量筛选、DIC outlier rejection，以及可能的 detector/sample tilt 参数后，再作为默认校正方法。

主要代码：

- `global_dic_pc_refinement.py`

### 7. 可视化输出

- 输出 raw pattern / preprocessed pattern 在高分辨率 Kikuchi sphere 上的最终空间匹配结果。
- 输出 PC/radius 搜索的 score landscape。
- 输出 batch contact sheet，方便横向检查多组 pattern。
- 输出全局 DIC 平均残差场、PC/radius 灵敏度场、line search 结果和校正前后 contact sheet。
- 可视化输出默认写入 `outputs/`，不上传 GitHub。

## 版本改动

### 2026-05-23

- 新增 `global_dic_pc_refinement.py`，用于尝试论文中的全局 DIC 几何 refinement 思路。
- 新脚本支持从多张 pattern 建立平均 DIC 位移残差场，有限差分估计 `pcx / pcy / radius scale` 灵敏度，并用阻尼最小二乘 + line search 给出全局更新。
- 对 5 张相对可信的 Area1 pattern 测试时，平均 DIC 残差从 `8.313 px` 降到 `8.105 px`，对应修正约为 `dx=-1.33 px, dy=-2.06 px, radius_scale=0.9931`。
- 对 10 张 linspace 样本测试时，混入错误初始匹配会导致 line search 选择不更新，说明后续必须先做可靠匹配筛选和 outlier rejection。
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
