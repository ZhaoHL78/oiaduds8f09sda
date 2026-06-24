# EBSD Project Core Summary

本项目的核心目标是把 EDAX EBSD 采集数据中的单张 Kikuchi pattern、软件 indexing 结果、OHP/Hough 菊池带、pattern center 和实验几何统一到一个可审计的球面坐标链中，最终服务于 Kikuchi sphere 重建、取向校准和多张 EBSD mapping 的对应关系分析。

## 数据主线

- `EDAX H5`: 读取 EBSD map、`ANG/DATA` 软件 indexing、`OHP/DATA` 菊池带、pattern center、IQ/CI、SEM Signal 与 PRIAS/BSE 图像。
- `UP2`: 读取原始 Kikuchi pattern，保持 raw pattern 与预处理 pattern 分离。
- `master sphere`: 使用 kikuchipy/EMsoft master pattern 作为标准 Kikuchi sphere 参考。
- `outputs/`: 只保存本地可视化和中间结果，不提交 GitHub。

## 当前核心脚本

- `h5_band_enhanced_match.py`: 数据读取、pattern 预处理、OHP band 光栅化、detector-to-sphere 投影、master sphere 匹配的基础模块。
- `visualize_calibration_pipeline.py`: 输出预处理、球面投影和最终空间匹配的可视化。
- `pc_radius_bias_correction.py` / `batch_pc_radius_bias_correction.py` / `batch_pc_radius_bias_correction_gpu.py`: PC 与投影半径偏差搜索。
- `continuous_band_geometric_refinement.py`: 以 H5-band-enhanced 匹配为初值，连续优化旋转、半径和可选 PC。
- `hough_point_spatial_reconstruction.py`: 将 OHP band 转为 detector Hough 点后做几何匹配和三维 band 点还原。
- `spherical_hough_expansion_refinement.py`: 球面 Hough 点的膨胀/收缩迭代匹配。
- `spherical_radon_graph_pipeline.py`: 球面 Radon/Hough 峰图、三角候选、partial OT 和局部 refinement 原型。
- `direct_hkl_sphere_localization.py`: 直接用 H5 软件 orientation 与 phase HKL families 验证 OHP band 的 HKL 几何解释。
- `software_orientation_sphere_projection.py`: 不做搜索，直接用 PC 和软件 orientation 将 pattern 放到标准 master sphere。
- `geometry_only_pc_orientation_projection.py`: 几何-only 的干净坐标闭环基线。
- `closed_loop_crystal_frame_mapping.py`: `detector -> sample -> crystal/master` 的闭环 forward validation。
- `export_h5_ipf_bse_maps.py`: 从 H5 导出 IPF-Z、IQ/CI、SEM/BSE、FOV 和 montage。
- `export_publication_h5_kikuchi_bands.py`: 导出带透明背景的 Kikuchi pattern 与 H5/OHP band 叠加图，便于论文图和报告图使用。

## 当前推荐路线

1. 用 `export_h5_ipf_bse_maps.py` 检查 EBSD map 与 BSE/SEM 图像质量。
2. 用 `export_h5_band_examples.py` 或 `export_publication_h5_kikuchi_bands.py` 检查 OHP band 是否投影到真实 Kikuchi line 上。
3. 用 `h5_band_enhanced_match.py` 或 `visualize_calibration_pipeline.py` 做单张 pattern 的 baseline 球面匹配。
4. 用 `batch_pc_radius_bias_correction_gpu.py` 做 batch PC/radius bias 检查，默认优先 weighted image/band route。
5. 对坐标链路有疑问时，先跑 `geometry_only_pc_orientation_projection.py` 和 `closed_loop_crystal_frame_mapping.py`，再决定是否引入更自由的 refinement。
6. 对结构化 band peak 匹配实验，使用 `spherical_radon_graph_pipeline.py`，但目前它仍是研究原型，最终 image score 仍需和 weighted route 对照。

## 多角度 mapping align 方向

多张 EBSD mapping 的 2D 对应关系目前在相邻目录 `../EBSD-align/` 中实验。当前可行的原则是：

- 已知采样旋转角时，先把 in-plane rotation 当作强先验。
- BSE/SEM 图像只用于估计轻微残余畸变，不应替代物理角度先验。
- 局部细匹配应输出像素坐标映射和 valid mask，而不只是 warped image。
- LightGlue/SuperPoint 或其它稀疏特征方法应受小位移和质量门控约束，防止 BSE 通道衬度或条纹把 EBSD mapping 扭坏。

## GitHub 规则

- 提交代码、文档和轻量配置。
- 不提交 `.edaxh5`、`.up2`、master pattern H5、输出图、缓存、权重和 notebook runtime 噪声。
- 算法默认值或推荐路线变化时，同步更新 `README.md` 或本文件。
