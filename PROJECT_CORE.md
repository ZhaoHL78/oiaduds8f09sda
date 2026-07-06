# EBSD Project Core Summary

本项目的核心目标是把 EDAX EBSD 采集数据中的单张 Kikuchi pattern、软件 indexing 结果、OHP/Hough 菊池带、pattern center 和实验几何统一到一个可审计的球面坐标链中，最终服务于 Kikuchi sphere 重建、取向校准和多张 EBSD mapping 的对应关系分析。

## 数据主线

- `EDAX H5`: 读取 EBSD map、`ANG/DATA` 软件 indexing、`OHP/DATA` 菊池带、pattern center、IQ/CI、SEM Signal 与 PRIAS/BSE 图像。
- `UP2`: 读取原始 Kikuchi pattern，保持 raw pattern 与预处理 pattern 分离。
- `master sphere`: 使用 kikuchipy/EMsoft master pattern 作为标准 Kikuchi sphere 参考。
- `outputs/`: 只保存本地可视化和中间结果，不提交 GitHub。

## 当前核心脚本

- `project_edax_oim_to_sphere.py`: 当前主线模块，从 EDAX H5/UP2 读取 raw pattern、PC、OIM orientation 和实验几何，并投影到 master sphere。
- `diagnose_edax_transform_chain.py`: 系统诊断 EDAX PC convention、图像 flip/rotate、sample tilt/camera elevation 和 orientation 矩阵解释。
- `single_kikuchi_pc_finetune.py`: 单张 Kikuchi pattern 的端到端验证入口，包含图像预处理、EDAX PC 球面标定、orientation 投影到 master sphere、局部 PC finetune 和可视化。
- `pt3_same_face_spherical_calibration.py`: Pt-3 四组 90° in-plane EBSD 的同一晶面 Kikuchi 球形标定流程，包含 SEM 同晶面 ROI 选点、保守圆形 mask 预处理、H5 orientation 单图定位、cubic symmetry 等价落点选择、共同轴线闭包、PC 在 pattern 上的位置标注、PC 方向投影到同一个 master sphere 的锚点诊断、同一 master sphere 合成贴图、高清球面 Kikuchi 图和正视 3D master sphere 贴合可视化。
- `pt_highres_30deg_lightglue_calibration.py`: Pt high-resolution 十二组 30° in-plane EBSD 流程，使用相邻角度 LightGlue/SuperPoint SEM 配准链把全部 mapping 对齐到 0°，自动选取同一晶面上的同一物理位置，读取对应 UP2 Kikuchi，执行球面标定、PC finetune、固定球面落点可视化、相邻 30° 固定落点旋转诊断和 PC 晶体学锚点可视化。30° 物理条件只用于从已匹配好的球面落点中估计旋转轴线，不再移动 Kikuchi patch。
- `visualize_edax_projection_sets.py`: 批量展示多张 pattern 在同一固定链路下投影到 master sphere 的结果。
- `visualize_edax_match_3d.py`: 用 PyVista 输出标准 Kikuchi sphere 与实验 pattern patch 的 3D 叠加图。
- `visualize_scan_position_pc_correction.py` / `score_scan_position_pc_correction.py`: 用扫描位置估计 PC 漂移，并对不同 PC correction 模型评分。
- `export_h5_ipf_bse_maps.py`: 从 H5 导出 IPF-Z、IQ/CI、SEM/BSE、FOV 和 montage。
- `export_publication_h5_kikuchi_bands.py`: 导出带透明背景的 Kikuchi pattern 与 H5/OHP band 叠加图，便于论文图和报告图使用。
- `simulate_111_tilt_kikuchi_patterns.py`: 固定 PC 下模拟单晶 zone-axis 倾斜或 PCx 漂移的 Kikuchi pattern。
- `annotate_simulated_ipf_points.py`: 将模拟 pattern 的表观方向标注到立方晶体反极图。
- `simulate_123_detector_sample_tilt_cases.py`: 分离模拟 `(1,2,3)` zone axis 的探测器倾斜和样品倾斜工况。
- `classify_pt1_ebsd_data.py`: 读取 `D:\EBSD-data\Pt-1` 的 H5/UP2 元数据，并把可匹配组和孤立 UP2 组分类。
- `export_pt1_sem_kikuchi_pairs.py`: 对 Pt-1 的每个可靠 H5-UP2 对应组，导出 SEM 标注图和同 index 的 raw Kikuchi pattern。
- `export_h5_mapping_sem_correspondence.py`: 导出 H5 中全部 OIM mapping 的 SEM 图，并和本机/回收站中可找到的 UP2 原始文件名建立对应表。
- `export_up2_ebsd_kikuchi_correspondence.py`: 以 UP2 为主线，逐文件输出其对应 H5 EBSD mapping、SEM 标注点和 Kikuchi 示例。
- `align_pt1_inplane_sem_common_circle.py`: 对 Pt-1 四组 in-plane EBSD 的 SEM 进行反向旋转、平移微调，并计算四组共同有效区域的最大圆形 ROI。

## 当前推荐路线

1. 用 `export_h5_ipf_bse_maps.py` 检查 EBSD map 与 BSE/SEM 图像质量。
2. 用 `export_publication_h5_kikuchi_bands.py` 检查 OHP band 是否投影到真实 Kikuchi line 上。
3. 用 `project_edax_oim_to_sphere.py` 做单张 pattern 的 PC/orientation/master sphere 投影。
4. 需要把单张 pattern 的“预处理 -> 球面标定 -> PC finetune -> 可视化”先跑通时，跑 `single_kikuchi_pc_finetune.py`。
5. 对 Pt-3 四组 in-plane EBSD 的同一晶面做批处理时，跑 `pt3_same_face_spherical_calibration.py`，先审查 SEM ROI 和选点，再看 H5 orientation 落点、cubic symmetry 等价落点、PC 在 pattern 与同一 master sphere 上的锚点位置，以及 3D master sphere 上的共同旋转轴。
6. 对坐标链路有疑问时，先跑 `diagnose_edax_transform_chain.py`，枚举图像方向、tilt 和 orientation matrix convention。
7. 用 `visualize_edax_projection_sets.py` / `visualize_edax_match_3d.py` 做批量和三维可视化。
8. 对新的 Pt-1 多角度数据，先跑 `classify_pt1_ebsd_data.py`，只把 H5 与 UP2 精确匹配的组用于后续 Kikuchi pattern、PC、orientation、OHP band 联合分析。
9. 需要快速核对 EBSD map 与 raw Kikuchi 对应关系时，跑 `export_pt1_sem_kikuchi_pairs.py`，用 SEM 上的标注点检查所选 pattern 的空间来源。
10. 需要从 H5 全局整理 mapping/SEM/UP2 文件名关系时，跑 `export_h5_mapping_sem_correspondence.py`，先看 `h5_mapping_sem_contact_sheet.png` 和 `h5_mapping_sem_correspondence.csv`。
11. 需要按“每个 UP2 文件”核对 EBSD 和 Kikuchi 时，跑 `export_up2_ebsd_kikuchi_correspondence.py`，优先查看 `up2_ebsd_kikuchi_correspondence.csv`。
12. 需要对 Pt-1 四组 in-plane 旋转 EBSD 取共同分析区域时，跑 `align_pt1_inplane_sem_common_circle.py`，使用输出的最大圆形 ROI。
13. 需要分析 Pt high-resolution 十二组 30° in-plane EBSD 时，跑 `pt_highres_30deg_lightglue_calibration.py`，先检查 LightGlue SEM align 和同一点选择，再看每张 Kikuchi pattern 用 EDAX PC + H5 orientation + PC finetune 得到的固定 master-sphere 落点。当前原则是：单张 pattern 和 master sphere 的最佳贴合位置不能被 30° 先验改写；30° 物理旋转只用于事后测量这些固定落点之间是否存在共同旋转轴，并输出相邻 pair 的实际旋转角、轴线散布和最佳轴模型残差。如果残差很大，说明 orientation/坐标 convention/PC/样品旋转轴解释仍有误差，而不是把 patch 硬移动到 30°。

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
