# Interactive Gaussian Beam XY Simulation

本项目建议在 VS Code 中使用 Python 3.10 Conda 环境，以 Python 和 X/Y 分离的复数 q 参数、ABCD 矩阵计算 355 nm 相干高斯光束。当前版本按工件表面垂直入射计算，不使用折转反射镜。

完整的软件功能与操作说明请参阅：[SOFTWARE_GUIDE_CN.md](SOFTWARE_GUIDE_CN.md)。


## 快速启动目录版 EXE

目录版程序位于：

```text
dist\OpticalSimulation\OpticalSimulation.exe
```

运行或复制程序时必须保留整个 `dist\OpticalSimulation` 文件夹，不能只复制其中的 EXE；其 `_internal` 子目录包含 Python、NumPy、Matplotlib 和 Tcl/Tk 依赖。目录版无需在每次启动时解压全部组件，因此比单文件版启动更快。


界面支持：

- 窗口自动适配当前屏幕，左侧参数区可滚动，不会超过屏幕可用范围；
- 任意波长、输入 X/Y 光斑全直径、X/Y 发散半角、功率和折射率；
- 用 X-Z、Y-Z 两个方向角设置任意中心光线传播方向；
- 在任意非负 Z 位置添加任意数量薄透镜；
- 每片透镜的 `fx`、`fy` 独立设置，留空表示该方向无光焦度；
- “添加新透镜”和“更新选中透镜”为两个独立操作；只有选中表格行后才能更新；
- 可直接导入软件导出的透镜 CSV；包含 `TARGET` 行时同时读取输出 Z；
- 可选择自定义输出 Z 平面，或直接选择任意一片透镜的前表面光场；
- 二维光场支持 X/Y 相同物理比例；小于一个屏幕像素的光斑方向保留一像素标记；
- 光路图固定使用 mm，分别显示 X-Z、Y-Z 中心光线、投影光束包络和束半径；
- 保存 320 DPI 光场图、光路图、NPZ 数值数据与透镜 CSV。

方向角定义为中心光线在 X-Z、Y-Z 平面相对全局 +Z 轴的角度。透镜中心位于中心光线与对应全局 Z 平面的交点，薄透镜面假定垂直于中心光线。两 Z 平面之间用于 q 参数传播的真实光程为：

```text
ds = dz * sqrt(1 + tan(theta_x)^2 + tan(theta_y)^2)
```

输出二维图计算在全局 `Z = 常数` 平面上；斜向光束会自动投影到该平面。输出 Z 之前的透镜参与计算，位于输出面之后的透镜自动忽略。

X/Y 发散角输入单位为 mrad，定义为激光器输出面 `1/e^2` 半径处的局部发散半角。正值表示光束扩张，负值表示会聚，0 表示该方向在输出面处为束腰。模型通过 `tan(alpha) = dw/ds = w/R` 将发散半角换算为初始波前曲率，再进入 q 参数传播。

## 输入与约定

- 波长：355 nm；
- 激光输出光斑：X、Y 均为 1.2 mm；
- 目标工件面光斑：X = 0.8 um，Y = 1500 um；
- 入射光轴与工件表面法线夹角：0 deg；
- 所有光斑尺寸均为强度 `1/e^2` 全直径；
- 当前 355 nm 预设的 X/Y 发散半角均为 0，输入束腰位于 X1 面；理想 TEM00、M2 = 1、空气折射率 n = 1；
- X/Y 扩束采用独立柱面镜，C1/C2 为整体扩束镜，F 仅在 X 方向聚焦。

## 计算处方

| 序号 | 透镜 | z/mm | fx/mm | fy/mm | 到下一面/mm |
|---:|---|---:|---:|---:|---:|
| 1 | X1：X 负柱面镜 | 0.000000 | -15.000000 | -- | 153.153102 |
| 2 | X2：X 正柱面镜 | 153.153102 | +168.153102 | -- | 30.000000 |
| 3 | Y1：Y 负柱面镜 | 183.153102 | -- | -50.000000 | 9.349040 |
| 4 | Y2：Y 正柱面镜 | 192.502142 | -- | +59.349040 | 30.000000 |
| 5 | C1：整体负透镜 | 222.502142 | -100.000000 | -100.000000 | 5.000000 |
| 6 | C2：整体正透镜 | 227.502142 | +105.000000 | +105.000000 | 20.000000 |
| 7 | F：X 聚焦柱面镜 | 247.502142 | +25.000000 | -- | 25.000006 |
|  | 工件靶面 | 272.502149 |  |  |  |

模型输出为：

- 靶面 X 全直径：0.800000 um；
- 靶面 Y 全直径：1500.000000 um；
- X 束腰半径：0.400000 um；
- X 近轴斜率参数：0.282500；
- X 等效几何 NA：0.271860。

斜率参数低于项目设置的 0.3 有效性阈值，但已经接近阈值。工程定型仍应加入真实透镜厚度、材料、像差、M2、装调误差、损伤阈值，并用矢量衍射或物理光学软件复核 0.8 um 聚焦结果。


## 输出文件

所有本设计的 PNG 使用 320 DPI 输出；最终二维光场为 4000 x 2400 px，逐透镜光场与光路图的长边也均超过 4000 px。

- `outputs/stripe_355nm_surface.png`：最终工件面二维光场，X/Y 坐标统一为 um，并保持相同物理比例；
- `outputs/stripe_355nm_field.npz`：最终复电场与强度数据，包含 X/Y 的 m 与 um 坐标；
- `outputs/stripe_355nm_lens_prescription.csv`：透镜焦距、位置、间距与口径；
- `outputs/lens_planes/01_X1_after.png` 至 `07_F_after.png`：7 个透镜后的强度与相位；
- 同名 `.npz`：各透镜面坐标、复电场、强度、q 参数和曲率；
- `outputs/lens_planes/lens_planes_manifest.csv`：逐透镜输出清单；
- `outputs/stripe_355nm_lens_planes_overview.png`：7 个透镜面光场总览；
- `outputs/stripe_355nm_optical_layout.png`：从激光输出到最终聚焦的 X/Y 光束包络。

运行测试：

```powershell
conda activate <environment-name>
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
```

## 180 mm 双支路四工况初算

项目包含一个等效的一阶模型，用于在真实镜组处方尚未齐全时启动计算。当前工况为：

- 近法向入射 `1 deg`，倍率 A：`1.44 um x 5.2 mm`；
- 近法向入射 `1 deg`，倍率 B：`1.05 um x 2.667 mm`；
- 斜入射 `75 deg`，倍率 A：`1.44 um x 5.2 mm`；
- 斜入射 `75 deg`，倍率 B：`1.05 um x 2.667 mm`。

四个工况均使用从激光器输出参考面到晶圆面的 `180 mm` 展开中心光程。近法向中路采用 `s=110 mm` 的 E1 光束整形等效面、`s=166 mm` 的 Normal LFC/Y 向补偿等效面和 `s=174 mm`、等效焦距 `30 mm` 的理想物镜面；斜入射支路采用 `s=110 mm` 的 E1 面和 `s=174 mm` 的 Oblique LFC/E2 面。四工况由“共用前端 + FIBE 倍率 A/B + Prism Diverter 支路选择 + 支路专用后端”组成。运行命令：

```powershell
conda activate NIST2
$env:PYTHONPATH = "$PWD\src"
python examples\run_dual_branch_180mm.py
```

完整方案在公共 Waist Relay 后增加一维 Y 向匀化器，以 36 阶超高斯近似平顶输出；四工况要求中央 90% 有效线长内 `I_min/I_max >= 90%`。有用光学效率按激光器输出参考面到晶圆目标矩形内的功率定义，要求 `>=65%`，并按逐元件镀膜/衍射效率进行标称值和分配下限预算。

输出目录为 `outputs/dual_branch_180mm`。其中 `case_path_matrix.csv` 标明每个工况的共用部分、倍率切换和支路切换，`estimated_layout.csv` 保存预估元件位置，`summary.csv` 保存四工况焦距和有效性指标，`performance_summary.csv` 保存均匀度和效率判定，`efficiency_budget.csv` 保存逐元件效率预算，`results.json` 保存完整假设与结果；PNG 文件给出拓扑、光束包络、晶圆面光强和性能合规图。

完整三维工程预布局见 `docs/照明光路三维布局完整方案.md`。运行 `examples/generate_complete_3d_layout.py` 会导出逐元件三维坐标、镜面法向、路径段、四工况等效焦度矩阵以及三维/俯视/侧视图。坐标以晶圆交点为原点，严格满足两条 180 mm 几何光程和 1°/75°末段入射角；未提供真实机械与镜片处方的项目均标记为工程预估。

`examples/render_computed_ray_trace.py` 对与高斯 q 参数一致的 1/e² 相空间光线束执行逐段 ABCD 追迹，输出 `computed_ray_trace.png/.svg/.pdf` 以及可复算的 `computed_ray_trace_data.npz`。近法向中路追迹 E1、Normal LFC/补偿面和 `30 mm` 理想物镜，斜入射支路追迹 E1、Oblique LFC/E2；在真实曲率、厚度和三维坐标补齐前，不应解释为真实表面非序列追迹。

该模型是等效初算，不是真实镜片处方。75 度工况的 X 向近轴斜率超过项目阈值，必须用真实光学处方和物理光学方法复核。
