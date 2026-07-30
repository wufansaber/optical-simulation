"""Tkinter 交互式任意方向像散高斯光束仿真界面。"""

from __future__ import annotations

import csv
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .interactive_model import (
    InteractiveConfig,
    InteractiveLens,
    InteractiveResult,
    OutputPlaneIntensity,
    PropagationDirection,
    intensity_on_output_z_plane,
    sample_interactive_path,
    simulate_to_z,
    stripe_355nm_preset,
)
from .lens_csv import load_lenses_csv


def _optional_focal_mm(text: str) -> Optional[float]:
    value = text.strip().lower()
    if value in ("", "--", "none", "inf", "+inf", "infinity"):
        return None
    focal_mm = float(value)
    if focal_mm == 0.0:
        raise ValueError("焦距不能为零；留空表示该方向无光焦度。")
    return focal_mm * 1e-3


def _format_focal_mm(value: Optional[float]) -> str:
    return "" if value is None else "{:.6g}".format(value * 1e3)


def _display_unit(maximum_m: float) -> Tuple[float, str]:
    maximum_m = abs(maximum_m)
    if maximum_m < 1e-6:
        return 1e9, "nm"
    if maximum_m < 5e-3:
        return 1e6, "um"
    if maximum_m < 1.0:
        return 1e3, "mm"
    return 1.0, "m"


class OpticalSimulationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Gaussian Beam XY Interactive Simulator")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(1500, max(1, screen_width - 80))
        window_height = min(900, max(1, screen_height - 120))
        window_x = max(0, (screen_width - window_width) // 2)
        window_y = max(0, (screen_height - window_height) // 2)
        self.root.geometry(
            "{}x{}+{}+{}".format(window_width, window_height, window_x, window_y)
        )
        self.root.minsize(min(900, window_width), min(620, window_height))
        self.variables: Dict[str, tk.StringVar] = {
            "wavelength_nm": tk.StringVar(value="355"),
            "diameter_x_mm": tk.StringVar(value="1.2"),
            "diameter_y_mm": tk.StringVar(value="1.2"),
            "divergence_x_mrad": tk.StringVar(value="0"),
            "divergence_y_mrad": tk.StringVar(value="0"),
            "power_w": tk.StringVar(value="1.0"),
            "refractive_index": tk.StringVar(value="1.0"),
            "theta_x_deg": tk.StringVar(value="0"),
            "theta_y_deg": tk.StringVar(value="0"),
            "observation_z_mm": tk.StringVar(value="272.502149"),
            "grid_points": tk.StringVar(value="401"),
            "span_radii": tk.StringVar(value="2.5"),
            "lens_name": tk.StringVar(value="L1"),
            "lens_z_mm": tk.StringVar(value="0"),
            "lens_fx_mm": tk.StringVar(value=""),
            "lens_fy_mm": tk.StringVar(value=""),
        }
        self.equal_scale = tk.BooleanVar(value=True)
        self.output_plane_mode = tk.StringVar(value="自定义 Z 输出面")
        self.output_plane_options: Dict[str, Tuple[float, Tuple[InteractiveLens, ...]]] = {}
        self.simulation_lenses: Tuple[InteractiveLens, ...] = ()
        self.status = tk.StringVar(value="就绪")
        self.summary = tk.StringVar(value="设置参数后单击“计算光场”。")
        self.lenses: List[InteractiveLens] = []
        self.selected_lens_index: Optional[int] = None
        self.result: Optional[InteractiveResult] = None
        self.output_field: Optional[OutputPlaneIntensity] = None
        self.minimum_pixel_marker_used = False
        self._build_interface()
        self.load_stripe_preset(calculate=False)
        self.root.after(200, self.calculate)

    def _entry(self, parent, row: int, label: str, key: str, unit: str = "") -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=3)
        ttk.Entry(parent, textvariable=self.variables[key], width=13).grid(
            row=row, column=1, sticky="ew", padx=5, pady=3
        )
        ttk.Label(parent, text=unit).grid(row=row, column=2, sticky="w", padx=(0, 5), pady=3)

    def _build_interface(self) -> None:
        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)
        controls_outer = ttk.Frame(main, width=390)
        results_outer = ttk.Frame(main)
        main.add(controls_outer, weight=0)
        main.add(results_outer, weight=1)

        controls_canvas = tk.Canvas(controls_outer, highlightthickness=0, width=380)
        controls_scroll = ttk.Scrollbar(
            controls_outer, orient=tk.VERTICAL, command=controls_canvas.yview
        )
        controls = ttk.Frame(controls_canvas, padding=8)
        controls_window = controls_canvas.create_window((0, 0), window=controls, anchor="nw")
        controls_canvas.configure(yscrollcommand=controls_scroll.set)
        controls_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        controls_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        controls.bind(
            "<Configure>",
            lambda event: controls_canvas.configure(scrollregion=controls_canvas.bbox("all")),
        )
        controls_canvas.bind(
            "<Configure>",
            lambda event: controls_canvas.itemconfigure(controls_window, width=event.width),
        )

        laser = ttk.LabelFrame(controls, text="激光器输出端", padding=6)
        laser.pack(fill=tk.X, pady=(0, 7))
        self._entry(laser, 0, "波长", "wavelength_nm", "nm")
        self._entry(laser, 1, "X 1/e2 全直径", "diameter_x_mm", "mm")
        self._entry(laser, 2, "Y 1/e2 全直径", "diameter_y_mm", "mm")
        self._entry(laser, 3, "X 发散半角", "divergence_x_mrad", "mrad")
        self._entry(laser, 4, "Y 发散半角", "divergence_y_mrad", "mrad")
        self._entry(laser, 5, "功率", "power_w", "W")
        self._entry(laser, 6, "折射率", "refractive_index")
        laser.columnconfigure(1, weight=1)

        direction = ttk.LabelFrame(controls, text="传播方向与输出面", padding=6)
        direction.pack(fill=tk.X, pady=7)
        self._entry(direction, 0, "X-Z 方向角", "theta_x_deg", "deg")
        self._entry(direction, 1, "Y-Z 方向角", "theta_y_deg", "deg")
        self._entry(direction, 2, "输出面 Z", "observation_z_mm", "mm")
        ttk.Label(direction, text="输出位置选择").grid(
            row=3, column=0, sticky="w", padx=5, pady=3
        )
        self.output_plane_selector = ttk.Combobox(
            direction,
            textvariable=self.output_plane_mode,
            state="readonly",
            width=25,
        )
        self.output_plane_selector.grid(
            row=3, column=1, columnspan=2, sticky="ew", padx=5, pady=3
        )
        self.output_plane_selector.bind("<<ComboboxSelected>>", self._select_output_plane)
        self._entry(direction, 4, "光场网格", "grid_points", "奇数")
        self._entry(direction, 5, "显示半径范围", "span_radii", "x w")
        ttk.Checkbutton(
            direction,
            text="二维图 X/Y 使用相同物理比例",
            variable=self.equal_scale,
        ).grid(row=6, column=0, columnspan=3, sticky="w", padx=5, pady=4)
        direction.columnconfigure(1, weight=1)

        lens_frame = ttk.LabelFrame(controls, text="XY 独立焦距薄透镜", padding=6)
        lens_frame.pack(fill=tk.BOTH, expand=True, pady=7)
        columns = ("name", "z", "fx", "fy")
        self.lens_table = ttk.Treeview(
            lens_frame, columns=columns, show="headings", height=8, selectmode="browse"
        )
        for column, title, width in (
            ("name", "名称", 58),
            ("z", "Z/mm", 82),
            ("fx", "fx/mm", 82),
            ("fy", "fy/mm", 82),
        ):
            self.lens_table.heading(column, text=title)
            self.lens_table.column(column, width=width, anchor="center")
        self.lens_table.pack(fill=tk.X)
        self.lens_table.bind("<<TreeviewSelect>>", self._select_lens)

        editor = ttk.Frame(lens_frame)
        editor.pack(fill=tk.X, pady=(6, 0))
        self._entry(editor, 0, "名称", "lens_name")
        self._entry(editor, 1, "Z 位置", "lens_z_mm", "mm")
        self._entry(editor, 2, "X 焦距 fx", "lens_fx_mm", "mm/空")
        self._entry(editor, 3, "Y 焦距 fy", "lens_fy_mm", "mm/空")
        editor.columnconfigure(1, weight=1)
        lens_actions = ttk.Frame(lens_frame)
        lens_actions.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(lens_actions, text="添加新透镜", command=self.add_lens).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        self.update_lens_button = ttk.Button(
            lens_actions,
            text="更新选中透镜",
            command=self.update_lens,
            state=tk.DISABLED,
        )
        self.update_lens_button.pack(side=tk.LEFT, padx=4)
        lens_list_actions = ttk.Frame(lens_frame)
        lens_list_actions.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(
            lens_list_actions,
            text="导入透镜 CSV",
            command=self.import_lenses_csv,
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(lens_list_actions, text="删除", command=self.remove_lens).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(lens_list_actions, text="清空", command=self.clear_lenses).pack(
            side=tk.LEFT, padx=4
        )

        actions = ttk.Frame(controls)
        actions.pack(fill=tk.X, pady=7)
        ttk.Button(actions, text="加载 355 nm 条形预设", command=self.load_stripe_preset).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(actions, text="计算光场", command=self.calculate).pack(fill=tk.X, pady=2)
        ttk.Button(actions, text="保存当前结果", command=self.save_results).pack(fill=tk.X, pady=2)
        ttk.Label(controls, textvariable=self.summary, wraplength=345, justify=tk.LEFT).pack(
            fill=tk.X, pady=6
        )

        notebook = ttk.Notebook(results_outer)
        notebook.pack(fill=tk.BOTH, expand=True)
        field_tab = ttk.Frame(notebook)
        path_tab = ttk.Frame(notebook)
        notebook.add(field_tab, text="输出面二维光强")
        notebook.add(path_tab, text="光束包络与中心轨迹")

        self.field_figure = Figure(figsize=(9.0, 7.0), dpi=100)
        self.field_canvas = FigureCanvasTkAgg(self.field_figure, master=field_tab)
        self.field_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.field_canvas, field_tab, pack_toolbar=True)

        self.path_figure = Figure(figsize=(9.0, 7.0), dpi=100)
        self.path_canvas = FigureCanvasTkAgg(self.path_figure, master=path_tab)
        self.path_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.path_canvas, path_tab, pack_toolbar=True)
        ttk.Label(results_outer, textvariable=self.status, anchor="w").pack(fill=tk.X, padx=6, pady=3)
        self.root.bind("<Return>", lambda _event: self.calculate())

    def _refresh_lens_table(self) -> None:
        self.lenses.sort(key=lambda lens: (lens.z_m, lens.name))
        for item in self.lens_table.get_children():
            self.lens_table.delete(item)
        for index, lens in enumerate(self.lenses):
            self.lens_table.insert(
                "",
                tk.END,
                iid=str(index),
                values=(
                    lens.name,
                    "{:.6f}".format(lens.z_m * 1e3),
                    _format_focal_mm(lens.focal_x_m) or "--",
                    _format_focal_mm(lens.focal_y_m) or "--",
                ),
            )
        self.selected_lens_index = None
        self.update_lens_button.configure(state=tk.DISABLED)
        self._refresh_output_plane_options()

    def _refresh_output_plane_options(self) -> None:
        custom_label = "自定义 Z 输出面"
        options: Dict[str, Tuple[float, Tuple[InteractiveLens, ...]]] = {}
        labels = [custom_label]
        for index, lens in enumerate(self.lenses):
            label = "透镜前 #{:02d} {}  z={:.6f} mm".format(
                index + 1, lens.name, lens.z_m * 1e3
            )
            labels.append(label)
            options[label] = (lens.z_m, tuple(self.lenses[:index]))
        self.output_plane_options = options
        self.output_plane_selector.configure(values=labels)
        if self.output_plane_mode.get() not in labels:
            self.output_plane_mode.set(custom_label)

    def _select_output_plane(self, _event=None) -> None:
        selection = self.output_plane_options.get(self.output_plane_mode.get())
        if selection is not None:
            observation_z_m, _lenses_before = selection
            self.variables["observation_z_mm"].set("{:.9f}".format(observation_z_m * 1e3))
        self.calculate()

    def _select_lens(self, _event=None) -> None:
        selection = self.lens_table.selection()
        if not selection:
            return
        index = int(selection[0])
        lens = self.lenses[index]
        self.selected_lens_index = index
        self.update_lens_button.configure(state=tk.NORMAL)
        self.variables["lens_name"].set(lens.name)
        self.variables["lens_z_mm"].set("{:.9g}".format(lens.z_m * 1e3))
        self.variables["lens_fx_mm"].set(_format_focal_mm(lens.focal_x_m))
        self.variables["lens_fy_mm"].set(_format_focal_mm(lens.focal_y_m))

    def _lens_from_editor(self) -> InteractiveLens:
        lens = InteractiveLens(
            name=self.variables["lens_name"].get().strip(),
            z_m=float(self.variables["lens_z_mm"].get()) * 1e-3,
            focal_x_m=_optional_focal_mm(self.variables["lens_fx_mm"].get()),
            focal_y_m=_optional_focal_mm(self.variables["lens_fy_mm"].get()),
        )
        if lens.focal_x_m is None and lens.focal_y_m is None:
            raise ValueError("至少需要填写一个方向的焦距。")
        return lens

    def add_lens(self) -> None:
        try:
            self.lenses.append(self._lens_from_editor())
            self._refresh_lens_table()
            self.variables["lens_name"].set("L{}".format(len(self.lenses) + 1))
            self.status.set("已添加新透镜；单击“计算光场”应用参数。")
        except Exception as error:
            messagebox.showerror("透镜参数错误", str(error), parent=self.root)

    def update_lens(self) -> None:
        if self.selected_lens_index is None:
            messagebox.showinfo("更新透镜", "请先在表格中选择要更新的透镜。", parent=self.root)
            return
        try:
            index = self.selected_lens_index
            self.lenses[index] = self._lens_from_editor()
            self._refresh_lens_table()
            self.status.set("已更新选中透镜；单击“计算光场”应用参数。")
        except Exception as error:
            messagebox.showerror("透镜参数错误", str(error), parent=self.root)

    def import_lenses_csv(self) -> None:
        csv_path = filedialog.askopenfilename(
            title="选择透镜处方 CSV",
            initialdir=str(Path.cwd() / "outputs"),
            filetypes=(("透镜 CSV", "*.csv"), ("所有文件", "*.*")),
            parent=self.root,
        )
        if not csv_path:
            return
        try:
            imported = load_lenses_csv(Path(csv_path))
            if self.lenses and not messagebox.askyesno(
                "导入透镜 CSV",
                "导入将替换当前 {} 片透镜，是否继续？".format(len(self.lenses)),
                parent=self.root,
            ):
                return
            self.lenses = list(imported.lenses)
            self._refresh_lens_table()
            if imported.observation_z_m is not None:
                self.output_plane_mode.set("自定义 Z 输出面")
                self.variables["observation_z_mm"].set(
                    "{:.9f}".format(imported.observation_z_m * 1e3)
                )
            self.status.set(
                "已从 {} 导入 {} 片透镜；单击“计算光场”应用参数。".format(
                    Path(csv_path).name, len(self.lenses)
                )
            )
            messagebox.showinfo(
                "导入完成",
                "已导入 {} 片透镜{}。".format(
                    len(self.lenses),
                    "，并读取 TARGET 输出位置" if imported.observation_z_m is not None else "",
                ),
                parent=self.root,
            )
        except Exception as error:
            messagebox.showerror("透镜 CSV 导入失败", str(error), parent=self.root)

    def remove_lens(self) -> None:
        selection = self.lens_table.selection()
        if not selection:
            messagebox.showinfo("删除透镜", "请先在表格中选择透镜。", parent=self.root)
            return
        del self.lenses[int(selection[0])]
        self._refresh_lens_table()
        self.status.set("透镜已删除；单击“计算光场”应用参数。")

    def clear_lenses(self) -> None:
        self.lenses = []
        self._refresh_lens_table()
        self.status.set("已清空透镜列表。")

    def load_stripe_preset(self, calculate: bool = True) -> None:
        self.variables["wavelength_nm"].set("355")
        self.variables["diameter_x_mm"].set("1.2")
        self.variables["diameter_y_mm"].set("1.2")
        self.variables["divergence_x_mrad"].set("0")
        self.variables["divergence_y_mrad"].set("0")
        self.variables["power_w"].set("1.0")
        self.variables["refractive_index"].set("1.0")
        self.variables["theta_x_deg"].set("0")
        self.variables["theta_y_deg"].set("0")
        self.variables["observation_z_mm"].set("272.502149")
        self.output_plane_mode.set("自定义 Z 输出面")
        self.lenses = list(stripe_355nm_preset())
        self._refresh_lens_table()
        self.status.set("已加载 355 nm 条形光斑七透镜预设。")
        if calculate:
            self.calculate()

    def _configuration(self) -> Tuple[InteractiveConfig, float, int, float]:
        config = InteractiveConfig(
            wavelength_m=float(self.variables["wavelength_nm"].get()) * 1e-9,
            input_diameter_x_m=float(self.variables["diameter_x_mm"].get()) * 1e-3,
            input_diameter_y_m=float(self.variables["diameter_y_mm"].get()) * 1e-3,
            input_divergence_x_mrad=float(self.variables["divergence_x_mrad"].get()),
            input_divergence_y_mrad=float(self.variables["divergence_y_mrad"].get()),
            power_w=float(self.variables["power_w"].get()),
            refractive_index=float(self.variables["refractive_index"].get()),
            direction=PropagationDirection(
                theta_x_deg=float(self.variables["theta_x_deg"].get()),
                theta_y_deg=float(self.variables["theta_y_deg"].get()),
            ),
        )
        observation_z_m = float(self.variables["observation_z_mm"].get()) * 1e-3
        grid_points = int(self.variables["grid_points"].get())
        span_radii = float(self.variables["span_radii"].get())
        return config, observation_z_m, grid_points, span_radii

    def calculate(self) -> None:
        try:
            self.root.configure(cursor="watch")
            self.status.set("正在计算二维光场与光束包络……")
            self.root.update_idletasks()
            config, observation_z_m, grid_points, span_radii = self._configuration()
            selected_plane = self.output_plane_options.get(self.output_plane_mode.get())
            if selected_plane is None:
                simulation_lenses = tuple(self.lenses)
            else:
                observation_z_m, simulation_lenses = selected_plane
                self.variables["observation_z_mm"].set(
                    "{:.9f}".format(observation_z_m * 1e3)
                )
            self.simulation_lenses = simulation_lenses
            result = simulate_to_z(config, simulation_lenses, observation_z_m)
            output_field = intensity_on_output_z_plane(result, grid_points, span_radii)
            self.result = result
            self.output_field = output_field
            self._plot_output_field(result, output_field)
            self._plot_path(config, observation_z_m, simulation_lenses)
            beam = result.observation.beam
            self.summary.set(
                "输出 Z = {:.6f} mm；光程 = {:.6f} mm\n"
                "中心 X = {:.6f} mm，Y = {:.6f} mm\n"
                "输入发散半角：X = {:.6f} mrad，Y = {:.6f} mrad\n"
                "光轴垂直截面 1/e2 直径：X = {:.6f} um，Y = {:.6f} um\n"
                "峰值光强 = {:.6e} W/m2；已应用 {} 片透镜，未通过 {} 片\n"
                "输出位置：{}".format(
                    observation_z_m * 1e3,
                    result.observation.path_length_m * 1e3,
                    result.observation.center_x_m * 1e3,
                    result.observation.center_y_m * 1e3,
                    config.input_divergence_x_mrad,
                    config.input_divergence_y_mrad,
                    2.0 * beam.radius_x * 1e6,
                    2.0 * beam.radius_y * 1e6,
                    float(np.max(output_field.intensity_w_m2)),
                    len(result.applied_lenses),
                    len(self.lenses) - len(result.applied_lenses),
                    self.output_plane_mode.get(),
                )
            )
            if self.minimum_pixel_marker_used:
                self.status.set("计算完成；小于一个屏幕像素的光斑方向已用一像素标记保留。")
            else:
                self.status.set("计算完成。可改变任意参数或输出 Z 后重新计算。")
        except Exception as error:
            self.status.set("计算失败。")
            messagebox.showerror("仿真参数错误", str(error), parent=self.root)
        finally:
            self.root.configure(cursor="")

    def _plot_output_field(
        self,
        result: InteractiveResult,
        field: OutputPlaneIntensity,
    ) -> None:
        self.field_figure.clear()
        self.field_figure.set_constrained_layout(True)
        grid = self.field_figure.add_gridspec(2, 2, width_ratios=(1.5, 1.0))
        map_axis = self.field_figure.add_subplot(grid[:, 0])
        x_axis = self.field_figure.add_subplot(grid[0, 1])
        y_axis = self.field_figure.add_subplot(grid[1, 1])

        common_display_limit = None
        if self.equal_scale.get():
            common_maximum = max(
                float(np.max(np.abs(field.x_offset_m))),
                float(np.max(np.abs(field.y_offset_m))),
            )
            x_scale, x_unit = _display_unit(common_maximum)
            y_scale, y_unit = x_scale, x_unit
            common_display_limit = common_maximum * x_scale
        else:
            x_scale, x_unit = _display_unit(float(np.max(np.abs(field.x_offset_m))))
            y_scale, y_unit = _display_unit(float(np.max(np.abs(field.y_offset_m))))
        x_values = field.x_offset_m * x_scale
        y_values = field.y_offset_m * y_scale
        threshold = np.exp(-2.0)
        mesh = map_axis.pcolormesh(
            x_values,
            y_values,
            field.intensity_w_m2,
            shading="auto",
            cmap="cividis",
            vmin=0.0,
            vmax=float(np.max(field.intensity_w_m2)),
        )
        map_axis.contour(
            x_values,
            y_values,
            field.normalized_intensity,
            levels=[threshold],
            colors=["#d97706"],
            linewidths=2.0,
        )
        map_axis.set(
            title="Output Z-plane intensity",
            xlabel="X - X center ({})".format(x_unit),
            ylabel="Y - Y center ({})".format(y_unit),
        )
        if common_display_limit is not None:
            map_axis.set_xlim(-common_display_limit, common_display_limit)
            map_axis.set_ylim(-common_display_limit, common_display_limit)
        map_axis.set_aspect("equal" if self.equal_scale.get() else "auto", adjustable="box")
        self.field_figure.colorbar(mesh, ax=map_axis, label="Intensity (W/m2)")

        center_x = len(x_values) // 2
        center_y = len(y_values) // 2
        x_axis.plot(
            x_values,
            field.normalized_intensity[center_y, :],
            color="#1f5a94",
            linewidth=1.8,
        )
        x_axis.axhline(threshold, color="#374151", linestyle=":", label="1/e2")
        x_axis.set(
            title="X center profile",
            xlabel="X - X center ({})".format(x_unit),
            ylabel="Normalized intensity",
            ylim=(-0.02, 1.04),
        )
        y_axis.plot(
            y_values,
            field.normalized_intensity[:, center_x],
            color="#a16207",
            linewidth=1.8,
        )
        y_axis.axhline(threshold, color="#374151", linestyle=":", label="1/e2")
        y_axis.set(
            title="Y center profile",
            xlabel="Y - Y center ({})".format(y_unit),
            ylabel="Normalized intensity",
            ylim=(-0.02, 1.04),
        )
        for axis in (x_axis, y_axis):
            axis.grid(color="#d1d5db", alpha=0.45)
            axis.legend(loc="upper right")

        beam = result.observation.beam
        self.field_figure.suptitle(
            "z={:.6f} mm | theta_x={:.3f} deg, theta_y={:.3f} deg | "
            "beam-normal diameters: X={:.6f} um, Y={:.6f} um".format(
                result.observation_z_m * 1e3,
                result.config.direction.theta_x_deg,
                result.config.direction.theta_y_deg,
                2.0 * beam.radius_x * 1e6,
                2.0 * beam.radius_y * 1e6,
            )
        )
        self.field_canvas.draw()
        self.minimum_pixel_marker_used = False
        footprint_rows, footprint_columns = np.nonzero(
            field.normalized_intensity >= threshold
        )
        if footprint_rows.size and footprint_columns.size:
            footprint_x_min = x_values[int(np.min(footprint_columns))]
            footprint_x_max = x_values[int(np.max(footprint_columns))]
            footprint_y_min = y_values[int(np.min(footprint_rows))]
            footprint_y_max = y_values[int(np.max(footprint_rows))]
            x_pixels = abs(
                map_axis.transData.transform((footprint_x_max, 0.0))[0]
                - map_axis.transData.transform((footprint_x_min, 0.0))[0]
            )
            y_pixels = abs(
                map_axis.transData.transform((0.0, footprint_y_max))[1]
                - map_axis.transData.transform((0.0, footprint_y_min))[1]
            )
            one_pixel_points = 72.0 / self.field_figure.dpi
            if x_pixels < 1.0 and y_pixels < 1.0:
                map_axis.plot(
                    [0.0],
                    [0.0],
                    marker="s",
                    markersize=one_pixel_points,
                    color="#d97706",
                    clip_on=False,
                    zorder=10,
                )
                self.minimum_pixel_marker_used = True
            elif x_pixels < 1.0:
                map_axis.plot(
                    [0.0, 0.0],
                    [footprint_y_min, footprint_y_max],
                    color="#d97706",
                    linewidth=one_pixel_points,
                    solid_capstyle="butt",
                    clip_on=False,
                    zorder=10,
                )
                self.minimum_pixel_marker_used = True
            elif y_pixels < 1.0:
                map_axis.plot(
                    [footprint_x_min, footprint_x_max],
                    [0.0, 0.0],
                    color="#d97706",
                    linewidth=one_pixel_points,
                    solid_capstyle="butt",
                    clip_on=False,
                    zorder=10,
                )
                self.minimum_pixel_marker_used = True
        self.field_canvas.draw_idle()

    def _plot_path(
        self,
        config: InteractiveConfig,
        observation_z_m: float,
        simulation_lenses: Tuple[InteractiveLens, ...],
    ) -> None:
        samples = sample_interactive_path(config, simulation_lenses, observation_z_m)
        z_mm = np.array([sample.z_m for sample in samples]) * 1e3
        radius_x_m = np.array([sample.radius_x_m for sample in samples])
        radius_y_m = np.array([sample.radius_y_m for sample in samples])
        radius_x_mm = radius_x_m * 1e3
        radius_y_mm = radius_y_m * 1e3
        center_x_mm = np.array([sample.center_x_m for sample in samples]) * 1e3
        center_y_mm = np.array([sample.center_y_m for sample in samples]) * 1e3

        basis_x, basis_y = config.direction.transverse_basis
        mapping = np.array(
            [[basis_x[0], basis_x[1]], [basis_y[0], basis_y[1]]],
            dtype=np.float64,
        )
        inverse_mapping = np.linalg.inv(mapping)
        projected_half_x_mm = np.sqrt(
            (inverse_mapping[0, 0] * radius_x_m) ** 2
            + (inverse_mapping[0, 1] * radius_y_m) ** 2
        ) * 1e3
        projected_half_y_mm = np.sqrt(
            (inverse_mapping[1, 0] * radius_x_m) ** 2
            + (inverse_mapping[1, 1] * radius_y_m) ** 2
        ) * 1e3

        self.path_figure.clear()
        self.path_figure.set_constrained_layout(True)
        axes = self.path_figure.subplots(3, 1, sharex=True)
        axes[0].fill_between(
            z_mm,
            center_x_mm - projected_half_x_mm,
            center_x_mm + projected_half_x_mm,
            color="#90b7dc",
            alpha=0.45,
            label="Projected X 1/e2 envelope",
        )
        axes[0].plot(z_mm, center_x_mm, color="#1f5a94", linewidth=1.5, label="Center ray")
        axes[0].plot(
            z_mm, center_x_mm - projected_half_x_mm, color="#1f5a94", linewidth=0.9
        )
        axes[0].plot(
            z_mm, center_x_mm + projected_half_x_mm, color="#1f5a94", linewidth=0.9
        )
        axes[0].set_ylabel("Global X (mm)")
        axes[0].set_title("Central ray and projected beam envelope in X-Z")
        axes[0].legend(loc="best")

        axes[1].fill_between(
            z_mm,
            center_y_mm - projected_half_y_mm,
            center_y_mm + projected_half_y_mm,
            color="#efd38b",
            alpha=0.5,
            label="Projected Y 1/e2 envelope",
        )
        axes[1].plot(z_mm, center_y_mm, color="#a16207", linewidth=1.5, label="Center ray")
        axes[1].plot(
            z_mm, center_y_mm - projected_half_y_mm, color="#a16207", linewidth=0.9
        )
        axes[1].plot(
            z_mm, center_y_mm + projected_half_y_mm, color="#a16207", linewidth=0.9
        )
        axes[1].set_ylabel("Global Y (mm)")
        axes[1].set_title("Central ray and projected beam envelope in Y-Z")
        axes[1].legend(loc="best")

        axes[2].plot(z_mm, radius_x_mm, color="#1f5a94", label="Beam-normal X radius")
        axes[2].plot(
            z_mm,
            radius_y_mm,
            color="#a16207",
            linestyle="--",
            label="Beam-normal Y radius",
        )
        axes[2].set_yscale("log")
        axes[2].set_ylabel("1/e2 radius (mm, log)")
        axes[2].set_xlabel("Global Z (mm)")
        axes[2].legend(loc="best")

        for lens in simulation_lenses:
            if lens.z_m > observation_z_m:
                continue
            for axis in axes:
                axis.axvline(lens.z_m * 1e3, color="#6b7280", linewidth=0.8, alpha=0.7)
            axes[0].text(
                lens.z_m * 1e3,
                0.96,
                lens.name,
                rotation=90,
                transform=axes[0].get_xaxis_transform(),
                ha="right",
                va="top",
                fontsize=8,
            )
        for axis in axes:
            axis.grid(color="#d1d5db", alpha=0.45)
            axis.axvline(
                observation_z_m * 1e3,
                color="#202124",
                linestyle=":",
                linewidth=1.3,
            )
        selected_plane = self.output_plane_options.get(self.output_plane_mode.get())
        if selected_plane is None:
            output_description = "custom Z plane"
        else:
            lens_index = len(simulation_lenses)
            lens_name = self.lenses[lens_index].name if lens_index < len(self.lenses) else "lens"
            output_description = "front of {}".format(lens_name)
        self.path_figure.suptitle(
            "All propagation coordinates in mm | output: {}".format(output_description)
        )
        self.path_canvas.draw_idle()



    def save_results(self) -> None:
        if self.result is None or self.output_field is None:
            messagebox.showinfo("保存结果", "请先计算光场。", parent=self.root)
            return
        directory_text = filedialog.askdirectory(
            title="选择结果保存目录",
            initialdir=str(Path.cwd() / "outputs"),
            parent=self.root,
        )
        if not directory_text:
            return
        try:
            directory = Path(directory_text)
            directory.mkdir(parents=True, exist_ok=True)
            field_png = directory / "interactive_output_field.png"
            path_png = directory / "interactive_beam_path.png"
            field_npz = directory / "interactive_output_field.npz"
            lens_csv = directory / "interactive_lenses.csv"
            self.field_figure.savefig(field_png, dpi=320, facecolor="white")
            self.path_figure.savefig(path_png, dpi=320, facecolor="white")
            field = self.output_field
            result = self.result
            np.savez_compressed(
                field_npz,
                x_offset_m=field.x_offset_m,
                y_offset_m=field.y_offset_m,
                intensity_W_per_m2=field.intensity_w_m2,
                normalized_intensity=field.normalized_intensity,
                observation_z_m=result.observation_z_m,
                optical_path_length_m=result.observation.path_length_m,
                center_x_m=field.center_x_m,
                center_y_m=field.center_y_m,
                input_divergence_x_mrad=result.config.input_divergence_x_mrad,
                input_divergence_y_mrad=result.config.input_divergence_y_mrad,
                wavelength_m=result.config.wavelength_m,
                theta_x_deg=result.config.direction.theta_x_deg,
                theta_y_deg=result.config.direction.theta_y_deg,
                qx=result.observation.beam.qx,
                qy=result.observation.beam.qy,
            )
            with lens_csv.open("w", newline="", encoding="utf-8-sig") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["name", "z_mm", "focal_x_mm", "focal_y_mm"])
                for lens in self.lenses:
                    writer.writerow(
                        [
                            lens.name,
                            "{:.9f}".format(lens.z_m * 1e3),
                            _format_focal_mm(lens.focal_x_m),
                            _format_focal_mm(lens.focal_y_m),
                        ]
                    )
            self.status.set("结果已保存到 {}".format(directory))
            messagebox.showinfo(
                "保存完成",
                "已输出高分辨率光场图、光路图、NPZ 数据和透镜 CSV。",
                parent=self.root,
            )
        except Exception as error:
            messagebox.showerror("保存失败", str(error), parent=self.root)


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style(root).theme_use("vista")
    except tk.TclError:
        pass
    OpticalSimulationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
