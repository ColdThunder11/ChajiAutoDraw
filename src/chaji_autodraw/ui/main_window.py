"""主窗口：图片二值化预览、小程序窗口检测、画布框选与自动绘制控制。"""

from __future__ import annotations


import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QGuiApplication, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QDoubleSpinBox,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from chaji_autodraw.core.drawer import DrawWorker
from chaji_autodraw.core.imaging import (
    BinarizeParams,
    binarize,
    is_colorful,
    load_grayscale,
    mask_to_preview,
)
from chaji_autodraw.core.input_backends import (
    ForegroundMouseBackend,
    PostMessageBackend,
)
from chaji_autodraw.core.tracing import (
    connect_nearby_strokes,
    count_points,
    deduplicate_strokes,
    extend_short_strokes,
    filter_strokes,
    path_length,
    trace_paths,
)
from chaji_autodraw.core.winfinder import (
    WindowInfo,
    capture,
    get_bounds,
    list_matches,
)
from chaji_autodraw.ui.preview_window import PreviewWindow
from chaji_autodraw.ui.window_picker import (
    detect_white_canvas,
    pil_to_pixmap,
    validate_canvas_position,
)

#: 参数标签的悬浮说明（标签文字 -> 详细解释）
_LABEL_TIPS: dict[str, str] = {
    "阈值": "二值化阈值（0~255）：灰度低于该值的像素视为要画的黑色，其余为白色。"
        "调低→只画最深的线条；调高→更多浅色区域被涂黑。配合右侧二值化预览调整。",
    "模糊": "二值化前的高斯模糊半径（0~5px）：平滑噪点与细碎纹理，减少碎笔画；"
        "过大会丢失细节。照片类图片建议 1~2px。",
    "抽稀": "笔画顶点的最小间距（1~10px）：直线段内丢弃过密的中间点（转角始终保留）。"
        "越大点数越少、绘制越快，细节越少。",
    "画笔粗细": "规划的画笔线宽，请与小程序里设置的画笔大小一致："
        ">1px 时自动合并被粗笔覆盖的冗余笔画（笔画数明显减少），预览按该粗细渲染。",
    "最短笔画": "路径长度小于该值的碎笔画直接丢弃（0=不过滤）："
        "用于去掉缩放产生的孤立点与微段；调到 4px 以上可避开小程序的丢步区。",
    "最小拖拽": "短笔画延伸/往返拖拽的目标位移（默认 6px）："
        "小程序画布约需 3~4px 位移才落笔，太小仍可能丢步。",
    "连接距离": "端点距离不超过该值的相邻笔画首尾相连成一笔（默认 1.5px，覆盖 1px 间隙）："
        "显著减少落笔次数；过大会把本应分离的线条连在一起。",
    "线条阈值": "线条模式的检测灵敏度（1~60）：越小线条越多越密，越大只保留最深的线。"
        "仅在线条模式下生效。",
    "阴影剔除": "仅线条模式：识别并剔除光影分界线，防止把阴影边缘当成线条画出（默认开启）。"
        "三道判据全部满足才删除：几何上是阶跃边缘而非线条（没有任何方向两侧恢复亮背景）、"
        "邻近宽暗区、两侧色度一致（同一材质被整体调暗的光照变化）。"
        "因此发色/肤色等材质分界线、浅色低对比的边界、阴影区域内部的细节线、密集排线都会保留；"
        "带明显冷暖色偏的阴影可能不被滤除，可关闭本项找回旧行为。"
        "注意：超过 300 万像素的大图加载时会默认关闭本项（滤波开销大，防卡顿），可手动勾选。",
    "保护强度": "阴影剔除的保守程度（0~100，默认 40）：值越大越重视保留线条——只有色度几乎"
        "完全一致、明暗落差很小的分界才被当成阴影滤除，可能漏掉部分阴影边；值越小滤除越"
        "激进——色度容差与落差上限放宽，会滤除更多边界，也可能误删浅色材质分界线。"
        "50 为标准强度，默认取 40 略偏激进；若仍有线条被误删调大，若阴影边没滤干净调小。",
    "窗口关键词": "窗口标题包含该文字才会被检测，例如“霸王茶姬”。",
    "进程名": "窗口须属于该进程（默认 WeChatAppEx.exe，微信小程序）才会自动选中；"
        "多个同进程窗口时取面积最大的主窗口。留空则仅按标题匹配。",
    "步长": "绘制时相邻移动点的插值间距（像素，默认 2px）：不影响笔画形状。"
        "调小更连贯、更慢；调大更快，过大可能断线。",
    "绘制速度": "1x=每步 6ms、无笔间停顿；低于 1x 延时加倍并加笔间停顿"
        "（窗口响应不过来/丢笔时用），高于 1x 加速。",
    "倒计时": "点「开始绘制」后的等待秒数：利用这段时间把鼠标挪开（前台模式）"
        "或确认小程序窗口就绪（后台模式）。",
    "输入方式": "前台鼠标：移动真实光标，兼容性最好；"
        "后台消息：向小程序窗口投递鼠标消息，不动鼠标、窗口可被遮挡，但不能最小化。",
}


def _tip_label(text: str, parent: QWidget | None = None) -> QLabel:
    """创建参数文字标签：鼠标悬浮显示 _LABEL_TIPS 中的详细说明。"""
    label = QLabel(text, parent)
    tip = _LABEL_TIPS.get(text)
    if tip:
        label.setToolTip(tip)
    return label


#: 小程序画布大约需要 3~4px 的位移才落笔，单笔短于此值可能丢步
_RISKY_STROKE_PX = 4.0
#: 笔画数超过该值时，小程序可能在保存时崩溃
_STROKE_COUNT_WARN = 2000
#: 超过该像素数的大图默认关闭阴影剔除（形态学滤波开销大，防加载卡顿）
_LARGE_IMAGE_PIXELS = 3_000_000
#: 绘制速度档位 → (每步延时秒, 笔间停顿秒)
_SPEED_TABLE: dict[str, tuple[float, float]] = {
    "0.25x": (0.024, 0.010),
    "0.5x": (0.012, 0.005),
    "0.75x": (0.008, 0.003),
    "1x": (0.006, 0.0),
    "1.5x": (0.004, 0.0),
    "2x": (0.003, 0.0),
    "3x": (0.002, 0.0),
    "4x": (0.0015, 0.0),
}


class _PreviewLabel(QLabel):
    """按可用空间等比缩放显示图片的预览标签。"""

    def __init__(
        self, placeholder: str, popup_title: str, parent: QWidget | None = None
    ) -> None:
        """创建预览标签。

        Args:
            placeholder: 未加载图片时显示的占位文字。
            popup_title: 双击弹出的独立预览窗口标题。
            parent: 父组件。
        """
        super().__init__(parent)
        self._source: QPixmap | None = None
        self._popup_title = popup_title
        self._popup: PreviewWindow | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(180, 110)
        self.setFrameShape(QLabel.Shape.StyledPanel)
        self.setText(placeholder)
        self.setToolTip("双击打开独立预览窗口：滚轮缩放、按住拖动平移查看细节。")

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt 签名
        """双击：打开/刷新独立预览窗口（全分辨率原图，可缩放平移）。"""
        if self._source is None or self._source.isNull():
            return
        if self._popup is None:
            self._popup = PreviewWindow(self._popup_title, self._source, self)
        else:
            self._popup.set_pixmap(self._source)
        self._popup.show()
        self._popup.raise_()
        self._popup.activateWindow()

    def set_image(self, pixmap: QPixmap) -> None:
        """设置要显示的源图，并按当前尺寸等比缩放。"""
        self._source = QPixmap(pixmap)
        self._rescale()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt 签名
        """窗口尺寸变化时重新缩放显示。"""
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        """按标签当前尺寸等比缩放源图。"""
        if self._source is None or self._source.isNull():
            return
        self.setPixmap(
            self._source.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class MainWindow(QMainWindow):
    """霸王茶姬杯贴自动绘制主窗口。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化状态与界面。"""
        super().__init__(parent)
        self.setWindowTitle("霸王茶姬杯贴自动绘制")
        # 初始尺寸按屏幕可用区域收窄，保证 720p（含任务栏）也能完整显示
        screen = QGuiApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else QRect(0, 0, 1280, 720)
        self.resize(min(1150, avail.width() - 40), min(780, avail.height() - 40))

        # ---- 运行状态 ----
        self._image_path: str | None = None
        self._gray: PILImage.Image | None = None  # 原图灰度版（用于二值化）
        self._rgb: PILImage.Image | None = None  # 原图彩色版（线条模式阴影剔除用）
        self._window: WindowInfo | None = None  # 当前选中的目标窗口
        self._canvas_rel: QRect | None = None  # 画布相对窗口左上角的矩形
        self._bounds_at_pick: tuple[int, int, int, int] | None = None  # 框选时窗口 bounds
        self._worker: DrawWorker | None = None  # 绘制线程（保引用防 GC）
        self._risk_mode = False  # 丢步警告预览模式：标红风险笔画、其余变灰
        self._eta_seconds: int | None = None  # 绘制剩余时间估算（秒）

        self._build_ui()
        self._connect_signals()
        self.statusBar().showMessage(
            "就绪：① 检测窗口 → ② 截图框选画布 → ③ 打开图片调参数 → ④ 开始绘制"
        )

    # ------------------------------------------------------------------ 界面
    def _build_ui(self) -> None:
        """构建三栏布局：左「图片」、右「画布」、底部「绘制」。"""
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.addWidget(splitter, 1)
        layout.addWidget(self._build_draw_panel())
        self.setCentralWidget(central)

    def _build_left_panel(self) -> QGroupBox:
        """构建左栏「图片」：打开图片、原图/二值化预览与二值化参数。"""
        group = QGroupBox("图片", self)
        layout = QVBoxLayout(group)

        self._open_btn = QPushButton("打开图片...", group)
        layout.addWidget(self._open_btn)

        self._original_preview = _PreviewLabel("原图预览", "原图预览", group)
        self._bin_preview = _PreviewLabel("绘制预览", "绘制预览", group)
        layout.addWidget(self._original_preview, 1)
        layout.addWidget(self._bin_preview, 1)
        self._stroke_info = QLabel("未框选画布：预览为原图分辨率", group)
        self._stroke_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._stroke_info)

        form_area = QWidget(group)
        form_col = QVBoxLayout(form_area)
        form_col.setContentsMargins(0, 0, 0, 0)
        form = QHBoxLayout()
        form_col.addLayout(form)

        # 阈值 0~255，默认 128
        form.addWidget(_tip_label("阈值", form_area))
        self._threshold_slider = QSlider(Qt.Orientation.Horizontal, form_area)
        self._threshold_slider.setRange(0, 255)
        self._threshold_slider.setValue(128)
        form.addWidget(self._threshold_slider, 1)
        self._threshold_value = QLabel("128", form_area)
        self._threshold_value.setMinimumWidth(32)
        form.addWidget(self._threshold_value)

        # 模糊 0~50 映射 0~5.0 半径，默认 0
        form.addWidget(_tip_label("模糊", form_area))
        self._blur_slider = QSlider(Qt.Orientation.Horizontal, form_area)
        self._blur_slider.setRange(0, 50)
        self._blur_slider.setValue(0)
        form.addWidget(self._blur_slider, 1)
        self._blur_value = QLabel("0.0", form_area)
        self._blur_value.setMinimumWidth(32)
        form.addWidget(self._blur_value)

        # 抽稀 1~10 像素，默认 2
        form.addWidget(_tip_label("抽稀", form_area))
        self._thin_slider = QSlider(Qt.Orientation.Horizontal, form_area)
        self._thin_slider.setRange(1, 10)
        self._thin_slider.setValue(2)
        form.addWidget(self._thin_slider, 1)
        self._thin_value = QLabel("2 px", form_area)
        self._thin_value.setMinimumWidth(44)
        form.addWidget(self._thin_value)

        # 画笔粗细 1~10 像素，默认 1：规划用线宽，>1 时自动合并冗余笔画
        form.addWidget(_tip_label("画笔粗细", form_area))
        self._pensize_spin = QSpinBox(form_area)
        self._pensize_spin.setRange(1, 10)
        self._pensize_spin.setValue(1)
        self._pensize_spin.setSuffix(" px")
        self._pensize_spin.setToolTip(
            "规划的画笔线宽，请与小程序里设置的画笔大小一致；"
            ">1 时自动合并被粗笔覆盖的冗余笔画，预览按该粗细渲染。"
        )
        form.addWidget(self._pensize_spin)

        # 最短笔画 0~20 像素，默认 2：过滤缩放产生的孤立点/碎段
        form.addWidget(_tip_label("最短笔画", form_area))
        self._minstroke_spin = QSpinBox(form_area)
        self._minstroke_spin.setRange(0, 20)
        self._minstroke_spin.setValue(2)
        self._minstroke_spin.setSuffix(" px")
        self._minstroke_spin.setToolTip(
            "路径长度小于该值的碎笔画（孤立点、微段）直接丢弃，0 = 不过滤。"
        )
        form.addWidget(self._minstroke_spin)

        self._invert_check = QCheckBox("反相（绘制浅色像素）", form_area)
        self._invert_check.setToolTip(
            "勾选后黑白判定反转：绘制图片中的浅色像素（适合深底浅色线条的图）。"
        )
        form.addWidget(self._invert_check)

        # 短笔画重复拖拽（防位移过短画布不落笔）
        exp_row = QHBoxLayout()
        self._mindrag_check = QCheckBox("短笔画延伸拖拽", form_area)
        self._mindrag_check.setChecked(True)
        self._mindrag_check.setToolTip(
            "画布按「按下点到最远拖拽点」的距离判定有效笔画。路径长度不足"
            "「最小拖拽」的短笔画自动沿最优方向延伸（优先延伸进就近的已涂黑"
            "区域，溢出无痕迹；附近无黑色时回退原地往返）。"
        )
        exp_row.addWidget(self._mindrag_check)
        exp_row.addWidget(_tip_label("最小拖拽", form_area))
        self._mindrag_spin = QSpinBox(form_area)
        self._mindrag_spin.setRange(3, 20)
        self._mindrag_spin.setValue(6)
        self._mindrag_spin.setSuffix(" px")
        exp_row.addWidget(self._mindrag_spin)
        exp_row.addStretch(1)
        self._mindrag_spin.setEnabled(self._mindrag_check.isChecked())
        self._mindrag_check.toggled.connect(self._mindrag_spin.setEnabled)
        form_col.addLayout(exp_row)

        # 自动连接相近线条（减少落笔次数）
        conn_row = QHBoxLayout()
        self._connect_check = QCheckBox("自动连接相近线条", form_area)
        self._connect_check.setChecked(True)
        self._connect_check.setToolTip(
            "端点距离不超过「连接距离」的相邻笔画首尾相连成一笔，"
            "减少落笔次数；对追踪器拆散的 1px 间隙线条尤其有效。"
        )
        conn_row.addWidget(self._connect_check)
        conn_row.addWidget(_tip_label("连接距离", form_area))
        self._connect_gap_spin = QDoubleSpinBox(form_area)
        self._connect_gap_spin.setRange(0.5, 5.0)
        self._connect_gap_spin.setSingleStep(0.5)
        self._connect_gap_spin.setValue(1.5)
        self._connect_gap_spin.setSuffix(" px")
        conn_row.addWidget(self._connect_gap_spin)
        conn_row.addStretch(1)
        self._connect_gap_spin.setEnabled(self._connect_check.isChecked())
        self._connect_check.toggled.connect(self._connect_gap_spin.setEnabled)
        form_col.addLayout(conn_row)

        # 线条模式（彩色插画）：DoG 线稿提取替代阈值二值化
        line_row = QHBoxLayout()
        self._line_mode_check = QCheckBox("线条模式（彩色插画）", form_area)
        self._line_mode_check.setChecked(False)
        self._line_mode_check.setToolTip(
            "适合彩色二次元插画：用 DoG 差分高斯检测比周边暗的线条结构，"
            "只画线条不涂色块；开启后「阈值/反相」不参与，线条色块将不再被涂黑。"
        )
        line_row.addWidget(self._line_mode_check)
        self._line_thresh_label = _tip_label("线条阈值", form_area)
        line_row.addWidget(self._line_thresh_label)
        self._line_thresh_spin = QSpinBox(form_area)
        self._line_thresh_spin.setRange(1, 60)
        self._line_thresh_spin.setValue(12)
        line_row.addWidget(self._line_thresh_spin)
        # 阴影剔除：过滤光影分界线（线条模式下默认开启，可关）；
        # 右侧紧跟「保护强度」短滑条
        self._shadow_check = QCheckBox("阴影剔除", form_area)
        self._shadow_check.setChecked(True)
        self._shadow_check.setToolTip(_LABEL_TIPS["阴影剔除"])
        line_row.addWidget(self._shadow_check)
        self._shadow_prot_label = _tip_label("保护强度", form_area)
        line_row.addWidget(self._shadow_prot_label)
        self._shadow_prot_slider = QSlider(Qt.Orientation.Horizontal, form_area)
        self._shadow_prot_slider.setRange(0, 100)
        self._shadow_prot_slider.setValue(40)
        self._shadow_prot_slider.setFixedWidth(80)
        line_row.addWidget(self._shadow_prot_slider)
        self._shadow_prot_value = QLabel("40", form_area)
        self._shadow_prot_value.setMinimumWidth(24)
        line_row.addWidget(self._shadow_prot_value)
        line_row.addStretch(1)
        self._line_thresh_spin.setEnabled(False)
        self._line_thresh_label.setEnabled(False)
        self._shadow_check.setEnabled(False)
        self._shadow_prot_slider.setEnabled(False)
        self._shadow_prot_label.setEnabled(False)
        self._shadow_prot_value.setEnabled(False)
        form_col.addLayout(line_row)

        layout.addWidget(form_area)
        return group

    def _build_right_panel(self) -> QGroupBox:
        """构建右栏「画布」：窗口检测、自动重设尺寸与画布框选。"""
        group = QGroupBox("画布", self)
        layout = QVBoxLayout(group)

        # 窗口检测
        kw_row = QHBoxLayout()
        kw_row.addWidget(_tip_label("窗口关键词", group))
        self._keyword_edit = QLineEdit("霸王茶姬", group)
        kw_row.addWidget(self._keyword_edit, 1)
        self._detect_btn = QPushButton("检测窗口", group)
        kw_row.addWidget(self._detect_btn)
        layout.addLayout(kw_row)

        proc_row = QHBoxLayout()
        proc_row.addWidget(_tip_label("进程名", group))
        self._process_edit = QLineEdit("WeChatAppEx.exe", group)
        self._process_edit.setToolTip(
            "标题与进程名都匹配的窗口会被自动选中（多个时取面积最大的主窗口），"
            "不再弹窗；留空则仅按标题匹配，多结果时弹窗选择。"
        )
        proc_row.addWidget(self._process_edit, 1)
        layout.addLayout(proc_row)

        self._window_label = QLabel("未检测窗口", group)
        self._window_label.setWordWrap(True)
        layout.addWidget(self._window_label)

        self._capture_btn = QPushButton("截取窗口并自动定位画布", group)
        layout.addWidget(self._capture_btn)

        self._canvas_label = QLabel("未框选画布", group)
        self._canvas_label.setWordWrap(True)
        layout.addWidget(self._canvas_label)

        layout.addStretch(1)
        return group

    def _build_draw_panel(self) -> QGroupBox:
        """构建底部「绘制」参数栏与开始/停止按钮、进度条。"""
        group = QGroupBox("绘制", self)
        layout = QHBoxLayout(group)

        layout.addWidget(_tip_label("步长", group))
        self._step_spin = QSpinBox(group)
        self._step_spin.setRange(1, 10)
        self._step_spin.setValue(2)
        self._step_spin.setSuffix(" px")
        layout.addWidget(self._step_spin)

        # 绘制速度：档位 → (每步延时, 笔间停顿)，见 _SPEED_TABLE
        layout.addWidget(_tip_label("绘制速度", group))
        self._speed_combo = QComboBox(group)
        self._speed_combo.addItems(
            ("0.25x", "0.5x", "0.75x", "1x", "1.5x", "2x", "3x", "4x")
        )
        self._speed_combo.setCurrentIndex(3)  # 默认 1x
        self._speed_combo.setToolTip(
            "1x=每步 6ms、无笔间停顿；低于 1x 逐步加倍延时并加笔间停顿"
            "（窗口响应不过来/丢笔时用），高于 1x 加速。"
        )
        layout.addWidget(self._speed_combo)

        layout.addWidget(_tip_label("倒计时", group))
        self._countdown_spin = QSpinBox(group)
        self._countdown_spin.setRange(0, 10)
        self._countdown_spin.setValue(3)
        self._countdown_spin.setSuffix(" s")
        layout.addWidget(self._countdown_spin)

        # 输入方式：前台真实鼠标 / 后台 PostMessage 消息
        layout.addWidget(_tip_label("输入方式", group))
        self._backend_combo = QComboBox(group)
        self._backend_combo.addItems(
            (
                "前台鼠标（移动真实光标）",
                "后台消息（窗口不可最小化）",
            )
        )
        self._backend_combo.setCurrentIndex(1)  # 默认后台消息
        layout.addWidget(self._backend_combo)



        self._start_btn = QPushButton("开始绘制", group)
        layout.addWidget(self._start_btn)
        self._stop_btn = QPushButton("停止", group)
        self._stop_btn.setEnabled(False)
        layout.addWidget(self._stop_btn)

        self._progress = QProgressBar(group)
        self._progress.setValue(0)
        layout.addWidget(self._progress, 1)
        self._progress_label = QLabel("第 -/- 笔", group)
        layout.addWidget(self._progress_label)
        return group

    def _connect_signals(self) -> None:
        """连接所有信号与槽。"""
        # 左栏
        self._open_btn.clicked.connect(self._open_image)
        self._threshold_slider.valueChanged.connect(self._on_params_changed)
        self._blur_slider.valueChanged.connect(self._on_params_changed)
        self._thin_slider.valueChanged.connect(self._on_params_changed)
        self._minstroke_spin.valueChanged.connect(self._on_params_changed)
        self._pensize_spin.valueChanged.connect(self._on_params_changed)
        self._mindrag_check.toggled.connect(self._on_params_changed)
        self._mindrag_spin.valueChanged.connect(self._on_params_changed)
        self._connect_check.toggled.connect(self._on_params_changed)
        self._connect_gap_spin.valueChanged.connect(self._on_params_changed)
        self._line_mode_check.toggled.connect(self._on_line_mode_toggled)
        self._line_thresh_spin.valueChanged.connect(self._on_params_changed)
        self._shadow_check.toggled.connect(self._on_params_changed)
        self._shadow_check.toggled.connect(self._sync_shadow_prot_enabled)
        self._shadow_prot_slider.valueChanged.connect(self._on_params_changed)
        self._invert_check.toggled.connect(self._on_params_changed)
        # 右栏
        self._detect_btn.clicked.connect(self._detect_windows)
        self._capture_btn.clicked.connect(self._capture_and_pick)
        # 底部
        self._start_btn.clicked.connect(self._start_draw)
        self._stop_btn.clicked.connect(self._stop_draw)
        # 丢步警告（富文本链接）
        self._stroke_info.linkActivated.connect(self._on_risk_link_clicked)

    # ------------------------------------------------------------------ 图片
    def _current_params(self) -> BinarizeParams:
        """从界面控件读取当前二值化参数。"""
        return BinarizeParams(
            threshold=self._threshold_slider.value(),
            blur_radius=self._blur_slider.value() / 10.0,
            line_mode=self._line_mode_check.isChecked(),
            line_threshold=float(self._line_thresh_spin.value()),
            reject_shadow=self._shadow_check.isChecked(),
            shadow_protection=float(self._shadow_prot_slider.value()),
            invert=self._invert_check.isChecked(),
        )

    def _sync_param_labels(self) -> None:
        """刷新阈值/模糊/抽稀/保护强度的数值标签。"""
        self._threshold_value.setText(str(self._threshold_slider.value()))
        self._blur_value.setText(f"{self._blur_slider.value() / 10.0:.1f}")
        self._thin_value.setText(f"{self._thin_slider.value()} px")
        self._shadow_prot_value.setText(str(self._shadow_prot_slider.value()))

    def _sync_shadow_prot_enabled(self) -> None:
        """保护强度滑条仅在线条模式开启且勾选阴影剔除时可用。"""
        enabled = (
            self._line_mode_check.isChecked() and self._shadow_check.isChecked()
        )
        self._shadow_prot_slider.setEnabled(enabled)
        self._shadow_prot_label.setEnabled(enabled)
        self._shadow_prot_value.setEnabled(enabled)

    def _on_line_mode_toggled(self, checked: bool) -> None:
        """线条模式开关：切换可用控件并刷新预览。"""
        self._threshold_slider.setEnabled(not checked)
        self._threshold_value.setEnabled(not checked)
        self._invert_check.setEnabled(not checked)
        self._line_thresh_spin.setEnabled(checked)
        self._line_thresh_label.setEnabled(checked)
        self._shadow_check.setEnabled(checked)
        self._sync_shadow_prot_enabled()
        self._on_params_changed()

    def _on_params_changed(self) -> None:
        """任一二值化参数变化：刷新标签，若已加载图片则重算预览。"""
        self._sync_param_labels()
        if self._gray is not None:
            self._update_binarize_preview()

    def _open_image(self) -> None:
        """弹出文件对话框打开图片，并生成原图与二值化预览。"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*.*)",
        )
        if not path:
            return
        try:
            self._gray = load_grayscale(path)
            with PILImage.open(path) as img:
                rgb = img.convert("RGB")  # 转换副本，脱离文件句柄独立存在
                original_pix = pil_to_pixmap(rgb)
                colorful = is_colorful(img)
        except Exception as e:  # noqa: BLE001 - 用户选择的文件可能任意损坏
            QMessageBox.critical(self, "打开图片", f"无法读取图片：{e}")
            return
        self._image_path = path
        self._rgb = rgb
        self._original_preview.set_image(original_pix)
        # 超大图：线条模式检测（含阴影剔除滤波）在原图全分辨率执行，开销随
        # 像素数线性增长，默认关闭阴影剔除防卡顿（在开启线条模式前设置，
        # 避免首次预览就带滤镜重算）
        huge = rgb.size[0] * rgb.size[1] > _LARGE_IMAGE_PIXELS
        if huge:
            self._shadow_check.setChecked(False)
        # 自动检测：大面积彩色（如二次元插画）→ 开启线条模式；灰度图 → 关闭
        self._line_mode_check.setChecked(colorful)
        self._update_binarize_preview()
        hint = (
            "（检测到大面积彩色，已自动开启线条模式）" if colorful else ""
        )
        if huge:
            hint += "（图片超过 300 万像素，已默认关闭阴影剔除防卡顿）"
        self.statusBar().showMessage(f"已加载图片：{path}{hint}", 8000)

    def _compute_strokes(
        self,
    ) -> tuple[list[list[tuple[int, int]]], tuple[int, int] | None]:
        """按当前参数规划笔画：二值化(画布尺寸) → 抽稀追踪 → 碎笔画过滤
        → 按画笔粗细合并冗余。

        预览与绘制共用本方法，保证「预览即实绘」。未框选画布时 target
        为 None（按原图尺寸），不做碎笔画过滤与冗余合并。

        Raises:
            Exception: binarize/trace_paths 参数或图像异常时原样抛出。
        """
        assert self._gray is not None
        target: tuple[int, int] | None = None
        if self._canvas_rel is not None:
            target = (self._canvas_rel.width(), self._canvas_rel.height())
        mask, _offset = binarize(
            self._gray,
            self._current_params(),
            target_w=target[0] if target else None,
            target_h=target[1] if target else None,
            rgb=self._rgb,
        )
        strokes = trace_paths(mask, min_step=float(self._thin_slider.value()))
        if target is not None:
            strokes = filter_strokes(
                strokes, float(self._minstroke_spin.value())
            )
            strokes = deduplicate_strokes(
                strokes, int(self._pensize_spin.value())
            )
            if self._connect_check.isChecked():
                strokes = connect_nearby_strokes(
                    strokes, float(self._connect_gap_spin.value())
                )
            if self._mindrag_check.isChecked():
                strokes = extend_short_strokes(
                    strokes, float(self._mindrag_spin.value()), mask
                )
        return strokes, target

    def _render_strokes_preview(
        self,
        strokes: list[list[tuple[int, int]]],
        size: tuple[int, int],
        pen_width: int,
        risky_idx: frozenset[int] = frozenset(),
        dim_normal: bool = False,
    ) -> PILImage.Image:
        """把笔画按规划笔粗渲染到白底画布上（即实际绘制效果）。

        Args:
            strokes: 笔画列表。
            size: 画布尺寸 (w, h)。
            pen_width: 规划笔粗（像素）。
            risky_idx: 丢步风险笔画的下标集合，渲染为红色。
            dim_normal: True 时非风险笔画变浅灰（标红模式下突出风险笔画）。
        """
        normal = (210, 210, 210) if dim_normal else (0, 0, 0)
        img = PILImage.new("RGB", size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        for i, stroke in enumerate(strokes):
            color = (211, 47, 47) if i in risky_idx else normal
            if len(stroke) > 1:
                draw.line(
                    [(float(x), float(y)) for x, y in stroke],
                    fill=color,
                    width=max(1, pen_width),
                    joint="curve",
                )
            elif stroke:
                if pen_width <= 1:
                    draw.point((float(stroke[0][0]), float(stroke[0][1])), fill=color)
                else:
                    r = pen_width / 2.0
                    cx, cy = float(stroke[0][0]), float(stroke[0][1])
                    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
        return img

    def _update_binarize_preview(self) -> None:
        """按当前参数重算绘制预览。

        已框选画布时规划实际笔画并按笔粗渲染——预览即实绘结果（含缩放、
        抽稀、碎笔画过滤与冗余合并）。单笔长度小于 _RISKY_STROKE_PX 的
        笔画在小程序画布上可能丢步（位移不足不落笔），存在时显示可点击
        的红色警告，点击后预览中标红这些笔画、其余变灰；未框选时退回
        原图分辨率的二值化预览。
        """
        assert self._gray is not None
        try:
            strokes, target = self._compute_strokes()
        except Exception as e:  # noqa: BLE001 - 参数可能越界，兜底提示
            self.statusBar().showMessage(f"二值化失败：{e}")
            return
        if target is None:
            mask, _offset = binarize(
                self._gray, self._current_params(), rgb=self._rgb
            )
            self._bin_preview.set_image(pil_to_pixmap(mask_to_preview(mask)))
            self._stroke_info.setText("未框选画布：预览为原图分辨率")
            self._risk_mode = False
            return
        pen = int(self._pensize_spin.value())
        risky_idx = frozenset(
            i
            for i, s in enumerate(strokes)
            if path_length(s) < _RISKY_STROKE_PX
        )
        base = (
            f"实际笔画: {len(strokes)} 笔 / {count_points(strokes)} 点"
            f"（笔粗 {pen}px，最短 {self._minstroke_spin.value()}px）"
        )
        crash_warn = ""
        if len(strokes) > _STROKE_COUNT_WARN:
            crash_warn = (
                "<br><span style='color:#d32f2f;font-weight:bold;'>"
                f"⚠ 请注意，笔画数大于{_STROKE_COUNT_WARN}时，小程序可能会在"
                "保存时崩溃（建议依次调大「自动连接相近线条→连接距离」"
                "→「最短笔画」→「抽稀」减少笔画）"
                "</span>"
            )
        if risky_idx:
            tip = (
                f"{base}{crash_warn}　<a href='risk' "
                f"style='color:#d32f2f;font-weight:bold;'>"
                f"⚠ {len(risky_idx)} 笔可能丢步（单笔<{_RISKY_STROKE_PX:.0f}px），"
                f"点击{'恢复' if self._risk_mode else '标红查看'}</a>"
            )
            self._stroke_info.setTextFormat(Qt.TextFormat.RichText)
            self._stroke_info.setText(tip)
            self._bin_preview.set_image(
                pil_to_pixmap(
                    self._render_strokes_preview(
                        strokes, target, pen, risky_idx, self._risk_mode
                    )
                )
            )
        else:
            self._risk_mode = False
            self._stroke_info.setTextFormat(Qt.TextFormat.RichText)
            self._stroke_info.setText(base + crash_warn)
            self._bin_preview.set_image(
                pil_to_pixmap(self._render_strokes_preview(strokes, target, pen))
            )

    def _on_risk_link_clicked(self, _link: str) -> None:
        """点击丢步警告：切换「标红风险笔画」预览模式并刷新。"""
        self._risk_mode = not self._risk_mode
        if self._gray is not None:
            self._update_binarize_preview()

    # ------------------------------------------------------------------ 窗口
    def _detect_windows(self) -> None:
        """按标题关键词 + 进程名强制匹配窗口，无任何回退。

        只认「标题与进程名都匹配」的窗口（多个时取面积最大的主窗口）；
        没有匹配直接报错——标题命中但进程不符的窗口（例如本程序自己
        的窗口）会被明确忽略。
        """
        keyword = self._keyword_edit.text().strip()
        if not keyword:
            self.statusBar().showMessage("请输入窗口关键词。")
            return
        process = self._process_edit.text().strip()
        if not process:
            self.statusBar().showMessage("请输入进程名（如 WeChatAppEx.exe）。")
            return
        try:
            matches = list_matches(keyword)
        except Exception as e:  # noqa: BLE001 - Win32 枚举可能异常
            QMessageBox.critical(self, "检测窗口", f"窗口检测失败：{e}")
            return

        exact = [m for m in matches if m.process_name.lower() == process.lower()]
        if not exact:
            detail = (
                f"（忽略 {len(matches)} 个标题命中但进程不符的窗口）"
                if matches
                else ""
            )
            QMessageBox.warning(
                self,
                "未找到目标窗口",
                f"未找到标题包含“{keyword}”且进程为“{process}”的窗口{detail}。"
                "请确认小程序已在电脑微信中打开。",
            )
            return
        # 双重匹配：自动选面积最大的（主窗口）
        info = max(exact, key=lambda m: (m.rect[2] - m.rect[0]) * (m.rect[3] - m.rect[1]))
        extra = f"（{len(exact)} 个同进程窗口中取最大）" if len(exact) > 1 else ""
        self._window = info
        self._refresh_window_label()
        self.statusBar().showMessage(f"已自动选中窗口：{info.title}{extra}", 5000)
        # 检测成功后自动执行一次画布定位
        self._capture_and_pick()

    def _refresh_window_label(self) -> None:
        """刷新窗口信息标签（标题/进程/句柄/当前尺寸）。"""
        info = self._window
        if info is None:
            self._window_label.setText("未检测窗口")
            return
        w = info.rect[2] - info.rect[0]
        h = info.rect[3] - info.rect[1]
        self._window_label.setText(
            f"标题：{info.title}\n进程：{info.process_name}（PID {info.pid}）\n"
            f"句柄：{info.hwnd}\n当前尺寸：{w}×{h}"
        )

    def _capture_and_pick(self) -> None:
        """截取窗口画面并自动定位画布（居中且宽高占比 ≥50%），不弹窗。

        定位失败时提示用户检查小程序界面状态。
        """
        if self._window is None:
            self.statusBar().showMessage("请先点击“检测窗口”选择目标窗口。")
            return
        try:
            bounds = get_bounds(self._window.hwnd)
            shot = capture(self._window.hwnd)
        except Exception as e:  # noqa: BLE001 - Win32 调用可能失败
            QMessageBox.critical(self, "定位画布", f"截取窗口画面失败：{e}")
            return
        box = detect_white_canvas(np.asarray(shot.convert("RGB")))
        window_size = (bounds[2] - bounds[0], bounds[3] - bounds[1])
        if box is None or not validate_canvas_position(box, window_size):
            QMessageBox.warning(
                self,
                "无法定位画布",
                "无法正确定位画布，请确保已经进入定制界面，并且正确选择了画笔后，"
                "隐藏画笔大小调整界面使得画布完整出现不被遮盖",
            )
            return
        x0, y0, x1, y1 = box
        rel = QRect(x0, y0, x1 - x0 + 1, y1 - y0 + 1)
        self._canvas_rel = rel
        self._bounds_at_pick = bounds
        left = bounds[0] + rel.x()
        top = bounds[1] + rel.y()
        self._canvas_label.setText(
            f"已定位画布: 相对({rel.x()},{rel.y()},{rel.width()}×{rel.height()}) / "
            f"屏幕({left},{top},{left + rel.width()},{top + rel.height()})"
        )
        # 画布尺寸变化 → 绘制预览改按实际绘制分辨率渲染
        if self._gray is not None:
            self._update_binarize_preview()
        self.statusBar().showMessage("画布定位完成，可打开图片调整参数。", 5000)

    def _start_draw(self) -> None:
        """开始绘制：校验 → 生成笔画 → 启动后台线程。"""
        if self._worker is not None and self._worker.isRunning():
            self.statusBar().showMessage("绘制已在进行中。")
            return
        # ① 前置校验
        if self._window is None:
            self.statusBar().showMessage("请先点击“检测窗口”选择目标窗口。")
            return
        if self._canvas_rel is None or self._bounds_at_pick is None:
            self.statusBar().showMessage("请先“截取窗口并自动定位画布”。")
            return
        if self._gray is None:
            self.statusBar().showMessage("请先打开一张图片。")
            return
        # ② 窗口位置一致性检查
        try:
            bounds = get_bounds(self._window.hwnd)
        except Exception as e:  # noqa: BLE001 - Win32 调用可能失败
            QMessageBox.critical(self, "开始绘制", f"获取窗口位置失败：{e}")
            return
        if bounds != self._bounds_at_pick:
            answer = QMessageBox.warning(
                self,
                "窗口已变化",
                "检测到窗口位置或大小与框选画布时不一致，画布位置可能已偏移。\n"
                "建议重新“截取窗口画面并框选画布”。是否仍要继续绘制？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        # ③ 画布绝对区域
        origin_x = bounds[0] + self._canvas_rel.x()
        origin_y = bounds[1] + self._canvas_rel.y()
        # ⑤ 生成笔画（与预览共用同一规划：缩放二值化 → 抽稀追踪 → 碎笔画过滤）
        try:
            strokes, _target = self._compute_strokes()
        except Exception as e:  # noqa: BLE001 - 参数可能异常
            QMessageBox.critical(self, "开始绘制", f"生成笔画失败：{e}")
            return
        total = len(strokes)
        if total == 0:
            self.statusBar().showMessage(
                "没有生成任何笔画：请调整阈值/反相等参数，或确认框选的画布区域正确。"
            )
            return
        # ⑥ 构造输入后端（后台消息需在 UI 线程先解析 Chromium 子窗口，
        #    失败直接报错中止，不启动 worker）
        backend_desc = ""
        if self._backend_combo.currentIndex() == 1:
            if self._window is None:
                self.statusBar().showMessage("后台消息输入需要先检测并选中目标窗口。")
                return
            backend = PostMessageBackend(self._window.hwnd)
            try:
                backend_desc = backend.resolve()
            except Exception as e:  # noqa: BLE001 - Win32 枚举可能异常
                QMessageBox.critical(self, "开始绘制", f"解析后台消息目标失败：{e}")
                return
        else:
            backend = ForegroundMouseBackend()
        # ⑦ 启动后台绘制线程
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._progress.setRange(0, total)
        self._progress.setValue(0)
        self._eta_seconds = None
        self._progress_label.setText(f"第 0/{total} 笔")
        prefix = f"{backend_desc}；" if backend_desc else ""
        if self._backend_combo.currentIndex() == 1:
            move_hint = "绘制期间请勿把光标移到小程序窗口显示的区域内"
        else:
            move_hint = "请把鼠标移到不碍事的位置，绘制期间不要移动鼠标"
        self.statusBar().showMessage(
            prefix
            + f"共 {total} 笔 / {count_points(strokes)} 点，"
            f"{self._countdown_spin.value()} 秒后开始，{move_hint}；"
            "绘制中随时按 Esc 急停"
        )
        worker = DrawWorker(
            strokes,
            origin_x,
            origin_y,
            backend,
            step_px=float(self._step_spin.value()),
            delay_s=_SPEED_TABLE[self._speed_combo.currentText()][0],
            stroke_gap_s=_SPEED_TABLE[self._speed_combo.currentText()][1],
            countdown_s=float(self._countdown_spin.value()),
            parent=self,
        )
        worker.progress.connect(self._on_draw_progress)
        worker.remaining.connect(self._on_draw_remaining)
        worker.done.connect(self._on_draw_done)
        self._worker = worker
        worker.start()

    def _on_draw_progress(self, current: int, total: int) -> None:
        """绘制进度槽：更新进度条与标签（含剩余时间）。"""
        self._progress.setValue(current)
        self._update_progress_label()

    def _on_draw_remaining(self, seconds: int) -> None:
        """剩余时间估算槽：记录并刷新进度标签。"""
        self._eta_seconds = seconds
        self._update_progress_label()

    def _update_progress_label(self) -> None:
        """按当前进度与剩余时间估算刷新进度标签（始终显示剩余时间）。"""
        total = self._progress.maximum()
        current = self._progress.value()
        text = f"第 {current}/{total} 笔"
        if self._eta_seconds is not None:
            text += f" · 剩余约 {self._fmt_duration(self._eta_seconds)}"
        self._progress_label.setText(text)

    @staticmethod
    def _fmt_duration(seconds: int) -> str:
        """把秒数格式化为「X分Y秒」或「Y秒」。"""
        total = max(0, int(seconds))
        if total < 60:
            return f"{total} 秒"
        return f"{total // 60} 分 {total % 60} 秒"

    def _on_draw_done(self, ok: bool, message: str) -> None:
        """绘制结束槽：恢复按钮并显示结果消息。"""
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self.statusBar().showMessage(message if ok else f"绘制结束：{message}")
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.wait(5000)
            worker.deleteLater()

    def _stop_draw(self) -> None:
        """请求停止绘制线程。"""
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.stop()
            self.statusBar().showMessage("正在停止绘制……")

    def closeEvent(self, event) -> None:  # noqa: ANN001 - Qt 事件对象类型宽泛
        """关闭窗口：若绘制线程在跑先请求停止并等待。"""
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.stop()
            worker.wait(3000)
        super().closeEvent(event)
        event.accept()
