import sys
import os
import shutil
from typing import List, Optional, TextIO, Any
import uuid

import soundfile as sf
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QTextEdit,
    QFileDialog, QGroupBox, QScrollArea, QMessageBox, QTabWidget, QFormLayout, QFrame,
)
from PySide6.QtCore import (
    Qt, Signal, Slot, QObject,
)
from PySide6.QtGui import QTextCursor, QCloseEvent

from ..Utils.TextSplitter import TextSplitter
from .Utils import (
    generate_output_filenames, FileSelectorWidget, FileSelectionMode, MyComboBox, sanitize_filename,
    MyTextEdit
)
from .AudioPlayer import AudioPlayer
from .PresetManager import PresetManager
from .ServerManager import InferenceWorker
from .ConverterWidget import ConverterWidget

"""
抄自 Genie CUDA Runtime
"""

CACHE_DIR = './UserData/Cache/GenieGUI'
os.makedirs(CACHE_DIR, exist_ok=True)

TEMP_MODEL_NAME: str = 'Genie-TTS GUI'


# ==================== 后台工作线程 ====================

class LogRedirector(QObject):
    """重定向 stdout 到 Signal"""
    textWritten = Signal(str)

    def __init__(self):
        super().__init__()
        self._old_stdout: TextIO = sys.stdout

    def write(self, text: Any):
        text = str(text)
        self.textWritten.emit(text)
        if self._old_stdout is not None:
            self._old_stdout.write(text)

    def flush(self):
        pass


# ==================== UI 组件实现 ====================

class PreviewItemWidget(QFrame):
    """单条音频预览组件"""

    def __init__(
            self,
            index: int,
            text: str,
            file_path: str,
            player: AudioPlayer,
            parent: QWidget = None
    ):
        super().__init__(parent)
        self.text: str = text
        self.file_path: str = file_path
        self.player: AudioPlayer = player
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # 编号
        lbl_id = QLabel(f"#{index}")
        lbl_id.setFixedWidth(40)
        lbl_id.setStyleSheet("font-weight: bold; color: #555;")

        # 文本
        lbl_text = QLabel(text)
        lbl_text.setFixedWidth(240)
        lbl_text.setToolTip(text)

        # 按钮 - 播放
        btn_play = QPushButton("▶ 播放")
        btn_play.setFixedWidth(80)
        btn_play.clicked.connect(self._play_audio)

        # 按钮 - 保存
        btn_save = QPushButton("⬇ 保存")
        btn_save.setFixedWidth(80)
        btn_save.clicked.connect(self._save_file)

        # 按钮 - 删除 (新增)
        btn_del = QPushButton("🗑 删除")
        btn_del.setFixedWidth(80)
        btn_del.setStyleSheet("color: #ff4d4d;")  # 以此区分删除按钮
        btn_del.clicked.connect(self._delete_item)

        layout.addWidget(lbl_id)
        layout.addWidget(lbl_text, 1)  # Stretch
        layout.addWidget(btn_play)
        layout.addWidget(btn_save)
        layout.addWidget(btn_del)  # 添加到布局

    def _play_audio(self):
        # 播放前先停止其他播放
        self.player.stop()
        self.player.play(self.file_path)

    def _save_file(self):
        filename = sanitize_filename(self.text)
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存音频", f"{filename}.wav", "WAV Audio (*.wav)"
        )
        if save_path:
            try:
                shutil.copy(self.file_path, save_path)
                QMessageBox.information(self, "成功", "文件保存成功！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def _delete_item(self):
        """删除当前条目及对应的文件"""
        # 1. 停止播放
        self.player.stop()

        # 2. 尝试删除物理文件 (避免垃圾堆积)
        try:
            if os.path.exists(self.file_path):
                os.remove(self.file_path)
                print(f"[INFO] 已删除文件: {self.file_path}")
        except Exception as e:
            print(f"[WARN] 删除文件失败: {e}")

        # 3. 从界面移除自身
        self.deleteLater()


class LogWidget(QWidget):
    """日志显示Tab"""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.text_edit: QTextEdit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet(
            "background-color: #1e1e1e;"
            "color: #ecf0f1;"
            "font-family: Consolas;"
            "font-size: 12pt;"
        )
        layout.addWidget(self.text_edit)

    @Slot(str)
    def append_log(self, text: str):
        self.text_edit.moveCursor(QTextCursor.MoveOperation.End)
        self.text_edit.insertPlainText(text)
        self.text_edit.moveCursor(QTextCursor.MoveOperation.End)


class TTSWidget(QWidget):
    """TTS 主交互界面"""

    def __init__(self, player: AudioPlayer, parent: QWidget = None):
        super().__init__(parent)
        self.player: AudioPlayer = player
        self.splitter: TextSplitter = TextSplitter()
        self.current_gen_id: int = 0
        self.current_worker: Optional[InferenceWorker] = None

        main_layout = QVBoxLayout(self)

        # ---------------- 顶部：预设管理器 ----------------
        self.preset_manager = PresetManager(
            presets_file='./UserData/GenieGuiConfig.json',
            state_getter=self.get_ui_state,
        )
        self.preset_manager.sig_load_state.connect(self.apply_ui_state)
        main_layout.addWidget(self.preset_manager)

        # ---------------- 中间：滚动设置区 ----------------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()

        content_layout = QHBoxLayout(content_widget)
        left_column_layout = QVBoxLayout()
        right_column_layout = QVBoxLayout()

        # ==================== 左侧列内容 ====================

        # 模型设置组
        group_model = QGroupBox("模型设置")
        self.layout_model = QFormLayout()
        self.combo_model_type = MyComboBox()
        self.combo_model_type.addItems(["Genie-TTS"])
        self.combo_model_type.currentTextChanged.connect(self._update_model_ui_visibility)
        self.combo_model_type.setEnabled(False)
        self.file_gpt = FileSelectorWidget("gpt_path", FileSelectionMode.FILE, "Checkpoints (*.ckpt)")
        self.file_vits = FileSelectorWidget("vits_path", FileSelectionMode.FILE, "Models (*.pth)")
        self.file_genie = FileSelectorWidget("genie_dir", FileSelectionMode.DIRECTORY)
        self.file_gpt.pathChanged.connect(self._on_gpt_path_changed)
        self.file_vits.pathChanged.connect(self._on_vits_path_changed)
        self.layout_model.addRow("模型类型:", self.combo_model_type)
        self.layout_model.addRow("GPT模型 (.ckpt):", self.file_gpt)
        self.layout_model.addRow("VITS模型 (.pth):", self.file_vits)
        self.layout_model.addRow("Genie模型目录:", self.file_genie)
        group_model.setLayout(self.layout_model)
        # 参考音频组
        group_ref = QGroupBox("参考音频")
        layout_ref = QFormLayout()
        self.file_ref_audio = FileSelectorWidget("ref_audio", FileSelectionMode.FILE, "Audio (*.wav *.mp3)")
        self.input_ref_text = QLineEdit()
        self.input_ref_text.setPlaceholderText("请输入参考音频对应的文本...")
        btn_play_ref = QPushButton("▶️")
        btn_play_ref.setFixedWidth(30)
        btn_play_ref.clicked.connect(self._play_ref_audio)
        hbox_ref_text = QHBoxLayout()
        hbox_ref_text.addWidget(self.input_ref_text)
        hbox_ref_text.addWidget(btn_play_ref)
        layout_ref.addRow("音频文件:", self.file_ref_audio)
        layout_ref.addRow("音频文本:", hbox_ref_text)
        group_ref.setLayout(layout_ref)

        left_column_layout.addWidget(group_model)
        left_column_layout.addWidget(group_ref)
        left_column_layout.addStretch()

        # ==================== 右侧列内容 ====================

        # === 推理设置组 ===
        group_infer = QGroupBox("推理参数")
        layout_infer = QFormLayout()
        self.combo_device = MyComboBox()
        self.combo_device.addItems(["CPU"])
        self.combo_device.setEnabled(False)
        self.combo_quality = MyComboBox()
        self.combo_quality.addItems(["质量优先"])
        self.combo_quality.setEnabled(False)
        self.combo_split = MyComboBox()
        self.combo_split.addItems(["不切分", "智能切分", "按行切分"])
        self.combo_mode = MyComboBox()
        self.combo_mode.addItems(["串行推理"])
        self.combo_mode.setEnabled(False)
        self.combo_lang = MyComboBox()
        self.combo_lang.addItems(["Chinese", "English", "Japanese"])
        layout_infer.addRow("推理设备:\n(重启生效)", self.combo_device)
        layout_infer.addRow("推理需求:", self.combo_quality)
        layout_infer.addRow("分句方式:", self.combo_split)
        layout_infer.addRow("推理模式:", self.combo_mode)
        layout_infer.addRow("目标语言:", self.combo_lang)
        group_infer.setLayout(layout_infer)

        # === 自动保存组 ===
        group_save = QGroupBox("自动保存设置")
        self.layout_save = QFormLayout()
        self.combo_save_mode = MyComboBox()
        self.combo_save_mode.addItems(["禁用自动保存", "保存为单个文件", "保存为多个文件"])
        self.combo_save_mode.currentIndexChanged.connect(self._update_save_ui_state)
        default_out_path = os.path.join(os.path.expanduser("~"), "Desktop", "Genie 输出语音")
        self.file_out_dir = FileSelectorWidget("out_dir", FileSelectionMode.DIRECTORY)
        self.file_out_dir.set_path(default_out_path)
        self.layout_save.addRow("保存方式:", self.combo_save_mode)
        self.layout_save.addRow("输出文件夹:", self.file_out_dir)
        group_save.setLayout(self.layout_save)

        right_column_layout.addWidget(group_infer)
        right_column_layout.addWidget(group_save)
        right_column_layout.addStretch()

        content_layout.addLayout(left_column_layout, 1)
        content_layout.addLayout(right_column_layout, 1)
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll, 5)

        # ==================== 底部：输入控制 + 输出预览 ====================

        # 创建底部容器 widget
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)  # 去除边距让它贴合

        # --- 输入控制组 ---
        group_input = QGroupBox("目标文本")
        layout_input = QVBoxLayout()
        self.text_input = MyTextEdit()
        self.text_input.setPlaceholderText("请输入要合成的目标文本...")
        self.text_input.setFixedHeight(300)
        self.btn_start = QPushButton("开始推理")
        self.btn_start.setFixedHeight(40)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold; 
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.btn_start.clicked.connect(self._start_inference)
        layout_input.addWidget(self.text_input)
        layout_input.addWidget(self.btn_start)
        group_input.setLayout(layout_input)

        # --- 输出预览组 ---
        group_preview = QGroupBox("输出音频预览")
        preview_layout = QVBoxLayout()
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_container = QWidget()
        self.preview_list_layout = QVBoxLayout(self.preview_container)
        self.preview_list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.preview_scroll.setWidget(self.preview_container)
        preview_layout.addWidget(self.preview_scroll)
        group_preview.setLayout(preview_layout)

        bottom_layout.addWidget(group_input, 1)
        bottom_layout.addWidget(group_preview, 1)
        main_layout.addWidget(bottom_widget, 3)

        self.apply_ui_state(self.preset_manager.current_preset_data)

    # ==================== 状态管理接口 (供 PresetManager 调用) ====================

    @property
    def current_preset_name(self) -> str:
        return self.preset_manager.current_preset_name

    @property
    def current_preset_data(self) -> dict:
        return self.preset_manager.current_preset_data

    def get_ui_state(self) -> dict:
        """收集当前UI状态为字典"""
        return {
            "model_type": self.combo_model_type.currentText(),
            "gpt_path": self.file_gpt.get_path(),
            "vits_path": self.file_vits.get_path(),
            "genie_dir": self.file_genie.get_path(),
            "ref_audio": self.file_ref_audio.get_path(),
            "ref_text": self.input_ref_text.text(),
            "device": self.combo_device.currentText().lower(),
            "quality": self.combo_quality.currentText(),
            "split": self.combo_split.currentText(),
            "mode": self.combo_mode.currentText(),
            "lang": self.combo_lang.currentText(),
            "save_mode": self.combo_save_mode.currentText(),
            "out_dir": self.file_out_dir.get_path()
        }

    @Slot(dict)
    def apply_ui_state(self, data: dict) -> None:
        """将字典数据应用到UI"""

        def set_combo_text(combo: MyComboBox, text: str) -> None:
            index = combo.findText(text)
            if index >= 0:
                combo.setCurrentIndex(index)

        set_combo_text(self.combo_model_type, data.get("model_type", ""))
        self.file_gpt.set_path(data.get("gpt_path", ""), block_signals=True)
        self.file_vits.set_path(data.get("vits_path", ""), block_signals=True)
        self.file_genie.set_path(data.get("genie_dir", ""))
        self.file_ref_audio.set_path(data.get("ref_audio", ""))
        self.input_ref_text.setText(data.get("ref_text", ""))

        set_combo_text(self.combo_device, data.get("device", ""))
        set_combo_text(self.combo_quality, data.get("quality", ""))
        set_combo_text(self.combo_split, data.get("split", ""))
        set_combo_text(self.combo_mode, data.get("mode", ""))
        set_combo_text(self.combo_lang, data.get("lang", ""))
        set_combo_text(self.combo_save_mode, data.get("save_mode", ""))

        self.file_out_dir.set_path(data.get("out_dir", ""))

        # 确保UI显隐状态正确
        self._update_model_ui_visibility()
        self._update_save_ui_state()

    # ==================== UI 逻辑处理 ====================

    def _update_model_ui_visibility(self, *args) -> None:
        """根据模型类型控制文件选择器的显隐"""
        is_gpt = self.combo_model_type.currentText() == "GPT-SoVITS"
        self.layout_model.setRowVisible(self.file_gpt, is_gpt)
        self.layout_model.setRowVisible(self.file_vits, is_gpt)
        self.layout_model.setRowVisible(self.file_genie, not is_gpt)

    @Slot(str)
    def _on_gpt_path_changed(self, path: str):
        if path and os.path.exists(path) and not self.file_vits.get_path():
            self._try_auto_fill_sibling(path, ".pth", self.file_vits)

    @Slot(str)
    def _on_vits_path_changed(self, path: str):
        if path and os.path.exists(path) and not self.file_gpt.get_path():
            self._try_auto_fill_sibling(path, ".ckpt", self.file_gpt)

    @staticmethod
    def _try_auto_fill_sibling(current_path: str, target_ext: str, target_widget: FileSelectorWidget):
        try:
            directory = os.path.dirname(current_path)
            if not os.path.exists(directory):
                return
            for f in os.listdir(directory):
                if f.lower().endswith(target_ext.lower()):
                    full_path = os.path.join(directory, f)
                    target_widget.set_path(full_path)
                    print(f"[INFO] 自动关联模型文件: {full_path}")
                    break
        except Exception as e:
            print(f"[WARN] 自动关联文件失败: {e}")

    def _update_save_ui_state(self) -> None:
        enabled = self.combo_save_mode.currentText() != "禁用自动保存"
        self.layout_save.setRowVisible(self.file_out_dir, enabled)

    def _play_ref_audio(self) -> None:
        path = self.file_ref_audio.get_path()
        if os.path.exists(path):
            self.player.stop()
            self.player.play(path)
        else:
            QMessageBox.warning(self, "错误", "参考音频文件不存在")

    def _get_split_texts(self, text: str) -> List[str]:
        method = self.combo_split.currentText()
        if method == "不切分":
            return [text]
        elif method == "按行切分":
            return [line.strip() for line in text.split('\n') if line.strip()]
        elif method == "智能切分":
            return self.splitter.split(text)
        return [text]

    def _start_inference(self) -> None:
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "提示", "请输入目标文本")
            return

        ref_path = self.file_ref_audio.get_path()
        ref_text = self.input_ref_text.text().strip()
        if not ref_path or not ref_text:
            QMessageBox.warning(self, "提示", "请设置参考音频")
            return

        if not self.file_genie.get_path():
            QMessageBox.warning(self, "提示", "请选择Genie模型目录")
            return

        out_dir = self.file_out_dir.get_path()
        save_mode = self.combo_save_mode.currentText()
        if not out_dir and save_mode != "禁用自动保存":
            desktop = os.path.join(os.path.expanduser("~"), "Desktop", "Genie Output")
            self.file_out_dir.set_path(desktop)
            print(f"[INFO] 未设置输出文件夹, 将在桌面创建!")

        self.btn_start.setEnabled(False)
        self.btn_start.setText("推理中...")
        self._chain_import_model()

    # ==================== 推理链式调用 ====================

    def _chain_import_model(self) -> None:
        req = {
            "character_name": TEMP_MODEL_NAME,
            "onnx_model_dir": self.file_genie.get_path(),
            "language": self.combo_lang.currentText(),
        }
        worker = InferenceWorker(req, mode="load_character")
        worker.finished.connect(lambda s, m, d: self._on_import_finished(s, m))
        worker.start()
        self.current_worker = worker

    @Slot(bool, str)
    def _on_import_finished(self, success: bool, msg: str) -> None:
        if not success:
            self._reset_ui_state()
            QMessageBox.critical(self, "模型加载失败", msg)
            return
        print(f"[INFO] {msg}")
        self._chain_set_ref()

    def _chain_set_ref(self) -> None:
        req = {
            "character_name": TEMP_MODEL_NAME,
            "audio_path": self.file_ref_audio.get_path(),
            "audio_text": self.input_ref_text.text().strip(),
            "language": self.combo_lang.currentText(),
        }
        worker = InferenceWorker(req, mode="set_reference_audio")
        worker.finished.connect(lambda s, m, d: self._on_set_ref_finished(s, m))
        worker.start()
        self.current_worker = worker

    @Slot(bool, str)
    def _on_set_ref_finished(self, success: bool, msg: str) -> None:
        if not success:
            self._reset_ui_state()
            QMessageBox.critical(self, "设置参考音频失败", msg)
            return
        print(f"[INFO] {msg}")
        self._chain_tts()

    def _chain_tts(self) -> None:
        text_full = self.text_input.toPlainText().strip()
        text_list = self._get_split_texts(text_full)

        print(f"[INFO] 开始串行推理, 分句结果: {text_list}")
        self._process_serial_step(0, text_list, [], 32000)

    def _process_serial_step(
            self,
            index: int,
            text_list: List[str],
            audio_accumulator: List[np.ndarray],
            sample_rate: int
    ) -> None:
        # 1. 终止条件：所有句子处理完毕
        if index >= len(text_list):
            save_mode = self.combo_save_mode.currentText()
            out_dir = self.file_out_dir.get_path()

            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            if audio_accumulator and save_mode != "保存为多个文件":
                full_text = ''.join(text_list)
                full_audio = np.concatenate(audio_accumulator, axis=0)
                if save_mode == "保存为单个文件":
                    target_names = generate_output_filenames(folder=out_dir, original_texts=[full_text])
                    save_path = os.path.join(out_dir, target_names[0])
                else:  # "禁用自动保存"
                    save_path = os.path.join(CACHE_DIR, f"{uuid.uuid4().hex}.wav")
                sf.write(save_path, data=full_audio, samplerate=sample_rate, subtype='PCM_16')
                self._add_to_preview(full_text, save_path)

            print(f"\n[INFO] 串行推理全部完成，共 {len(text_list)} 句。")
            self._reset_ui_state()
            return

        # 2. 递归进行：发起当前句子的请求
        req = {
            "character_name": TEMP_MODEL_NAME,
            "text": text_list[index],
        }
        worker = InferenceWorker(req, mode="tts")
        worker.finished.connect(
            lambda s, m, d: self._on_serial_step_finished(s, m, d, index, text_list, audio_accumulator)
        )
        worker.start()
        self.current_worker = worker

    @Slot(bool, str, object, int, object, object, object)
    def _on_serial_step_finished(
            self,
            success: bool,
            msg: str,
            return_data: dict,
            index: int,
            text_list: List[str],
            audio_accumulator: List[np.ndarray]
    ) -> None:
        if not success:
            self._reset_ui_state()
            QMessageBox.critical(self, "推理失败", f"第 {index + 1} 句出错: {msg}")
            return

        sr = return_data.get("sample_rate", 32000)
        audio_list = return_data.get("audio_list", [])
        save_mode = self.combo_save_mode.currentText()
        out_dir = self.file_out_dir.get_path()
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if audio_list:
            audio_accumulator.append(audio_list[0])
            if save_mode == "保存为多个文件":
                target_names = generate_output_filenames(folder=out_dir, original_texts=[text_list[index]])
                save_path = os.path.join(out_dir, target_names[0])
                sf.write(save_path, data=audio_list[0], samplerate=sr, subtype='PCM_16')
                self._add_to_preview(text_list[index], save_path)
        else:
            print(f"[WARN] 第 {index + 1} 句返回空音频")

        # 继续处理下一句
        self._process_serial_step(index + 1, text_list, audio_accumulator, sr)

    def _add_to_preview(self, text: str, path: str) -> None:
        item = PreviewItemWidget(self.current_gen_id, text, path, self.player)
        self.preview_list_layout.insertWidget(0, item)
        self.current_gen_id += 1

    def _reset_ui_state(self) -> None:
        self.btn_start.setEnabled(True)
        self.btn_start.setText("开始推理")

    def closeEvent(self, event: QCloseEvent) -> None:
        # 委托 PresetManager 处理保存逻辑
        self.preset_manager.shutdown()
        super().closeEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Genie TTS Inference GUI")
        self.resize(1300, 900)

        # 初始化音频播放器
        self.player: AudioPlayer = AudioPlayer()

        # 初始化日志重定向
        self.log_widget: LogWidget = LogWidget()
        sys.stdout = LogRedirector()
        sys.stdout.textWritten.connect(self.log_widget.append_log)

        # 初始化主界面
        self.tabs: QTabWidget = QTabWidget()
        self.tts_widget = TTSWidget(self.player)
        self.conv_widget = ConverterWidget()

        self.tabs.addTab(self.log_widget, "GUI Log")
        self.tabs.addTab(self.tts_widget, "TTS Inference")
        self.tabs.addTab(self.conv_widget, "Converter")
        self.tabs.setCurrentIndex(1)  # 默认显示TTS页

        self.setCentralWidget(self.tabs)

    def closeEvent(self, event: QCloseEvent) -> None:
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
        if hasattr(self, 'player'):
            self.player.stop()
        # 线程安全退出后，再恢复 stdout
        sys.stdout = sys.__stdout__
        if hasattr(self, 'tts_widget'):
            self.tts_widget.closeEvent(event)
        event.accept()
