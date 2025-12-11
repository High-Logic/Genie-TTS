import os
import re
from enum import Enum
from datetime import datetime
import socket
from typing import List

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QLineEdit, QFileDialog, QMessageBox, QComboBox, QTextEdit
)
from PySide6.QtCore import (Qt, Signal, QSettings, Property, QObject, QEvent, QMimeData)
from PySide6.QtGui import QFont


def sanitize_filename(filename: str, replacement: str = '') -> str:
    """
    将文本清理为合法的 Windows 文件名。

    Args:
        filename (str): 原始文件名。
        replacement (str): 非法字符的替换字符，默认为空，建议使用 "_"。

    Returns:
        str: 清理后的文件名。
    """
    # 1. 去除非法字符
    # \/:*?"<>| 是标准非法字符
    # \x00-\x1f 是控制字符 (如换行符、制表符等)，Windows 也不允许
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]', replacement, filename)

    # 2. 去除首尾的空格和点
    # Windows 文件名不能以空格或点结尾，也不能以空格开头(虽然允许但通常不推荐)
    cleaned = cleaned.strip().rstrip('.')

    # 3. 处理 Windows 保留文件名 (CON, PRN, AUX, NUL, COM1-9, LPT1-9)
    # 这些名字不论加什么扩展名都是非法的 (例如 con.txt 也是非法的)
    reserved_names = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
    }

    # 如果文件名(全大写)是保留字，或者文件名是保留字+扩展名(如 con.txt)，则加下划线前缀
    filename_upper = cleaned.upper()
    file_stem = filename_upper.split('.')[0]  # 获取不带后缀的主文件名

    if filename_upper in reserved_names or file_stem in reserved_names:
        cleaned = "_" + cleaned

    # 4. 处理空文件名 (如果输入全是乱码被删光了)
    if not cleaned:
        cleaned = "unnamed_file"

    # 5. 限制长度 (Windows API 通常限制 255 字符，但在某些路径下更短)
    cleaned = truncate_text(cleaned)

    return cleaned


def is_port_free(port: int) -> bool:
    """返回端口是否可用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('', port))
        except OSError:
            return False
    return True


def find_free_port(preferred: int = 8000) -> int:
    # 先尝试 preferred 端口
    if is_port_free(preferred):
        return preferred

    # 否则用系统自动分配
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def truncate_text(text: str, max_len: int = 30) -> str:
    """
    按指定宽度截断文本
    """
    result = ""
    cur_len = 0
    for ch in text:
        char_len = 2 if ('\u4e00' <= ch <= '\u9fff') else 1
        if cur_len + char_len > max_len:
            break
        result += ch
        cur_len += char_len
    return result


def generate_output_filenames(folder: str, original_texts: List[str]) -> List[str]:
    """
    批量生成文件名：
    输入 original_texts 列表
    输出等长 filenames 列表
    """
    today = datetime.now().strftime("%Y%m%d")
    pattern = re.compile(rf'^\[{today}]\[(\d{{3}})]')

    # ① 查找当天现存最大编号
    max_n = 0
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            m = pattern.match(name)
            if m:
                n = int(m.group(1))
                max_n = max(max_n, n)

    filenames = []
    cur_n = max_n

    # ② 依次生成新文件名
    for text in original_texts:
        cur_n += 1
        n_str = f"{cur_n:03d}"

        cleaned = sanitize_filename(text)

        filename = f"[{today}][{n_str}]{cleaned}.wav"
        filenames.append(filename)

    return filenames


# ==================== 通用组件 ====================

class FileSelectionMode(Enum):
    FILE = 0
    DIRECTORY = 1


class FileSelectorWidget(QWidget):
    """一个包含行编辑和浏览按钮的复合控件，支持文件和文件夹选择。（原样引用并稍作适配）"""
    pathChanged = Signal(str)

    def __init__(
            self,
            setting_key: str,
            selection_mode: FileSelectionMode = FileSelectionMode.DIRECTORY,
            file_filter: str = "All Files (*)",
            parent: QWidget = None,
    ):
        super().__init__(parent)
        self.setting_key: str = setting_key
        self.selection_mode: FileSelectionMode = selection_mode
        self.file_filter: str = file_filter

        # 使用 QSettings 模拟 STUDIO_SETTINGS
        self.settings = QSettings("MyTTS", "GUI")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self.path_edit: QLineEdit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("未选择路径")

        self.browse_button: QPushButton = QPushButton("📁")
        self.browse_button.setCursor(Qt.CursorShape.PointingHandCursor)
        default_font = QFont()
        default_font.setPointSize(10)
        self.browse_button.setFont(default_font)
        self.browse_button.setFixedSize(30, 30)

        self.clear_button: QPushButton = QPushButton("❌")
        self.clear_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_button.setFont(default_font)
        self.clear_button.setFixedSize(30, 30)

        layout.addWidget(self.path_edit)
        layout.addWidget(self.browse_button)
        layout.addWidget(self.clear_button)

        self.browse_button.clicked.connect(self._open_dialog)
        self.clear_button.clicked.connect(self._clear_path)
        self.path_edit.textChanged.connect(self.pathChanged)

    def _open_dialog(self):
        path = self.path_edit.text()
        if path and os.path.exists(path):
            start_path = path
        else:
            # 【修改点】默认路径改为 Desktop
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            start_path = str(self.settings.value(
                f"last_path_{self.setting_key}", defaultValue=desktop_path
            ))

        if self.selection_mode == FileSelectionMode.DIRECTORY:
            selected_path = QFileDialog.getExistingDirectory(
                self, "选择文件夹", start_path
            )
        else:
            selected_path, _ = QFileDialog.getOpenFileName(
                self, "选择文件", start_path, self.file_filter
            )

        if selected_path:
            self.set_path(selected_path)
            parent_path = os.path.dirname(selected_path)
            if parent_path:
                self.settings.setValue(
                    f"last_path_{self.setting_key}", parent_path)

    def _clear_path(self):
        if not self.path_edit.text():
            return
        reply = QMessageBox.question(self, '确认', '您确定要清空路径吗？',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.set_path("")

    def get_path(self) -> str:
        text = self.path_edit.text()
        # 即使路径不存在（可能是输入时），也返回文本供逻辑判断，或者严格校验
        return text

    def set_path(self, path: str, block_signals: bool = False):
        if block_signals:
            self.path_edit.blockSignals(True)
        self.path_edit.setText(path)
        if block_signals:
            self.path_edit.blockSignals(False)

    path = Property(str, fget=get_path, fset=set_path, notify=pathChanged)  # type: ignore


class WheelEventFilter(QObject):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel and isinstance(obj, QComboBox):
            return True  # 阻止默认滚轮行为
        return super().eventFilter(obj, event)


class MyComboBox(QComboBox):
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._wheelFilter = WheelEventFilter()
        self.installEventFilter(self._wheelFilter)


class MyTextEdit(QTextEdit):
    def insertFromMimeData(self, source: QMimeData) -> None:
        # 仅取纯文本
        if source.hasText():
            self.insertPlainText(source.text())
        else:
            super().insertFromMimeData(source)
