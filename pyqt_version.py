import sys
import os
import shutil
import time
from datetime import datetime
import numpy as np
import open3d as o3d

from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
                             QFileDialog, QHBoxLayout, QProgressBar, QScrollArea, QSizePolicy,
                             QFrame, QGraphicsDropShadowEffect)
from PyQt6.QtCore import Qt
import process
# ---------- Карточка файла ----------
class FileCard(QFrame):
    def __init__(self, file_path, remove_callback, preview_callback):
        super().__init__()
        self.file_path = file_path
        self.remove_callback = remove_callback
        self.preview_callback = preview_callback
        self.processed = False

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(Qt.GlobalColor.black)
        self.setGraphicsEffect(shadow)

        self.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 12%);
                border-radius: 6px;
            }
            QFrame:hover {
                background-color: rgba(255, 255, 255, 20%);
            }
        """)

        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(10, 5, 10, 5)

        # Название файла
        self.label = QLabel(os.path.basename(file_path))
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.label.setStyleSheet("padding-left: 8px;")
        self.layout.addWidget(self.label)

        # Статус
        self.status_label = QLabel("В очереди")
        self.status_label.setFixedHeight(28)
        self.status_label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            color: #AAAAAA; 
            font-weight: bold; 
            border-radius:4px; 
            padding-left:8px; 
            padding-right:8px;
        """)
        self.layout.addWidget(self.status_label)

        # Кнопка предпросмотра
        self.preview_btn = QPushButton("Посмотреть результат")
        self.preview_btn.setVisible(False)
        self.preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #03DAC5;
                color: black;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #00BFA5;
            }
        """)
        self.preview_btn.clicked.connect(self.preview)
        self.layout.addWidget(self.preview_btn)

        # Кнопка удаления
        self.remove_btn = QPushButton("✖")
        self.remove_btn.setMaximumWidth(30)
        self.remove_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #FF5555;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                color: red;
            }
        """)
        self.remove_btn.clicked.connect(self.remove)
        self.layout.addWidget(self.remove_btn)

        self.setLayout(self.layout)

    def remove(self):
        self.remove_callback(self)

    def preview(self):
        self.preview_callback(self.file_path)

    def set_status(self, status, ready=False):
        self.status_label.setText(status)
        self.processed = ready
        base_style = "font-weight:bold; border-radius:4px; padding-left:8px; padding-right:8px;"
        if ready:
            self.status_label.setStyleSheet(f"""
                {base_style}
                color: white;
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #00FF00, stop:1 #00CC00);
            """)
            self.preview_btn.setVisible(True)
        elif status == "В работе":
            self.status_label.setStyleSheet(f"""
                {base_style}
                color: white;
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #FFFF00, stop:1 #FFD700);
            """)
        else:
            self.status_label.setStyleSheet(f"""
                {base_style}
                color: white;
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #AAAAAA, stop:1 #888888);
            """)

# ---------- Основное окно ----------
class PCDApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCD Обработчик")
        self.resize(900, 650)
        self.setMinimumSize(800, 500)
        self.setAcceptDrops(True)

        self.output_dir = os.path.join(os.getcwd(), "processed")
        os.makedirs(self.output_dir, exist_ok=True)

        self.file_cards = []

        self.init_ui()

    def init_ui(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: white;
                font-size: 14px;
            }
            QPushButton#mainBtn {
                background-color: #6200EE;
                color: white;
                border-radius: 6px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton#mainBtn:hover {
                background-color: #3700B3;
            }
        """)

        main_layout = QVBoxLayout()

        self.select_btn = QPushButton("📂 Выбрать файлы")
        self.select_btn.setObjectName("mainBtn")
        self.select_btn.clicked.connect(self.load_files)
        main_layout.addWidget(self.select_btn)

        # Scroll area для файлов
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollBar:vertical {
                background: transparent;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255,255,255,30%);
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255,255,255,60%);
            }
        """)
        self.files_widget = QWidget()
        self.files_layout = QVBoxLayout()
        self.files_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.files_widget.setLayout(self.files_layout)
        self.scroll_area.setWidget(self.files_widget)
        main_layout.addWidget(self.scroll_area)

        # Прогрессбар
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(24)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.hide()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border-radius: 12px;
                background-color: rgba(255, 255, 255, 12%);
                color: white;
                text-align: center;
            }
            QProgressBar::chunk {
                border-radius: 12px;
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #6200EE, stop:1 #3700B3);
            }
        """)
        main_layout.addWidget(self.progress_bar)

        self.process_btn = QPushButton("🚀 Обработать все файлы")
        self.process_btn.setObjectName("mainBtn")
        self.process_btn.clicked.connect(self.process_files)
        self.process_btn.hide()
        main_layout.addWidget(self.process_btn)

        self.setLayout(main_layout)

    # ---------- Drag & Drop ----------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            f = url.toLocalFile()
            if f.lower().endswith(".pcd") and not any(fc.file_path == f for fc in self.file_cards):
                self.add_file_card(f)
        self.update_process_button_visibility()

    # ---------- Загрузка файлов ----------
    def load_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Выберите PCD файлы", "", "PCD Files (*.pcd)")
        for f in files:
            if not any(fc.file_path == f for fc in self.file_cards):
                self.add_file_card(f)
        self.update_process_button_visibility()

    def add_file_card(self, file_path):
        card = FileCard(file_path, self.remove_file_card, self.preview_file)
        self.file_cards.append(card)
        self.files_layout.addWidget(card)

    def remove_file_card(self, card):
        self.file_cards.remove(card)
        card.setParent(None)
        self.update_process_button_visibility()

    def update_process_button_visibility(self):
        self.process_btn.setVisible(bool(self.file_cards))

    # ---------- Обработка ----------
    def process_files(self):
        self.progress_bar.setMaximum(len(self.file_cards))
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        QApplication.processEvents()

        for idx, card in enumerate(self.file_cards, start=1):
            card.set_status("В работе", ready=False)
            QApplication.processEvents()

            time.sleep(0.5)  # имитация обработки
            
            file_name = os.path.basename(card.file_path)
            file_path = card.file_path
            process.main(file_path, file_name)
            timestamp = datetime.now().strftime("%d%m%y-%H%M%S")
            out_path = os.path.join(self.output_dir, f"{os.path.splitext(file_name)[0]}_{timestamp}.pcd")
            shutil.copy(card.file_path, out_path)
            card.file_path = out_path
            card.set_status("Готов", ready=True)

            self.progress_bar.setValue(idx)
            QApplication.processEvents()

        self.progress_bar.hide()

    # ---------- Предпросмотр ----------
    def preview_file(self, file_path):
        pcd = o3d.io.read_point_cloud(file_path)
        if not pcd.has_colors():
            colors = np.ones_like(np.asarray(pcd.points))
            pcd.colors = o3d.utility.Vector3dVector(colors)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=os.path.basename(file_path))
        vis.add_geometry(pcd)
        ro = vis.get_render_option()
        ro.point_size = 1.2
        ro.background_color = [0,0,0]
        vis.run()
        vis.destroy_window()

# ---------- Запуск ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PCDApp()
    window.show()
    sys.exit(app.exec())
