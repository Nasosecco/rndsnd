import sys
import os
import time
import sqlite3
import datetime
import random
import numpy as np
import librosa
import soundfile as sf
import tempfile
import ctypes
import traceback
import warnings
import shutil

# Import Pydub
from pydub import AudioSegment

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")

# --- LOAD NEURAL ENGINE ---
AI_AVAILABLE = False
try:
    import torch
    from panns_inference import AudioTagging
    AI_AVAILABLE = True
except ImportError:
    pass

# Fix ALSA errors on Linux
def alsa_error_handler(filename, line, function, err, fmt): pass
try:
    asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    CMPFUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
    c_alsa_error_handler = CMPFUNC(alsa_error_handler)
    asound.snd_lib_error_set_handler(c_alsa_error_handler)
except: pass

os.environ['QT_API'] = 'pyside6'
os.environ['PYTHONWARNINGS'] = 'ignore'

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import SpanSelector

from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QWidget, QLineEdit, QTableWidget, QTableWidgetItem,
                             QLabel, QTabWidget, QSplitter, QFrame, QRadioButton, QSpinBox, 
                             QComboBox, QSplashScreen, QAbstractItemView, QFileDialog, QMessageBox, 
                             QHeaderView, QProgressBar, QFileSystemModel, QTreeView, QMenu)
from PySide6.QtCore import Qt, QMimeData, QUrl, QTimer, QSize, QPoint, QThread, Signal, QElapsedTimer, QDir
from PySide6.QtGui import QDrag, QColor, QPixmap, QIcon, QAction
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

# --- RESOURCE MANAGEMENT ---
def get_base_path():
    try: base_path = sys._MEIPASS
    except: base_path = os.path.dirname(os.path.abspath(__file__))
    return base_path

def resource_path(relative_path): return os.path.join(get_base_path(), relative_path)
DB_PATH = os.path.join(get_base_path(), "audio.db")

# --- INTELLIGENT SCANNER (MULTI-SAMPLE) ---
class ScanWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal(int)

    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        self.ai_model = None

    def run(self):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        
        valid_exts = ('.wav', '.mp3', '.flac', '.aiff', '.ogg', '.m4a', '.wma', '.aac', '.opus', '.aif')
        file_list = []
        for root, dirs, files in os.walk(self.folder):
            for f in files:
                if f.lower().endswith(valid_exts):
                    file_list.append(os.path.join(root, f))
        
        total = len(file_list)
        if total == 0:
            self.finished.emit(0)
            conn.close()
            return

        count = 0
        new_files = 0
        model_loaded = False

        for i, path in enumerate(file_list):
            filename = os.path.basename(path)
            
            # Check if file exists
            cur.execute("SELECT id FROM files WHERE path = ?", (path,))
            if cur.fetchone():
                if i % 10 == 0: self.progress.emit(int((i / total) * 100))
                continue

            # Load AI only if needed
            if AI_AVAILABLE and not model_loaded:
                self.log.emit("Loading AI Model...")
                try:
                    self.ai_model = AudioTagging(checkpoint_path=None, device='cpu') 
                    model_loaded = True
                except: pass

            tags = "Unknown"
            duration = 0.0
            
            try:
                duration = librosa.get_duration(path=path)
                size = os.path.getsize(path)

                if self.ai_model:
                    chunk_dur = 5.0
                    offsets = []
                    
                    if duration < 10:
                        offsets = [0]
                    else:
                        offsets = [0, (duration / 2) - (chunk_dur / 2), duration - chunk_dur]
                        offsets = [max(0, o) for o in offsets]

                    tag_accumulator = {}

                    for off in offsets:
                        y, _ = librosa.load(path, sr=32000, mono=True, offset=off, duration=chunk_dur)
                        y = y[None, :]
                        clipwise_output, _ = self.ai_model.inference(y)
                        scores = clipwise_output[0]
                        
                        for idx, score in enumerate(scores):
                            label = self.ai_model.labels[idx]
                            if label in tag_accumulator:
                                tag_accumulator[label] += score
                            else:
                                tag_accumulator[label] = score
                    
                    sorted_tags = sorted(tag_accumulator.items(), key=lambda x: x[1], reverse=True)
                    top_3_tags = [t[0] for t in sorted_tags[:3]]
                    tags = ", ".join(top_3_tags)
                    
                else:
                    tags = "No AI"

                cur.execute("INSERT OR IGNORE INTO files (filename, path, folder, tags, size, duration) VALUES (?, ?, ?, ?, ?, ?)",
                            (filename, path, os.path.dirname(path), tags, size, duration))
                conn.commit()
                new_files += 1
                
                self.progress.emit(int((i / total) * 100))
                self.log.emit(f"Analyzed: {filename[:15]}... [{tags}]")
                
            except Exception as e: 
                print(f"Error analyzing {filename}: {e}")
        
        conn.close()
        self.finished.emit(new_files)

# --- STYLES ---
COMMON_BUTTON_STYLE = """
    QPushButton { background-color: #e65100; color: #ffffff; border-radius: 4px; padding: 8px; font-weight: bold; border: none; } 
    QPushButton:hover { background-color: #ff9800; } 
    QPushButton:pressed { background-color: #bf4500; }
"""

DARK_STYLE = COMMON_BUTTON_STYLE + """
    QMainWindow { background-color: #121212; } 
    QTabWidget::pane { border: 1px solid #333; background: #1e1e1e; } 
    QTabBar::tab { background: #2d2d2d; color: #bbb; padding: 10px 20px; } 
    QTabBar::tab:selected { background: #1e1e1e; color: #ff9800; border-bottom: 2px solid #ff9800; } 
    
    QTableWidget { background-color: #1e1e1e; color: #ffffff; gridline-color: #333; selection-background-color: #e65100; } 
    
    QHeaderView::section { 
        background-color: #ff9800; 
        color: #000000; 
        padding: 4px; 
        border: 1px solid #c66900;
        font-weight: bold;
    }
    QTableCornerButton::section {
        background-color: #ff9800;
        border: 1px solid #c66900;
    }
    
    QLabel { color: #4caf50; font-weight: bold; } 
    QRadioButton { color: #4caf50; font-weight: bold; } 
    QLineEdit, QSpinBox, QComboBox { background-color: #252525; color: #4caf50; border: 1px solid #3d3d3d; border-radius: 4px; padding: 5px; } 
    QTreeView { background-color: #1e1e1e; color: #ddd; border: 1px solid #333; }
"""

# --- INTERFACE COMPONENTS ---
class DragButton(QPushButton):
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setAcceptDrops(False); self.main_window = None
    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.LeftButton and self.main_window: self.main_window.start_drag_operation()
        super().mouseMoveEvent(e)

class RndSndApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("rndsnd v0.8.0")
        self.resize(1600, 950)
        self.init_db()
        
        # Audio Engine
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)
        self.audio_data = None
        self.sr = 44100
        self.duration = 0.0
        self.selection_range = (0, 0)
        self.is_looping = False
        self.playhead_line = None
        self.current_browsing_path = ""
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_playhead_and_loop)
        
        # Main Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        self.setup_header()
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        self.setup_explorer_tab()
        self.setup_mixer_tab()
        
        self.switch_theme("Dark")

    def init_db(self):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS files (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, path TEXT UNIQUE, folder TEXT, tags TEXT, duration REAL, size INTEGER)")
        try: cur.execute("SELECT folder FROM files LIMIT 1")
        except: cur.execute("ALTER TABLE files ADD COLUMN folder TEXT")
        conn.commit()
        conn.close()

    def setup_header(self):
        header = QHBoxLayout()
        self.title_label = QLabel("<h2>rndsnd v0.8.0</h2>")
        header.addWidget(self.title_label)
        header.addStretch()
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.currentTextChanged.connect(self.switch_theme)
        header.addWidget(QLabel("THEME:"))
        header.addWidget(self.theme_combo)
        self.main_layout.addLayout(header)

    def switch_theme(self, theme):
        if theme == "Dark":
            self.setStyleSheet(DARK_STYLE)
            self.canvas_bg, self.wf_color, self.cursor_color = '#1e1e1e', '#ff9800', 'white'
        else:
            self.setStyleSheet(COMMON_BUTTON_STYLE)
            self.canvas_bg, self.wf_color, self.cursor_color = 'white', '#ff9800', 'black'
        if self.audio_data is not None: self.plot_waveform()

    # --- EXPLORER TAB ---
    def setup_explorer_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        main_splitter = QSplitter(Qt.Horizontal)
        
        # 1. LEFT: SYSTEM TREE
        tree_frame = QFrame()
        tree_layout = QVBoxLayout(tree_frame)
        tree_layout.setContentsMargins(0,0,0,0)
        
        lbl_tree = QLabel("SYSTEM BROWSER")
        lbl_tree.setStyleSheet("padding: 5px; background: #333; color: white;")
        tree_layout.addWidget(lbl_tree)

        self.dir_model = QFileSystemModel()
        self.dir_model.setRootPath("") # Root
        self.dir_model.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs | QDir.Drives) 
        
        self.tree = QTreeView()
        self.tree.setModel(self.dir_model)
        
        self.tree.expand(self.dir_model.index(QDir.homePath()))
        self.tree.scrollTo(self.dir_model.index(QDir.homePath()))
        
        self.tree.setColumnHidden(1, True)
        self.tree.setColumnHidden(2, True)
        self.tree.setColumnHidden(3, True)
        self.tree.setHeaderHidden(True)
        
        self.tree.clicked.connect(self.on_folder_clicked)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.open_tree_context_menu)
        
        tree_layout.addWidget(self.tree)
        main_splitter.addWidget(tree_frame)

        # 2. RIGHT: CONTENT
        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)
        
        self.scan_info_lbl = QLabel("Select a folder to view files.")
        self.scan_progress = QProgressBar()
        self.scan_progress.setVisible(False)
        right_layout.addWidget(self.scan_info_lbl)
        right_layout.addWidget(self.scan_progress)

        self.file_table = QTableWidget()
        self.file_table.setColumnCount(4)
        self.file_table.setHorizontalHeaderLabels(["Filename", "Tags", "Duration", "Size"])
        self.file_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.file_table.horizontalHeader().setStretchLastSection(True)
        self.file_table.setColumnWidth(0, 250)
        self.file_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.file_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.file_table.itemClicked.connect(self.load_selected_file)
        
        right_layout.addWidget(self.file_table)

        self.figure, self.ax = plt.subplots(figsize=(10, 3))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.canvas.mpl_connect('scroll_event', self.on_scroll_zoom)
        right_layout.addWidget(self.canvas)
        
        self.span = SpanSelector(self.ax, self.on_select, 'horizontal', useblit=True, 
                               props=dict(alpha=0.4, facecolor='#ff9800'), interactive=True, drag_from_anywhere=True)

        # Transport
        t_lyt = QHBoxLayout()
        self.btn_rew = QPushButton("⏪ -5s")
        self.btn_rew.clicked.connect(lambda: self.seek_relative(-5000))
        
        self.btn_play = QPushButton("▶ PLAY")
        self.btn_play.clicked.connect(self.toggle_play)
        
        self.btn_stop = QPushButton("⏹ STOP")
        self.btn_stop.clicked.connect(self.stop_audio)
        
        self.btn_fwd = QPushButton("⏩ +5s")
        self.btn_fwd.clicked.connect(lambda: self.seek_relative(5000))

        self.btn_drag = DragButton("📦 DRAG")
        self.btn_drag.main_window = self
        self.btn_drag.setFixedWidth(100)
        self.btn_drag.setStyleSheet("background-color: #444; border: 1px dashed #777;")
        
        t_lyt.addWidget(self.btn_rew)
        t_lyt.addWidget(self.btn_play)
        t_lyt.addWidget(self.btn_stop)
        t_lyt.addWidget(self.btn_fwd)
        t_lyt.addStretch()
        t_lyt.addWidget(self.btn_drag)
        right_layout.addLayout(t_lyt)

        main_splitter.addWidget(right_frame)
        main_splitter.setSizes([400, 1000])
        
        layout.addWidget(main_splitter)
        self.tabs.addTab(tab, "Library Editor")

    def setup_mixer_tab(self):
        tab = QWidget()
        l = QVBoxLayout(tab)
        l.setAlignment(Qt.AlignCenter)
        
        l.addWidget(QLabel("<h2>Generative Mixer</h2>"))
        l.addWidget(QLabel("Create a chaotic layered mix from your database."))
        
        self.radio_tags = QRadioButton("Use filtered list (Search Filter)")
        self.radio_tags.setChecked(True)
        self.radio_chaos = QRadioButton("Full Chaos (Use Entire Library)")
        
        h_params = QHBoxLayout()
        h_params.addWidget(QLabel("Total Duration (sec):"))
        self.spin_dur = QSpinBox()
        self.spin_dur.setRange(5, 300)
        self.spin_dur.setValue(30)
        h_params.addWidget(self.spin_dur)
        
        # Layers Control
        h_params.addWidget(QLabel("Layers / Tracks:"))
        self.spin_layers = QSpinBox()
        self.spin_layers.setRange(1, 20)
        self.spin_layers.setValue(4)
        h_params.addWidget(self.spin_layers)
        
        self.mix_btn = QPushButton("GENERATE MIX")
        self.mix_btn.setFixedSize(300, 60)
        self.mix_btn.clicked.connect(self.generate_mix)
        
        self.status_lbl = QLabel("")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        
        l.addWidget(self.radio_tags)
        l.addWidget(self.radio_chaos)
        l.addLayout(h_params)
        l.addWidget(self.mix_btn)
        l.addWidget(self.status_lbl)
        
        self.tabs.addTab(tab, "Mixer")

    # --- TREE VIEW LOGIC ---
    def on_folder_clicked(self, index):
        path = self.dir_model.fileInfo(index).absoluteFilePath()
        self.current_browsing_path = path
        self.update_table_from_db(path)

    def open_tree_context_menu(self, position):
        indexes = self.tree.selectedIndexes()
        if len(indexes) > 0:
            index = indexes[0]
            path = self.dir_model.fileInfo(index).absoluteFilePath()
            
            menu = QMenu()
            scan_action = QAction(f"✨ Scan '{os.path.basename(path)}' with AI", self)
            scan_action.triggered.connect(lambda: self.start_scan(path))
            menu.addAction(scan_action)
            
            menu.exec_(self.tree.viewport().mapToGlobal(position))

    def start_scan(self, folder_path):
        self.scan_progress.setVisible(True)
        self.scan_progress.setValue(0)
        self.scan_info_lbl.setText(f"Scanning: {folder_path}...")
        
        self.scan_thread = ScanWorker(folder_path)
        self.scan_thread.progress.connect(self.scan_progress.setValue)
        self.scan_thread.log.connect(self.scan_info_lbl.setText)
        self.scan_thread.finished.connect(lambda count: self.on_scan_completed(count, folder_path))
        self.scan_thread.start()

    def on_scan_completed(self, count, path):
        self.scan_progress.setVisible(False)
        self.scan_info_lbl.setText(f"✅ Scan Complete. {count} new files.")
        if self.current_browsing_path == path:
            self.update_table_from_db(path)

    def update_table_from_db(self, folder_path):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        query_path = folder_path + "%"
        cur.execute("SELECT filename, tags, duration, size, path FROM files WHERE path LIKE ?", (query_path,))
        rows = cur.fetchall()
        conn.close()

        self.file_table.setRowCount(len(rows))
        
        if len(rows) == 0:
            self.scan_info_lbl.setText("Folder not in DB. Right-click folder to SCAN.")
        else:
            self.scan_info_lbl.setText(f"Viewing: {os.path.basename(folder_path)} ({len(rows)} scanned files)")

        for row_idx, (name, tags, dur, size, path) in enumerate(rows):
            dur_str = f"{int(dur//60)}:{int(dur%60):02d}"
            size_str = f"{size/(1024*1024):.2f} MB"
            
            # --- COLOR CHANGED TO #39df0f ---
            name_item = QTableWidgetItem(name)
            name_item.setForeground(QColor("#39df0f")) 
            self.file_table.setItem(row_idx, 0, name_item)
            
            self.file_table.setItem(row_idx, 1, QTableWidgetItem(tags))
            self.file_table.setItem(row_idx, 2, QTableWidgetItem(dur_str))
            self.file_table.setItem(row_idx, 3, QTableWidgetItem(size_str))
            self.file_table.item(row_idx, 0).setData(Qt.UserRole, path)

    # --- AUDIO LOGIC ---
    def load_selected_file(self, item):
        path = self.file_table.item(item.row(), 0).data(Qt.UserRole)
        if not path or not os.path.exists(path): return
        
        try:
            audio = AudioSegment.from_file(path)
            y = np.array(audio.get_array_of_samples())
            if audio.channels == 2: y = y.reshape((-1, 2)).mean(axis=1)
            
            if audio.sample_width == 2: self.audio_data = y.astype(np.float32) / 32768.0
            else: self.audio_data = y.astype(np.float32) / (1 << (8 * audio.sample_width - 1))
                
            self.sr = audio.frame_rate
            self.duration = len(self.audio_data) / self.sr
            
            self.player.setSource(QUrl.fromLocalFile(path))
            self.is_looping = False; self.selection_range = (0, 0)
            self.btn_drag.setText("📦 DRAG")
            self.plot_waveform()
        except Exception as e: print(f"Error loading audio: {e}")

    def plot_waveform(self):
        self.ax.clear(); self.ax.set_facecolor(self.canvas_bg); self.figure.patch.set_facecolor(self.canvas_bg)
        if self.audio_data is not None:
            downsample = 100 
            t = np.linspace(0, self.duration, len(self.audio_data[::downsample]))
            self.ax.plot(t, self.audio_data[::downsample], color=self.wf_color, lw=0.7)
            self.ax.set_xlim(0, self.duration); self.ax.set_ylim(-1.1, 1.1)
            self.ax.axis('off')
            self.playhead_line = self.ax.axvline(x=0, color=self.cursor_color, lw=2)
        self.canvas.draw()

    def on_scroll_zoom(self, event):
        if self.audio_data is None or event.inaxes != self.ax: return
        cur_xlim = self.ax.get_xlim(); scale = 1/1.2 if event.button == 'up' else 1.2
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale
        center = event.xdata if event.xdata else (cur_xlim[0] + cur_xlim[1]) / 2
        new_min = max(0, center - (center - cur_xlim[0]) * scale)
        new_max = min(self.duration, center + (cur_xlim[1] - center) * scale)
        self.ax.set_xlim([new_min, new_max]); self.canvas.draw_idle()

    def on_select(self, xmin, xmax):
        self.selection_range = (xmin, xmax); self.is_looping = True; self.btn_drag.setText(f"📦 {xmax-xmin:.1f}s")
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            pos_sec = self.player.position() / 1000.0
            if pos_sec < xmin or pos_sec > xmax: self.player.setPosition(int(xmin * 1000))

    def on_mouse_click(self, event):
        if event.inaxes != self.ax or self.audio_data is None: return
        if event.button == 1:
            click_time = max(0, min(event.xdata, self.duration))
            self.player.setPosition(int(click_time * 1000))
            self.update_playhead_and_loop()

    def update_playhead_and_loop(self):
        pos_sec = self.player.position() / 1000.0
        
        if self.is_looping and self.player.playbackState() == QMediaPlayer.PlayingState:
            s, e = self.selection_range
            if pos_sec >= e: self.player.setPosition(int(s * 1000)); pos_sec = s
            
        if self.playhead_line: 
            self.playhead_line.set_xdata([pos_sec])
            self.canvas.draw_idle()

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause(); self.timer.stop(); self.btn_play.setText("▶ PLAY")
        else:
            self.player.play(); self.timer.start(30); self.btn_play.setText("⏸ PAUSE")

    def stop_audio(self):
        self.player.stop(); self.timer.stop(); self.btn_play.setText("▶ PLAY")
        if self.playhead_line: self.playhead_line.set_xdata([0]); self.canvas.draw()
    
    def seek_relative(self, ms): self.player.setPosition(max(0, self.player.position() + ms))

    def start_drag_operation(self):
        if self.audio_data is None: return
        s, e = (self.selection_range if self.is_looping else (0, self.duration))
        chunk = self.audio_data[int(s*self.sr):int(e*self.sr)]
        tp = os.path.join(tempfile.gettempdir(), f"rnd_{int(time.time())}.wav")
        sf.write(tp, chunk, self.sr)
        drag = QDrag(self.btn_drag); mime = QMimeData(); mime.setUrls([QUrl.fromLocalFile(tp)])
        drag.setMimeData(mime); drag.setPixmap(QPixmap(32, 32)); drag.exec_(Qt.CopyAction)

    def generate_mix(self):
        if not os.path.exists("output"): os.makedirs("output")
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f"rndsnd_mix_{timestamp}"; ext = "mp3"
        
        conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
        cur.execute("SELECT path FROM files")
        src = [x[0] for x in cur.fetchall()]
        conn.close()
            
        if not src: 
            self.status_lbl.setText("❌ Database empty. Scan folders first!")
            return

        self.status_lbl.setText("⏳ Mixing..."); QApplication.processEvents()
        
        target_duration_ms = self.spin_dur.value() * 1000
        num_layers = self.spin_layers.value()
        used_files_log = []
        
        # Helper for timecode
        def fmt_ms(ms): return f"{int(ms/1000/60):02d}:{int(ms/1000)%60:02d}"

        if num_layers == 1:
            # Linear DJ Mode
            mix = AudioSegment.silent(duration=0)
            crossfade_time = 2000
            
            while len(mix) < target_duration_ms:
                try:
                    chosen_file = random.choice(src)
                    a = AudioSegment.from_file(chosen_file)
                    
                    clip_len = random.randint(10000, 30000) 
                    
                    start_pos = 0
                    if len(a) < clip_len: seg = a
                    else:
                        start_pos = random.randint(0, len(a) - clip_len)
                        seg = a[start_pos:start_pos+clip_len]
                        
                    seg = seg.fade_in(50).fade_out(50)
                    
                    if len(mix) == 0: mix = seg
                    else: mix = mix.append(seg, crossfade=min(len(mix), len(seg), crossfade_time))
                    
                    # LOGGING RESTORED FOR LINEAR MODE
                    timecode = f"{fmt_ms(start_pos)}-{fmt_ms(start_pos+len(seg))}"
                    used_files_log.append(f"Track: {os.path.basename(chosen_file)} [{timecode}]")
                except: continue
            
            base_mix = mix[:target_duration_ms]
            
        else:
            # Chaos Mode
            base_mix = AudioSegment.silent(duration=target_duration_ms)
            for layer_idx in range(num_layers):
                layer_audio = AudioSegment.silent(duration=target_duration_ms)
                current_pos = 0
                while current_pos < target_duration_ms:
                    try:
                        chosen_file = random.choice(src)
                        a = AudioSegment.from_file(chosen_file)
                        clip_len = random.randint(3000, 15000)
                        
                        start_pos = 0
                        if len(a) < clip_len: seg = a
                        else:
                            start_pos = random.randint(0, len(a) - clip_len)
                            seg = a[start_pos:start_pos+clip_len]
                        
                        seg = seg.fade_in(100).fade_out(100)
                        seg = seg.pan(random.uniform(-0.5, 0.5))
                        seg = seg - random.uniform(0, 6) 
                        
                        if current_pos + len(seg) > target_duration_ms:
                            seg = seg[:target_duration_ms - current_pos]
                        
                        layer_audio = layer_audio.overlay(seg, position=current_pos)
                        
                        timecode = f"{fmt_ms(start_pos)}-{fmt_ms(start_pos+len(seg))}"
                        used_files_log.append(f"Layer {layer_idx+1}: {os.path.basename(chosen_file)} [{timecode}]")
                        
                        current_pos += len(seg)
                    except: continue
                base_mix = base_mix.overlay(layer_audio)

        base_mix.export(f"output/{fname}.{ext}", format=ext)
        
        try:
            with open(f"output/{fname}.txt", "w", encoding="utf-8") as f:
                f.write(f"RNDSND LOG\nDate: {timestamp}\nFile: {fname}.{ext}\n")
                f.write(f"Duration: {self.spin_dur.value()}s | Layers: {num_layers}\n" + "-"*40 + "\n")
                for line in sorted(used_files_log): f.write(f"- {line}\n")
        except: pass
            
        self.status_lbl.setText(f"✅ Created: {fname}.{ext} + Log")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    sp = resource_path("rndsnd_splash.png")
    ic = resource_path("rndsnd_icon.png")
    
    splash = None
    if os.path.exists(sp):
        pix = QPixmap(sp)
        if not pix.isNull():
            if pix.width() > 800: pix = pix.scaledToWidth(800, Qt.SmoothTransformation)
            splash = QSplashScreen(pix)
            splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
            splash.show(); splash.repaint()
            t = QElapsedTimer(); t.start()
            while t.elapsed() < 2000: app.processEvents(); time.sleep(0.01)

    win = RndSndApp()
    if os.path.exists(ic): win.setWindowIcon(QIcon(ic))
    win.show()
    if splash: splash.finish(win)
    sys.exit(app.exec())
