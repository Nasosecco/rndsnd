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

# Import Pydub
from pydub import AudioSegment

# --- CONFIGURAZIONE ---
warnings.filterwarnings("ignore")

# --- CARICAMENTO MOTORE NEURALE ---
AI_AVAILABLE = False
try:
    import torch
    from panns_inference import AudioTagging
    AI_AVAILABLE = True
except ImportError:
    pass

# Fix ALSA
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
                             QComboBox, QButtonGroup, QSplashScreen, QAbstractItemView, QFileDialog, QMessageBox, QHeaderView, QProgressBar)
from PySide6.QtCore import Qt, QMimeData, QUrl, QTimer, QSize, QPoint, QThread, Signal
from PySide6.QtGui import QDrag, QColor, QPixmap, QIcon
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

# --- GESTIONE RISORSE ---
def get_base_path():
    try: base_path = sys._MEIPASS
    except: base_path = os.path.dirname(os.path.abspath(__file__))
    return base_path

def resource_path(relative_path): return os.path.join(get_base_path(), relative_path)
DB_PATH = os.path.join(get_base_path(), "audio.db")

# --- SCANNER INTELLIGENTE ---
class ScanWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal(int)

    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        self.ai_model = None

    def run(self):
        if AI_AVAILABLE:
            self.log.emit("🧠 Caricamento Modello AI...")
            try:
                self.ai_model = AudioTagging(checkpoint_path=None, device='cpu') 
                self.log.emit("✅ AI Pronta! Inizio scansione...")
            except Exception as e:
                self.log.emit(f"❌ Errore AI: {e}")
                self.ai_model = None
        
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("DELETE FROM files")
        conn.commit()

        valid_exts = ('.wav', '.mp3', '.flac', '.aiff', '.ogg', '.m4a', '.wma', '.aac', '.opus', '.aif')
        file_list = []
        for root, dirs, files in os.walk(self.folder):
            for f in files:
                if f.lower().endswith(valid_exts):
                    file_list.append(os.path.join(root, f))
        
        total = len(file_list)
        count = 0
        
        for i, path in enumerate(file_list):
            filename = os.path.basename(path)
            tags = "Unknown"
            duration = 0.0
            try:
                size = os.path.getsize(path)
                if self.ai_model:
                    y, _ = librosa.load(path, sr=32000, mono=True, duration=10.0)
                    y = y[None, :]
                    clipwise_output, _ = self.ai_model.inference(y)
                    top_indices = np.argsort(clipwise_output[0])[::-1][:3]
                    detected = [self.ai_model.labels[k] for k in top_indices]
                    tags = ", ".join(detected)
                    duration = librosa.get_duration(path=path)
                else:
                    duration = librosa.get_duration(path=path)
                    tags = "No AI Mode"

                cur.execute("INSERT OR IGNORE INTO files (filename, path, tags, size, duration) VALUES (?, ?, ?, ?, ?)",
                            (filename, path, tags, size, duration))
                count += 1
                if count % 2 == 0 or count == total:
                    self.progress.emit(int((count / total) * 100))
                    self.log.emit(f"Analisi: {filename[:25]}... [{tags}]")
            except: pass
        
        conn.commit()
        conn.close()
        self.finished.emit(count)

# --- INTERFACCIA ---
class DragButton(QPushButton):
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setAcceptDrops(False); self.main_window = None
    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.LeftButton and self.main_window: self.main_window.start_drag_operation()
        super().mouseMoveEvent(e)

COMMON_BUTTON_STYLE = "QPushButton { background-color: #e65100; color: #ffffff; border-radius: 4px; padding: 8px; font-weight: bold; border: none; } QPushButton:hover { background-color: #ff9800; } QPushButton:pressed { background-color: #bf4500; }"
DARK_STYLE = COMMON_BUTTON_STYLE + "QMainWindow { background-color: #121212; } QTabWidget::pane { border: 1px solid #333; background: #1e1e1e; } QTabBar::tab { background: #2d2d2d; color: #bbb; padding: 10px 20px; } QTabBar::tab:selected { background: #1e1e1e; color: #ff9800; border-bottom: 2px solid #ff9800; } QTableWidget { background-color: #1e1e1e; color: #ffffff; gridline-color: #333; selection-background-color: #e65100; } QLabel { color: #4caf50; font-weight: bold; } QRadioButton { color: #4caf50; font-weight: bold; } QLineEdit, QSpinBox, QComboBox { background-color: #252525; color: #4caf50; border: 1px solid #3d3d3d; border-radius: 4px; padding: 5px; }"

class RndSndApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("rndsnd v0.6.5 // Full Features Restored")
        self.resize(1550, 950)
        self.init_db()
        
        # Audio
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)
        
        self.audio_data = None
        self.sr = 44100
        self.duration = 0.0
        
        # State
        self.selection_range = (0, 0)
        self.is_looping = False
        self.last_folder = None
        self.playhead_line = None
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_playhead_and_loop)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        self.setup_header()
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        self.setup_archive_tab()
        self.setup_mixer_tab()
        self.switch_theme("Dark")
        self.refresh_list()

    def init_db(self):
        conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
        try: cur.execute("SELECT size FROM files LIMIT 1")
        except: cur.execute("DROP TABLE IF EXISTS files")
        cur.execute("CREATE TABLE IF NOT EXISTS files (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, path TEXT UNIQUE, tags TEXT, duration REAL, size INTEGER)")
        conn.commit(); conn.close()

    def setup_header(self):
        header = QHBoxLayout(); self.title_label = QLabel("<h2>rndsnd v0.6.5</h2>")
        header.addWidget(self.title_label); header.addStretch()
        self.theme_combo = QComboBox(); self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.currentTextChanged.connect(self.switch_theme)
        header.addWidget(QLabel("TEMA:")); header.addWidget(self.theme_combo)
        self.main_layout.addLayout(header)

    def switch_theme(self, theme):
        if theme == "Dark":
            self.setStyleSheet(DARK_STYLE); self.canvas_bg, self.wf_color, self.axis_color, self.cursor_color = '#1e1e1e', '#ff9800', 'white', 'white'
        else:
            self.setStyleSheet(COMMON_BUTTON_STYLE); self.canvas_bg, self.wf_color, self.axis_color, self.cursor_color = 'white', '#2e7d32', 'black', 'black'
        if self.audio_data is not None: self.plot_waveform()

    def setup_archive_tab(self):
        tab = QWidget(); layout = QHBoxLayout(tab); splitter = QSplitter(Qt.Horizontal)
        left = QFrame(); l_lyt = QVBoxLayout(left)
        
        lh = QHBoxLayout()
        self.btn_load = QPushButton("📂 CAMBIA"); self.btn_load.clicked.connect(self.open_load_dialog); self.btn_load.setStyleSheet("background-color: #6200ea; color: white;")
        self.btn_refresh = QPushButton("🔄 RILANCIA"); self.btn_refresh.clicked.connect(self.refresh_current_library); self.btn_refresh.setStyleSheet("background-color: #2e7d32; color: white;")
        lh.addWidget(self.btn_load); lh.addWidget(self.btn_refresh); l_lyt.addLayout(lh)
        
        self.archive_status_lbl = QLabel("Pronto."); self.archive_status_lbl.setStyleSheet("color: #aaa; font-style: italic;")
        l_lyt.addWidget(self.archive_status_lbl)
        self.scan_progress = QProgressBar(); self.scan_progress.setVisible(False); l_lyt.addWidget(self.scan_progress)
        
        self.search_input = QLineEdit(); self.search_input.setPlaceholderText("🔍 Cerca tag...")
        self.search_input.textChanged.connect(self.refresh_list); l_lyt.addWidget(self.search_input)
        
        self.file_table = QTableWidget(); self.file_table.setColumnCount(5)
        self.file_table.setHorizontalHeaderLabels(["Nome", "Durata", "Size", "Data", "Tags"])
        self.file_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.file_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.file_table.setColumnWidth(0, 300) 
        self.file_table.itemClicked.connect(self.load_selected_file_table); l_lyt.addWidget(self.file_table)
        
        right = QFrame(); r_lyt = QVBoxLayout(right)
        self.figure, self.ax = plt.subplots(); self.canvas = FigureCanvas(self.figure)
        # EVENTI MOUSE
        self.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.canvas.mpl_connect('scroll_event', self.on_scroll_zoom)
        r_lyt.addWidget(self.canvas)
        
        self.span = SpanSelector(self.ax, self.on_select, 'horizontal', useblit=True, props=dict(alpha=0.4, facecolor='#ff9800'), interactive=True, drag_from_anywhere=True)
        
        # --- TRANSPORT BAR ---
        t_lyt = QHBoxLayout()
        self.btn_rew = QPushButton("⏪ -5s"); self.btn_play = QPushButton("▶ PLAY")
        self.btn_stop = QPushButton("⏹ STOP"); self.btn_fwd = QPushButton("⏩ +5s")
        self.btn_drag = DragButton("📦 DRAG"); self.btn_drag.main_window = self
        self.btn_drag.setFixedWidth(120); self.btn_drag.setStyleSheet("background-color: #333; border: 1px dashed #666; color: #aaa;")
        
        self.btn_rew.clicked.connect(lambda: self.seek_relative(-5000))
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_stop.clicked.connect(self.stop_audio)
        self.btn_fwd.clicked.connect(lambda: self.seek_relative(5000))
        
        t_lyt.addWidget(self.btn_rew); t_lyt.addWidget(self.btn_play); t_lyt.addWidget(self.btn_stop); t_lyt.addWidget(self.btn_fwd)
        t_lyt.addStretch(); t_lyt.addWidget(self.btn_drag)
        r_lyt.addLayout(t_lyt)
        
        splitter.addWidget(left); splitter.addWidget(right); layout.addWidget(splitter); self.tabs.addTab(tab, "Editor")

    def setup_mixer_tab(self):
        tab = QWidget(); l = QVBoxLayout(tab); l.setAlignment(Qt.AlignCenter)
        l.addWidget(QLabel("<h2>rndsnd mixer</h2>"))
        self.radio_tags = QRadioButton("Usa lista filtrata (Ricerca)"); self.radio_tags.setChecked(True)
        self.radio_chaos = QRadioButton("Full Chaos (Tutto il DB)")
        h_params = QHBoxLayout()
        h_params.addWidget(QLabel("Min:")); self.spin_dur = QSpinBox(); self.spin_dur.setRange(1, 60); self.spin_dur.setValue(5); h_params.addWidget(self.spin_dur)
        self.mix_btn = QPushButton("🚀 GENERATE MIX"); self.mix_btn.setFixedSize(300, 60); self.mix_btn.clicked.connect(self.generate_mix)
        self.status_lbl = QLabel(""); self.status_lbl.setAlignment(Qt.AlignCenter)
        l.addWidget(self.radio_tags); l.addWidget(self.radio_chaos); l.addLayout(h_params); l.addWidget(self.mix_btn); l.addWidget(self.status_lbl)
        self.tabs.addTab(tab, "Mixer")

    def open_load_dialog(self):
        f = QFileDialog.getExistingDirectory(self, "Seleziona cartella audio")
        if f: self.last_folder = f; self.start_scan_process(f)

    def refresh_current_library(self):
        if self.last_folder and os.path.exists(self.last_folder): self.start_scan_process(self.last_folder)
        else: self.open_load_dialog()

    def start_scan_process(self, folder):
        self.scan_progress.setValue(0); self.scan_progress.setVisible(True)
        self.btn_load.setEnabled(False); self.btn_refresh.setEnabled(False)
        self.archive_status_lbl.setText("🚀 Analisi AI in corso...")
        self.scan_thread = ScanWorker(folder)
        self.scan_thread.progress.connect(self.scan_progress.setValue)
        self.scan_thread.log.connect(self.archive_status_lbl.setText)
        self.scan_thread.finished.connect(self.on_scan_finished); self.scan_thread.start()

    def on_scan_finished(self, count):
        self.scan_progress.setVisible(False); self.btn_load.setEnabled(True); self.btn_refresh.setEnabled(True)
        self.refresh_list(); self.archive_status_lbl.setText(f"✅ Completato: {count} file analizzati.")

    def refresh_list(self):
        filter_text = self.search_input.text().lower(); self.file_table.setRowCount(0); conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
        if filter_text: cur.execute("SELECT filename, tags, path, duration, size FROM files WHERE lower(filename) LIKE ? OR lower(tags) LIKE ? LIMIT 500", (f"%{filter_text}%", f"%{filter_text}%"))
        else: cur.execute("SELECT filename, tags, path, duration, size FROM files LIMIT 200")
        rows = cur.fetchall(); self.file_table.setRowCount(len(rows))
        for row, (name, tags, path, dur, size_b) in enumerate(rows):
            dur_str = f"{int(dur//60)}:{int(dur%60):02d}" if dur else "--:--"
            size = f"{size_b/(1024*1024):.1f}MB" if size_b else "N/A"
            ni = QTableWidgetItem(name); ni.setData(Qt.UserRole, path); ni.setForeground(QColor('#4caf50'))
            self.file_table.setItem(row, 0, ni); self.file_table.setItem(row, 1, QTableWidgetItem(dur_str))
            self.file_table.setItem(row, 2, QTableWidgetItem(size)); self.file_table.setItem(row, 3, QTableWidgetItem("Today"))
            self.file_table.setItem(row, 4, QTableWidgetItem(tags))
        conn.close()

    def load_selected_file_table(self, item):
        path = self.file_table.item(item.row(), 0).data(Qt.UserRole)
        try:
            audio = AudioSegment.from_file(path)
            y = np.array(audio.get_array_of_samples())
            if audio.channels == 2: y = y.reshape((-1, 2)).mean(axis=1)
            self.audio_data = y.astype(np.float32) / (1 << (8 * audio.sample_width - 1))
            self.sr = audio.frame_rate; self.duration = len(self.audio_data) / self.sr
            self.player.setSource(QUrl.fromLocalFile(path))
            self.is_looping = False; self.selection_range = (0, 0)
            self.plot_waveform()
        except: traceback.print_exc()

    def plot_waveform(self):
        self.ax.clear(); self.ax.set_facecolor(self.canvas_bg); self.figure.patch.set_facecolor(self.canvas_bg)
        t = np.linspace(0, self.duration, len(self.audio_data))
        self.ax.plot(t[::100], self.audio_data[::100], color=self.wf_color, lw=0.7)
        self.ax.set_xlim(0, self.duration); self.ax.set_ylim(-1.1, 1.1)
        self.playhead_line = self.ax.axvline(x=0, color=self.cursor_color, lw=2); self.canvas.draw()

    def on_scroll_zoom(self, event):
        if self.audio_data is None or event.inaxes != self.ax: return
        cur_xlim = self.ax.get_xlim(); scale = 1/1.2 if event.button == 'up' else 1.2
        new_xr = (cur_xlim[1] - cur_xlim[0]) * scale
        new_xm = event.xdata + new_xr * ((cur_xlim[1] - event.xdata) / (cur_xlim[1] - cur_xlim[0]))
        self.ax.set_xlim([max(0, new_xm - new_xr), min(self.duration, new_xm)]); self.canvas.draw_idle()

    def on_select(self, xmin, xmax):
        self.selection_range = (xmin, xmax); self.is_looping = True; self.btn_drag.setText(f"📦 {xmax-xmin:.1f}s")
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            cp = self.player.position() / 1000.0
            if cp < xmin or cp > xmax: self.player.setPosition(int(xmin * 1000))

    def on_mouse_click(self, event):
        if event.inaxes != self.ax or self.audio_data is None: return
        # Click per posizionare cursore
        click_time = max(0, min(event.xdata, self.duration))
        self.player.setPosition(int(click_time * 1000))
        # Se clicco fuori dalla selezione, tolgo il loop
        if self.is_looping:
            s, e = self.selection_range
            if click_time < s or click_time > e:
                self.is_looping = False
                self.btn_drag.setText("📦 DRAG")
        self.update_playhead_and_loop()

    def update_playhead_and_loop(self):
        cp = self.player.position(); cs = cp / 1000.0
        if self.is_looping:
            s, e = self.selection_range
            if cs >= e or cs < s: self.player.setPosition(int(s * 1000)); cs = s
        if self.playhead_line: self.playhead_line.set_xdata([cs]); self.canvas.draw_idle()

    def start_drag_operation(self):
        if self.audio_data is None: return
        s, e = (self.selection_range if self.is_looping else (0, self.duration))
        chunk = self.audio_data[int(s*self.sr):int(e*self.sr)]
        tp = os.path.join(tempfile.gettempdir(), f"rnd_{int(time.time())}.wav")
        sf.write(tp, chunk, self.sr)
        drag = QDrag(self.btn_drag); mime = QMimeData(); mime.setUrls([QUrl.fromLocalFile(tp)])
        drag.setMimeData(mime); drag.exec_(Qt.CopyAction)

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState: self.player.pause(); self.timer.stop(); self.btn_play.setText("▶ PLAY")
        else: self.player.play(); self.timer.start(50); self.btn_play.setText("⏸ PAUSE")
    def stop_audio(self): self.player.stop(); self.timer.stop(); self.btn_play.setText("▶ PLAY"); self.update_playhead_and_loop()
    def seek_relative(self, ms): self.player.setPosition(max(0, self.player.position() + ms))

    def generate_mix(self):
        if not os.path.exists("output"): os.makedirs("output")
        fname = f"rndsnd_mix_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"; ext = "mp3"
        src = []
        if self.radio_tags.isChecked():
            for r in range(self.file_table.rowCount()): src.append(self.file_table.item(r, 0).data(Qt.UserRole))
        else:
            c = sqlite3.connect(DB_PATH); cur = c.cursor(); cur.execute("SELECT path FROM files"); src = [x[0] for x in cur.fetchall()]; c.close()
        if not src: return
        self.status_lbl.setText("⏳ Mixing..."); QApplication.processEvents()
        mix = AudioSegment.silent(duration=0); target = self.spin_dur.value() * 60 * 1000
        while len(mix) < target:
            try:
                a = AudioSegment.from_file(random.choice(src))
                c_len = random.randint(10000, 30000)
                seg = a[0:c_len] if len(a) < c_len else a[random.randint(0, len(a)-c_len):][:c_len]
                mix = mix.append(seg, crossfade=500) if len(mix) > 0 else seg
            except: continue
        mix.export(f"output/{fname}.{ext}", format=ext)
        self.status_lbl.setText(f"✅ Creato: {fname}.{ext}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Percorsi icone e splash
    sp = resource_path("rndsnd_splash.png")
    ic = resource_path("rndsnd_icon.png")
    
    splash = None
    if os.path.exists(sp):
        pix = QPixmap(sp)
        if not pix.isNull():
            # Ridimensiona se troppo grande (opzionale)
            if pix.width() > 800:
                pix = pix.scaledToWidth(800, Qt.SmoothTransformation)
            
            splash = QSplashScreen(pix)
            # Forza lo splash a stare sopra tutte le finestre
            splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
            splash.show()
            
            # Manteniamo lo splash visibile mentre carichiamo l'app
            app.processEvents()
            # Un piccolo ritardo di 2 secondi per godersi l'immagine
            time.sleep(2) 

    # Carica la finestra principale
    win = RndSndApp()
    if os.path.exists(ic):
        win.setWindowIcon(QIcon(ic))
        
    win.show()
    
    # Chiudi lo splash solo dopo che la finestra principale è visibile
    if splash:
        splash.finish(win)
        
    sys.exit(app.exec())
