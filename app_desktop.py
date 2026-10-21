import sys
import os
import time
import math
import sqlite3
import datetime
import random
import threading
import tempfile
import ctypes
import traceback
import warnings
import shutil
import json
import copy
import importlib.util
import urllib.request
import urllib.error
import http.client

import numpy as np
import librosa
import soundfile as sf

# Pydub: usato solo come fallback per formati che libsndfile non legge (m4a, wma, aac...)
try:
    from pydub import AudioSegment
    HAVE_PYDUB = True
except Exception:  # es. Python 3.13 senza audioop
    AudioSegment = None
    HAVE_PYDUB = False

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")
APP_VERSION = "0.11.0"

# --- MOTORE IA (PANNs) ---
# PyTorch NON viene importato all'avvio (e' lento e pesante): qui si controlla solo se e' installato.
# Il vero import avviene nel thread di scansione, quando serve (vedi load_ai_model).
def _module_available(name):
    try: return importlib.util.find_spec(name) is not None
    except Exception: return False

# Fix ALSA errors on Linux
def alsa_error_handler(filename, line, function, err, fmt): pass
try:
    asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    CMPFUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
    c_alsa_error_handler = CMPFUNC(alsa_error_handler)
    asound.snd_lib_error_set_handler(c_alsa_error_handler)
except Exception:
    pass

os.environ['QT_API'] = 'pyside6'
os.environ['PYTHONWARNINGS'] = 'ignore'

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import SpanSelector

from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QPushButton, QWidget, QLineEdit, QTableWidget, QTableWidgetItem,
                             QLabel, QTabWidget, QSplitter, QFrame, QRadioButton, QSpinBox,
                             QComboBox, QSplashScreen, QAbstractItemView, QHeaderView,
                             QProgressBar, QFileSystemModel, QTreeView, QMenu, QCheckBox,
                             QSlider, QScrollArea, QSizePolicy, QListWidget, QListWidgetItem, QFileDialog,
                             QDoubleSpinBox, QInputDialog, QScrollBar, QMessageBox)
from PySide6.QtCore import (Qt, QMimeData, QUrl, QTimer, QThread, Signal, QElapsedTimer,
                            QDir, QIODevice, QRectF, QPointF)
from PySide6.QtGui import (QDrag, QColor, QPixmap, QIcon, QAction, QPainter, QPen, QFont,
                           QDesktopServices, QShortcut, QKeySequence, QBrush)
from PySide6.QtMultimedia import (QMediaPlayer, QAudioOutput, QMediaDevices, QAudioSink,
                                  QAudioFormat)

# =====================================================================
#  PERCORSI
#  FIX: il DB stava in sys._MEIPASS (cartella dell'app / temporanea).
#  Ora dati e output stanno in ~/rndsnd, indipendenti dall'installazione.
# =====================================================================
def get_base_path():
    try: return sys._MEIPASS
    except Exception: return os.path.dirname(os.path.abspath(__file__))

def resource_path(relative_path): return os.path.join(get_base_path(), relative_path)

APP_DIR = os.path.join(os.path.expanduser("~"), "rndsnd")
OUTPUT_DIR = os.path.join(APP_DIR, "output")
DRAG_DIR = os.path.join(tempfile.gettempdir(), "rndsnd_drag")
DB_PATH = os.path.join(APP_DIR, "audio.db")

def prepare_dirs():
    for d in (APP_DIR, OUTPUT_DIR, DRAG_DIR):
        try: os.makedirs(d, exist_ok=True)
        except Exception: pass
    # migra il vecchio database (accanto all'app) se il nuovo non esiste ancora
    legacy = os.path.join(get_base_path(), "audio.db")
    if not os.path.exists(DB_PATH) and os.path.exists(legacy):
        try: shutil.copy2(legacy, DB_PATH)
        except Exception: pass
    # pulizia dei file di drag più vecchi di 1 giorno
    try:
        limit = time.time() - 86400
        for f in os.listdir(DRAG_DIR):
            fp = os.path.join(DRAG_DIR, f)
            if os.path.getmtime(fp) < limit:
                if os.path.isdir(fp): shutil.rmtree(fp, ignore_errors=True)
                else: os.remove(fp)
    except Exception:
        pass

prepare_dirs()

# =====================================================================
#  UTILITY DB
# =====================================================================
def esc_like(s):
    """Escape dei caratteri speciali per LIKE ... ESCAPE '\\'."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

def folder_prefix(folder):
    folder = os.path.normpath(folder)
    return folder if folder.endswith(os.sep) else folder + os.sep

def fmt_time(sec):
    sec = max(0, int(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"

# =====================================================================
#  MOTORE AUDIO "OFFLINE" (condiviso da Mixer e Live Mix)
# =====================================================================
MIX_SR = 44100
RMS_TARGET = 0.2   # livello medio di ogni clip dopo la normalizzazione (~ -14 dBFS)

GRAINS = [("Micro (0.2s - 2s)", 200, 2000), ("Short (2s - 5s)", 2000, 5000),
          ("Medium (5s - 15s)", 5000, 15000), ("Long (15s - 30s)", 15000, 30000),
          ("Extra Long (30s - 60s)", 30000, 60000)]

class Cancelled(Exception): pass

def _to_stereo_44k(data, sr):
    if data.ndim == 1: data = data[:, None]
    if data.shape[1] == 1: data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2: data = data[:, :2]
    if sr != MIX_SR:
        data = librosa.resample(np.ascontiguousarray(data.T), orig_sr=sr, target_sr=MIX_SR).T
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(data, dtype=np.float32)

def read_clip(path, clip_ms, rng):
    """Legge SOLO la porzione casuale richiesta (non l'intero file).
    Ritorna (array stereo float32 a 44.1k, start_ms) oppure None se illeggibile."""
    try:
        info = sf.info(path)
        sr, frames = info.samplerate, info.frames
        if frames <= 0 or sr <= 0: raise RuntimeError("empty")
        want = max(1, int(clip_ms * sr / 1000))
        start = 0 if frames <= want else rng.randint(0, frames - want)
        stop = min(frames, start + want)
        data, _ = sf.read(path, start=start, stop=stop, dtype='float32', always_2d=True)
        return _to_stereo_44k(data, sr), start * 1000.0 / sr
    except Exception:
        pass
    if HAVE_PYDUB:  # fallback (richiede ffmpeg per m4a/wma/aac)
        try:
            a = AudioSegment.from_file(path)
            start = 0 if len(a) <= clip_ms else rng.randint(0, len(a) - clip_ms)
            s = a[start:start + clip_ms]
            ch = s.channels
            arr = np.array(s.get_array_of_samples(), dtype=np.float32) / float(1 << (8 * s.sample_width - 1))
            return _to_stereo_44k(arr.reshape(-1, ch), s.frame_rate), float(start)
        except Exception:
            return None
    return None

def rms_normalize(x, target=RMS_TARGET, max_gain=10.0):
    r = float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0
    if r < 1e-4: return x
    return x * min(target / r, max_gain)

def apply_pan(x, p):
    """Pan equal-power, p in [-1, 1] (0 = centro, guadagno unitario)."""
    ang = (p + 1.0) * math.pi / 4.0
    y = x.copy()
    y[:, 0] *= math.cos(ang) * math.sqrt(2.0)
    y[:, 1] *= math.sin(ang) * math.sqrt(2.0)
    return y

def _fade(n, rising):
    c = np.sin if rising else np.cos
    return c(np.linspace(0.0, math.pi / 2, max(1, n), dtype=np.float32)).astype(np.float32)

def build_layer(src, n_frames, min_ms, max_ms, xf_ms, randomize, ctx, label, rng, prog=None, grid_frames=0):
    """Costruisce UN layer lungo n_frames mettendo clip casuali una dopo l'altra.
    xf_ms > 0  -> crossfade da DJ tra clip consecutive (modo 1 layer)
    randomize  -> pan e volume casuali per clip (modo multi-layer)
    grid_frames > 0 -> la lunghezza di ogni clip diventa un multiplo di questa griglia (tagli a tempo)"""
    buf = np.zeros((n_frames, 2), np.float32)
    log = []
    pos = 0; fails = 0; first = True; prev_len = 0
    xf_max = int(MIX_SR * xf_ms / 1000)
    fade = int(MIX_SR * (0.05 if xf_ms else 0.1))
    min_len = int(MIX_SR * 0.05)
    while pos < n_frames:
        ctx.check()
        path = rng.choice(src)
        got = read_clip(path, rng.randint(min_ms, max_ms), rng)
        if got is None or len(got[0]) < min_len:
            fails += 1
            # FIX: prima, se nessun file era leggibile, il ciclo girava all'infinito
            if fails >= 40:
                raise RuntimeError("Nessun file leggibile tra le sorgenti scelte "
                                   "(file corrotti o formato non supportato / ffmpeg mancante?).")
            continue
        fails = 0
        seg, start_ms = got
        if grid_frames > 0:
            k = max(1, int(round(len(seg) / grid_frames)))
            if k * grid_frames > len(seg): k = max(1, len(seg) // grid_frames)
            target = k * grid_frames
            seg = seg[:target] if len(seg) >= target else np.pad(seg, ((0, target - len(seg)), (0, 0)))
        seg = rms_normalize(seg)
        if randomize:
            seg = apply_pan(seg, rng.uniform(-0.5, 0.5))
            seg = seg * np.float32(10 ** (-rng.uniform(0, 6) / 20.0))
        n = len(seg)
        if xf_max and not first:
            x = max(1, min(xf_max, n // 2, prev_len // 2))
            start = pos - x
            seg[:x] *= _fade(x, True)[:, None]
            buf[start:pos] *= _fade(x, False)[:, None]
        else:
            h = max(1, min(fade, n // 2))
            seg[:h] *= _fade(h, True)[:, None]
            if not xf_max:
                seg[-h:] *= _fade(h, False)[:, None]
            start = pos
        end = min(n_frames, start + n)
        buf[start:end] += seg[:end - start]
        nominal = pos   # posizione "musicale" della clip (prima del crossfade)
        log.append(f"{label}: {os.path.basename(path)} [{fmt_time(start_ms/1000)}-{fmt_time(start_ms/1000 + n/MIX_SR)}] @ t={nominal / MIX_SR:.2f}s")
        pos = (nominal + n) if grid_frames > 0 else (start + n)   # con la griglia non c'è deriva
        prev_len = n; first = False
        if prog: prog(min(1.0, pos / n_frames))
    f = max(1, min(fade, n_frames // 2))
    buf[:f] *= _fade(f, True)[:, None]
    buf[-f:] *= _fade(f, False)[:, None]
    return buf, log

def compute_peaks(stereo, bins=2048):
    """Min/max per bin (mono) per disegnare la forma d'onda."""
    mono = (stereo[:, 0] + stereo[:, 1]) * 0.5
    n = len(mono); step = max(1, n // bins); m = (n // step) * step
    r = mono[:m].reshape(-1, step)
    return np.stack([r.min(axis=1), r.max(axis=1)], axis=1).astype(np.float32)

def render_mix(layers, gains):
    acc = np.zeros_like(layers[0])
    for lay, g in zip(layers, gains):
        if g > 0: acc += lay * np.float32(g)
    return acc

def export_mix(audio, base_name, header_lines, log_lines, soft=True):
    """Esporta in MP3 (libsndfile -> pydub/ffmpeg) con fallback automatico a WAV. + file .txt di log."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # nome unico: due mix nello stesso secondo non si sovrascrivono
    stem, k = base_name, 1
    while any(os.path.exists(os.path.join(OUTPUT_DIR, stem + e)) for e in (".mp3", ".wav", ".txt")):
        k += 1; stem = f"{base_name}_{k}"
    base_name = stem
    # soft=True: soft-limiter (mix generativi). soft=False: nessuna colorazione, solo protezione da overflow (multitraccia)
    data = (np.tanh(audio) if soft else np.clip(audio, -1.0, 1.0)).astype(np.float32)
    path = None; note = ""
    mp3 = os.path.join(OUTPUT_DIR, base_name + ".mp3")
    try:
        if 'MP3' in sf.available_formats():
            sf.write(mp3, data, MIX_SR, format='MP3')
            if os.path.getsize(mp3) > 0: path = mp3
    except Exception:
        pass
    if path is None and HAVE_PYDUB and (shutil.which("ffmpeg") or shutil.which("avconv")):
        try:
            pcm = (data * 32767).astype(np.int16)
            AudioSegment(data=pcm.tobytes(), sample_width=2, frame_rate=MIX_SR, channels=2).export(mp3, format="mp3")
            if os.path.getsize(mp3) > 0: path = mp3
        except Exception:
            pass
    if path is None:
        path = os.path.join(OUTPUT_DIR, base_name + ".wav")
        sf.write(path, data, MIX_SR, subtype='PCM_16')
        note = " (MP3 non disponibile: salvato in WAV)"
    try:
        with open(os.path.join(OUTPUT_DIR, base_name + ".txt"), "w", encoding="utf-8") as f:
            f.write("RNDSND LOG\n")
            for h in header_lines: f.write(h + "\n")
            f.write("-" * 50 + "\n")
            for line in log_lines: f.write(f"- {line}\n")
    except Exception:
        pass
    return path, note

# =====================================================================
#  THREAD GENERICO (per non bloccare mai la GUI)
# =====================================================================
class TaskThread(QThread):
    progress = Signal(int)
    status = Signal(str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self._cancel = False

    def cancel(self): self._cancel = True
    def check(self):
        if self._cancel: raise Cancelled()

    def run(self):
        try:
            res = self.fn(self)
        except Cancelled:
            self.failed.emit("cancelled"); return
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e)); return
        self.done.emit(res)

# --- INTELLIGENT SCANNER (MULTI-SAMPLE) ---
def get_duration(path):
    try: return float(sf.info(path).duration)
    except Exception: pass
    try: return float(librosa.get_duration(path=path))
    except TypeError: return float(librosa.get_duration(filename=path))

# =====================================================================
#  IA: stato, download del modello (senza wget), caricamento, etichettatura
# =====================================================================
PANNS_DIR = os.path.join(os.path.expanduser("~"), "panns_data")           # stessa cartella che usa la libreria
PANNS_CKPT = os.path.join(PANNS_DIR, "Cnn14_mAP=0.431.pth")
PANNS_LABELS = os.path.join(PANNS_DIR, "class_labels_indices.csv")
PANNS_CKPT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
PANNS_LABELS_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
PANNS_CKPT_MIN = 3e8        # stessa soglia della libreria: sotto, il file e' considerato incompleto
UNTAGGED = ("No AI", "Scan Error", "Unknown")   # tag che "Rianalizza con IA" sostituisce (i tag scritti a mano restano)

def ai_status():
    """'missing_libs' (PyTorch/panns_inference non installati) | 'need_model' (file da scaricare) | 'ready'."""
    if not (_module_available('torch') and _module_available('panns_inference')): return 'missing_libs'
    ok = (os.path.isfile(PANNS_LABELS) and os.path.getsize(PANNS_LABELS) > 0 and
          os.path.isfile(PANNS_CKPT) and os.path.getsize(PANNS_CKPT) >= PANNS_CKPT_MIN)
    return 'ready' if ok else 'need_model'

def download_file(url, dest, progress=None, check=None, min_bytes=0, timeout=30, chunk=1 << 18):
    """Scarica in dest con ripresa: i dati arrivano in dest.part e, se il download si interrompe,
    il tentativo successivo riparte da dove era arrivato (se il server supporta Range)."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {'User-Agent': 'rndsnd'}
    if have: headers['Range'] = f'bytes={have}-'
    resp = None
    try:
        resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)
    except urllib.error.HTTPError as e:
        if not (e.code == 416 and have):      # 416 + .part esistente: il file era gia' completo
            raise RuntimeError(f"server answered HTTP {e.code}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"network error: {e.reason}") from None
    except OSError as e:
        raise RuntimeError(f"network error: {e}") from None
    total = 0
    if resp is not None:
        with resp:
            clen = int(resp.headers.get('Content-Length') or 0)
            if have and getattr(resp, 'status', 200) == 206: mode = 'ab'; total = have + clen if clen else 0
            else: mode = 'wb'; have = 0; total = clen       # il server ignora Range: si riparte da zero
            done = have
            try:
                with open(part, mode) as f:
                    while True:
                        if check: check()
                        buf = resp.read(chunk)
                        if not buf: break
                        f.write(buf); done += len(buf)
                        if progress: progress(done, total)
            except (http.client.IncompleteRead, OSError) as e:
                raise RuntimeError(f"connection interrupted ({e.__class__.__name__}); press the button again to resume") from None
    size = os.path.getsize(part)
    if total and size != total:
        raise RuntimeError(f"incomplete download ({size / 1e6:.0f} of {total / 1e6:.0f} MB); press the button again to resume")
    if size < min_bytes:
        os.remove(part)
        raise RuntimeError("the downloaded file is too small (probably an error page); try again later")
    os.replace(part, dest)

def load_ai_model():
    """Importa PyTorch e panns_inference SOLO ora. I file sono gia' scaricati da noi, quindi la libreria
    non chiama wget. Usa la GPU se PyTorch ne trova una, altrimenti la CPU. Ritorna (modello, device)."""
    import torch
    from panns_inference import AudioTagging
    try:
        model = AudioTagging(checkpoint_path=PANNS_CKPT, device='cuda')
    except Exception as e:
        # PyTorch >= 2.6 carica i checkpoint in modalita' "weights_only": se il file ufficiale lo rifiuta, si ritenta
        if 'weights_only' not in str(e) and 'npickl' not in str(e): raise
        orig = torch.load
        torch.load = lambda *a, **k: orig(*a, **{**k, 'weights_only': False})
        try: model = AudioTagging(checkpoint_path=PANNS_CKPT, device='cuda')
        finally: torch.load = orig
    return model, str(getattr(model, 'device', 'cpu'))

def tag_audio(model, path, duration):
    """Le 3 categorie piu' probabili su fino a 3 frammenti da 5 s (inizio, meta', fine)."""
    chunk_dur = 5.0
    if duration < 10: offsets = [0]
    else: offsets = [max(0, o) for o in (0, duration / 2 - chunk_dur / 2, duration - chunk_dur)]
    acc = None
    for off in offsets:
        y, _ = librosa.load(path, sr=32000, mono=True, offset=off, duration=chunk_dur)
        if len(y) < 32000: y = np.pad(y, (0, 32000 - len(y)), mode='constant')
        clipwise, _ = model.inference(y[None, :])
        acc = clipwise[0].copy() if acc is None else acc + clipwise[0]
    top = np.argsort(acc)[::-1][:3]
    return ", ".join(model.labels[int(k)] for k in top)

class ScanWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    scan_done = Signal(int)   # FIX: prima ridefiniva QThread.finished

    def __init__(self, folder, use_ai=True, retag=False):
        super().__init__()
        self.folder = folder
        self.use_ai = use_ai          # False: scansione senza IA (tag "No AI")
        self.retag = retag            # True: rianalizza solo i file con tag "No AI" / "Scan Error" / "Unknown"
        self.ai_model = None
        self.ai_error = None          # motivo per cui l'IA non e' stata usata (mostrato all'utente)
        self._ai_tried = False

    def run(self):
        count = 0
        conn = None
        try:
            conn = sqlite3.connect(DB_PATH)
            count = self._retag(conn) if self.retag else self._scan(conn)
        except Exception:
            traceback.print_exc()
        finally:
            if conn is not None:
                try: conn.commit(); conn.close()
                except Exception: pass
            self.scan_done.emit(count)   # sempre emesso (prima poteva restare "Scanning..." per sempre)

    def _ensure_model(self):
        """Carica il modello UNA volta sola. Ritorna True se e' disponibile."""
        if self.ai_model is not None: return True
        if self._ai_tried or not self.use_ai: return False
        self._ai_tried = True
        st = ai_status()
        if st != 'ready':
            self.ai_error = ("PyTorch / panns-inference not installed" if st == 'missing_libs'
                             else "AI model files not downloaded")
            return False
        self.log.emit("🧠 Loading AI model...")
        try:
            self.ai_model, dev = load_ai_model()
            self.log.emit(f"🧠 AI model ready ({'GPU' if 'cuda' in dev else 'CPU'})")
            return True
        except Exception as e:
            traceback.print_exc()
            self.ai_error = f"model could not be loaded: {str(e)[:120]}"
            return False

    def _scan(self, conn):
        cur = conn.cursor()
        valid_exts = ('.wav', '.mp3', '.flac', '.aiff', '.ogg', '.m4a', '.wma', '.aac', '.opus', '.aif')
        file_list = []
        for root, dirs, files in os.walk(self.folder):
            for f in files:
                if f.lower().endswith(valid_exts):
                    file_list.append(os.path.normpath(os.path.join(root, f)))
        total = len(file_list)

        # rimuove dal DB i file di questa cartella che non esistono più
        prefix = folder_prefix(self.folder)
        cur.execute("SELECT path FROM files WHERE substr(path, 1, ?) = ?", (len(prefix), prefix))
        present = set(file_list)
        stale = [(p,) for (p,) in cur.fetchall() if p not in present]
        if stale: cur.executemany("DELETE FROM files WHERE path = ?", stale)
        if total == 0: return 0

        new_files = 0
        for i, path in enumerate(file_list):
            if self.isInterruptionRequested(): break
            filename = os.path.basename(path)
            cur.execute("SELECT 1 FROM files WHERE path = ?", (path,))
            if cur.fetchone():
                if i % 10 == 0: self.progress.emit(int((i / total) * 100))
                continue

            have_ai = self._ensure_model()
            tags = "Unknown"; duration = 0.0; size = 0   # FIX: 'size' ora è sempre definita
            try:
                size = os.path.getsize(path)
                duration = get_duration(path)
                tags = tag_audio(self.ai_model, path, duration) if have_ai else "No AI"
            except Exception as e:
                print(f"Error analyzing {filename}: {e}")
                tags = "Scan Error"
            cur.execute("INSERT OR IGNORE INTO files (filename, path, folder, tags, size, duration) VALUES (?, ?, ?, ?, ?, ?)",
                        (filename, path, os.path.dirname(path), tags, size, duration))
            new_files += 1
            if new_files % 20 == 0: conn.commit()
            self.progress.emit(int(((i + 1) / total) * 100))
            self.log.emit(f"Analyzed: {filename[:15]}... [{tags}]")
        return new_files

    def _retag(self, conn):
        """Rianalizza con l'IA i file gia' nel database marcati "No AI" / "Scan Error" / "Unknown".
        I tag scritti a mano non vengono toccati."""
        cur = conn.cursor()
        prefix = folder_prefix(self.folder)
        cur.execute(f"SELECT id, path, duration FROM files WHERE substr(path, 1, ?) = ? AND tags IN ({','.join('?' * len(UNTAGGED))})",
                    (len(prefix), prefix) + UNTAGGED)
        rows = [r for r in cur.fetchall() if os.path.exists(r[1])]
        if not rows: return 0
        if not self._ensure_model(): return 0          # senza modello non si sovrascrive nulla
        done = 0
        for i, (fid, path, dur) in enumerate(rows):
            if self.isInterruptionRequested(): break
            try:
                tags = tag_audio(self.ai_model, path, dur or get_duration(path))
            except Exception as e:
                print(f"Error analyzing {os.path.basename(path)}: {e}"); continue
            cur.execute("UPDATE files SET tags = ? WHERE id = ?", (tags, fid))
            done += 1
            if done % 20 == 0: conn.commit()
            self.progress.emit(int((i + 1) / len(rows) * 100))
            self.log.emit(f"🧠 {os.path.basename(path)[:18]}... [{tags}]")
        return done


# =====================================================================
#  ANALISI BPM / TONALITA'
# =====================================================================
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
ALL_KEYS = [f"{n} {m}" for n in NOTE_NAMES for m in ("major", "minor")]
_KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_CAMELOT_MAJ = [8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1]   # numero Camelot della tonalita' maggiore per pitch class

def estimate_key(chroma_mean):
    """Krumhansl-Schmuckler: ritorna (correlazione, nome tonalita')."""
    best = (-2.0, None)
    for pc in range(12):
        for prof, mode in ((_KS_MAJ, "major"), (_KS_MIN, "minor")):
            c = np.corrcoef(chroma_mean, np.roll(prof, pc))[0, 1]
            if c > best[0]: best = (float(c), f"{NOTE_NAMES[pc]} {mode}")
    return best

def _camelot(key):
    note, mode = key.split()
    pc = NOTE_NAMES.index(note)
    return (_CAMELOT_MAJ[pc], "B") if mode == "major" else (_CAMELOT_MAJ[(pc + 3) % 12], "A")

def compatible_keys(key):
    """Tonalita' mixabili secondo la ruota di Camelot: stessa, +-1 quinta, relativa maggiore/minore."""
    if key not in ALL_KEYS: return [key]
    num, let = _camelot(key)
    want = {(num, let), ((num % 12) + 1, let), (((num - 2) % 12) + 1, let), (num, "A" if let == "B" else "B")}
    return [k for k in ALL_KEYS if _camelot(k) in want]

MIN_RHYTHM_CLARITY = 0.30   # sotto: niente BPM (rumore, pad, ambienti senza pulsazione)
MIN_CHROMA_CONTRAST = 0.30  # sotto: niente tonalita' (rumore, percussioni)

def analyze_bpm_key(path, max_sec=30):
    """Ritorna (bpm|None, tonalita'|None). Analizza max_sec secondi presi dal centro del file.
    Se il segnale non ha una pulsazione o un'armonia chiare, il valore resta None."""
    dur = get_duration(path)
    if dur < 3: return None, None
    offset = max(0.0, dur / 2 - max_sec / 2) if dur > max_sec else 0.0
    y, sr = librosa.load(path, sr=22050, mono=True, offset=offset, duration=max_sec)
    if len(y) < sr * 3 or float(np.max(np.abs(y))) < 1e-3: return None, None
    bpm = None; key = None
    try:
        hop = 256
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        tempo, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop)
        t = float(np.atleast_1d(tempo)[0])
        bt = librosa.frames_to_time(beats, sr=sr, hop_length=hop)
        if len(bt) >= 8:   # media su tutti i battiti: elimina l'errore di quantizzazione
            refined = 60.0 * (len(bt) - 1) / (bt[-1] - bt[0])
            if abs(refined - t) / max(t, 1e-6) < 0.04: t = refined
        # regolarita' del ritmo: autocorrelazione dell'envelope al periodo del battito
        e = env - env.mean(); ac = librosa.autocorrelate(e)
        if ac[0] > 0 and t > 0:
            ac = ac / ac[0]
            lag = int(round(60.0 / t * sr / hop))
            clarity = float(ac[max(1, lag - 2):lag + 3].max()) if lag + 3 < len(ac) else 0.0
            if clarity >= MIN_RHYTHM_CLARITY:
                while t > 200: t /= 2
                while 0 < t < 60: t *= 2
                bpm = round(t, 1)
    except Exception:
        pass
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
        if float(np.mean(chroma)) > 0 and float(np.std(chroma) / np.mean(chroma)) >= MIN_CHROMA_CONTRAST:
            corr, name = estimate_key(chroma)
            if corr >= 0.5: key = name
    except Exception:
        pass
    return bpm, key

class AnalyzeWorker(QThread):
    """Calcola BPM e tonalita' dei file di una cartella che non sono ancora stati analizzati."""
    progress = Signal(int)
    log = Signal(str)
    scan_done = Signal(int)

    def __init__(self, folder):
        super().__init__()
        self.folder = folder

    def run(self):
        done = 0; conn = None
        try:
            conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
            prefix = folder_prefix(self.folder)
            cur.execute("SELECT id, path FROM files WHERE substr(path, 1, ?) = ? AND bpm IS NULL", (len(prefix), prefix))
            todo = cur.fetchall(); total = len(todo)
            for i, (fid, path) in enumerate(todo):
                if self.isInterruptionRequested(): break
                try: bpm, key = analyze_bpm_key(path)
                except Exception as e:
                    print(f"BPM/key error {os.path.basename(path)}: {e}"); bpm, key = None, None
                # 0 / '-' = "analizzato ma indeterminato" (non verra' rianalizzato ogni volta)
                cur.execute("UPDATE files SET bpm = ?, tonality = ? WHERE id = ?", (bpm or 0.0, key or "-", fid))
                done += 1
                if done % 10 == 0: conn.commit()
                self.progress.emit(int((i + 1) / total * 100))
                self.log.emit(f"🎼 {os.path.basename(path)[:22]}... {bpm or '-'} BPM, {key or '-'}")
        except Exception:
            traceback.print_exc()
        finally:
            if conn is not None:
                try: conn.commit(); conn.close()
                except Exception: pass
            self.scan_done.emit(done)


# =====================================================================
#  MULTITRACCIA: modello del progetto + mix a blocchi
#  Il progetto e' fatto di dizionari semplici (facili da salvare in JSON e da annullare):
#    track = {name, gain, pan, mute, solo, clips:[clip]}
#    clip  = {id, name, path, media_a, media_b, pos, in, len, gain, fade_in, fade_out}
#  "media_a..media_b" e' il materiale inserito (secondi del file originale); "in"/"len" e' la
#  finestra visibile dentro quel materiale; "pos" e' l'inizio sulla linea del tempo.
#  I file originali non vengono mai modificati.
# =====================================================================
TRACK_COLORS = ['#ff9800', '#29b6f6', '#66bb6a', '#ec407a', '#ab47bc', '#ffee58', '#26a69a', '#ef5350']
MT_BLOCK = 64   # campioni per colonna della piramide dei picchi

def new_track(n):
    return {'name': f"Track {n}", 'gain': 1.0, 'pan': 0.0, 'mute': False, 'solo': False, 'clips': []}

def new_project():
    return {'version': 1, 'tracks': [new_track(i + 1) for i in range(4)], 'next_id': 1, 'bpm': 120, 'snap': True}

def media_key(path, a, b):
    return (path, round(float(a), 3), round(float(b), 3))

def read_segment(path, a, b):
    """Decodifica SOLO [a, b] secondi del file, come stereo float32 a 44.1 kHz."""
    try:
        info = sf.info(path); sr = info.samplerate
        data, _ = sf.read(path, start=int(a * sr), stop=int(b * sr), dtype='float32', always_2d=True)
    except Exception:
        y, sr = librosa.load(path, sr=None, mono=False, offset=a, duration=b - a)   # fallback (m4a ecc.)
        data = (y.T if y.ndim > 1 else y[:, None]).astype(np.float32)
    if len(data) == 0: raise RuntimeError("segmento vuoto")
    return _to_stereo_44k(data, sr)

class MtMedia:
    """Audio decodificato di un segmento + piramide dei picchi per disegnare la forma d'onda in fretta."""
    def __init__(self, arr):
        self.arr = arr
        mono = (arr[:, 0] + arr[:, 1]) * 0.5
        n = (len(mono) // MT_BLOCK) * MT_BLOCK
        if n > 0:
            r = mono[:n].reshape(-1, MT_BLOCK)
            self.pmin, self.pmax = r.min(axis=1), r.max(axis=1)
        else:
            self.pmin = self.pmax = mono[:1] if len(mono) else np.zeros(1, np.float32)
        self.mono = mono
        self.duration = len(arr) / MIX_SR

    def columns(self, t0, length, width):
        """Min/max per ciascuna delle `width` colonne di pixel per l'intervallo [t0, t0+length] (secondi)."""
        s0 = max(0, int(t0 * MIX_SR)); s1 = min(len(self.mono), int((t0 + length) * MIX_SR))
        if width <= 0 or s1 <= s0: return None
        spp = (s1 - s0) / width                      # campioni per pixel
        if spp >= MT_BLOCK * 2:
            b0, b1 = s0 // MT_BLOCK, max(s0 // MT_BLOCK + 1, s1 // MT_BLOCK)
            mn, mx = self.pmin[b0:b1], self.pmax[b0:b1]
        else:
            mn = mx = self.mono[s0:s1]
        N = len(mn)
        if N == 0: return None
        idx = (np.arange(width) * N // width).astype(np.intp)
        return np.minimum.reduceat(mn, idx), np.maximum.reduceat(mx, idx)

def pan_gains(p):
    ang = (p + 1.0) * math.pi / 4.0
    return math.cos(ang) * math.sqrt(2.0), math.sin(ang) * math.sqrt(2.0)

def mt_length(project):
    """Durata del progetto in secondi (fine dell'ultimo clip)."""
    end = 0.0
    for t in project['tracks']:
        for c in t['clips']: end = max(end, c['pos'] + c['len'])
    return end

def build_snapshot(project, media):
    """Lista ordinata di clip "pronti da suonare" (guadagni, pan, mute e solo gia' risolti)."""
    any_solo = any(t['solo'] for t in project['tracks'])
    items = []
    for t in project['tracks']:
        audible = (not t['mute']) and (t['solo'] or not any_solo)
        if not audible: continue
        pl, pr = pan_gains(t['pan'])
        for c in t['clips']:
            m = media.get(media_key(c['path'], c['media_a'], c['media_b']))
            if m is None: continue
            in_f = int(round(c['in'] * MIX_SR))
            n = min(int(round(c['len'] * MIX_SR)), len(m.arr) - in_f)
            if n <= 0: continue
            g = t['gain'] * c['gain']
            items.append({'start': int(round(c['pos'] * MIX_SR)), 'n': n, 'in_f': in_f, 'arr': m.arr,
                          'gl': g * pl, 'gr': g * pr,
                          'fi': min(int(round(c['fade_in'] * MIX_SR)), n),
                          'fo': min(int(round(c['fade_out'] * MIX_SR)), n)})
    items.sort(key=lambda it: it['start'])
    return items

def mt_mix_block(items, pos, frames):
    """Somma i clip che cadono nel blocco [pos, pos+frames). Stessa funzione per ascolto ed esportazione."""
    out = np.zeros((frames, 2), np.float32)
    end = pos + frames
    for it in items:
        s = it['start']
        if s >= end: break                   # ordinati per inizio: i successivi sono tutti dopo
        n = it['n']
        if s + n <= pos: continue
        a = max(pos, s); b = min(end, s + n)
        loc0, loc1 = a - s, b - s
        seg = it['arr'][it['in_f'] + loc0: it['in_f'] + loc1]
        k = len(seg)
        if k <= 0: continue
        fi, fo = it['fi'], it['fo']
        if (fi > 0 and loc0 < fi) or (fo > 0 and loc1 > n - fo):
            env = np.ones(k, np.float32)
            if fi > 0 and loc0 < fi:
                m = min(k, fi - loc0)
                env[:m] = np.arange(loc0, loc0 + m, dtype=np.float32) / fi
            if fo > 0 and loc1 > n - fo:
                i0 = max(0, (n - fo) - loc0)
                env[i0:] *= (n - (loc0 + np.arange(i0, k, dtype=np.float32))) / fo
            out[a - pos:a - pos + k, 0] += seg[:, 0] * env * np.float32(it['gl'])
            out[a - pos:a - pos + k, 1] += seg[:, 1] * env * np.float32(it['gr'])
        else:
            out[a - pos:a - pos + k, 0] += seg[:, 0] * np.float32(it['gl'])
            out[a - pos:a - pos + k, 1] += seg[:, 1] * np.float32(it['gr'])
    return out

CLIP_MIME = "application/x-rndsnd-clip"

# --- STYLES ---
COMMON_BUTTON_STYLE = """
    QPushButton { background-color: #e65100; color: #ffffff; border-radius: 4px; padding: 8px; font-weight: bold; border: none; }
    QPushButton:hover { background-color: #ff9800; }
    QPushButton:pressed { background-color: #bf4500; }
    QPushButton:disabled { background-color: #555; color: #999; }
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
    QRadioButton, QCheckBox { color: #4caf50; font-weight: bold; }
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background-color: #252525; color: #4caf50; border: 1px solid #3d3d3d; border-radius: 4px; padding: 5px; }
    QTreeView { background-color: #1e1e1e; color: #ddd; border: 1px solid #333; }
    QScrollArea { border: none; background: #1e1e1e; }
    QScrollArea > QWidget > QWidget { background: #1e1e1e; }
    QListWidget { background-color: #1e1e1e; color: #ddd; border: 1px solid #333; }
    QListWidget::item { padding: 5px; }
    QListWidget::item:selected { background-color: #e65100; color: #ffffff; }
    QListWidget QLineEdit { padding: 1px 3px; color: #ffffff; background-color: #333333; border: 1px solid #ff9800; }
    QProgressBar { border: 1px solid #333; border-radius: 3px; text-align: center; color: #fff; background: #252525; }
    QProgressBar::chunk { background-color: #e65100; }
"""

LIGHT_STYLE = COMMON_BUTTON_STYLE + """
    QScrollArea { border: none; }
"""

# --- INTERFACE COMPONENTS ---
class SortItem(QTableWidgetItem):
    """Cella ordinabile per valore (numeri, durate, dimensioni) e non per testo."""
    def __lt__(self, other):
        a = self.data(Qt.UserRole + 1); b = other.data(Qt.UserRole + 1)
        if a is not None and b is not None:
            try: return a < b
            except TypeError: pass
        return self.text().lower() < other.text().lower()


class DragButton(QPushButton):
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setAcceptDrops(False); self.main_window = None
        self._press = None

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton: self._press = e.position().toPoint()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        # FIX: il drag partiva ad ogni minimo movimento; ora serve la distanza di drag di sistema
        if (e.buttons() & Qt.LeftButton) and self.main_window and self._press is not None:
            if (e.position().toPoint() - self._press).manhattanLength() >= QApplication.startDragDistance():
                self._press = None
                self.main_window.start_drag_operation()
                self.setDown(False)
                return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._press = None
        super().mouseReleaseEvent(e)


class ClipList(QListWidget):
    """Elenco dei clip: trascinabile verso altri programmi (es. Audacity), Canc per eliminare."""
    def __init__(self, main, editable=True):
        super().__init__()
        self.main = main
        self.editable = editable      # False: copia di consultazione (pannello del Multitrack)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection if editable else QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed if editable
                             else QAbstractItemView.NoEditTriggers)
        self.setContextMenuPolicy(Qt.CustomContextMenu)

    def startDrag(self, actions):
        it = self.currentItem()
        if it is not None: self.main.drag_clip(it)

    def keyPressEvent(self, e):
        if self.editable and e.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.state() != QAbstractItemView.EditingState:
            self.main.delete_selected_clips(); return
        super().keyPressEvent(e)


class WaveformWidget(QWidget):
    """Forma d'onda leggera (QPainter + cache) con playhead e click/drag per spostarsi."""
    seekRequested = Signal(float)

    def __init__(self, height=48, ruler=False, parent=None):
        super().__init__(parent)
        self.ruler = ruler
        self.mins = None; self.maxs = None
        self.duration = 0.0
        self.pos_frac = 0.0
        self.dim = False
        self.grid_sec = 0.0
        self.bg = QColor('#1e1e1e'); self.fg = QColor('#ff9800')
        self.cursor_c = QColor('white'); self.grid = QColor('#444'); self.text_c = QColor('#aaa')
        self._cache = None
        self.setMinimumHeight(height)
        if ruler:
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        else:
            self.setFixedHeight(height)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_theme(self, bg, fg, cursor, grid, text):
        self.bg, self.fg, self.cursor_c = QColor(bg), QColor(fg), QColor(cursor)
        self.grid, self.text_c = QColor(grid), QColor(text)
        self._cache = None; self.update()

    def set_peaks(self, mins, maxs, duration=None):
        self.mins, self.maxs = mins, maxs
        if duration is not None: self.duration = duration
        self._cache = None; self.update()

    def set_grid(self, beat_sec):
        """Griglia ritmica (battiti + battuta da 4). 0 = nascosta."""
        beat_sec = float(beat_sec or 0.0)
        if abs(beat_sec - self.grid_sec) > 1e-9: self.grid_sec = beat_sec; self._cache = None; self.update()

    def set_dim(self, dim):
        if dim != self.dim: self.dim = dim; self._cache = None; self.update()

    def set_playhead(self, frac):
        frac = min(1.0, max(0.0, frac))
        if abs(frac - self.pos_frac) * max(1, self.width()) >= 0.5:
            self.pos_frac = frac; self.update()

    def resizeEvent(self, e):
        self._cache = None
        super().resizeEvent(e)

    def _rebuild(self):
        w, h = max(1, self.width()), max(1, self.height())
        pm = QPixmap(w, h); pm.fill(self.bg)
        p = QPainter(pm)
        top = 14 if self.ruler else 0
        mid = top + (h - top) / 2.0
        amp = (h - top) / 2.0 - 2
        p.setPen(QPen(self.grid)); p.drawLine(0, int(mid), w, int(mid))
        if self.mins is not None and len(self.mins) > 0:
            N = len(self.mins)
            idx = (np.arange(w) * N // w).astype(np.intp)
            mn = np.minimum.reduceat(self.mins, idx)
            mx = np.maximum.reduceat(self.maxs, idx)
            col = QColor(self.fg)
            if self.dim: col.setAlpha(60)
            p.setPen(QPen(col))
            for x in range(w):
                y1 = mid - float(mx[x]) * amp
                y2 = mid - float(mn[x]) * amp
                if y2 - y1 < 1: y2 = y1 + 1
                p.drawLine(x, int(y1), x, int(y2))
        if self.grid_sec > 0 and self.duration > 0 and self.grid_sec / self.duration * w >= 5:
            beat_col = QColor(self.grid); beat_col.setAlpha(110)
            bar_col = QColor(self.fg); bar_col.setAlpha(120)
            i = 0
            while i * self.grid_sec <= self.duration:
                x = int(i * self.grid_sec / self.duration * w)
                p.setPen(QPen(bar_col if i % 4 == 0 else beat_col)); p.drawLine(x, top, x, h)
                i += 1
        if self.ruler and self.duration > 0:
            p.setFont(QFont("sans", 8))
            step = next((s for s in (1, 2, 5, 10, 15, 30, 60, 120, 300) if s * w / self.duration >= 70), 600)
            t = 0
            while t <= self.duration:
                x = int(t / self.duration * w)
                p.setPen(QPen(self.grid)); p.drawLine(x, 0, x, h)
                p.setPen(QPen(self.text_c)); p.drawText(x + 3, 11, fmt_time(t))
                t += step
        p.end()
        self._cache = pm

    def paintEvent(self, e):
        if self._cache is None or self._cache.size() != self.size(): self._rebuild()
        p = QPainter(self)
        p.drawPixmap(0, 0, self._cache)
        x = int(self.pos_frac * self.width())
        p.setPen(QPen(self.cursor_c, 2)); p.drawLine(x, 0, x, self.height())
        p.end()

    def _seek(self, e):
        self.seekRequested.emit(min(1.0, max(0.0, e.position().x() / max(1, self.width()))))
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton: self._seek(e)
    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton: self._seek(e)


LANE_COLORS = ['#ff9800', '#4caf50', '#29b6f6', '#ec407a', '#ab47bc', '#ffee58', '#26a69a', '#ef5350']

class LiveLane(QFrame):
    """Una corsia (layer) del Live Mix: mute / solo / volume / re-roll + forma d'onda."""
    changed = Signal()      # fader volume: sempre immediato
    toggled = Signal()      # mute / solo: puo' essere quantizzato a tempo
    reroll = Signal(int)
    seekRequested = Signal(float)

    def __init__(self, idx, color):
        super().__init__()
        self.idx = idx; self.color = color
        h = QHBoxLayout(self); h.setContentsMargins(2, 1, 2, 1); h.setSpacing(4)
        lbl = QLabel(f"L{idx + 1}"); lbl.setFixedWidth(26)
        lbl.setStyleSheet(f"color:{color}; font-weight:bold;")
        self.btn_m = QPushButton("M"); self.btn_s = QPushButton("S")
        for b, on in ((self.btn_m, "#c62828"), (self.btn_s, "#f9a825")):
            b.setCheckable(True); b.setFixedSize(26, 26)
            b.setStyleSheet(f"QPushButton{{padding:0;background:#333;color:#ddd;border-radius:3px;}}"
                            f"QPushButton:checked{{background:{on};color:#000;}}")
            b.toggled.connect(lambda *_: self.toggled.emit())
        self.gain = QSlider(Qt.Horizontal); self.gain.setRange(0, 150); self.gain.setValue(100)
        self.gain.setFixedWidth(90); self.gain.setToolTip("Volume layer")
        self.gain.valueChanged.connect(self._on_change)
        self.btn_r = QPushButton("🎲"); self.btn_r.setFixedSize(30, 26)
        self.btn_r.setStyleSheet("QPushButton{padding:0;} QPushButton:disabled{background:#555;}")
        self.btn_r.setToolTip("Rigenera solo questo layer")
        self.btn_r.clicked.connect(lambda: self.reroll.emit(self.idx))
        self.wave = WaveformWidget(48)
        self.wave.seekRequested.connect(self.seekRequested)
        for w in (lbl, self.btn_m, self.btn_s, self.gain, self.btn_r): h.addWidget(w)
        h.addWidget(self.wave, 1)

    def _on_change(self, *_):
        self.changed.emit()
    def user_gain(self): return self.gain.value() / 100.0
    def muted(self): return self.btn_m.isChecked()
    def solo(self): return self.btn_s.isChecked()

    def set_theme(self, bg, cursor, grid, text):
        self.wave.set_theme(bg, self.color, cursor, grid, text)


def compute_eff(user, mute, solo):
    """Guadagno effettivo di ogni layer (0 se muto o escluso da un solo)."""
    L = len(user); base = 1.0 / math.sqrt(max(1, L)); any_solo = any(solo)
    return [base * user[i] if ((not mute[i]) and (solo[i] or not any_solo)) else 0.0 for i in range(L)]


class LiveEngine(QIODevice):
    """Mixer in tempo reale: somma i layer (già in RAM) blocco per blocco e li passa a QAudioSink.
    Loop sample-accurate, mute/solo/volume applicabili durante la riproduzione."""
    reachedEnd = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock = threading.Lock()
        self.layers = []; self._user = []; self._mute = []; self._solo = []; self._eff = []
        self._pending = []
        self.n = 0; self.pos = 0; self.loop = True; self._end_sent = False

    # --- stato del mix ---
    def _recompute(self):
        L = len(self.layers)
        if not (len(self._user) == len(self._mute) == len(self._solo) == L): self._eff = [0.0] * L; return
        self._eff = compute_eff(self._user, self._mute, self._solo)

    def load(self, layers, user, mute, solo):
        with self._lock:
            self.layers = list(layers)
            self.n = len(layers[0]) if layers else 0
            self.pos = self.pos % self.n if self.n else 0
            self._user, self._mute, self._solo = list(user), list(mute), list(solo)
            self._pending = []
            self._end_sent = False
            self._recompute()

    def replace(self, idx, arr):
        with self._lock: self._apply('replace', (idx, arr))

    def set_state(self, user, mute, solo):
        with self._lock:
            self._user, self._mute, self._solo = list(user), list(mute), list(solo)
            self._recompute()

    def set_gains(self, user):
        """Solo i volumi (immediati); mute/solo restano come sono, anche se ne e' in attesa un cambio."""
        with self._lock:
            self._user = list(user); self._recompute()

    def schedule(self, target_frame, kind, payload):
        """Esegue l'azione esattamente al frame indicato del flusso audio (sample-accurate).
        target_frame == n significa 'al giro di loop'."""
        with self._lock: self._pending.append((int(target_frame), kind, payload))

    def pending_count(self):
        with self._lock: return len(self._pending)

    def _apply(self, kind, payload):          # chiamare con il lock gia' preso
        if kind == 'replace':
            idx, arr = payload
            if 0 <= idx < len(self.layers) and len(arr) == self.n: self.layers[idx] = arr
        elif kind == 'state':
            mute, solo = payload
            if len(mute) == len(self.layers):
                self._mute, self._solo = list(mute), list(solo)
            self._recompute()

    def _apply_due(self, upto, include_end=False):
        if not self._pending: return
        keep = []
        for t, kind, payload in self._pending:
            if t <= upto and (t < self.n or include_end): self._apply(kind, payload)
            else: keep.append((t, kind, payload))
        self._pending = keep

    def eff_gains(self):
        with self._lock: return list(self._eff)

    def seek_frac(self, f):
        with self._lock:
            self.pos = min(max(0, int(f * self.n)), max(0, self.n - 1)); self._end_sent = False

    def reset(self):
        with self._lock: self.pos = 0; self._end_sent = False

    # --- QIODevice ---
    def isSequential(self): return True
    def bytesAvailable(self): return 65536 + super().bytesAvailable()
    def writeData(self, data): return 0

    def readData(self, maxlen):
        frames = min(int(maxlen) // 4, 16384)
        if frames <= 0: return b""
        acc = np.zeros((frames, 2), np.float32)
        ended = False
        with self._lock:
            n = self.n
            if n > 0 and self.layers:
                done = 0
                while done < frames:
                    self._apply_due(self.pos)          # azioni programmate che scadono ora
                    nxt = min((t for t, _, _ in self._pending if self.pos < t < n), default=n)
                    take = min(frames - done, n - self.pos, nxt - self.pos)   # il blocco si spezza sul punto di scadenza
                    if take > 0:
                        for lay, g in zip(self.layers, self._eff):
                            if g > 0: acc[done:done + take] += lay[self.pos:self.pos + take] * np.float32(g)
                        done += take; self.pos += take
                    if self.pos >= n:
                        self._apply_due(n, include_end=True)   # azioni "al giro di loop"
                        if self.loop: self.pos = 0
                        else:
                            self.pos = n; ended = not self._end_sent; self._end_sent = True; break
        if ended: self.reachedEnd.emit()
        return (np.tanh(acc) * 32767.0).astype(np.int16).tobytes()



# =====================================================================
#  MULTITRACCIA: motore di riproduzione + widget della linea del tempo
# =====================================================================
class MtEngine(QIODevice):
    """Riproduzione del multitraccia: mixa a blocchi i clip (mt_mix_block) e li passa a QAudioSink."""
    reachedEnd = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock = threading.Lock()
        self.items = []; self.length = 0; self.pos = 0; self.loop = False; self._end_sent = False

    def set_items(self, items, length):
        with self._lock:
            self.items = items; self.length = int(length)
            self.pos = min(self.pos, self.length)

    def seek(self, frame):
        with self._lock:
            self.pos = max(0, min(int(frame), self.length)); self._end_sent = False

    def isSequential(self): return True
    def bytesAvailable(self): return 65536 + super().bytesAvailable()
    def writeData(self, data): return 0

    def readData(self, maxlen):
        frames = min(int(maxlen) // 4, 16384)
        if frames <= 0: return b""
        out = np.zeros((frames, 2), np.float32)
        ended = False
        with self._lock:
            L = self.length
            if L > 0:
                done = 0
                while done < frames:
                    if self.pos >= L:
                        if self.loop: self.pos = 0
                        else: ended = not self._end_sent; self._end_sent = True; break
                    take = min(frames - done, L - self.pos)
                    out[done:done + take] = mt_mix_block(self.items, self.pos, take)
                    done += take; self.pos += take
        if ended: self.reachedEnd.emit()
        return (np.clip(out, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class RulerWidget(QWidget):
    """Righello del tempo (clic / trascinamento = sposta il cursore)."""
    def __init__(self, main):
        super().__init__()
        self.m = main
        self.setFixedHeight(26)

    def paintEvent(self, e):
        m = self.m; p = QPainter(self); th = m.mt_theme
        p.fillRect(self.rect(), QColor(th['ruler']))
        pps, x0 = m.mt_pps, m.mt_x0
        step = next((s for s in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600) if s * pps >= 70), 1200)
        p.setFont(QFont("sans", 8))
        t = math.floor(x0 / step) * step
        while (t - x0) * pps <= self.width():
            x = int((t - x0) * pps)
            p.setPen(QPen(QColor(th['grid']))); p.drawLine(x, 14, x, 26)
            p.setPen(QPen(QColor(th['text'])))
            p.drawText(x + 3, 11, (f"{int(t // 60)}:{t % 60:05.2f}" if step < 1 else fmt_time(t)))
            t += step
        xp = int((m.mt_playhead - x0) * pps)
        p.setPen(QPen(QColor(th['cursor']), 2)); p.drawLine(xp, 0, xp, 26)
        p.end()

    def _seek(self, e): self.m.mt_seek(max(0.0, e.position().x() / self.m.mt_pps + self.m.mt_x0))
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton: self._seek(e)
    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton: self._seek(e)


class MtTrackHeader(QFrame):
    """Intestazione di una traccia: nome, mute, solo, volume, pan."""
    def __init__(self, main, idx):
        super().__init__()
        self.m = main; self.idx = idx
        self.setFixedHeight(TimelineWidget.TH); self.setFixedWidth(200)
        tr = main.mt_project['tracks'][idx]
        g = QVBoxLayout(self); g.setContentsMargins(4, 3, 4, 3); g.setSpacing(2)
        r1 = QHBoxLayout()
        self.name = QLineEdit(tr['name']); self.name.setStyleSheet(f"color:{TRACK_COLORS[idx % len(TRACK_COLORS)]}; font-weight:bold; padding:1px 3px;")
        self.name.editingFinished.connect(lambda: self.m.mt_track_edit(self.idx, 'name', self.name.text().strip() or f"Track {self.idx + 1}"))
        self.btn_m = QPushButton("M"); self.btn_s = QPushButton("S")
        for b, on in ((self.btn_m, "#c62828"), (self.btn_s, "#f9a825")):
            b.setCheckable(True); b.setFixedSize(24, 22)
            b.setStyleSheet(f"QPushButton{{padding:0;background:#333;color:#ddd;border-radius:3px;}}QPushButton:checked{{background:{on};color:#000;}}")
        self.btn_m.setChecked(tr['mute']); self.btn_s.setChecked(tr['solo'])
        self.btn_m.toggled.connect(lambda v: self.m.mt_track_edit(self.idx, 'mute', bool(v)))
        self.btn_s.toggled.connect(lambda v: self.m.mt_track_edit(self.idx, 'solo', bool(v)))
        r1.addWidget(self.name, 1); r1.addWidget(self.btn_m); r1.addWidget(self.btn_s)
        g.addLayout(r1)
        r2 = QHBoxLayout(); r2.setSpacing(3)
        lv = QLabel("Vol"); lv.setStyleSheet("font-size:10px;font-weight:normal;")
        self.gain = QSlider(Qt.Horizontal); self.gain.setRange(0, 150); self.gain.setValue(int(tr['gain'] * 100))
        self.gain.valueChanged.connect(lambda v: self.m.mt_track_edit(self.idx, 'gain', v / 100.0))
        r2.addWidget(lv); r2.addWidget(self.gain, 1); g.addLayout(r2)
        r3 = QHBoxLayout(); r3.setSpacing(3)
        lp = QLabel("Pan"); lp.setStyleSheet("font-size:10px;font-weight:normal;")
        self.pan = QSlider(Qt.Horizontal); self.pan.setRange(-100, 100); self.pan.setValue(int(tr['pan'] * 100))
        self.pan.valueChanged.connect(lambda v: self.m.mt_track_edit(self.idx, 'pan', v / 100.0))
        self.pan.setToolTip("Doppio clic sul valore: centro")
        r3.addWidget(lp); r3.addWidget(self.pan, 1); g.addLayout(r3)

    def mousePressEvent(self, e):
        self.m.mt_select_track(self.idx); super().mousePressEvent(e)

    def set_selected(self, sel):
        self.setStyleSheet("MtTrackHeader{border:2px solid #ff9800;}" if sel else "MtTrackHeader{border:1px solid #777777;}")


class TimelineWidget(QWidget):
    TH = 84   # altezza di una traccia

    def __init__(self, main):
        super().__init__()
        self.m = main
        self.drag = None
        self.setAcceptDrops(True); self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setMinimumHeight(self.TH * 4)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

    # --- coordinate ---
    def x_of(self, t): return (t - self.m.mt_x0) * self.m.mt_pps
    def t_of(self, x): return x / self.m.mt_pps + self.m.mt_x0
    def clip_rect(self, ti, c):
        return QRectF(self.x_of(c['pos']), ti * self.TH + 4, max(1.0, c['len'] * self.m.mt_pps), self.TH - 8)

    # --- disegno ---
    def paintEvent(self, e):
        m = self.m; th = m.mt_theme; pps = m.mt_pps
        p = QPainter(self); W = self.width()
        p.fillRect(self.rect(), QColor(th['bg']))
        tracks = m.mt_project['tracks']
        for ti, tr in enumerate(tracks):
            y = ti * self.TH
            if ti == m.mt_sel_track: p.fillRect(0, y, W, self.TH, QColor(th['lane_sel']))
            elif ti % 2: p.fillRect(0, y, W, self.TH, QColor(th['lane_alt']))
            p.setPen(QPen(QColor(th['grid']))); p.drawLine(0, y + self.TH - 1, W, y + self.TH - 1)
        if m.mt_project.get('snap'):      # griglia ritmica
            beat = 60.0 / max(1, m.mt_project.get('bpm', 120))
            if beat * pps >= 8:
                i = max(0, int(m.mt_x0 / beat))
                while True:
                    x = (i * beat - m.mt_x0) * pps
                    if x > W: break
                    c = QColor(th['grid']); c.setAlpha(150 if i % 4 == 0 else 60)
                    p.setPen(QPen(c)); p.drawLine(int(x), 0, int(x), self.height()); i += 1
        for ti, tr in enumerate(tracks):
            col = QColor(TRACK_COLORS[ti % len(TRACK_COLORS)])
            dim = tr['mute']
            for c in tr['clips']:
                r = self.clip_rect(ti, c)
                if r.right() < 0 or r.left() > W: continue
                self._paint_clip(p, r, c, col, dim, c['id'] == m.mt_sel_clip, W)
        xp = int(self.x_of(m.mt_playhead))
        p.setPen(QPen(QColor(th['cursor']), 2)); p.drawLine(xp, 0, xp, self.height())
        p.end()

    def _paint_clip(self, p, r, c, col, dim, selected, W):
        m = self.m
        media = m.mt_media.get(media_key(c['path'], c['media_a'], c['media_b']))
        fill = QColor(col); fill.setAlpha(45 if dim else 80)
        p.fillRect(r, fill)
        if media is None:                       # file mancante: tratteggio rosso
            p.fillRect(r, QBrush(QColor(220, 50, 50, 140), Qt.BDiagPattern))
        else:
            x_a = max(0.0, r.left()); x_b = min(float(W), r.right())
            width = int(x_b - x_a)
            if width > 0:
                t0 = c['in'] + (x_a - r.left()) / m.mt_pps
                cols = media.columns(t0, width / m.mt_pps, width)
                if cols is not None:
                    mn, mx = cols
                    wc = QColor(col); wc.setAlpha(110 if dim else 230)
                    p.setPen(QPen(wc))
                    mid = r.center().y(); amp = (r.height() - 18) / 2.0
                    xa = int(x_a)
                    for i in range(width):
                        y1 = mid - float(mx[i]) * amp; y2 = mid - float(mn[i]) * amp
                        if y2 - y1 < 1: y2 = y1 + 1
                        p.drawLine(xa + i, int(y1), xa + i, int(y2))
        # dissolvenze
        fi = c['fade_in'] * m.mt_pps; fo = c['fade_out'] * m.mt_pps
        p.save(); p.setClipRect(r)
        p.setPen(QPen(QColor(255, 255, 255, 170), 1))
        if fi > 0: p.drawLine(QPointF(r.left(), r.bottom()), QPointF(r.left() + fi, r.top() + 6))
        if fo > 0: p.drawLine(QPointF(r.right() - fo, r.top() + 6), QPointF(r.right(), r.bottom()))
        p.setPen(QPen(QColor(m.mt_theme['text'])))
        p.setFont(QFont("sans", 8)); p.drawText(QPointF(r.left() + 4, r.top() + 12), c['name'])
        p.restore()
        p.setPen(QPen(QColor(255, 255, 255) if selected else col, 2 if selected else 1))
        p.setBrush(Qt.NoBrush); p.drawRect(r)
        if selected:                               # maniglie delle dissolvenze
            p.setBrush(QColor(255, 255, 255)); p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(r.left() + fi, r.top() + 6), 4, 4)
            p.drawEllipse(QPointF(r.right() - fo, r.top() + 6), 4, 4)

    # --- hit test ---
    def hit_test(self, pos):
        tracks = self.m.mt_project['tracks']
        ti = int(pos.y() // self.TH)
        if ti < 0 or ti >= len(tracks): return None
        for c in reversed(tracks[ti]['clips']):
            r = self.clip_rect(ti, c)
            if not r.contains(pos): continue
            pps = self.m.mt_pps
            top = pos.y() - r.top() <= 14
            if top and abs(pos.x() - (r.left() + c['fade_in'] * pps)) <= 7: return ('fade_in', ti, c)
            if top and abs(pos.x() - (r.right() - c['fade_out'] * pps)) <= 7: return ('fade_out', ti, c)
            if pos.x() - r.left() <= 6: return ('trim_l', ti, c)
            if r.right() - pos.x() <= 6: return ('trim_r', ti, c)
            return ('move', ti, c)
        return None

    # --- mouse ---
    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton: return
        m = self.m; pos = e.position()
        hit = self.hit_test(pos)
        if hit:
            kind, ti, c = hit
            m.mt_select_clip(c['id'], ti)
            m.mt_checkpoint()                      # un solo punto di annullamento per l'intero trascinamento
            self.drag = {'kind': kind, 'id': c['id'], 'x0': pos.x(), 'orig': dict(c), 'changed': False}
        else:
            m.mt_select_clip(None, max(0, min(int(pos.y() // self.TH), len(m.mt_project['tracks']) - 1)))
            m.mt_seek(max(0.0, self.t_of(pos.x())))
            self.drag = {'kind': 'seek'}
        self.update()

    def mouseMoveEvent(self, e):
        m = self.m; pos = e.position(); d = self.drag
        if d is None:
            hit = self.hit_test(pos)
            self.setCursor(Qt.SizeHorCursor if hit and hit[0] in ('trim_l', 'trim_r', 'fade_in', 'fade_out')
                           else (Qt.OpenHandCursor if hit else Qt.ArrowCursor))
            return
        if not (e.buttons() & Qt.LeftButton): return
        if d['kind'] == 'seek':
            m.mt_seek(max(0.0, self.t_of(pos.x()))); return
        found = m.mt_find_clip(d['id'])
        if not found: return
        ti, c = found; o = d['orig']; pps = m.mt_pps
        dt = (pos.x() - d['x0']) / pps
        tol = 8.0 / pps
        dur = m.mt_media_duration(c)
        if d['kind'] == 'move':
            newpos = max(0.0, o['pos'] + dt)
            snapped = m.mt_snap_time(newpos, c['id'], tol)
            if snapped is None:
                end_s = m.mt_snap_time(newpos + o['len'], c['id'], tol)
                if end_s is not None: snapped = end_s - o['len']
            if snapped is not None: newpos = max(0.0, snapped)
            c['pos'] = newpos
            nt = max(0, min(int(pos.y() // self.TH), len(m.mt_project['tracks']) - 1))
            if nt != ti:
                m.mt_project['tracks'][ti]['clips'].remove(c)
                m.mt_project['tracks'][nt]['clips'].append(c); m.mt_sel_track = nt
        elif d['kind'] == 'trim_l':
            newpos = o['pos'] + dt
            sn = m.mt_snap_time(newpos, c['id'], tol)
            if sn is not None: newpos = sn
            dd = newpos - o['pos']
            dd = max(-o['in'], -o['pos'], min(dd, o['len'] - 0.05))
            c['in'] = o['in'] + dd; c['pos'] = o['pos'] + dd; c['len'] = o['len'] - dd
        elif d['kind'] == 'trim_r':
            end = o['pos'] + o['len'] + dt
            sn = m.mt_snap_time(end, c['id'], tol)
            if sn is not None: end = sn
            c['len'] = max(0.05, min(end - o['pos'], dur - o['in']))
        elif d['kind'] == 'fade_in':
            c['fade_in'] = max(0.0, min((pos.x() - self.x_of(c['pos'])) / pps, c['len'] - c['fade_out']))
        elif d['kind'] == 'fade_out':
            c['fade_out'] = max(0.0, min((self.x_of(c['pos'] + c['len']) - pos.x()) / pps, c['len'] - c['fade_in']))
        c['fade_in'] = min(c['fade_in'], c['len']); c['fade_out'] = min(c['fade_out'], max(0.0, c['len'] - c['fade_in']))
        d['changed'] = True
        m.mt_changed()

    def mouseReleaseEvent(self, e):
        d = self.drag; self.drag = None
        if d and d.get('kind') != 'seek':
            self.m.mt_changed(); self.m.mt_update_props()
        self.update()

    def mouseDoubleClickEvent(self, e):
        hit = self.hit_test(e.position())
        if hit: self.m.mt_rename_clip(hit[2]['id'])

    def wheelEvent(self, e):
        dy = e.angleDelta().y()
        if e.modifiers() & Qt.ShiftModifier:
            self.m.mt_scroll_by(-dy / 120.0 * 60.0 / self.m.mt_pps * 5)
        else:
            self.m.mt_zoom_at(1.18 ** (dy / 120.0), e.position().x())
        e.accept()

    def contextMenuEvent(self, e):
        hit = self.hit_test(e.pos())
        if not hit: return
        m = self.m; c = hit[2]; m.mt_select_clip(c['id'], hit[1])
        menu = QMenu()
        for label, fn in (("✂ Split at cursor (S)", m.mt_split), ("⧉ Duplicate (Ctrl+D)", m.mt_duplicate),
                          ("✏️ Rename", lambda: m.mt_rename_clip(c['id'])), ("🗑 Delete (Del)", m.mt_delete)):
            a = QAction(label, self); a.triggered.connect(fn); menu.addAction(a)
        menu.exec(e.globalPos())

    # --- drop dal Clip Bin / dal file manager ---
    def dragEnterEvent(self, e):
        md = e.mimeData()
        if md.hasFormat(CLIP_MIME) or md.hasUrls(): e.acceptProposedAction()
    def dragMoveEvent(self, e): e.acceptProposedAction()
    def dropEvent(self, e):
        md = e.mimeData(); pos = e.position()
        ti = max(0, min(int(pos.y() // self.TH), len(self.m.mt_project['tracks']) - 1))
        t = max(0.0, self.t_of(pos.x()))
        if md.hasFormat(CLIP_MIME):
            self.m.mt_add_bin_clip(int(bytes(md.data(CLIP_MIME)).decode()), ti, t); e.acceptProposedAction()
        elif md.hasUrls():
            for u in md.urls():
                if u.isLocalFile(): self.m.mt_add_file(u.toLocalFile(), ti, t)
            e.acceptProposedAction()


class RndSndApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"rndsnd v{APP_VERSION}")
        self.resize(1600, 950)
        self.init_db()

        # Defaults tema (servono prima di costruire le schede)
        self.canvas_bg, self.wf_color, self.cursor_color = '#1e1e1e', '#ff9800', 'white'
        self.current_theme = "Dark"

        # Audio Engine (Library Editor)
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)
        self.player.playbackStateChanged.connect(self.on_player_state)
        self.player.mediaStatusChanged.connect(self.on_media_status)
        self.play_once_end = None     # fine del clip in riproduzione singola
        self._pending_clip = None     # inizio del clip da suonare appena il file e' caricato
        self._loading_clips = False
        self.audio_data = None
        self.current_path = ""
        self.sr = 44100
        self.duration = 0.0
        self.selection_range = (0, 0)
        self.is_looping = False
        self.playhead_line = None
        self.wf_artist = None
        self.span = None
        self.current_browsing_path = ""
        self._tasks = []          # thread in esecuzione (riferimenti vivi)
        self.scan_thread = None
        self.mix_task = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_playhead_and_loop)

        # Main Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.setup_header()
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        self.mt_init_state()
        self.setup_explorer_tab()
        self.setup_mixer_tab()
        self.setup_live_tab()
        self.setup_multitrack_tab()

        QShortcut(QKeySequence("Ctrl+B"), self, activated=self.add_clip_from_selection)
        self.ai_task = None
        self.switch_theme("Dark")
        self.refresh_ai_status()

    def init_db(self):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS files (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, path TEXT UNIQUE, folder TEXT, tags TEXT, duration REAL, size INTEGER)")
        cols = {r[1] for r in cur.execute("PRAGMA table_info(files)")}
        for name, typ in (("folder", "TEXT"), ("tags", "TEXT"), ("duration", "REAL"), ("size", "INTEGER"),
                           ("bpm", "REAL"), ("tonality", "TEXT")):
            if name not in cols: cur.execute(f"ALTER TABLE files ADD COLUMN {name} {typ}")
        cur.execute("CREATE TABLE IF NOT EXISTS clips (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, path TEXT, "
                    "start REAL, end REAL, created TEXT)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_clips_path ON clips(path)")
        conn.commit()
        conn.close()

    def _start_task(self, task):
        """Tiene un riferimento al thread finché non finisce."""
        self._tasks.append(task)
        task.finished.connect(lambda t=task: self._tasks.remove(t) if t in self._tasks else None)
        task.start()

    def setup_header(self):
        header = QHBoxLayout()
        self.title_label = QLabel(f"<h2>rndsnd v{APP_VERSION}</h2>")
        header.addWidget(self.title_label)
        header.addStretch()
        self.ai_lbl = QLabel(""); header.addWidget(self.ai_lbl)
        self.ai_btn = QPushButton(""); self.ai_btn.clicked.connect(self.on_ai_button); header.addWidget(self.ai_btn)
        header.addSpacing(20)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.currentTextChanged.connect(self.switch_theme)
        header.addWidget(QLabel("THEME:"))
        header.addWidget(self.theme_combo)
        self.main_layout.addLayout(header)

    def switch_theme(self, theme):
        self.current_theme = theme
        if theme == "Dark":
            self.setStyleSheet(DARK_STYLE)
            self.canvas_bg, self.wf_color, self.cursor_color = '#1e1e1e', '#ff9800', 'white'
            text_color = "#39df0f"
        else:
            self.setStyleSheet(LIGHT_STYLE)
            self.canvas_bg, self.wf_color, self.cursor_color = 'white', '#ff9800', 'black'
            text_color = "black"

        if self.audio_data is not None: self.plot_waveform(keep_view=True)
        else: self.plot_waveform()

        for row in range(self.file_table.rowCount()):
            item = self.file_table.item(row, 0)
            if item: item.setForeground(QColor(text_color))
        self.live_apply_theme()
        self.mt_apply_theme()

    # =================================================================
    #  EXPLORER TAB
    # =================================================================
    def setup_explorer_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        main_splitter = QSplitter(Qt.Horizontal)

        # 1. LEFT: SYSTEM TREE
        tree_frame = QFrame()
        tree_layout = QVBoxLayout(tree_frame)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        lbl_tree = QLabel("📂 SYSTEM BROWSER")
        lbl_tree.setStyleSheet("padding: 5px; background: #333; color: white;")
        tree_layout.addWidget(lbl_tree)

        self.dir_model = QFileSystemModel()
        self.dir_model.setRootPath("")
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
        self.scan_info_lbl.setWordWrap(True)      # i messaggi lunghi (es. errori dell'IA) devono restare leggibili
        self.scan_progress = QProgressBar()
        self.scan_progress.setVisible(False)
        right_layout.addWidget(self.scan_info_lbl)
        right_layout.addWidget(self.scan_progress)

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("🔍 Search / Filter files...")
        self.search_bar.textChanged.connect(self.filter_file_table)
        right_layout.addWidget(self.search_bar)

        self._populating = False
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(6)
        self.file_table.setHorizontalHeaderLabels(["Filename", "Tags (double-click to edit)", "Duration", "Size", "BPM", "Key"])
        hdr = self.file_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Interactive)
        hdr.setStretchLastSection(True)
        hdr.setSortIndicatorShown(True)
        self.file_table.setColumnWidth(0, 250); self.file_table.setColumnWidth(1, 220)
        self.file_table.setColumnWidth(2, 70); self.file_table.setColumnWidth(3, 80); self.file_table.setColumnWidth(4, 60)
        self.file_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        # solo la colonna Tags e' modificabile (le altre celle non hanno il flag ItemIsEditable)
        self.file_table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.file_table.itemClicked.connect(self.load_selected_file)
        self.file_table.itemChanged.connect(self.on_tags_edited)
        self.file_table.setSortingEnabled(True)
        hdr.setSortIndicator(0, Qt.AscendingOrder)
        # il flag "riga nascosta" non segue le righe quando si riordina: si riapplica il filtro dopo l'ordinamento
        hdr.sortIndicatorChanged.connect(lambda *_: QTimer.singleShot(0, lambda: self.filter_file_table(self.search_bar.text())))
        right_layout.addWidget(self.file_table)

        self.figure, self.ax = plt.subplots(figsize=(10, 3))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.canvas.mpl_connect('scroll_event', self.on_scroll_zoom)
        right_layout.addWidget(self.canvas)

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

        for w in (self.btn_rew, self.btn_play, self.btn_stop, self.btn_fwd): t_lyt.addWidget(w)
        t_lyt.addStretch()
        t_lyt.addWidget(self.btn_drag)
        right_layout.addLayout(t_lyt)

        main_splitter.addWidget(right_frame)
        main_splitter.addWidget(self.build_clip_panel())
        main_splitter.setSizes([340, 900, 320])
        layout.addWidget(main_splitter)
        self.refresh_clips()
        self.tabs.addTab(tab, "Library Editor")

    def filter_file_table(self, text):
        text = text.lower()
        for row in range(self.file_table.rowCount()):
            item = self.file_table.item(row, 0)
            tags = self.file_table.item(row, 1)
            match = bool(item and tags and (text in item.text().lower() or text in tags.text().lower()))
            self.file_table.setRowHidden(row, not match)

    def editor_visible_paths(self, text="", bpm_min=0, bpm_max=0, keys=None):
        """Percorsi delle righe visibili nella tabella dell'Editor, opz. filtrate per testo, BPM e tonalita'."""
        text = text.lower(); out = []
        for row in range(self.file_table.rowCount()):
            if self.file_table.isRowHidden(row): continue
            it = self.file_table.item(row, 0)
            tg = self.file_table.item(row, 1)
            p = it.data(Qt.UserRole) if it else None
            if not p: continue
            if text and text not in it.text().lower() and not (tg and text in tg.text().lower()): continue
            if bpm_min > 0 or bpm_max > 0:
                bi = self.file_table.item(row, 4)
                b = bi.data(Qt.UserRole + 1) if bi else None
                if not b or b <= 0 or (bpm_min > 0 and b < bpm_min) or (bpm_max > 0 and b > bpm_max): continue
            if keys:
                ki = self.file_table.item(row, 5)
                if not ki or ki.text() not in keys: continue
            out.append(p)
        return out

    def on_tags_edited(self, item):
        """Salva nel database i tag modificati a mano (colonna 1)."""
        if self._populating or item.column() != 1: return
        cell = self.file_table.item(item.row(), 0)
        path = cell.data(Qt.UserRole) if cell else None
        if not path: return
        new = ", ".join(t.strip() for t in item.text().split(",") if t.strip()) or "Unknown"
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("UPDATE files SET tags = ? WHERE path = ?", (new, path))
            conn.commit(); conn.close()
            self.scan_info_lbl.setText(f"✏️ Tags saved for {os.path.basename(path)}: {new}")
        except Exception as e:
            self.scan_info_lbl.setText(f"❌ Could not save tags: {e}")

    def on_folder_clicked(self, index):
        path = os.path.normpath(self.dir_model.fileInfo(index).absoluteFilePath())
        self.current_browsing_path = path
        self.update_table_from_db(path)

    def open_tree_context_menu(self, position):
        indexes = self.tree.selectedIndexes()
        if len(indexes) > 0:
            path = os.path.normpath(self.dir_model.fileInfo(indexes[0]).absoluteFilePath())
            menu = QMenu()
            name = os.path.basename(path) or path
            scan_action = QAction(f"✨ Scan '{name}' with AI", self)
            scan_action.triggered.connect(lambda: self.start_scan(path))
            menu.addAction(scan_action)
            retag_action = QAction(f"🧠 Re-run AI on untagged files in '{name}'", self)
            retag_action.triggered.connect(lambda: self.start_scan(path, mode="retag"))
            menu.addAction(retag_action)
            bpm_action = QAction(f"🎼 Analyze BPM / Key in '{name}'", self)
            bpm_action.triggered.connect(lambda: self.start_scan(path, mode="bpm"))
            menu.addAction(bpm_action)
            menu.exec(self.tree.viewport().mapToGlobal(position))

    # --- stato e download dell'IA ---
    def refresh_ai_status(self):
        st = ai_status(); self.ai_state = st
        downloading = self.ai_task is not None
        if downloading:
            self.ai_lbl.setText("🧠 AI: downloading model..."); self.ai_btn.setText("✖ Cancel download"); self.ai_btn.setVisible(True)
        elif st == 'ready':
            self.ai_lbl.setText("🧠 AI: ready"); self.ai_btn.setVisible(False)
        elif st == 'need_model':
            self.ai_lbl.setText("🧠 AI: model not downloaded"); self.ai_btn.setText("⬇ Download AI model (~330 MB)"); self.ai_btn.setVisible(True)
        else:
            self.ai_lbl.setText("🧠 AI: not installed"); self.ai_btn.setText("How to enable the AI"); self.ai_btn.setVisible(True)
        # senza questo, con la finestra stretta etichetta e pulsante si schiacciano e il testo viene tagliato
        self.ai_lbl.setMinimumWidth(self.ai_lbl.fontMetrics().horizontalAdvance(self.ai_lbl.text()) + 24)
        self.ai_btn.setMinimumWidth(self.ai_btn.fontMetrics().horizontalAdvance(self.ai_btn.text()) + 40)

    AI_INSTALL_HELP = ("The AI tagging needs PyTorch and panns-inference, which are not installed in this Python.\n\n"
                       "In a terminal:\n    pip install torch panns-inference\n\n"
                       "(For a smaller CPU-only PyTorch:  pip install torch --index-url https://download.pytorch.org/whl/cpu )\n\n"
                       "Then restart rndsnd. The ~330 MB model is downloaded from inside the program with the button in the top bar. "
                       "Until then, files are scanned without AI and tagged \"No AI\".")

    def on_ai_button(self):
        if self.ai_task is not None:
            self.ai_task.cancel(); self.scan_info_lbl.setText("⏹ Cancelling download (the partial file is kept)..."); return
        if self.ai_state == 'need_model': self.ai_download()
        elif self.ai_state == 'missing_libs': QMessageBox.information(self, "Enable the AI", self.AI_INSTALL_HELP)

    def ai_download(self, then=None):
        """Scarica etichette e modello (~330 MB) dentro il programma, con avanzamento e ripresa. Niente wget."""
        if self.ai_task is not None: return

        def work(ctx):
            os.makedirs(PANNS_DIR, exist_ok=True)
            if not (os.path.isfile(PANNS_LABELS) and os.path.getsize(PANNS_LABELS) > 0):
                ctx.status.emit("⬇ Downloading AI labels...")
                download_file(PANNS_LABELS_URL, PANNS_LABELS, check=ctx.check, min_bytes=1000)
            if not (os.path.isfile(PANNS_CKPT) and os.path.getsize(PANNS_CKPT) >= PANNS_CKPT_MIN):
                def prog(done, total):
                    if total: ctx.progress.emit(int(done / total * 100))
                    ctx.status.emit(f"⬇ Downloading AI model: {done / 1e6:.0f}" + (f" / {total / 1e6:.0f} MB" if total else " MB"))
                download_file(PANNS_CKPT_URL, PANNS_CKPT, progress=prog, check=ctx.check, min_bytes=PANNS_CKPT_MIN)
            return True

        task = TaskThread(work)
        task.progress.connect(self.scan_progress.setValue); task.status.connect(self.scan_info_lbl.setText)
        task.done.connect(lambda _r: self._ai_download_finished(None, then))
        task.failed.connect(lambda msg: self._ai_download_finished(msg, None))
        self.ai_task = task
        self.scan_progress.setValue(0); self.scan_progress.setVisible(True)
        self.refresh_ai_status(); self._start_task(task)

    def _ai_download_finished(self, error, then):
        self.ai_task = None; self.scan_progress.setVisible(False); self.refresh_ai_status()
        if error == "cancelled":
            self.scan_info_lbl.setText("⏹ Download cancelled. Press the button again to resume where it stopped.")
        elif error:
            self.scan_info_lbl.setText(f"❌ AI model download failed: {error}")
        else:
            self.scan_info_lbl.setText("✅ AI model downloaded. It is loaded the first time you scan a folder.")
            if then: then()

    def _ask_model_choice(self):
        """Il modello manca: 'download' | 'noai' | 'cancel'."""
        box = QMessageBox(self); box.setWindowTitle("AI model not downloaded")
        box.setText("The AI model (~330 MB) has not been downloaded yet.\n\nDownload it now, or scan without AI tags (files are tagged \"No AI\" and can be re-analyzed later)?")
        b_dl = box.addButton("Download now", QMessageBox.AcceptRole)
        b_no = box.addButton("Scan without AI", QMessageBox.DestructiveRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.exec(); c = box.clickedButton()
        return 'download' if c is b_dl else ('noai' if c is b_no else 'cancel')

    def start_scan(self, folder_path, mode="ai"):
        if self.scan_thread is not None and self.scan_thread.isRunning():
            self.scan_info_lbl.setText("⏳ A scan is already running...")
            return
        use_ai = True
        if mode in ("ai", "retag"):
            st = ai_status()
            if st == 'need_model':
                choice = self._ask_model_choice() if mode == "ai" else 'download'
                if choice == 'cancel': return
                if choice == 'download':
                    if self.ai_task is not None:
                        self.scan_info_lbl.setText("⏳ The AI model is still downloading..."); return
                    self.ai_download(then=lambda: self.start_scan(folder_path, mode)); return
                use_ai = False
            elif st == 'missing_libs':
                if mode == "retag": self.scan_info_lbl.setText("⚠️ The AI is not installed: use the 'How to enable the AI' button."); return
                use_ai = False
        self.scan_progress.setVisible(True)
        self.scan_progress.setValue(0)
        label = {"bpm": "Analyzing BPM/Key", "retag": "Re-running AI"}.get(mode, "Scanning")
        self.scan_info_lbl.setText(f"{label}: {folder_path}...")
        self.scan_thread = (AnalyzeWorker(folder_path) if mode == "bpm"
                            else ScanWorker(folder_path, use_ai=use_ai, retag=(mode == "retag")))
        if mode == "ai" and not use_ai and ai_status() == 'missing_libs':
            self.scan_thread.ai_error = "PyTorch / panns-inference not installed"
        self.scan_thread.progress.connect(self.scan_progress.setValue)
        self.scan_thread.log.connect(self.scan_info_lbl.setText)
        self.scan_thread.scan_done.connect(lambda count: self.on_scan_completed(count, folder_path, mode))
        self.scan_thread.start()

    def on_scan_completed(self, count, path, mode="ai"):
        self.scan_progress.setVisible(False)
        err = getattr(self.scan_thread, 'ai_error', None)
        if mode == "bpm": self.scan_info_lbl.setText(f"✅ BPM/Key analysis complete. {count} files analyzed.")
        elif mode == "retag":
            self.scan_info_lbl.setText(f"✅ AI re-run complete: {count} files tagged." if not err
                                       else f"⚠️ AI could not run: {err}. Nothing was changed.")
        else:
            msg = f"✅ Scan Complete. {count} new files."
            if err and count: msg += f" ⚠️ AI unavailable ({err}): they are tagged \"No AI\" — fix it, then right-click the folder > Re-run AI."
            self.scan_info_lbl.setText(msg)
        if self.current_browsing_path == path:
            self.update_table_from_db(path)
        self.refresh_ai_status()

    def update_table_from_db(self, folder_path):
        prefix = folder_prefix(folder_path)   # "Music" non include "Music2" e % _ non sono jolly
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT filename, tags, duration, size, path, bpm, tonality FROM files WHERE substr(path, 1, ?) = ?",
                    (len(prefix), prefix))
        rows = cur.fetchall()
        conn.close()

        if len(rows) == 0:
            self.scan_info_lbl.setText("Folder not in DB. Right-click folder to SCAN.")
        else:
            self.scan_info_lbl.setText(f"Viewing: {os.path.basename(folder_path) or folder_path} ({len(rows)} scanned files)")

        text_color = "black" if getattr(self, "current_theme", "Dark") == "Light" else "#39df0f"
        hdr = self.file_table.horizontalHeader()
        sort_col, sort_ord = hdr.sortIndicatorSection(), hdr.sortIndicatorOrder()
        self._populating = True
        self.file_table.setSortingEnabled(False)       # senza questo le righe si spostano mentre si riempie
        self.file_table.setUpdatesEnabled(False)
        self.file_table.setRowCount(len(rows))
        ro = ~Qt.ItemIsEditable
        for row_idx, (name, tags, dur, size, path, bpm, ton) in enumerate(rows):
            dur = dur or 0.0; size = size or 0
            name = name or os.path.basename(path)
            bpm_txt = "" if bpm is None else (f"{bpm:.0f}" if bpm > 0 else "–")
            key_txt = "" if ton is None else ("–" if ton == "-" else ton)
            cells = [(name, name.lower()), (tags or "", (tags or "").lower()),
                     (f"{int(dur // 60)}:{int(dur % 60):02d}", dur), (f"{size / (1024 * 1024):.2f} MB", size),
                     (bpm_txt, bpm if bpm else 0.0), (key_txt, key_txt)]
            for col, (txt, sk) in enumerate(cells):
                it = SortItem(txt); it.setData(Qt.UserRole + 1, sk)
                if col != 1: it.setFlags(it.flags() & ro)
                if col == 0:
                    it.setForeground(QColor(text_color)); it.setData(Qt.UserRole, path)
                self.file_table.setItem(row_idx, col, it)
        self.file_table.setUpdatesEnabled(True)
        self.file_table.setSortingEnabled(True)
        self.file_table.sortItems(sort_col, sort_ord)
        self._populating = False
        self.filter_file_table(self.search_bar.text())

    # --- caricamento file nell'editor ---
    def load_selected_file(self, item):
        cell = self.file_table.item(item.row(), 0)
        path = cell.data(Qt.UserRole) if cell else None
        self.load_audio_path(path)

    def load_audio_path(self, path):
        """Carica un file nell'editor (forma d'onda + player). Ritorna True se riuscito."""
        if not path or not os.path.exists(path):
            self.scan_info_lbl.setText("⚠️ File not found on disk.")
            return False
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            try:
                # soundfile a blocchi: memoria = solo il mono (float32), niente picco doppio
                chunks = [blk.mean(axis=1).astype(np.float32)
                          for blk in sf.blocks(path, blocksize=1 << 16, dtype='float32', always_2d=True)]
                data = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
                sr = sf.info(path).samplerate
            except Exception:
                data, sr = librosa.load(path, sr=None, mono=True)   # fallback (mp3/m4a ecc.)
                data = data.astype(np.float32)
            if len(data) == 0: raise RuntimeError("file vuoto")
        except Exception as e:
            self.scan_info_lbl.setText(f"❌ Cannot load '{os.path.basename(path)}': {e}")
            return False
        finally:
            QApplication.restoreOverrideCursor()

        self.player.stop()
        self.play_once_end = None; self._pending_clip = None
        self.audio_data, self.sr, self.current_path = data, sr, path
        self.duration = len(data) / sr
        self.player.setSource(QUrl.fromLocalFile(path))
        self.is_looping = False
        self.selection_range = (0, 0)
        self.btn_drag.setText("📦 DRAG")
        self.plot_waveform()
        if hasattr(self, 'clip_only_current') and self.clip_only_current.isChecked(): self.refresh_clips()
        return True

    def plot_waveform(self, keep_view=False):
        xl = self.ax.get_xlim() if (keep_view and self.audio_data is not None) else None
        if self.span is not None:
            try: self.span.disconnect_events()
            except Exception: pass
        self.ax.clear()
        self.ax.set_facecolor(self.canvas_bg); self.figure.patch.set_facecolor(self.canvas_bg)
        self.wf_artist = None; self.playhead_line = None
        self.ax.axis('off')   # niente assi 0-1 quando non c'è ancora un file
        if self.audio_data is not None:
            self.ax.set_xlim(*(xl if xl else (0, self.duration)))
            self.ax.set_ylim(-1.1, 1.1)
            self.draw_envelope()
            self.playhead_line = self.ax.axvline(x=0, color=self.cursor_color, lw=2)
        # FIX: dopo ax.clear() il SpanSelector va ricreato, altrimenti la selezione si rompe
        self.span = SpanSelector(self.ax, self.on_select, 'horizontal', useblit=True, minspan=0.02,
                                 props=dict(alpha=0.4, facecolor='#ff9800'), interactive=True, drag_from_anywhere=True)
        if self.is_looping and self.selection_range[1] > self.selection_range[0]:
            try: self.span.extents = self.selection_range
            except Exception: pass
        self.canvas.draw()

    def draw_envelope(self):
        """FIX: prima si disegnava 1 campione ogni 100 (aliasing, picchi persi).
        Ora inviluppo min/max ricalcolato sulla parte visibile (anche durante lo zoom)."""
        if self.wf_artist is not None:
            try: self.wf_artist.remove()
            except Exception: pass
            self.wf_artist = None
        if self.audio_data is None: return
        x0, x1 = self.ax.get_xlim()
        i0 = max(0, int(x0 * self.sr)); i1 = min(len(self.audio_data), int(x1 * self.sr) + 1)
        seg = self.audio_data[i0:i1]
        if len(seg) < 2: return
        bins = 2000
        if len(seg) <= bins * 2:
            t = (i0 + np.arange(len(seg))) / self.sr
            self.wf_artist = self.ax.plot(t, seg, color=self.wf_color, lw=0.8)[0]
        else:
            step = len(seg) // bins; m = (len(seg) // step) * step
            r = seg[:m].reshape(-1, step)
            t = (i0 + np.arange(r.shape[0]) * step + step / 2) / self.sr
            self.wf_artist = self.ax.fill_between(t, r.min(axis=1), r.max(axis=1),
                                                  facecolor=self.wf_color, edgecolor=self.wf_color, linewidth=0.6)

    def on_scroll_zoom(self, event):
        if self.audio_data is None or event.inaxes != self.ax: return
        cur_xlim = self.ax.get_xlim(); scale = 1 / 1.2 if event.button == 'up' else 1.2
        center = event.xdata if event.xdata is not None else (cur_xlim[0] + cur_xlim[1]) / 2
        new_min = max(0, center - (center - cur_xlim[0]) * scale)
        new_max = min(self.duration, center + (cur_xlim[1] - center) * scale)
        if new_max - new_min < 0.002: return
        self.ax.set_xlim([new_min, new_max])
        self.draw_envelope()
        self.canvas.draw_idle()

    def on_select(self, xmin, xmax):
        self.selection_range = (xmin, xmax); self.is_looping = True; self.play_once_end = None
        self.btn_drag.setText(f"📦 {xmax - xmin:.1f}s")
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            pos_sec = self.player.position() / 1000.0
            if pos_sec < xmin or pos_sec > xmax: self.player.setPosition(int(xmin * 1000))

    def on_mouse_click(self, event):
        if event.inaxes != self.ax or self.audio_data is None or event.xdata is None: return
        if event.button == 1:
            self.play_once_end = None
            click_time = max(0, min(event.xdata, self.duration))
            self.player.setPosition(int(click_time * 1000))
            self.update_playhead_and_loop()

    def update_playhead_and_loop(self):
        pos_sec = self.player.position() / 1000.0
        if self.is_looping and self.player.playbackState() == QMediaPlayer.PlayingState:
            s, e = self.selection_range
            if pos_sec >= e: self.player.setPosition(int(s * 1000)); pos_sec = s
        if (self.play_once_end is not None and self.player.playbackState() == QMediaPlayer.PlayingState
                and pos_sec >= self.play_once_end):
            # clip suonato una volta sola: ferma a fine clip e torna all'inizio (un nuovo clic lo riascolta)
            self.player.pause(); self.play_once_end = None
            pos_sec = self.selection_range[0]; self.player.setPosition(int(pos_sec * 1000))
        if self.playhead_line:
            self.playhead_line.set_xdata([pos_sec])
            self.canvas.draw_idle()

    def on_player_state(self, state):
        # FIX: il testo del pulsante restava "PAUSE" a fine file / cambio file
        if state == QMediaPlayer.PlayingState:
            self.btn_play.setText("⏸ PAUSE"); self.timer.start(30)
        else:
            self.btn_play.setText("▶ PLAY"); self.timer.stop()

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.live_pause(); self.mt_pause()          # un solo audio alla volta
            self.player.play()

    def stop_audio(self):
        self.play_once_end = None; self._pending_clip = None
        self.player.stop()
        if self.playhead_line: self.playhead_line.set_xdata([0]); self.canvas.draw_idle()

    def seek_relative(self, ms):
        top = int(self.duration * 1000) if self.duration else 0
        self.player.setPosition(min(max(0, self.player.position() + ms), top))

    @staticmethod
    def safe_filename(name, default="clip"):
        name = "".join("_" if c in '<>:"/\\|?*' or ord(c) < 32 else c for c in name).strip(" .")
        return name[:60] or default

    def export_segment(self, path, s, e, base, dest=None):
        """Esporta [s, e] secondi dal file ORIGINALE (stereo, 24 bit) in un WAV. Ritorna il percorso."""
        try:
            sr = sf.info(path).samplerate
            chunk, _ = sf.read(path, start=int(s * sr), stop=int(e * sr), dtype='float32', always_2d=True)
        except Exception:
            y, sr = librosa.load(path, sr=None, mono=False, offset=s, duration=e - s)   # fallback (m4a ecc.)
            chunk = y.T if y.ndim > 1 else y[:, None]
        if len(chunk) == 0: raise RuntimeError("selezione vuota")
        if dest is None:
            # cartella unica per ogni drag: il file mantiene il nome pulito (es. "Rumore pioggia.wav")
            folder = os.path.join(DRAG_DIR, str(time.time_ns() // 1000000))
            os.makedirs(folder, exist_ok=True)
            dest = os.path.join(folder, self.safe_filename(base) + ".wav")
        sf.write(dest, chunk, sr, subtype='PCM_24')
        return dest

    def _do_drag(self, source_widget, file_path, clip_id=None):
        drag = QDrag(source_widget); mime = QMimeData(); mime.setUrls([QUrl.fromLocalFile(file_path)])
        if clip_id is not None: mime.setData(CLIP_MIME, str(clip_id).encode())   # per il drop interno sul Multitrack
        drag.setMimeData(mime); drag.setPixmap(QPixmap(32, 32)); drag.exec(Qt.CopyAction)

    def start_drag_operation(self):
        if self.audio_data is None or not self.current_path: return
        try:
            if not self.is_looping:
                tp = self.current_path       # nessuna selezione: si trascina il file ORIGINALE, senza ricodifiche
            else:
                s, e = self.selection_range
                if e - s < 0.01: return
                base = os.path.splitext(os.path.basename(self.current_path))[0][:40] + f"_{s:.1f}-{e:.1f}"
                tp = self.export_segment(self.current_path, s, e, base)
        except Exception as ex:
            self.scan_info_lbl.setText(f"❌ Drag export failed: {ex}")
            return
        self._do_drag(self.btn_drag, tp)

    # =================================================================
    #  CLIP BIN  (parti di file salvate con un nome, legate al file originale)
    # =================================================================
    def build_clip_panel(self):
        frame = QFrame(); lay = QVBoxLayout(frame); lay.setContentsMargins(0, 0, 0, 0)
        head = QLabel("🎞 CLIP BIN"); head.setStyleSheet("padding: 5px; background: #333; color: white;")
        lay.addWidget(head)
        self.clip_hint = QLabel("Select a region on the waveform, then ➕ (Ctrl+B).\nClick = play • double-click = rename\nDrag a clip into Audacity / a DAW.")
        self.clip_hint.setWordWrap(True); self.clip_hint.setStyleSheet("font-weight: normal; font-size: 11px;")
        lay.addWidget(self.clip_hint)
        self.clip_add_btn = QPushButton("➕ ADD SELECTION  (Ctrl+B)")
        self.clip_add_btn.clicked.connect(self.add_clip_from_selection)
        lay.addWidget(self.clip_add_btn)
        opts = QHBoxLayout()
        self.clip_loop_chk = QCheckBox("Loop on play")
        self.clip_only_current = QCheckBox("Current file only")
        self.clip_only_current.toggled.connect(lambda *_: self.refresh_clips())
        opts.addWidget(self.clip_loop_chk); opts.addWidget(self.clip_only_current)
        lay.addLayout(opts)
        self.clip_list = ClipList(self)
        self.clip_list.itemClicked.connect(self.on_clip_clicked)
        self.clip_list.itemChanged.connect(self.on_clip_renamed)
        self.clip_list.currentItemChanged.connect(lambda cur, _prev: self.update_clip_details(cur))
        self.clip_list.customContextMenuRequested.connect(self.open_clip_menu)
        lay.addWidget(self.clip_list, 1)
        self.clip_info = QLabel(""); self.clip_info.setWordWrap(True)
        self.clip_info.setStyleSheet("font-weight: normal; font-size: 11px;")
        lay.addWidget(self.clip_info)
        return frame

    @staticmethod
    def fmt_pos(sec):
        return f"{int(sec // 60)}:{sec % 60:04.1f}"

    def clip_msg(self, text):
        """Messaggio visibile sia nell'Editor sia nel Multitrack."""
        self.scan_info_lbl.setText(text)
        if hasattr(self, 'mt_status'): self.mt_status.setText(text)

    def _make_clip_item(self, cid, name, path, st, en, editable):
        it = QListWidgetItem(name)
        it.setData(Qt.UserRole, cid)
        if editable: it.setFlags(it.flags() | Qt.ItemIsEditable)
        exists = os.path.exists(path)
        it.setToolTip(f"{os.path.basename(path)}\n{self.fmt_pos(st)} – {self.fmt_pos(en)}  ({en - st:.1f} s)"
                      + ("" if exists else "\n⚠️ source file not found"))
        if not exists: it.setForeground(QColor('#888888'))
        return it

    def refresh_clips(self, select_id=None):
        if not hasattr(self, 'clip_list'): return
        conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
        cur.execute("SELECT id, name, path, start, end FROM clips ORDER BY id")
        all_rows = cur.fetchall(); conn.close()
        self.clips = {cid: (name, path, st, en) for cid, name, path, st, en in all_rows}   # sempre TUTTI i clip
        rows = all_rows
        if self.clip_only_current.isChecked():       # il filtro vale solo per l'elenco dell'Editor
            rows = sorted([r for r in all_rows if r[2] == self.current_path], key=lambda r: r[3])
        self._loading_clips = True
        self.clip_list.clear()
        sel_item = None
        for cid, name, path, st, en in rows:
            it = self._make_clip_item(cid, name, path, st, en, True)
            self.clip_list.addItem(it)
            if cid == select_id: sel_item = it
        if hasattr(self, 'mt_clip_list'):            # pannello del Multitrack: stessi clip, sola lettura
            self.mt_clip_list.clear()
            for cid, name, path, st, en in all_rows:
                self.mt_clip_list.addItem(self._make_clip_item(cid, name, path, st, en, False))
            self.mt_apply_clip_filter()
        self._loading_clips = False
        if sel_item is not None: self.clip_list.setCurrentItem(sel_item)
        else: self.update_clip_details(None)

    def update_clip_details(self, item):
        if item is None or not hasattr(self, 'clips'):
            self.clip_info.setText(f"{self.clip_list.count()} clips"); return
        c = self.clips.get(item.data(Qt.UserRole))
        if not c: return
        name, path, st, en = c
        self.clip_info.setText(f"📄 {os.path.basename(path)}\n⏱ {self.fmt_pos(st)} – {self.fmt_pos(en)}  ({en - st:.1f} s)"
                               + ("" if os.path.exists(path) else "\n⚠️ source file not found"))

    def add_clip_from_selection(self):
        s, e = self.selection_range
        if self.audio_data is None or not self.current_path or e - s < 0.02:
            self.scan_info_lbl.setText("ℹ️ Select a region on the waveform first (drag on it), then add it to the Clip Bin.")
            return
        base = os.path.splitext(os.path.basename(self.current_path))[0]
        name = f"{base} {self.fmt_pos(s)}-{self.fmt_pos(e)}"
        conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
        cur.execute("INSERT INTO clips (name, path, start, end, created) VALUES (?, ?, ?, ?, ?)",
                    (name, self.current_path, float(s), float(e), datetime.datetime.now().isoformat(timespec='seconds')))
        cid = cur.lastrowid; conn.commit(); conn.close()
        self.refresh_clips(select_id=cid)
        self.scan_info_lbl.setText(f"🎞 Clip added ({e - s:.1f}s). Type a name and press Enter.")
        it = self.clip_list.currentItem()
        if it is not None:
            self.clip_list.editItem(it)      # si puo' scrivere subito il nome, come per le etichette di Audacity

    def on_clip_renamed(self, item):
        if self._loading_clips: return
        cid = item.data(Qt.UserRole)
        old = self.clips.get(cid, ("", "", 0, 0))[0]
        new = item.text().strip()
        if not new:                           # nome vuoto non ammesso: si ripristina
            self._loading_clips = True; item.setText(old); self._loading_clips = False; return
        conn = sqlite3.connect(DB_PATH)
        conn.execute("UPDATE clips SET name = ? WHERE id = ?", (new, cid)); conn.commit(); conn.close()
        n, p, s, e = self.clips[cid]; self.clips[cid] = (new, p, s, e)
        if hasattr(self, 'mt_clip_list'):             # tiene allineato il pannello del Multitrack
            for i in range(self.mt_clip_list.count()):
                mi = self.mt_clip_list.item(i)
                if mi.data(Qt.UserRole) == cid: mi.setText(new)
        self.scan_info_lbl.setText(f"✏️ Clip renamed: {new}")

    def delete_selected_clips(self):
        items = self.clip_list.selectedItems()
        if not items: return
        conn = sqlite3.connect(DB_PATH)
        conn.executemany("DELETE FROM clips WHERE id = ?", [(i.data(Qt.UserRole),) for i in items])
        conn.commit(); conn.close()
        self.refresh_clips()
        self.scan_info_lbl.setText(f"🗑 {len(items)} clip(s) removed from the bin (audio files untouched).")

    def open_clip_menu(self, pos):
        item = self.clip_list.itemAt(pos)
        if item is None: return
        self.clip_list.setCurrentItem(item)
        menu = QMenu()
        a1 = QAction("✏️ Rename", self); a1.triggered.connect(lambda: self.clip_list.editItem(item)); menu.addAction(a1)
        a2 = QAction("💾 Export as WAV...", self); a2.triggered.connect(lambda: self.export_clip(item)); menu.addAction(a2)
        a3 = QAction("🗑 Remove from bin", self); a3.triggered.connect(self.delete_selected_clips); menu.addAction(a3)
        menu.exec(self.clip_list.viewport().mapToGlobal(pos))

    def export_clip(self, item, dest=None):
        name, path, s, e = self.clips[item.data(Qt.UserRole)]
        if not os.path.exists(path):
            self.clip_msg("⚠️ Source file not found."); return
        if dest is None:
            dest, _ = QFileDialog.getSaveFileName(self, "Export clip", os.path.join(OUTPUT_DIR, self.safe_filename(name) + ".wav"),
                                                  "WAV (*.wav)")
            if not dest: return
        try:
            self.export_segment(path, s, e, name, dest=dest)
            self.scan_info_lbl.setText(f"💾 Exported: {dest}")
        except Exception as ex:
            self.scan_info_lbl.setText(f"❌ Export failed: {ex}")

    def drag_clip(self, item):
        name, path, s, e = self.clips[item.data(Qt.UserRole)]
        if not os.path.exists(path):
            self.clip_msg("⚠️ Source file not found."); return
        try: tp = self.export_segment(path, s, e, name)
        except Exception as ex:
            self.clip_msg(f"❌ Drag export failed: {ex}"); return
        self._do_drag(item.listWidget() or self.clip_list, tp, clip_id=item.data(Qt.UserRole))

    # --- ascolto al clic ---
    def on_clip_clicked(self, item):
        cid = item.data(Qt.UserRole)
        if cid not in self.clips: return
        name, path, s, e = self.clips[cid]
        if not os.path.exists(path):
            self.scan_info_lbl.setText(f"⚠️ Source file not found: {path}"); return
        same = (self.current_path == path and self.audio_data is not None)
        if not same and not self.load_audio_path(path): return
        loop = self.clip_loop_chk.isChecked()
        self.selection_range = (s, e); self.is_looping = loop
        self.play_once_end = None if loop else e
        self.btn_drag.setText(f"📦 {e - s:.1f}s")
        xl = self.ax.get_xlim()
        if s < xl[0] or e > xl[1]:            # il clip e' fuori dalla parte zoomata: si torna alla vista intera
            self.ax.set_xlim(0, self.duration); self.draw_envelope()
        try: self.span.extents = (s, e)
        except Exception: pass
        self.canvas.draw_idle()
        self.live_pause(); self.mt_pause()    # un solo audio alla volta
        if same: self._begin_clip_playback(s)
        else: self._pending_clip = s          # il file e' appena stato caricato: si parte quando il player e' pronto
        self.scan_info_lbl.setText(f"▶ {name}")

    def _begin_clip_playback(self, s):
        self.player.setPosition(int(s * 1000)); self.player.play()

    def on_media_status(self, status):
        if self._pending_clip is None: return
        if status in (QMediaPlayer.LoadedMedia, QMediaPlayer.BufferedMedia):
            s = self._pending_clip; self._pending_clip = None
            self._begin_clip_playback(s)
        elif status == QMediaPlayer.InvalidMedia:
            self._pending_clip = None
            self.scan_info_lbl.setText("❌ The player cannot open this file.")

    # =================================================================
    #  MIXER TAB (offline, ora in un thread)
    # =================================================================
    def setup_mixer_tab(self):
        tab = QWidget()
        l = QVBoxLayout(tab)
        l.setAlignment(Qt.AlignCenter)

        l.addWidget(QLabel("<h2>Generative Mixer</h2>"))
        l.addWidget(QLabel("Create a chaotic layered mix from your database."))

        self.radio_tags = QRadioButton("Use filtered list (Library Editor table)")
        self.radio_tags.setChecked(True)
        self.radio_chaos = QRadioButton("Full Chaos (Use Entire Library)")

        h_params = QHBoxLayout()
        h_params.addWidget(QLabel("Total Duration (sec):"))
        self.spin_dur = QSpinBox(); self.spin_dur.setRange(5, 300); self.spin_dur.setValue(30)
        h_params.addWidget(self.spin_dur)
        h_params.addWidget(QLabel("Clip Size:"))
        self.combo_grain = QComboBox(); self.combo_grain.addItems([g[0] for g in GRAINS]); self.combo_grain.setCurrentIndex(2)
        h_params.addWidget(self.combo_grain)
        h_params.addWidget(QLabel("Layers:"))
        self.spin_layers = QSpinBox(); self.spin_layers.setRange(1, 20); self.spin_layers.setValue(4)
        h_params.addWidget(self.spin_layers)

        self.mix_btn = QPushButton("🚀 GENERATE MIX")
        self.mix_btn.setFixedSize(300, 60)
        self.mix_btn.clicked.connect(self.generate_mix)
        self.mix_prog = QProgressBar(); self.mix_prog.setVisible(False); self.mix_prog.setFixedWidth(500)
        self.status_lbl = QLabel(""); self.status_lbl.setAlignment(Qt.AlignCenter)
        self.btn_open_out = QPushButton("📂 Open output folder")
        self.btn_open_out.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(OUTPUT_DIR)))

        l.addWidget(self.radio_tags); l.addWidget(self.radio_chaos); l.addLayout(h_params)
        l.addWidget(self.mix_btn, alignment=Qt.AlignCenter)
        l.addWidget(self.mix_prog, alignment=Qt.AlignCenter)
        l.addWidget(self.status_lbl)
        l.addWidget(self.btn_open_out, alignment=Qt.AlignCenter)
        self.tabs.addTab(tab, "Mixer")

    def generate_mix(self):
        if self.mix_task is not None:          # il pulsante diventa ANNULLA durante la generazione
            self.mix_task.cancel(); self.status_lbl.setText("⏹ Cancelling..."); return

        if self.radio_tags.isChecked():
            src = self.editor_visible_paths()
        else:
            conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
            cur.execute("SELECT path FROM files"); src = [x[0] for x in cur.fetchall()]; conn.close()
        src = sorted(set(src))
        if not src:
            self.status_lbl.setText("❌ No files found (Check filter or DB)."); return

        n_frames = self.spin_dur.value() * MIX_SR
        L = self.spin_layers.value()
        grain_name, min_ms, max_ms = GRAINS[self.combo_grain.currentIndex()]
        dur_s = self.spin_dur.value()
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f"rndsnd_mix_{timestamp}"
        seed = int(time.time() * 1000) & 0xFFFFFFFF

        def work(ctx):
            rng = random.Random(seed)
            acc = np.zeros((n_frames, 2), np.float32)
            logs = []
            base = 1.0 / math.sqrt(L)
            for k in range(L):
                ctx.status.emit(f"⏳ Mixing layer {k + 1}/{L}...")
                layer, log = build_layer(src, n_frames, min_ms, max_ms, 2000 if L == 1 else 0, L > 1, ctx,
                                         "Track" if L == 1 else f"Layer {k + 1}", rng,
                                         prog=lambda f, k=k: ctx.progress.emit(int((k + f) / L * 100)))
                acc += layer * np.float32(base); logs += log
                del layer
            ctx.check()
            ctx.status.emit("💾 Exporting...")
            header = [f"Date: {timestamp}", f"Seed: {seed}",
                      f"Duration: {dur_s}s | Layers: {L} | Grain: {grain_name} | Sources: {len(src)}"]
            return export_mix(acc, fname, header, logs)

        task = TaskThread(work)
        task.progress.connect(self.mix_prog.setValue)
        task.status.connect(self.status_lbl.setText)
        task.done.connect(self._mix_done)
        task.failed.connect(self._mix_failed)
        self.mix_task = task
        self.mix_btn.setText("✖ CANCEL"); self.mix_prog.setValue(0); self.mix_prog.setVisible(True)
        self.status_lbl.setText("⏳ Mixing...")
        self._start_task(task)

    def _mix_finished_ui(self):
        self.mix_task = None; self.mix_btn.setText("🚀 GENERATE MIX"); self.mix_prog.setVisible(False)

    def _mix_done(self, res):
        path, note = res
        self._mix_finished_ui()
        self.status_lbl.setText(f"✅ Created: {os.path.basename(path)} + Log{note}\n{OUTPUT_DIR}")

    def _mix_failed(self, msg):
        self._mix_finished_ui()
        self.status_lbl.setText("⏹ Cancelled." if msg == "cancelled" else f"❌ {msg}")

    # =================================================================
    #  LIVE MIX TAB
    # =================================================================
    def setup_live_tab(self):
        self.live_engine = LiveEngine(self)
        self.live_engine.reachedEnd.connect(self.live_on_end)
        self.live_sink = None
        self.live_state = 'stopped'      # stopped | playing | paused
        self.live_task = None
        self.live_lanes = []; self.live_audio = []; self.live_peaks = []; self.live_logs = []
        self.live_n = 0; self.live_seed = 0; self._live_gen = 0
        self.live_timer = QTimer(); self.live_timer.timeout.connect(self.live_tick)

        tab = QWidget(); root = QVBoxLayout(tab)

        # riga 1: sorgenti + parametri
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Source:"))
        self.live_src_combo = QComboBox()
        self.live_src_combo.addItems(["Entire library", "Library Editor list (visible rows)"])
        r1.addWidget(self.live_src_combo)
        self.live_filter = QLineEdit(); self.live_filter.setPlaceholderText("tag / name filter (es. bird, drum...)")
        self.live_filter.setMinimumWidth(220)
        r1.addWidget(self.live_filter)
        r1.addWidget(QLabel("Duration (s):"))
        self.live_dur = QSpinBox(); self.live_dur.setRange(10, 180); self.live_dur.setValue(60)
        r1.addWidget(self.live_dur)
        r1.addWidget(QLabel("Clip Size:"))
        self.live_grain = QComboBox(); self.live_grain.addItems([g[0] for g in GRAINS]); self.live_grain.setCurrentIndex(2)
        r1.addWidget(self.live_grain)
        r1.addWidget(QLabel("Layers:"))
        self.live_nlayers = QSpinBox(); self.live_nlayers.setRange(1, 8); self.live_nlayers.setValue(4)
        r1.addWidget(self.live_nlayers)
        r1.addStretch()
        self.live_gen_btn = QPushButton("🎲 NEW MIX")
        self.live_gen_btn.setMinimumWidth(160)
        self.live_gen_btn.clicked.connect(self.live_generate)
        r1.addWidget(self.live_gen_btn)
        root.addLayout(r1)

        # riga 1b: filtri musicali (richiedono "Analyze BPM / Key" sulla cartella)
        r1b = QHBoxLayout()
        r1b.addWidget(QLabel("BPM from:"))
        self.live_bpm_min = QSpinBox(); self.live_bpm_min.setRange(0, 250); self.live_bpm_min.setSpecialValueText("any")
        r1b.addWidget(self.live_bpm_min)
        r1b.addWidget(QLabel("to:"))
        self.live_bpm_max = QSpinBox(); self.live_bpm_max.setRange(0, 250); self.live_bpm_max.setSpecialValueText("any")
        r1b.addWidget(self.live_bpm_max)
        r1b.addWidget(QLabel("Key:"))
        self.live_key = QComboBox(); self.live_key.addItems(["Any key"] + ALL_KEYS)
        r1b.addWidget(self.live_key)
        self.live_compat = QCheckBox("+ compatible keys (Camelot)")
        r1b.addWidget(self.live_compat)
        r1b.addSpacing(30)
        r1b.addWidget(QLabel("Grid BPM:"))
        self.live_bpm = QSpinBox(); self.live_bpm.setRange(40, 240); self.live_bpm.setValue(120)
        r1b.addWidget(self.live_bpm)
        r1b.addWidget(QLabel("Quantize changes:"))
        self.live_quant = QComboBox(); self.live_quant.addItems(["Off", "1 beat", "1 bar", "2 bars"])
        self.live_quant.setCurrentIndex(2)
        self.live_quant.setToolTip("Mute / solo e layer rigenerati scattano sul prossimo battito / battuta")
        r1b.addWidget(self.live_quant)
        self.live_snap = QCheckBox("Snap clips to grid")
        self.live_snap.setToolTip("Le clip generate durano multipli di battito: i tagli cadono sulla griglia")
        r1b.addWidget(self.live_snap)
        r1b.addStretch()
        root.addLayout(r1b)
        for sig in (self.live_bpm.valueChanged, self.live_quant.currentIndexChanged, self.live_snap.toggled):
            sig.connect(lambda *_: self.live_update_grid())

        # riga 2: trasporto
        r2 = QHBoxLayout()
        self.live_play_btn = QPushButton("▶ PLAY"); self.live_play_btn.clicked.connect(self.live_play_pause)
        self.live_stop_btn = QPushButton("⏹ STOP"); self.live_stop_btn.clicked.connect(self.live_stop)
        self.live_loop_chk = QCheckBox("Loop"); self.live_loop_chk.setChecked(True)
        self.live_loop_chk.toggled.connect(lambda v: setattr(self.live_engine, 'loop', v))
        self.live_auto_chk = QCheckBox("Auto-play on new mix"); self.live_auto_chk.setChecked(True)
        r2.addWidget(self.live_play_btn); r2.addWidget(self.live_stop_btn)
        r2.addWidget(self.live_loop_chk); r2.addWidget(self.live_auto_chk)
        r2.addSpacing(20)
        r2.addWidget(QLabel("Vol:"))
        self.live_vol = QSlider(Qt.Horizontal); self.live_vol.setRange(0, 100); self.live_vol.setValue(80)
        self.live_vol.setFixedWidth(140); self.live_vol.valueChanged.connect(self.live_volume)
        r2.addWidget(self.live_vol)
        self.live_time_lbl = QLabel("00:00 / 00:00"); r2.addWidget(self.live_time_lbl)
        r2.addStretch()
        self.live_save_btn = QPushButton("💾 SAVE MIX"); self.live_save_btn.clicked.connect(self.live_save)
        self.live_open_btn = QPushButton("📂")
        self.live_open_btn.setToolTip("Open output folder")
        self.live_open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(OUTPUT_DIR)))
        r2.addWidget(self.live_save_btn); r2.addWidget(self.live_open_btn)
        root.addLayout(r2)

        self.live_prog = QProgressBar(); self.live_prog.setVisible(False)
        self.live_status = QLabel("Press NEW MIX to roll a random mix. Click a waveform to jump; M = mute, S = solo, 🎲 = re-roll a single layer.")
        root.addWidget(self.live_prog); root.addWidget(self.live_status)

        # forma d'onda master + corsie
        self.live_master = WaveformWidget(130, ruler=True)
        self.live_master.seekRequested.connect(self.live_seek)
        root.addWidget(self.live_master, 2)

        self.live_lane_box = QWidget()
        self.live_lane_layout = QVBoxLayout(self.live_lane_box)
        self.live_lane_layout.setContentsMargins(0, 0, 0, 0); self.live_lane_layout.setSpacing(2)
        self.live_lane_layout.addStretch()
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(self.live_lane_box)
        root.addWidget(scroll, 3)

        self.tabs.addTab(tab, "Live Mix")
        self._live_set_buttons()
        self.live_update_grid()

    def live_apply_theme(self):
        if not hasattr(self, 'live_master'): return
        dark = self.current_theme == "Dark"
        bg = '#1e1e1e' if dark else 'white'
        cur = 'white' if dark else 'black'
        grid = '#444' if dark else '#ccc'
        txt = '#aaa' if dark else '#333'
        self.live_master.set_theme(bg, '#ff9800', cur, grid, txt)
        for lane in self.live_lanes: lane.set_theme(bg, cur, grid, txt)

    # --- sorgenti ---
    def live_filters(self):
        bmin, bmax = self.live_bpm_min.value(), self.live_bpm_max.value()
        if bmin and bmax and bmin > bmax: bmin, bmax = bmax, bmin
        key = self.live_key.currentText(); keys = None
        if key != "Any key": keys = compatible_keys(key) if self.live_compat.isChecked() else [key]
        return bmin, bmax, keys

    def live_collect_sources(self):
        txt = self.live_filter.text().strip().lower()
        bmin, bmax, keys = self.live_filters()
        if self.live_src_combo.currentIndex() == 1:
            return sorted(set(self.editor_visible_paths(txt, bmin, bmax, keys)))
        conds, args = [], []
        if txt:
            pat = f"%{esc_like(txt)}%"
            conds.append("(lower(filename) LIKE ? ESCAPE '\\' OR lower(tags) LIKE ? ESCAPE '\\')"); args += [pat, pat]
        if bmin or bmax:
            conds.append("bpm > 0")
            if bmin: conds.append("bpm >= ?"); args.append(bmin)
            if bmax: conds.append("bpm <= ?"); args.append(bmax)
        if keys:
            conds.append(f"tonality IN ({','.join('?' * len(keys))})"); args += keys
        conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
        cur.execute("SELECT path FROM files" + (" WHERE " + " AND ".join(conds) if conds else ""), args)
        out = sorted({r[0] for r in cur.fetchall()}); conn.close()
        return out

    # --- griglia / quantizzazione ---
    def live_beat_frames(self): return max(1, int(round(60.0 / self.live_bpm.value() * MIX_SR)))

    def live_quant_frames(self):
        return [0, 1, 4, 8][self.live_quant.currentIndex()] * self.live_beat_frames()

    def live_update_grid(self):
        show = self.live_quant.currentIndex() > 0 or self.live_snap.isChecked()
        self.live_master.set_grid(60.0 / self.live_bpm.value() if show else 0.0)

    def live_next_grid(self, unit):
        """Prossimo punto della griglia (in frame del flusso audio); n = al giro di loop."""
        n = self.live_engine.n
        t = (self.live_engine.pos // unit + 1) * unit
        return min(t, n)

    # --- generazione ---
    def _live_set_buttons(self):
        busy = self.live_task is not None
        has = self.live_engine.n > 0
        self.live_gen_btn.setText("✖ CANCEL" if busy else "🎲 NEW MIX")
        self.live_play_btn.setEnabled(has); self.live_stop_btn.setEnabled(has)
        self.live_save_btn.setEnabled(has and not busy)
        for lane in self.live_lanes: lane.btn_r.setEnabled(not busy)

    def live_generate(self):
        if self.live_task is not None:
            self.live_task.cancel(); self.live_status.setText("⏹ Cancelling..."); return
        self._live_run(None)

    def live_reroll(self, idx):
        if self.live_task is None: self._live_run([idx])

    def _live_run(self, indices):
        """indices=None -> nuovo mix completo; altrimenti rigenera solo quei layer."""
        src = self.live_collect_sources()
        if not src:
            bmin, bmax, keys = self.live_filters()
            hint = " BPM/Key filters need files analyzed: right-click a folder > Analyze BPM / Key." if (bmin or bmax or keys) else ""
            self.live_status.setText("❌ No files match (check source / filters / scan some folders first)." + hint); return
        new_mix = indices is None
        grid = self.live_beat_frames() if self.live_snap.isChecked() else 0
        if new_mix:
            n_frames = self.live_dur.value() * MIX_SR; L = self.live_nlayers.value(); indices = list(range(L))
            if grid:   # la durata diventa un numero intero di battute, cosi' il loop resta a tempo
                bar = grid * 4; n_frames = max(bar, int(round(n_frames / bar)) * bar)
        else:
            n_frames = self.live_n; L = len(self.live_lanes)
        _, min_ms, max_ms = GRAINS[self.live_grain.currentIndex()]
        xf = 2000 if L == 1 else 0
        seed = int(time.time() * 1000) & 0xFFFFFFFF

        def work(ctx):
            rng = random.Random(seed); out = {}
            for k, idx in enumerate(indices):
                ctx.status.emit(f"🎲 Rolling layer {idx + 1}/{L} from {len(src)} files...")
                audio, log = build_layer(src, n_frames, min_ms, max_ms, xf, L > 1, ctx, f"Layer {idx + 1}", rng,
                                         prog=lambda f, k=k: ctx.progress.emit(int((k + f) / len(indices) * 100)),
                                         grid_frames=grid)
                out[idx] = (audio, compute_peaks(audio), log)
            return {'layers': out, 'n_frames': n_frames, 'L': L, 'seed': seed, 'new_mix': new_mix, 'nsrc': len(src)}

        task = TaskThread(work)
        task.progress.connect(self.live_prog.setValue)
        task.status.connect(self.live_status.setText)
        task.done.connect(self._live_on_done)
        task.failed.connect(self._live_on_failed)
        self.live_task = task
        self.live_prog.setValue(0); self.live_prog.setVisible(True)
        self._live_set_buttons()
        self._start_task(task)

    def _live_on_failed(self, msg):
        self.live_task = None; self.live_prog.setVisible(False)
        self.live_status.setText("⏹ Cancelled." if msg == "cancelled" else f"❌ {msg}")
        self._live_set_buttons()

    def _live_on_done(self, res):
        self.live_task = None; self.live_prog.setVisible(False)
        layers = res['layers']; note = ""
        if res['new_mix']:
            L = res['L']; self._live_gen += 1
            self.live_n = res['n_frames']; self.live_seed = res['seed']
            self.live_audio = [layers[i][0] for i in range(L)]
            self.live_peaks = [layers[i][1] for i in range(L)]
            self.live_logs = [layers[i][2] for i in range(L)]
            self._live_rebuild_lanes(L)
            self.live_engine.load(self.live_audio, [1.0] * L, [False] * L, [False] * L)
            self.live_engine.loop = self.live_loop_chk.isChecked()
        else:
            gen = self._live_gen
            q = self.live_quant_frames()
            when = None
            if q and self.live_state == 'playing': when = self.live_next_grid(q)
            for idx, (audio, pk, lg) in layers.items():
                self.live_audio[idx] = audio; self.live_logs[idx] = lg
                def show(idx=idx, pk=pk, gen=gen):
                    if gen != self._live_gen or idx >= len(self.live_lanes): return
                    self.live_peaks[idx] = pk
                    self.live_lanes[idx].wave.set_peaks(pk[:, 0], pk[:, 1], self.live_n / MIX_SR)
                    self.live_lanes[idx].setToolTip(f"{len(lg)} clips")
                    self.live_refresh_display()
                if when is None:
                    self.live_engine.replace(idx, audio); show()
                else:   # il layer nuovo entra sul prossimo battito / battuta, sample-accurate
                    self.live_engine.schedule(when, 'replace', (idx, audio))
                    ms = max(0, int((when - self.live_engine.pos) / MIX_SR * 1000))
                    QTimer.singleShot(ms, show)
                    note = f" — enters on the next {self.live_quant.currentText()} ({ms / 1000:.1f}s)"
        if res['new_mix']: self.live_mix_changed_all()
        else: self.live_refresh_display()   # non tocca mute/solo ancora in attesa di quantizzazione
        self.live_time_lbl.setText(f"00:00 / {fmt_time(self.live_n / MIX_SR)}")
        self.live_status.setText(f"✅ {'New mix' if res['new_mix'] else 'Layer re-rolled'} ready from {res.get('nsrc', '?')} files "
                                 f"({len(self.live_lanes)} layers, {self.live_n / MIX_SR:.1f}s, seed {res['seed']}){note}.")
        self._live_set_buttons()
        if res['new_mix'] and self.live_auto_chk.isChecked():
            if self.live_state != 'playing':
                self.live_stop(); self.live_play_pause()

    def _live_rebuild_lanes(self, L):
        for lane in self.live_lanes:
            self.live_lane_layout.removeWidget(lane); lane.deleteLater()
        self.live_lanes = []
        dur = self.live_n / MIX_SR
        for i in range(L):
            lane = LiveLane(i, LANE_COLORS[i % len(LANE_COLORS)])
            lane.changed.connect(self.live_gain_changed)
            lane.toggled.connect(self.live_toggle_changed)
            lane.reroll.connect(self.live_reroll)
            lane.seekRequested.connect(self.live_seek)
            pk = self.live_peaks[i]
            lane.wave.set_peaks(pk[:, 0], pk[:, 1], dur)
            lane.setToolTip(f"{len(self.live_logs[i])} clips")
            self.live_lane_layout.insertWidget(i, lane)
            self.live_lanes.append(lane)
        self.live_master.set_peaks(None, None, dur)
        self.live_apply_theme()

    def _live_ui_state(self):
        return ([l.user_gain() for l in self.live_lanes], [l.muted() for l in self.live_lanes],
                [l.solo() for l in self.live_lanes])

    def live_refresh_display(self):
        """Inviluppo master (stimato dai picchi dei layer, istantaneo) e corsie attenuate."""
        if not self.live_lanes: return
        eff = compute_eff(*self._live_ui_state())
        mn = sum(g * pk[:, 0] for g, pk in zip(eff, self.live_peaks))
        mx = sum(g * pk[:, 1] for g, pk in zip(eff, self.live_peaks))
        self.live_master.set_peaks(np.tanh(mn), np.tanh(mx), self.live_n / MIX_SR)
        for g, lane in zip(eff, self.live_lanes): lane.wave.set_dim(g == 0)

    def live_mix_changed_all(self):
        """Applica subito l'intero stato (dopo un nuovo mix / a riproduzione ferma)."""
        if not self.live_lanes: return
        self.live_engine.set_state(*self._live_ui_state())
        self.live_refresh_display()

    def live_gain_changed(self):            # fader: sempre immediato
        if not self.live_lanes: return
        self.live_engine.set_gains(self._live_ui_state()[0])
        self.live_refresh_display()

    def live_toggle_changed(self):          # mute / solo: sul prossimo battito se "Quantize" e' attivo
        if not self.live_lanes: return
        user, mute, solo = self._live_ui_state()
        q = self.live_quant_frames()
        if q and self.live_state == 'playing':
            self.live_engine.schedule(self.live_next_grid(q), 'state', (mute, solo))
        else:
            self.live_engine.set_state(user, mute, solo)
        self.live_refresh_display()

    # --- trasporto ---
    def _live_ensure_sink(self):
        if self.live_sink is not None: return True
        try:
            dev = QMediaDevices.defaultAudioOutput()
            if dev.isNull(): raise RuntimeError("no audio output device found")
            fmt = QAudioFormat()
            fmt.setSampleRate(MIX_SR); fmt.setChannelCount(2)
            fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
            if not dev.isFormatSupported(fmt): raise RuntimeError("audio device does not support 44.1 kHz / 16 bit stereo")
            sink = QAudioSink(dev, fmt)
            sink.setBufferSize(int(0.15 * MIX_SR) * 4)
            sink.setVolume(self.live_vol.value() / 100.0)
            self.live_sink = sink
            return True
        except Exception as e:
            self.live_status.setText(f"❌ Audio output error: {e}")
            return False

    def live_play_pause(self):
        if self.live_engine.n == 0: return
        if self.live_state == 'playing': self.live_pause(); return
        if not self._live_ensure_sink(): return
        self.player.pause(); self.mt_pause()      # ferma l'anteprima dell'Editor e il multitraccia
        if self.live_state == 'paused':
            self.live_sink.resume()
        else:
            if not self.live_engine.isOpen(): self.live_engine.open(QIODevice.OpenModeFlag.ReadOnly)
            if self.live_engine.pos >= self.live_engine.n: self.live_engine.reset()
            self.live_sink.start(self.live_engine)
        self.live_state = 'playing'
        self.live_play_btn.setText("⏸ PAUSE"); self.live_timer.start(40)

    def live_pause(self):
        if self.live_state != 'playing' or self.live_sink is None: return
        self.live_sink.suspend(); self.live_state = 'paused'
        self.live_play_btn.setText("▶ PLAY"); self.live_timer.stop()

    def live_stop(self):
        if self.live_sink is not None and self.live_state != 'stopped': self.live_sink.stop()
        self.live_engine.reset(); self.live_state = 'stopped'
        self.live_play_btn.setText("▶ PLAY"); self.live_timer.stop()
        self.live_tick()

    def live_on_end(self):
        if not self.live_engine.loop: self.live_stop()

    def live_seek(self, frac):
        self.live_engine.seek_frac(frac); self.live_tick()

    def live_volume(self, v):
        if self.live_sink is not None: self.live_sink.setVolume(v / 100.0)

    def live_tick(self):
        n = self.live_engine.n
        if n <= 0: return
        pos = self.live_engine.pos - int(0.075 * MIX_SR)   # compensa circa metà del buffer audio
        pos = pos % n if self.live_engine.loop else max(0, pos)
        frac = pos / n
        self.live_master.set_playhead(frac)
        for lane in self.live_lanes: lane.wave.set_playhead(frac)
        self.live_time_lbl.setText(f"{fmt_time(pos / MIX_SR)} / {fmt_time(n / MIX_SR)}")

    # --- salvataggio (renderizza esattamente quello che senti: mute/solo/volumi inclusi) ---
    def live_save(self):
        if self.live_engine.n == 0 or self.live_task is not None: return
        layers = list(self.live_audio); eff = self.live_engine.eff_gains()
        logs = [ln for lg in self.live_logs for ln in lg]
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        name = f"rndsnd_live_{timestamp}"
        header = [f"Date: {timestamp}", f"Seed (last roll): {self.live_seed}",
                  f"Duration: {self.live_n / MIX_SR:.0f}s | Layers: {len(layers)}",
                  "Layer state: " + ", ".join(f"L{i + 1}={'off' if g == 0 else f'{g:.2f}'}" for i, g in enumerate(eff))]

        def work(ctx):
            ctx.status.emit("💾 Rendering...")
            return export_mix(render_mix(layers, eff), name, header, logs)

        task = TaskThread(work)
        task.status.connect(self.live_status.setText)
        task.done.connect(lambda r: (self.live_status.setText(f"✅ Saved: {os.path.basename(r[0])} + Log{r[1]}\n{OUTPUT_DIR}"),
                                     self.live_save_btn.setEnabled(True)))
        task.failed.connect(lambda m: (self.live_status.setText(f"❌ {m}"), self.live_save_btn.setEnabled(True)))
        self.live_save_btn.setEnabled(False)
        self._start_task(task)


    # =================================================================
    #  MULTITRACK TAB  (montaggio non distruttivo dei clip)
    # =================================================================
    def mt_init_state(self):
        self.mt_project = new_project()
        self.mt_media = {}                 # (path, a, b) -> MtMedia
        self.mt_pps = 60.0                 # pixel per secondo
        self.mt_x0 = 0.0                   # secondo all'estremo sinistro della vista
        self.mt_playhead = 0.0
        self.mt_sel_clip = None; self.mt_sel_track = 0
        self.mt_undo = []; self.mt_redo = []; self._mt_last_key = None; self._mt_last_t = 0.0
        self.mt_engine = MtEngine(self); self.mt_engine.reachedEnd.connect(self.mt_on_end)
        self.mt_sink = None; self.mt_state = 'stopped'; self._mt_origin = 0.0
        self.mt_task = None
        self.mt_theme = {}
        self.mt_timer = QTimer(); self.mt_timer.timeout.connect(self.mt_tick)

    def setup_multitrack_tab(self):
        tab = QWidget(); self.mt_tab = tab; root = QVBoxLayout(tab)

        r1 = QHBoxLayout()
        self.mt_play_btn = QPushButton("▶ PLAY"); self.mt_play_btn.clicked.connect(self.mt_play_pause)
        self.mt_stop_btn = QPushButton("⏹ STOP"); self.mt_stop_btn.clicked.connect(self.mt_stop)
        self.mt_home_btn = QPushButton("⏮"); self.mt_home_btn.setToolTip("Torna all'inizio (Home)")
        self.mt_home_btn.clicked.connect(lambda: self.mt_seek(0.0))
        self.mt_loop_chk = QCheckBox("Loop"); self.mt_loop_chk.toggled.connect(lambda v: setattr(self.mt_engine, 'loop', v))
        r1.addWidget(self.mt_home_btn); r1.addWidget(self.mt_play_btn); r1.addWidget(self.mt_stop_btn); r1.addWidget(self.mt_loop_chk)
        r1.addWidget(QLabel("Vol:"))
        self.mt_vol = QSlider(Qt.Horizontal); self.mt_vol.setRange(0, 100); self.mt_vol.setValue(80); self.mt_vol.setFixedWidth(110)
        self.mt_vol.valueChanged.connect(lambda v: self.mt_sink.setVolume(v / 100.0) if self.mt_sink is not None else None)
        r1.addWidget(self.mt_vol)
        self.mt_time_lbl = QLabel("00:00.0 / 00:00"); self.mt_time_lbl.setMinimumWidth(130); r1.addWidget(self.mt_time_lbl)
        r1.addSpacing(16)
        self.mt_snap_chk = QCheckBox("Snap"); self.mt_snap_chk.setChecked(True)
        self.mt_snap_chk.setToolTip("Aggancia clip a griglia BPM, bordi degli altri clip e cursore")
        self.mt_snap_chk.toggled.connect(lambda v: (self.mt_project.__setitem__('snap', bool(v)), self.mt_timeline.update()))
        self.mt_bpm = QSpinBox(); self.mt_bpm.setRange(40, 240); self.mt_bpm.setValue(120)
        self.mt_bpm.valueChanged.connect(lambda v: (self.mt_project.__setitem__('bpm', int(v)), self.mt_timeline.update()))
        r1.addWidget(self.mt_snap_chk); r1.addWidget(QLabel("BPM:")); r1.addWidget(self.mt_bpm)
        r1.addSpacing(16)
        for txt, fn, tip in (("🔍−", lambda: self.mt_zoom_at(1 / 1.4, self.mt_timeline.width() / 2), "Zoom out"),
                             ("🔍+", lambda: self.mt_zoom_at(1.4, self.mt_timeline.width() / 2), "Zoom in (anche con la rotella)"),
                             ("Fit", self.mt_fit, "Mostra tutto il progetto")):
            b = QPushButton(txt); b.setToolTip(tip); b.clicked.connect(fn); b.setFixedWidth(52); r1.addWidget(b)
        r1.addSpacing(16)
        self.mt_undo_btn = QPushButton("↶"); self.mt_undo_btn.setToolTip("Annulla (Ctrl+Z)"); self.mt_undo_btn.clicked.connect(self.mt_undo_action)
        self.mt_redo_btn = QPushButton("↷"); self.mt_redo_btn.setToolTip("Ripeti (Ctrl+Shift+Z)"); self.mt_redo_btn.clicked.connect(self.mt_redo_action)
        for b in (self.mt_undo_btn, self.mt_redo_btn): b.setFixedWidth(40); r1.addWidget(b)
        r1.addStretch()
        root.addLayout(r1)

        r2 = QHBoxLayout()
        for txt, fn, tip in (("➕ Clip Bin item", self.mt_insert_bin_selected, "Inserisce il clip selezionato nel pannello Clip Bin al cursore, sulla traccia scelta"),
                             ("➕ Editor selection", self.mt_insert_editor, "Inserisce la selezione (o l'intero file) dell'Editor"),
                             ("✂ Split (S)", self.mt_split, "Divide il clip al cursore"),
                             ("⧉ Duplicate", self.mt_duplicate, "Duplica il clip dopo la sua fine (Ctrl+D)"),
                             ("🗑 Delete", self.mt_delete, "Elimina il clip selezionato (Canc)")):
            b = QPushButton(txt); b.setToolTip(tip); b.clicked.connect(fn); r2.addWidget(b)
        r2.addSpacing(16)
        for txt, fn in (("➕ Track", self.mt_add_track), ("➖ Track", self.mt_remove_track)):
            b = QPushButton(txt); b.clicked.connect(fn); r2.addWidget(b)
        r2.addStretch()
        for txt, fn in (("💾 Save project", self.mt_save_project), ("📂 Open project", self.mt_open_project)):
            b = QPushButton(txt); b.clicked.connect(fn); r2.addWidget(b)
        self.mt_export_btn = QPushButton("⬇ EXPORT MIX"); self.mt_export_btn.clicked.connect(self.mt_export)
        r2.addWidget(self.mt_export_btn)
        root.addLayout(r2)

        r3 = QHBoxLayout()
        self.mt_clip_lbl = QLabel("No clip selected"); self.mt_clip_lbl.setMinimumWidth(260)
        self.mt_gain_spin = QSpinBox(); self.mt_gain_spin.setRange(0, 200); self.mt_gain_spin.setSuffix(" %")
        self.mt_fin_spin = QDoubleSpinBox(); self.mt_fin_spin.setRange(0, 120); self.mt_fin_spin.setSingleStep(0.05); self.mt_fin_spin.setSuffix(" s")
        self.mt_fout_spin = QDoubleSpinBox(); self.mt_fout_spin.setRange(0, 120); self.mt_fout_spin.setSingleStep(0.05); self.mt_fout_spin.setSuffix(" s")
        self.mt_gain_spin.valueChanged.connect(lambda v: self.mt_prop_edit('gain', v / 100.0))
        self.mt_fin_spin.valueChanged.connect(lambda v: self.mt_prop_edit('fade_in', v))
        self.mt_fout_spin.valueChanged.connect(lambda v: self.mt_prop_edit('fade_out', v))
        r3.addWidget(self.mt_clip_lbl); r3.addWidget(QLabel("Clip gain:")); r3.addWidget(self.mt_gain_spin)
        r3.addWidget(QLabel("Fade in:")); r3.addWidget(self.mt_fin_spin); r3.addWidget(QLabel("Fade out:")); r3.addWidget(self.mt_fout_spin)
        r3.addStretch()
        self.mt_prog = QProgressBar(); self.mt_prog.setVisible(False); self.mt_prog.setFixedWidth(220); r3.addWidget(self.mt_prog)
        root.addLayout(r3)

        self.mt_status = QLabel("Drag clips from the Clip Bin panel (right) onto a track. Wheel = zoom • Shift+wheel = scroll • drag edges = trim • drag top dots = fades.")
        self.mt_status.setWordWrap(True)
        root.addWidget(self.mt_status)

        area = QWidget(); av = QVBoxLayout(area); av.setContentsMargins(0, 0, 0, 0)
        self.mt_ruler = RulerWidget(self)
        rr = QHBoxLayout(); rr.setSpacing(0); rr.setContentsMargins(0, 0, 0, 0)
        sp = QWidget(); sp.setFixedWidth(200); rr.addWidget(sp); rr.addWidget(self.mt_ruler, 1)
        av.addLayout(rr)

        self.mt_timeline = TimelineWidget(self)
        self.mt_headers_box = QWidget(); self.mt_headers_box.setFixedWidth(200)
        self.mt_headers_lay = QVBoxLayout(self.mt_headers_box); self.mt_headers_lay.setContentsMargins(0, 0, 0, 0); self.mt_headers_lay.setSpacing(0)
        container = QWidget(); cl = QHBoxLayout(container); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(0)
        cl.addWidget(self.mt_headers_box, 0, Qt.AlignTop); cl.addWidget(self.mt_timeline, 1, Qt.AlignTop)
        self.mt_scroll = QScrollArea(); self.mt_scroll.setWidgetResizable(True); self.mt_scroll.setWidget(container)
        self.mt_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        av.addWidget(self.mt_scroll, 1)

        self.mt_hscroll = QScrollBar(Qt.Horizontal); self.mt_hscroll.valueChanged.connect(self.mt_on_hscroll)
        hr = QHBoxLayout(); hr.setSpacing(0); hr.setContentsMargins(0, 0, 0, 0)
        sp2 = QWidget(); sp2.setFixedWidth(200); hr.addWidget(sp2); hr.addWidget(self.mt_hscroll, 1)
        av.addLayout(hr)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(area); split.addWidget(self.build_mt_clip_panel())
        split.setStretchFactor(0, 1); split.setStretchFactor(1, 0); split.setSizes([1150, 270])
        root.addWidget(split, 1)
        self.refresh_clips()

        self.tabs.addTab(tab, "Multitrack")
        for key, fn in ((Qt.Key_Space, self.mt_play_pause), (Qt.Key_S, self.mt_split), (Qt.Key_Delete, self.mt_delete),
                        ("Ctrl+D", self.mt_duplicate), ("Ctrl+Z", self.mt_undo_action), ("Ctrl+Shift+Z", self.mt_redo_action),
                        ("Ctrl+Y", self.mt_redo_action), (Qt.Key_Home, lambda: self.mt_seek(0.0))):
            sc = QShortcut(QKeySequence(key), tab, activated=fn); sc.setContext(Qt.WidgetWithChildrenShortcut)
        self.mt_apply_theme()
        self.mt_rebuild_headers()
        self.mt_changed()
        self.mt_update_props()

    def build_mt_clip_panel(self):
        """Copia del Clip Bin dentro il Multitrack: i clip si trascinano direttamente sulle tracce."""
        frame = QFrame(); lay = QVBoxLayout(frame); lay.setContentsMargins(0, 0, 0, 0)
        head = QLabel("🎞 CLIP BIN"); head.setStyleSheet("padding: 5px; background: #333; color: white;")
        hint = QLabel("Drag a clip onto a track,\nor double-click it to insert at the cursor\non the selected track.")
        hint.setWordWrap(True); hint.setStyleSheet("font-weight: normal; font-size: 11px;")
        self.mt_clip_filter = QLineEdit(); self.mt_clip_filter.setPlaceholderText("🔍 Filter clips...")
        self.mt_clip_filter.textChanged.connect(lambda *_: self.mt_apply_clip_filter())
        self.mt_clip_list = ClipList(self, editable=False)
        self.mt_clip_list.itemDoubleClicked.connect(lambda it: self.mt_add_bin_clip(it.data(Qt.UserRole), self.mt_sel_track, self.mt_playhead))
        ins = QPushButton("➕ Insert at cursor"); ins.clicked.connect(self.mt_insert_from_panel)
        for w in (head, hint, self.mt_clip_filter): lay.addWidget(w)
        lay.addWidget(self.mt_clip_list, 1); lay.addWidget(ins)
        return frame

    def mt_apply_clip_filter(self):
        if not hasattr(self, 'mt_clip_list'): return
        txt = self.mt_clip_filter.text().strip().lower()
        for i in range(self.mt_clip_list.count()):
            it = self.mt_clip_list.item(i); it.setHidden(bool(txt) and txt not in it.text().lower())

    def mt_insert_from_panel(self):
        it = self.mt_clip_list.currentItem()
        if it is None: self.mt_status.setText("ℹ️ Select a clip in the panel first."); return
        self.mt_add_bin_clip(it.data(Qt.UserRole), self.mt_sel_track, self.mt_playhead)

    def mt_apply_theme(self):
        if not hasattr(self, 'mt_timeline'): return
        if getattr(self, 'current_theme', 'Dark') == "Dark":
            self.mt_theme = dict(bg='#1b1b1b', lane_alt='#202020', lane_sel='#2a2a2a', grid='#444444', text='#dddddd', ruler='#2b2b2b', cursor='#ffffff')
        else:
            self.mt_theme = dict(bg='#fafafa', lane_alt='#f0f0f0', lane_sel='#e4e4e4', grid='#bbbbbb', text='#222222', ruler='#e0e0e0', cursor='#000000')
        self.mt_timeline.update(); self.mt_ruler.update()

    # --- tracce e selezione ---
    def mt_rebuild_headers(self):
        while self.mt_headers_lay.count():
            it = self.mt_headers_lay.takeAt(0)
            if it.widget(): it.widget().deleteLater()
        self.mt_headers = []
        for i in range(len(self.mt_project['tracks'])):
            h = MtTrackHeader(self, i); self.mt_headers.append(h); self.mt_headers_lay.addWidget(h)
        self.mt_headers_lay.addStretch()
        self.mt_sel_track = max(0, min(self.mt_sel_track, len(self.mt_project['tracks']) - 1))
        self.mt_timeline.setMinimumHeight(TimelineWidget.TH * max(4, len(self.mt_project['tracks'])))
        for i, h in enumerate(self.mt_headers): h.set_selected(i == self.mt_sel_track)
        self.mt_timeline.updateGeometry()

    def mt_select_track(self, idx):
        self.mt_sel_track = idx
        for i, h in enumerate(self.mt_headers): h.set_selected(i == idx)
        self.mt_timeline.update()

    def mt_select_clip(self, cid, ti=None):
        self.mt_sel_clip = cid
        if ti is not None: self.mt_select_track(ti)
        self.mt_update_props(); self.mt_timeline.update()

    def mt_find_clip(self, cid):
        for ti, t in enumerate(self.mt_project['tracks']):
            for c in t['clips']:
                if c['id'] == cid: return ti, c
        return None

    def mt_media_duration(self, c):
        m = self.mt_media.get(media_key(c['path'], c['media_a'], c['media_b']))
        return m.duration if m else (c['media_b'] - c['media_a'])

    def mt_track_edit(self, idx, field, value):
        tracks = self.mt_project['tracks']
        if idx >= len(tracks) or tracks[idx][field] == value: return
        self.mt_checkpoint(key=('trk', idx, field) if field in ('gain', 'pan') else None)
        tracks[idx][field] = value
        self.mt_changed()

    def mt_add_track(self):
        if len(self.mt_project['tracks']) >= 16:
            self.mt_status.setText("Maximum 16 tracks."); return
        self.mt_checkpoint(); self.mt_project['tracks'].append(new_track(len(self.mt_project['tracks']) + 1))
        self.mt_rebuild_headers(); self.mt_changed()

    def mt_remove_track(self):
        tracks = self.mt_project['tracks']
        if len(tracks) <= 1: self.mt_status.setText("At least one track is needed."); return
        t = tracks[self.mt_sel_track]
        if t['clips'] and QMessageBox.question(self, "Remove track", f"'{t['name']}' contains {len(t['clips'])} clip(s). Remove it anyway?") != QMessageBox.Yes:
            return
        self.mt_checkpoint(); tracks.pop(self.mt_sel_track)
        if self.mt_sel_clip is not None and not self.mt_find_clip(self.mt_sel_clip): self.mt_sel_clip = None
        self.mt_rebuild_headers(); self.mt_changed(); self.mt_update_props()

    # --- annulla / ripeti ---
    def mt_checkpoint(self, key=None):
        """Salva lo stato PRIMA di una modifica. Con `key`, modifiche ripetute ravvicinate (slider) = un solo annulla."""
        now = time.time()
        if key is not None and key == self._mt_last_key and now - self._mt_last_t < 1.0:
            self._mt_last_t = now; return
        self._mt_last_key, self._mt_last_t = key, now
        self.mt_undo.append(json.dumps(self.mt_project)); self.mt_undo = self.mt_undo[-100:]; self.mt_redo.clear()

    def _mt_restore(self, state):
        self.mt_project = json.loads(state)
        if self.mt_sel_clip is not None and not self.mt_find_clip(self.mt_sel_clip): self.mt_sel_clip = None
        self.mt_snap_chk.blockSignals(True); self.mt_snap_chk.setChecked(self.mt_project.get('snap', True)); self.mt_snap_chk.blockSignals(False)
        self.mt_bpm.blockSignals(True); self.mt_bpm.setValue(self.mt_project.get('bpm', 120)); self.mt_bpm.blockSignals(False)
        self._mt_last_key = None
        self.mt_rebuild_headers(); self.mt_changed(); self.mt_update_props()

    def mt_undo_action(self):
        if not self.mt_undo: self.mt_status.setText("Nothing to undo."); return
        self.mt_redo.append(json.dumps(self.mt_project)); self._mt_restore(self.mt_undo.pop())
        self.mt_status.setText("↶ Undone.")

    def mt_redo_action(self):
        if not self.mt_redo: self.mt_status.setText("Nothing to redo."); return
        self.mt_undo.append(json.dumps(self.mt_project)); self._mt_restore(self.mt_redo.pop())
        self.mt_status.setText("↷ Redone.")

    # --- aggiornamento ---
    def mt_changed(self):
        items = build_snapshot(self.mt_project, self.mt_media)
        self.mt_engine.set_items(items, int(round(mt_length(self.mt_project) * MIX_SR)))
        self.mt_update_scrollbar(); self.mt_update_time_label()
        self.mt_timeline.update(); self.mt_ruler.update()

    def mt_update_time_label(self):
        t = max(0.0, self.mt_playhead)
        self.mt_time_lbl.setText(f"{int(t // 60):02d}:{t % 60:04.1f} / {fmt_time(mt_length(self.mt_project))}")

    def mt_update_props(self):
        f = self.mt_find_clip(self.mt_sel_clip) if self.mt_sel_clip is not None else None
        for w in (self.mt_gain_spin, self.mt_fin_spin, self.mt_fout_spin): w.setEnabled(f is not None); w.blockSignals(True)
        if f:
            c = f[1]
            self.mt_clip_lbl.setText(f"🎞 {c['name']}  ({c['len']:.2f} s @ {c['pos']:.2f} s)")
            self.mt_gain_spin.setValue(int(round(c['gain'] * 100)))
            self.mt_fin_spin.setValue(c['fade_in']); self.mt_fout_spin.setValue(c['fade_out'])
        else:
            self.mt_clip_lbl.setText("No clip selected")
        for w in (self.mt_gain_spin, self.mt_fin_spin, self.mt_fout_spin): w.blockSignals(False)

    def mt_prop_edit(self, field, value):
        f = self.mt_find_clip(self.mt_sel_clip) if self.mt_sel_clip is not None else None
        if not f: return
        c = f[1]
        if abs(c[field] - value) < 1e-9: return
        self.mt_checkpoint(key=('prop', c['id'], field))
        c[field] = value
        c['fade_in'] = min(c['fade_in'], c['len']); c['fade_out'] = min(c['fade_out'], max(0.0, c['len'] - c['fade_in']))
        self.mt_changed()
        self.mt_clip_lbl.setText(f"🎞 {c['name']}  ({c['len']:.2f} s @ {c['pos']:.2f} s)")

    # --- vista: zoom, scorrimento, cursore ---
    def mt_visible_sec(self): return max(1, self.mt_timeline.width()) / self.mt_pps

    def mt_update_scrollbar(self):
        vis = self.mt_visible_sec()
        total = max(mt_length(self.mt_project) + 30, vis + 10)
        self.mt_hscroll.blockSignals(True)
        self.mt_hscroll.setMaximum(max(0, int((total - vis) * 100))); self.mt_hscroll.setPageStep(int(vis * 100))
        self.mt_hscroll.setValue(int(self.mt_x0 * 100)); self.mt_hscroll.blockSignals(False)

    def mt_on_hscroll(self, v):
        self.mt_x0 = v / 100.0; self.mt_timeline.update(); self.mt_ruler.update()

    def mt_scroll_by(self, dt):
        self.mt_x0 = max(0.0, self.mt_x0 + dt); self.mt_update_scrollbar(); self.mt_timeline.update(); self.mt_ruler.update()

    def mt_zoom_at(self, factor, x):
        t = x / self.mt_pps + self.mt_x0
        self.mt_pps = max(2.0, min(3000.0, self.mt_pps * factor))
        self.mt_x0 = max(0.0, t - x / self.mt_pps)
        self.mt_update_scrollbar(); self.mt_timeline.update(); self.mt_ruler.update()

    def mt_fit(self):
        L = max(mt_length(self.mt_project), 5.0)
        self.mt_pps = max(2.0, (self.mt_timeline.width() - 20) / L); self.mt_x0 = 0.0
        self.mt_update_scrollbar(); self.mt_timeline.update(); self.mt_ruler.update()

    def mt_seek(self, t):
        self.mt_playhead = max(0.0, t)
        self.mt_engine.seek(int(self.mt_playhead * MIX_SR))
        self.mt_update_time_label(); self.mt_timeline.update(); self.mt_ruler.update()

    def mt_snap_time(self, t, exclude_id, tol):
        """Punto di aggancio piu' vicino a t entro `tol` secondi (bordi dei clip, cursore, griglia), oppure None."""
        if not self.mt_project.get('snap'): return None
        cands = [0.0, self.mt_playhead]
        for tr in self.mt_project['tracks']:
            for c in tr['clips']:
                if c['id'] != exclude_id: cands += [c['pos'], c['pos'] + c['len']]
        beat = 60.0 / max(1, self.mt_project.get('bpm', 120))
        cands.append(round(t / beat) * beat)
        best = min(cands, key=lambda x: abs(x - t))
        return best if abs(best - t) <= tol else None

    # --- inserimento clip ---
    def mt_ensure_media(self, path, a, b):
        key = media_key(path, a, b)
        if key in self.mt_media: return True
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.mt_media[key] = MtMedia(read_segment(path, a, b)); return True
        except Exception as e:
            self.mt_status.setText(f"❌ Cannot read '{os.path.basename(path)}': {e}"); return False
        finally:
            QApplication.restoreOverrideCursor()

    def mt_add_clip(self, name, path, a, b, ti, t):
        if not os.path.exists(path): self.mt_status.setText(f"⚠️ Source file not found: {path}"); return None
        if not self.mt_ensure_media(path, a, b): return None
        media = self.mt_media[media_key(path, a, b)]
        self.mt_checkpoint()
        P = self.mt_project; cid = P['next_id']; P['next_id'] += 1
        c = {'id': cid, 'name': name, 'path': path, 'media_a': float(a), 'media_b': float(b), 'pos': max(0.0, float(t)),
             'in': 0.0, 'len': min(float(b - a), media.duration), 'gain': 1.0, 'fade_in': 0.0, 'fade_out': 0.0}
        ti = max(0, min(ti, len(P['tracks']) - 1))
        P['tracks'][ti]['clips'].append(c)
        self.mt_select_clip(cid, ti); self.mt_changed()
        self.mt_status.setText(f"➕ '{name}' added to {P['tracks'][ti]['name']} at {c['pos']:.2f}s ({c['len']:.2f}s).")
        return c

    def mt_add_bin_clip(self, clip_id, ti, t):
        c = getattr(self, 'clips', {}).get(clip_id)
        if not c: self.mt_status.setText("⚠️ Clip not found in the bin."); return
        name, path, s, e = c
        return self.mt_add_clip(name, path, s, e, ti, t)

    def mt_add_file(self, path, ti, t):
        if not path.lower().endswith(('.wav', '.mp3', '.flac', '.aiff', '.aif', '.ogg', '.m4a', '.wma', '.aac', '.opus')): return
        try: dur = get_duration(path)
        except Exception as e: self.mt_status.setText(f"❌ Cannot read '{os.path.basename(path)}': {e}"); return
        return self.mt_add_clip(os.path.splitext(os.path.basename(path))[0], path, 0.0, dur, ti, t)

    def mt_insert_bin_selected(self):
        it = self.mt_clip_list.currentItem() or self.clip_list.currentItem()
        if it is None: self.mt_status.setText("ℹ️ Select a clip in the Clip Bin panel first."); return
        self.mt_add_bin_clip(it.data(Qt.UserRole), self.mt_sel_track, self.mt_playhead)

    def mt_insert_editor(self):
        if self.audio_data is None or not self.current_path:
            self.mt_status.setText("ℹ️ Load a file in the Library Editor first."); return
        s, e = self.selection_range
        if not (self.is_looping and e - s > 0.02): s, e = 0.0, self.duration
        name = os.path.splitext(os.path.basename(self.current_path))[0]
        self.mt_add_clip(name if (s, e) == (0.0, self.duration) else f"{name} {self.fmt_pos(s)}", self.current_path, s, e,
                         self.mt_sel_track, self.mt_playhead)

    # --- modifica ---
    def _mt_selected(self):
        f = self.mt_find_clip(self.mt_sel_clip) if self.mt_sel_clip is not None else None
        return f

    def mt_split(self):
        t = self.mt_playhead
        f = self._mt_selected()
        if not f or not (f[1]['pos'] + 0.02 < t < f[1]['pos'] + f[1]['len'] - 0.02):
            f = None                                 # altrimenti: clip sotto il cursore nella traccia selezionata
            for c in self.mt_project['tracks'][self.mt_sel_track]['clips']:
                if c['pos'] + 0.02 < t < c['pos'] + c['len'] - 0.02: f = (self.mt_sel_track, c); break
        if not f: self.mt_status.setText("ℹ️ Place the cursor inside a clip (click the ruler or timeline) to split it."); return
        ti, c = f
        self.mt_checkpoint()
        d1 = t - c['pos']
        P = self.mt_project; nid = P['next_id']; P['next_id'] += 1
        b = dict(c); b.update(id=nid, pos=t, len=c['len'] - d1, fade_in=0.0)
        b['in'] = c['in'] + d1
        c['len'] = d1; c['fade_out'] = 0.0; c['fade_in'] = min(c['fade_in'], c['len'])
        P['tracks'][ti]['clips'].append(b)
        self.mt_select_clip(nid, ti); self.mt_changed()
        self.mt_status.setText(f"✂ Split at {t:.2f}s.")

    def mt_duplicate(self):
        f = self._mt_selected()
        if not f: self.mt_status.setText("ℹ️ Select a clip first."); return
        ti, c = f; self.mt_checkpoint()
        P = self.mt_project; nid = P['next_id']; P['next_id'] += 1
        d = dict(c); d.update(id=nid, pos=c['pos'] + c['len'])
        P['tracks'][ti]['clips'].append(d)
        self.mt_select_clip(nid, ti); self.mt_changed()

    def mt_delete(self):
        f = self._mt_selected()
        if not f: return
        self.mt_checkpoint(); f_t, c = f
        self.mt_project['tracks'][f_t]['clips'].remove(c)
        self.mt_sel_clip = None; self.mt_changed(); self.mt_update_props()
        self.mt_status.setText("🗑 Clip deleted from the timeline (source file untouched).")

    def mt_rename_clip(self, cid):
        f = self.mt_find_clip(cid)
        if not f: return
        name, ok = QInputDialog.getText(self, "Rename clip", "Name:", text=f[1]['name'])
        if ok and name.strip(): self.mt_set_clip_name(cid, name.strip())

    def mt_set_clip_name(self, cid, name):
        f = self.mt_find_clip(cid)
        if not f: return
        self.mt_checkpoint(); f[1]['name'] = name; self.mt_changed(); self.mt_update_props()

    # --- riproduzione ---
    def _mt_ensure_sink(self):
        if self.mt_sink is not None: return True
        try:
            dev = QMediaDevices.defaultAudioOutput()
            if dev.isNull(): raise RuntimeError("no audio output device found")
            fmt = QAudioFormat(); fmt.setSampleRate(MIX_SR); fmt.setChannelCount(2); fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
            if not dev.isFormatSupported(fmt): raise RuntimeError("audio device does not support 44.1 kHz / 16 bit stereo")
            sink = QAudioSink(dev, fmt); sink.setBufferSize(int(0.15 * MIX_SR) * 4); sink.setVolume(self.mt_vol.value() / 100.0)
            self.mt_sink = sink; return True
        except Exception as e:
            self.mt_status.setText(f"❌ Audio output error: {e}"); return False

    def mt_play_pause(self):
        if self.mt_state == 'playing': self.mt_pause(); return
        if self.mt_engine.length <= 0: self.mt_status.setText("ℹ️ The timeline is empty: add some clips first."); return
        if not self._mt_ensure_sink(): return
        self.player.pause(); self.live_pause()          # un solo audio alla volta
        if self.mt_state == 'paused':
            self.mt_sink.resume()
        else:
            if not self.mt_engine.isOpen(): self.mt_engine.open(QIODevice.OpenModeFlag.ReadOnly)
            if self.mt_playhead * MIX_SR >= self.mt_engine.length and not self.mt_engine.loop: self.mt_playhead = 0.0
            self._mt_origin = self.mt_playhead
            self.mt_engine.seek(int(self.mt_playhead * MIX_SR))
            self.mt_sink.start(self.mt_engine)
        self.mt_state = 'playing'; self.mt_play_btn.setText("⏸ PAUSE"); self.mt_timer.start(30)

    def mt_pause(self):
        if self.mt_state != 'playing' or self.mt_sink is None: return
        self.mt_sink.suspend(); self.mt_state = 'paused'; self.mt_play_btn.setText("▶ PLAY"); self.mt_timer.stop()

    def mt_stop(self):
        if self.mt_sink is not None and self.mt_state != 'stopped': self.mt_sink.stop()
        was = self.mt_state != 'stopped'
        self.mt_state = 'stopped'; self.mt_play_btn.setText("▶ PLAY"); self.mt_timer.stop()
        self.mt_seek(self._mt_origin if was else 0.0)     # come in Audacity: il cursore torna dove e' iniziata la riproduzione

    def mt_on_end(self):
        if not self.mt_engine.loop: self.mt_stop()

    def mt_tick(self):
        if self.mt_state != 'playing': return
        t = max(0.0, (self.mt_engine.pos - int(0.075 * MIX_SR)) / MIX_SR)
        self.mt_playhead = t
        vis = self.mt_visible_sec()
        if t < self.mt_x0 or t > self.mt_x0 + vis:           # il cursore esce dalla vista: la vista lo segue
            self.mt_x0 = max(0.0, t - vis * 0.1); self.mt_update_scrollbar()
        self.mt_update_time_label(); self.mt_timeline.update(); self.mt_ruler.update()

    # --- esportazione ---
    def mt_export(self):
        if self.mt_task is not None:
            self.mt_task.cancel(); self.mt_status.setText("⏹ Cancelling..."); return
        items = build_snapshot(self.mt_project, self.mt_media)
        n = int(round(mt_length(self.mt_project) * MIX_SR))
        if n <= 0 or not items: self.mt_status.setText("ℹ️ Nothing to export: the timeline is empty (or every track is muted)."); return
        P = self.mt_project
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S'); name = f"rndsnd_multitrack_{timestamp}"
        header = [f"Date: {timestamp}", f"Duration: {n / MIX_SR:.2f}s | Tracks: {len(P['tracks'])}"]
        logs = []
        for tr in P['tracks']:
            state = " [MUTED]" if tr['mute'] else (" [SOLO]" if tr['solo'] else "")
            logs.append(f"== {tr['name']} (vol {tr['gain']:.2f}, pan {tr['pan']:+.2f}){state}")
            for c in sorted(tr['clips'], key=lambda c: c['pos']):
                a = c['media_a'] + c['in']
                logs.append(f"   {c['name']}  <- {os.path.basename(c['path'])} [{self.fmt_pos(a)}-{self.fmt_pos(a + c['len'])}] "
                            f"@ {c['pos']:.2f}s, {c['len']:.2f}s, gain {c['gain']:.2f}, fade {c['fade_in']:.2f}/{c['fade_out']:.2f}")

        def work(ctx):
            out = np.zeros((n, 2), np.float32); blk = 1 << 16
            for p in range(0, n, blk):
                ctx.check(); k = min(blk, n - p)
                out[p:p + k] = mt_mix_block(items, p, k); ctx.progress.emit(int(p / n * 100))
            peak = float(np.abs(out).max()); note = ""
            if peak > 1.0:                                   # somma oltre 0 dB: si abbassa tutto invece di distorcere
                out *= 0.98 / peak; note = f" — peaks exceeded 0 dB, whole mix lowered by {20 * math.log10(peak / 0.98):.1f} dB"
            ctx.status.emit("💾 Encoding...")
            path, n2 = export_mix(out, name, header, logs, soft=False)
            return path, n2 + note

        task = TaskThread(work)
        task.progress.connect(self.mt_prog.setValue); task.status.connect(self.mt_status.setText)
        task.done.connect(self._mt_export_done); task.failed.connect(self._mt_export_failed)
        self.mt_task = task; self.mt_export_btn.setText("✖ CANCEL"); self.mt_prog.setValue(0); self.mt_prog.setVisible(True)
        self.mt_status.setText("⏳ Rendering..."); self._start_task(task)

    def _mt_export_ui(self): self.mt_task = None; self.mt_export_btn.setText("⬇ EXPORT MIX"); self.mt_prog.setVisible(False)
    def _mt_export_done(self, res):
        self._mt_export_ui(); self.mt_status.setText(f"✅ Exported: {os.path.basename(res[0])} + Log{res[1]}\n{OUTPUT_DIR}")
    def _mt_export_failed(self, msg):
        self._mt_export_ui(); self.mt_status.setText("⏹ Cancelled." if msg == "cancelled" else f"❌ {msg}")

    # --- salvataggio / apertura progetto ---
    def mt_save_project(self, path=None):
        if not path:
            os.makedirs(os.path.join(APP_DIR, "projects"), exist_ok=True)
            path, _ = QFileDialog.getSaveFileName(self, "Save project", os.path.join(APP_DIR, "projects", "project.rndmix"),
                                                  "rndsnd project (*.rndmix)")
            if not path: return
        if not path.endswith(".rndmix"): path += ".rndmix"
        try:
            with open(path, "w", encoding="utf-8") as f: json.dump(self.mt_project, f, indent=1)
            self.mt_status.setText(f"💾 Project saved: {path}")
        except Exception as e:
            self.mt_status.setText(f"❌ Could not save: {e}")

    def mt_open_project(self, path=None):
        if not path:
            os.makedirs(os.path.join(APP_DIR, "projects"), exist_ok=True)
            path, _ = QFileDialog.getOpenFileName(self, "Open project", os.path.join(APP_DIR, "projects"), "rndsnd project (*.rndmix)")
            if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f: P = json.load(f)
            assert isinstance(P.get('tracks'), list) and P['tracks']
            for t in P['tracks']:
                for k, v in new_track(1).items(): t.setdefault(k, v)
            ids = [c['id'] for t in P['tracks'] for c in t['clips']]
            P['next_id'] = max(ids + [0]) + 1; P.setdefault('bpm', 120); P.setdefault('snap', True)
        except Exception:
            self.mt_status.setText("❌ This is not a valid rndsnd project file (.rndmix)."); return
        self.mt_stop()
        keys = sorted({media_key(c['path'], c['media_a'], c['media_b']) for t in P['tracks'] for c in t['clips']})

        def work(ctx):
            media, missing = {}, []
            for i, k in enumerate(keys):
                ctx.check(); ctx.progress.emit(int(i / max(1, len(keys)) * 100))
                try: media[k] = MtMedia(read_segment(*k))
                except Exception: missing.append(os.path.basename(k[0]))
            return media, missing

        def done(res):
            self.mt_task = None; self.mt_prog.setVisible(False)
            self.mt_media, missing = res; self.mt_project = P; self.mt_undo.clear(); self.mt_redo.clear(); self._mt_last_key = None
            self.mt_sel_clip = None; self.mt_playhead = 0.0; self.mt_x0 = 0.0
            self.mt_snap_chk.blockSignals(True); self.mt_snap_chk.setChecked(P['snap']); self.mt_snap_chk.blockSignals(False)
            self.mt_bpm.blockSignals(True); self.mt_bpm.setValue(P['bpm']); self.mt_bpm.blockSignals(False)
            self.mt_rebuild_headers(); self.mt_changed(); self.mt_update_props(); self.mt_fit()
            self.mt_status.setText(f"📂 Project opened ({sum(len(t['clips']) for t in P['tracks'])} clips)."
                                   + (f" ⚠️ {len(set(missing))} source file(s) missing: {', '.join(sorted(set(missing))[:4])}" if missing else ""))
        task = TaskThread(work); task.progress.connect(self.mt_prog.setValue)
        task.done.connect(done); task.failed.connect(lambda m: (setattr(self, 'mt_task', None), self.mt_prog.setVisible(False), self.mt_status.setText(f"❌ {m}")))
        self.mt_task = task; self.mt_prog.setValue(0); self.mt_prog.setVisible(True); self.mt_status.setText("⏳ Loading project...")
        self._start_task(task)

    def closeEvent(self, event):
        try:
            if self.live_sink is not None: self.live_sink.stop()
            if self.mt_sink is not None: self.mt_sink.stop()
            self.player.stop()
            if self.scan_thread is not None and self.scan_thread.isRunning():
                self.scan_thread.requestInterruption(); self.scan_thread.wait(3000)
            for t in list(self._tasks):
                t.cancel(); t.wait(3000)
        except Exception:
            pass
        super().closeEvent(event)


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
