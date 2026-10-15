import os
import sqlite3
import json
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd

# --- CONFIGURAZIONE ---
DB_NAME = "audio.db"
YAMNET_MAP_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"

class AudioScanner:
    def __init__(self):
        self.conn = sqlite3.connect(DB_NAME)
        self.cursor = self.conn.cursor()
        self._setup_db()
        print("Caricamento modello YAMNet...")
        # Carica il modello YAMNet da TFHub
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        self.class_map = pd.read_csv(YAMNET_MAP_URL)['display_name'].tolist()

    def _setup_db(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS files 
            (path TEXT PRIMARY KEY, filename TEXT, tags TEXT, duration REAL, last_modified REAL)''')
        self.conn.commit()

    def analyze_audio(self, file_path):
        try:
            # Carica solo i primi 180s per risparmiare RAM ed evitare crash
            wav_data, sr = librosa.load(file_path, sr=16000, mono=True, duration=180)
            if len(wav_data) == 0: return None, None
            
            duration = librosa.get_duration(y=wav_data, sr=sr)
            
            # Inferenza AI
            scores, _, _ = self.model(wav_data)
            mean_scores = np.mean(scores, axis=0)
            top_5_indices = np.argsort(mean_scores)[-5:][::-1]
            tags = [self.class_map[i] for i in top_5_indices]
            
            return tags, duration
        except Exception as e:
            # Salta i file che danno errore (codec non supportati o corrotti)
            return None, None

    def scan(self, target_path):
        print(f"Inizio scansione Delta-Scan su: {target_path}")
        for root, dirs, files in os.walk(target_path):
            for file in files:
                # Supporto per tutti i formati richiesti (inclusi iPhone)
                if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.aac')):
                    full_path = os.path.join(root, file)
                    try:
                        mtime = os.path.getmtime(full_path)
                        
                        # Verifica se il file è già analizzato o modificato
                        self.cursor.execute("SELECT last_modified FROM files WHERE path=?", (full_path,))
                        row = self.cursor.fetchone()
                        
                        if row is None or row[0] < mtime:
                            print(f"Analisi IA: {file}")
                            tags, duration = self.analyze_audio(full_path)
                            if tags:
                                self.cursor.execute('''INSERT OR REPLACE INTO files 
                                    (path, filename, tags, duration, last_modified) VALUES (?, ?, ?, ?, ?)''',
                                    (full_path, file, json.dumps(tags), duration, mtime))
                                self.conn.commit()
                    except Exception:
                        continue
        print("Operazione completata.")

if __name__ == "__main__":
    # Test veloce se avviato da solo
    scanner = AudioScanner()
    scanner.scan("/media/pirxa/cardo/AUDIO")
