import streamlit as st
import sqlite3
import json
import os
import random
import pyperclip
from pydub import AudioSegment
from datetime import datetime
from database_manager import AudioScanner

# --- CONFIGURAZIONE ---
DB_NAME = "audio.db"
EXPORT_DIR = "exports"
DEFAULT_PATH = "/media/pirxa/cardo/AUDIO"

if not os.path.exists(EXPORT_DIR):
    os.makedirs(EXPORT_DIR)

def get_files(search_tag=None):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    if search_tag:
        cur.execute("SELECT * FROM files WHERE tags LIKE ?", (f'%{search_tag}%',))
    else:
        cur.execute("SELECT * FROM files")
    return cur.fetchall()

# --- INTERFACCIA STREAMLIT ---
st.set_page_config(page_title="Audio Curator Pro", layout="wide", page_icon="🎧")
st.title("🎧 Audio Curator Pro")

# --- SIDEBAR: GESTIONE SORGENTE ---
st.sidebar.header("📁 Sorgente Audio")
path_to_scan = st.sidebar.text_input("Percorso cartella da scansionare:", DEFAULT_PATH)

if st.sidebar.button("🔍 Avvia/Aggiorna Scansione"):
    if os.path.exists(path_to_scan):
        with st.spinner(f"L'IA sta analizzando i file in {path_to_scan}..."):
            scanner = AudioScanner()
            scanner.scan(path_to_scan)
            st.sidebar.success("Scansione terminata!")
            st.rerun()
    else:
        st.sidebar.error("Percorso non trovato!")

st.sidebar.markdown("---")
# Statistiche DB
conn = sqlite3.connect(DB_NAME)
total_files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
st.sidebar.metric("File in Libreria", total_files)

tab1, tab2 = st.tabs(["🔍 Archivio & Preascolto", "🎛️ Mixer Casuale"])

# --- TAB 1: ARCHIVIO ---
with tab1:
    search = st.text_input("Cerca per categoria (es: Rain, Piano, Speech, Tools)...")
    files = get_files(search if search else None)
    
    if files:
        selected_file_row = st.selectbox("Seleziona un file", files, 
                                         format_func=lambda x: f"{x[1]} | Tags: {x[2]}")
        
        path, name, tags, duration, _ = selected_file_row
        st.write(f"**Percorso:** `{path}`")
        st.audio(path)
        
        col1, col2 = st.columns(2)
        with col1:
            start = st.slider("Inizio (sec)", 0.0, float(duration), 0.0)
        with col2:
            end = st.slider("Fine (sec)", 0.0, float(duration), float(duration))
            
        if st.button("📋 Copia Percorso per Audacity"):
            pyperclip.copy(path)
            st.success("Copiato negli appunti!")

# --- TAB 2: MIXER ---
with tab2:
    st.header("Generatore di Mix")
    col_a, col_b = st.columns(2)
    with col_a:
        mode = st.radio("Modalità di pesca", ["Caos Totale", "Selezione Categorie"])
        format_choice = st.selectbox("Formato Export", ["WAV", "MP3"])
    with col_b:
        total_min = st.slider("Durata mix (minuti)", 1, 30, 5)
        cross_val = st.slider("Crossfade (ms)", 500, 3000, 1500)

    if st.button("🚀 Genera e Esporta Montaggio"):
        # Filtra i file
        source_files = get_files(search if mode == "Selezione Categorie" else None)
        
        if not source_files or len(source_files) < 2:
            st.error("Troppi pochi file trovati per creare un mix. Analizza più cartelle!")
        else:
            with st.spinner("Miscelazione audio in corso..."):
                combined = AudioSegment.empty()
                log_data = []
                target_ms = total_min * 60 * 1000
                id_mix = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                while len(combined) < target_ms:
                    f_path, f_name, f_tags, f_dur, _ = random.choice(source_files)
                    try:
                        audio = AudioSegment.from_file(f_path)
                        # Prendi un frammento tra 5 e 12 secondi
                        frag_ms = random.randint(5000, 12000)
                        if len(audio) > frag_ms:
                            start_p = random.uniform(0, len(audio) - frag_ms)
                            clip = audio[start_p:start_p + frag_ms]
                        else:
                            clip = audio
                        
                        clip = clip.fade_in(500).fade_out(500)
                        
                        if len(combined) == 0:
                            combined = clip
                        else:
                            combined = combined.append(clip, crossfade=min(cross_val, len(clip)-1))
                        
                        log_data.append(f"[{len(combined)/1000:.1f}s] {f_name} (Tag: {f_tags})")
                    except:
                        continue

                # Salvataggio
                audio_fn = f"MIX_{id_mix}.{format_choice.lower()}"
                log_fn = f"MIX_{id_mix}.txt"
                audio_path = os.path.join(EXPORT_DIR, audio_fn)
                log_path = os.path.join(EXPORT_DIR, log_fn)
                
                combined.export(audio_path, format=format_choice.lower())
                with open(log_path, "w") as f:
                    f.write(f"LOG MIX - {id_mix}\n" + "\n".join(log_data))
                
                st.success(f"✅ Mix creato con successo!")
                st.audio(audio_path)
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    with open(audio_path, "rb") as f:
                        st.download_button("Scarica Audio", f, file_name=audio_fn)
                with col_d2:
                    with open(log_path, "r") as f:
                        st.download_button("Scarica Log TXT", f, file_name=log_fn)
