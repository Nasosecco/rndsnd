# rndsnd - Audio Chaos Explorer & Generator

**rndsnd** is a desktop application written in Python (PySide6) designed for sound designers, experimental musicians, and producers. It functions as an "intelligent" sample library manager and a random sound texture generator (Chaos Mixer).

<img width="1392" height="768" alt="rndsnd_splash" src="https://github.com/user-attachments/assets/0997200c-f536-4aec-821b-45611e67d2d5" />

The goal is to rediscover your forgotten sample library, catalog it automatically via AI, and transform it into new soundscapes or samples ready for your DAW.

#### Don't want to build from source? You can download the latest pre-compiled version from the Releases page.

## What can rndsnd do?

rndsnd is a Python-based audio explorer and generative synthesizer designed to breathe new life into your local sound library. It combines neural network analysis with creative mixing tools.

### Key Capabilities:

- Deep AI Auto-Tagging: Uses the PANNs (Cnn14) neural model to analyze and tag your audio files automatically. It employs a smart multi-sampling strategy (scanning start, middle, and end) to accurately classify evolving field recordings and long tracks.

- Generative Mixer: Instantly create new audio assets using two distinct algorithms:
**Linear Mode (1 Layer):** Generates seamless, DJ-style continuous mixes with smooth crossfades.

**Chaos Mode (Multi-Layer):** Creates dense, experimental "walls of sound" by overlaying multiple tracks with random panning and volume.

**Visual Waveform Editor:** Preview files with a responsive waveform view, select loops, and inspect audio details.

**📦 Direct DAW Integration:** Features a dedicated Drag & Drop button to export selected audio snippets directly into Ableton, Reaper, Logic, or your desktop.

**Smart Logging:** Every generated mix is accompanied by a text log detailing exactly which files were used and their specific timecodes.

**🔒 Local & Persistent:** Builds a local SQLite database of your sounds for instant searching, working entirely offline without cloud dependencies.

## ✨ Key Features

* **Integrated System Explorer:** Navigate internal and external hard drives directly from the app.
* **Persistent Database:** Scan folders once; data is saved in a local SQLite database and remains available forever.
* **AI Auto-Tagging (PANNs):** Automatic content recognition (e.g., "Piano", "Synthesizer", "Rain", "Drum") without cloud dependencies.
* **Waveform Editor:** Waveform visualization with zoom, loop, and a **Drag & Drop** button to export selections directly to your favorite DAW.
* **Generative Mixer:** An engine that takes random fragments from your library and layers them to create unique textures.
* **Layered Logic:** The mixer doesn't just create a flat sequence; it overlays multiple audio tracks with random fades and panning for density and depth.
* **Logging:** Every generated mix is accompanied by a `.txt` file containing the exact timecodes of the samples used.

---
<img width="1600" height="972" alt="Screenshot_20260121_171001" src="https://github.com/user-attachments/assets/bb269987-f19c-47b8-8615-7ddee991f9c8" />

## How AI Analysis Works (Deep Scan)

The core of **rndsnd** is the neural analysis engine based on **Cnn14 (PANNs - Large-Scale Pretrained Audio Neural Networks)**.

Unlike other software that only analyzes the first few seconds of a file, **rndsnd v0.7.6** uses a **Weighted Multi-Sampling** algorithm to ensure accuracy even on long or evolving files (such as field recordings or full tracks).

### The Scanning Process:
1.  **Preliminary Check:** The system checks the file duration.
2.  **Sampling Strategy:**
    * **Short Files (< 10s):** Analyzed entirely in a single pass.
    * **Long Files (> 10s):** The algorithm extracts three "chunks" of 5 seconds each:
        * *Start*
        * *Middle*
        * *End*
3.  **Inference & Averaging:** The AI analyzes each chunk separately, calculating the probability of 527 possible audio tags.
4.  **Aggregation:** Scores are summed. If a sound (e.g., a siren) is present only at the very end of the file, it will still be detected thanks to the final chunk scan.
5.  **Tagging:** The 3 tags with the highest overall score are assigned to the file in the database.

### Mixer Logic: Linear vs. Chaos
The Generative Mixer changes its algorithm based on the "Layers / Tracks" setting:

#### 1 Layer (DJ Mode / Linear):

Creates a seamless, linear mix similar to a radio or DJ set.

Tracks are placed one after another with a smooth crossfade (no silence gaps).

Ideal for creating long, evolving soundscapes or continuous listening experiences.

#### 2+ Layers (Chaos Mode / Wall of Sound):

Creates a dense, textured "wall of sound."

Multiple audio clips play simultaneously, overlapping with random start times, panning, and volume levels.

Ideal for experimental textures, drone music, or complex background noise.

---

## 🛠️ Installation and Source Code

This application is written in **Python 3.10+**. Here is how to install and run it on your computer (Linux/macOS/Windows).

### 1. System Prerequisites
On Linux (Ubuntu/Debian), ensure you have the basic audio libraries installed:

`sudo apt update
sudo apt install python3-venv python3-pip libsndfile1 ffmpeg`

(FFmpeg is required to handle formats like MP3 and for export operations).

### 2. Clone or Download
Download the source files into a folder, for example, rndsnd_studio.

### 3. Create a Virtual Environment (Recommended)
To avoid interfering with your main Python system, use a virtual environment:

`cd rndsnd_studio
python3 -m venv venv
source venv/bin/activate`

### 4. Install Dependencies
Create a file named requirements.txt (see below) or install libraries manually:

`pip install PySide6 librosa pydub soundfile matplotlib panns-inference torch numpy`

### 5. Run the Application
With the virtual environment active:

`python app_desktop.py`

## User Guide

### Tab 1: Library & Editor
Navigation: Use the tree on the left ("System Browser") to explore your folders and drives.

Scanning: If a folder has never been analyzed, the table on the right will be empty.

Right-click on the folder in the tree -> "✨ Scan with AI".

Wait for completion.

Preview: Click on a file in the table to view the waveform.

Drag & Drop: Select a part of the waveform (orange area). Click and drag the "📦 DRAG" button directly into your DAW or onto your desktop to export that snippet.

### Tab 2: Generative Mixer
#### Mode:

Filtered List: Uses only files matching your current search in the Library tab.

Full Chaos: Picks randomly from the entire database (even files currently hidden).

#### Parameters:

Duration: The total length of the final mix.

Layers: How many tracks to overlay simultaneously (higher = denser/more chaotic).

Generate: Click the button. The file will be saved in the output/ folder along with a log .txt file.

### 📦 Project Structure
app_desktop.py: Main source code (GUI + Logic).

audio.db: SQLite database (generated automatically on first launch).

rndsnd_splash.png: Splash screen image (optional).

rndsnd_icon.png: App icon (optional).

output/: Folder where generated mixes are saved.

### License
MIT License - Feel free to modify, hack, and break this code to create new sounds.


***

### Extra Tip
Create a file named **`requirements.txt`** in the same folder as your script with the following content, so users can install everything easily:

`PySide6
librosa
pydub
soundfile
matplotlib
panns-inference
torch
numpy`

### ⚠️ Disclaimer

**rndsnd** is currently in active development (Beta). While every effort has been made to ensure the safety and stability of this software, it is provided **"as is"**, without warranty of any kind, express or implied.

**Data Safety:** This software scans and reads files from your hard drive. Although it operates in a read-only mode regarding your source files, the developer is not liable for any data loss or corruption that may occur. Always maintain backups of your important audio libraries.

**Copyright:** You are solely responsible for the audio files you scan and process. Ensure you have the legal right to use any samples or music tracks loaded into the software. The generated mixes are derivative works; their copyright status depends on the original material you provide.

**AI Accuracy:** The AI auto-tagging feature uses a pre-trained neural network (PANNs). Results are probabilistic and may not always be 100% accurate.
