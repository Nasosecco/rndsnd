# rndsnd
**Audio Chaos Explorer & Generative Multitrack Studio**

`rndsnd` is an experimental audio tool designed for sound designers, musicians, and audio chaos enthusiasts. It allows you to build a database from your local audio files, automatically analyze them using AI, and generate unpredictable, layered audio mixes or manually arrange them in a non-destructive multitrack environment.
![Splash Screen](rndsnd_splash.png)
## What it can do

* **Library Editor & AI Scanner:** Point the app to your audio folders. It will scan and extract duration, size, and optionally use AI to automatically tag your sounds and analyze their BPM and musical key.
* **Clip Bin:** Select interesting portions of an audio waveform and save them as clips. You can instantly drag and drop these clips directly into your favorite DAW (like Audacity, Ableton, or Reaper).
* **Generative Mixer:** Create offline, chaotic layered mixes. Set the duration, the number of layers, the "grain" size of the clips, and let the engine build a random soundscape from your filtered library.
* **Live Mix:** A real-time generative playback engine. It randomly pulls clips from your library based on your filters (including BPM and compatible Camelot keys), applies random panning/volume, and plays them live. You can solo/mute layers, quantize changes to a beat grid, and record the output.
* **Multitrack Editor:** A non-destructive timeline where you can drag and drop clips, trim them, add fades, adjust pan/volume, and export the final mix.

## ⚙️ How it works

1. **Scan your files:** Use the *System Browser* on the left to locate your audio folders. Right-click a folder to scan it. `rndsnd` will build a lightweight local database (`audio.db`).
2. **Filter & Select:** Use the search bar to filter sounds by name or tags.
3. **Generate & Play:** Switch to the *Mixer* or *Live Mix* tabs to let the software generate random compositions, or use the *Multitrack* tab to manually build your arrangement.
4. **Export:** All your generated mixes are saved in the `~/rndsnd/output` folder on your computer, along with a `.txt` log file detailing exactly which audio slices were used.

*(Supported formats include WAV, MP3, FLAC, OGG, and more).*

---

## Important Instructions for macOS Users

Since `rndsnd` is an open-source tool and is not signed with a paid Apple Developer Certificate, macOS's Gatekeeper security system will initially block the app, showing an error saying **"The app is damaged and can't be opened"** or that it is from an unidentified developer.

**This is normal.** To fix this and run the app safely, follow these quick steps:

1. Move the downloaded `rndsnd.app` file to your **Applications** folder (or any folder of your choice).
2. Open the **Terminal** app on your Mac (you can find it using Spotlight search).
3. Type the following command and make sure to leave a space at the end:
   `xattr -cr `
4. Drag and drop the `rndsnd.app` icon directly into the Terminal window (this will automatically paste the correct path).
5. Press **Enter**. 

You can now double-click the app to open it normally without any security warnings!
