# rndsnd
**Audio Chaos Explorer & Generative Multitrack Studio**

![Splash Screen](rndsnd_splash.png)

Un potente software per esplorare, gestire e remixare la tua libreria audio in modi nuovi e generativi. I dati, i file temporanei e i mix esportati vengono salvati comodamente nella tua cartella locale `~/rndsnd`[cite: 9].

## 🚀 Cosa può fare

* **Library Editor & AI Scanner:** Punta l'app verso le tue cartelle audio. Eseguirà una scansione per estrarre durata e dimensioni, utilizzando un modello di Intelligenza Artificiale integrato (PyTorch / PANNs) per taggare automaticamente i tuoi suoni e analizzare BPM e tonalità musicale[cite: 9].
* **Clip Bin & Integrazione DAW:** Isola solo le parti che ti interessano! Puoi selezionare porzioni specifiche di una forma d'onda audio e salvarle come clip nel "Clip Bin". Da lì, **puoi trascinare (drag and drop) queste clip direttamente su una traccia di Audacity** (o altre DAW come Ableton e Reaper) per accelerare drasticamente il tuo flusso di lavoro e la costruzione dei tuoi beat[cite: 9].
* **Generative Mixer:** Crea mix caotici e stratificati offline. Imposta la durata, il numero di tracce sovrapposte, la "grana" delle clip (da micro a extra-long) e lascia che il motore costruisca un paesaggio sonoro casuale partendo dalla tua libreria filtrata[cite: 9].
* **Live Mix:** Un motore di riproduzione generativa in tempo reale. Pesca casualmente clip dalla libreria in base ai tuoi filtri (inclusi BPM e chiavi Camelot compatibili), applica panning/volume casuali e le suona dal vivo. Puoi silenziare o isolare i livelli, rimescolare i singoli layer e quantizzare i cambi su una griglia ritmica (a tempo di beat o di battuta) per transizioni perfette[cite: 9].
* **Multitrack Editor:** Una timeline non distruttiva dove puoi posizionare le tue clip, tagliarle (trim), aggiungere dissolvenze in entrata e in uscita, regolare pan e volume di ogni traccia ed esportare il mix finale[cite: 9].

## 💻 Installazione

Il programma è stato progettato per essere "Plug & Play". Tutti i componenti complessi, inclusi i modelli di Intelligenza Artificiale da ~330 MB necessari allo scanner, vengono gestiti e scaricati comodamente dall'interno dell'app[cite: 9].

### Windows
1. Vai alla scheda **Releases** sulla destra della pagina GitHub.
2. Scarica il file `.exe` o il pacchetto `.zip` per Windows.
3. Se hai scaricato lo zip, estrailo. Fai doppio clic sul file eseguibile per lanciare il programma. Non è richiesta alcuna installazione!

### macOS
1. Vai alla scheda **Releases** su GitHub.
2. Scarica il pacchetto destinato a macOS ed estrailo.
3. Fai doppio clic sull'applicazione. *(Nota: al primo avvio potrebbe essere necessario fare clic destro sull'app e selezionare "Apri" per autorizzare l'apertura dalle impostazioni di sicurezza del Mac).*

### Linux
1. Vai alla scheda **Releases** e scarica l'eseguibile compilato per Linux.
2. Rendi il file eseguibile. Dal terminale digita: `chmod +x rndsnd_linux` (o il nome esatto del file).
3. Avvialo con `./rndsnd_linux` o facendoci doppio clic.

---

*In alternativa (per Sviluppatori):* puoi avviare l'app direttamente dal codice sorgente Python. Ti basterà installare le dipendenze di base (PySide6, librosa, soundfile, matplotlib) ed eseguire `python app_desktop.py`[cite: 9]. Per sbloccare lo scanner intelligente, assicurati di installare anche `torch` e `panns-inference` tramite pip[cite: 9].

## 📸 Screenshots

![Il Generative Mixer](assets/mixer.png)
![Il Multitrack Editor](assets/multitrack.png)
