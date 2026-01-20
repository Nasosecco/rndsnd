#!/bin/bash

# --- CONFIGURAZIONE ---
APP_NAME="rndsnd"
VERSION="0.6.5"
MAIN_SCRIPT="app_desktop.py"
ICON="rndsnd_icon.png"
SPLASH="rndsnd_splash.png"
ZIP_NAME="${APP_NAME}_v${VERSION}_linux_x64.tar.gz"

echo "🚀 Inizio processo di pacchettizzazione per $APP_NAME v$VERSION..."

# 1. Pulizia cartelle precedenti
echo "🧹 Pulizia vecchie build e archivi..."
rm -rf build dist "$ZIP_NAME"

# 2. Controllo dipendenze build
if ! command -v pyinstaller &> /dev/null
then
    echo "❌ PyInstaller non trovato. Lo installo ora..."
    pip install pyinstaller
fi

# 3. Lancio PyInstaller
echo "📦 Generazione pacchetto (One-Directory)..."
pyinstaller --noconfirm --onedir --windowed \
    --name "$APP_NAME" \
    --add-data "$SPLASH:." \
    --add-data "$ICON:." \
    --hidden-import="pytorch" \
    --hidden-import="panns_inference" \
    --hidden-import="librosa" \
    --hidden-import="sklearn.utils._cython_blas" \
    --hidden-import="sklearn.neighbors.typedefs" \
    --hidden-import="sklearn.neighbors.quad_tree" \
    --hidden-import="sklearn.tree._utils" \
    "$MAIN_SCRIPT"

# 4. Creazione file README (Bilingue)
if [ -d "dist/$APP_NAME" ]; then
    echo "📝 Generazione file README.txt..."
    cat << EOF > "dist/$APP_NAME/README.txt"
=======================================================
          $APP_NAME v$VERSION - Audio Intelligence
=======================================================

--- ITALIANO ---

ISTRUZIONI PER L'AVVIO:
1. Apri la cartella '$APP_NAME'.
2. Fai doppio clic sull'eseguibile '$APP_NAME' (o lancia ./$APP_NAME dal terminale).

NOTA IMPORTANTE SULL'IA:
Al primo avvio della funzione 'ANALIZZA LIBRERIA', l'app scaricherà 
automaticamente il modello neurale PANNs (circa 312MB). 
È necessaria una connessione internet attiva. Una volta scaricato,
il tagging funzionerà totalmente offline.

FUNZIONI PRINCIPALI:
- Editor: Seleziona e trascina (Drag & Drop) campioni audio.
- Mixer: Genera mix casuali con log testuale dei brani.
- AI: Catalogazione automatica in 527 categorie sonore.

--- ENGLISH ---

HOW TO RUN:
1. Open the '$APP_NAME' folder.
2. Double-click the '$APP_NAME' executable (or run ./$APP_NAME from terminal).

IMPORTANT NOTE ON AI:
The first time you use the 'ANALYZE LIBRARY' function, the app will 
automatically download the PANNs neural model (approx. 312MB). 
An active internet connection is required. Once downloaded, 
tagging will work entirely offline.

MAIN FEATURES:
- Editor: Select and export (Drag & Drop) audio samples.
- Mixer: Generate random mixes with detailed text logs.
- AI: Automatic cataloging into 527 sound categories.

=======================================================
EOF

    # 5. Creazione dell'archivio per la distribuzione
    echo "🎁 Creazione archivio compresso: $ZIP_NAME..."
    cd dist
    tar -czf "../$ZIP_NAME" "$APP_NAME"
    cd ..

    echo "---------------------------------------------------"
    echo "✅ PROCESSO COMPLETATO CON SUCCESSO!"
    echo "📦 Archivio pronto: $ZIP_NAME"
    echo "---------------------------------------------------"
else
    echo "❌ Errore: La build è fallita."
    exit 1
fi
