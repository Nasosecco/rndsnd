import PyInstaller.__main__
import os
import platform

separator = ';' if platform.system() == 'Windows' else ':'

# Rimuovere --onefile assicura che il pacchetto finale sia una cartella (One-Folder),
# così si avvia all'istante ed è più leggero da zippare e distribuire.
PyInstaller.__main__.run([
    'app_desktop.py',
    '--name=rndsnd_v0.11.0',
    '--onedir',
    '--windowed',
    '--noconfirm',
    f'--add-data=model/*{separator}model',
    # De-commenta e usa questa riga se hai l'icona e lo splash:
    # f'--add-data=rndsnd_splash.png{separator}.',
    # f'--add-data=rndsnd_icon.png{separator}.',
])
