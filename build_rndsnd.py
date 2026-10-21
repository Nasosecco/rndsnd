import PyInstaller.__main__
import os
import shutil
import tarfile

# --- CONFIGURAZIONE ---
APP_NAME = "rndsnd_v0.8.0"
OUTPUT_DIR = "dist"
ARCHIVE_NAME = f"{APP_NAME}_linux.tar.gz"

# --- 1. PULIZIA ---
print("Pulizia build precedenti...")
if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
if os.path.exists("build"): shutil.rmtree("build")

# --- 2. COMPILAZIONE ---
print(f"Inizio compilazione di {APP_NAME}...")

# Definizione dei dati da includere (Source : Destination)
datas = [
    ('rndsnd_splash.png', '.'),
    ('rndsnd_icon.png', '.')
]

# Configurazione PyInstaller
args = [
    'app_desktop.py',                  
    f'--name={APP_NAME}',                                     
    '--windowed',                      
    '--clean',                         
    '--hidden-import=sklearn.utils._cython_blas',
    '--hidden-import=sklearn.neighbors.typedefs',
    '--hidden-import=sklearn.neighbors.quad_tree',
    '--hidden-import=sklearn.tree',
    '--hidden-import=sklearn.tree._utils',
    '--collect-all=librosa',           
    '--collect-all=panns_inference',   
    '--collect-all=torch'              
]

# Aggiungi i dati con il separatore corretto per il sistema operativo
sep = ';' if os.name == 'nt' else ':'
for src, dest in datas:
    args.append(f'--add-data={src}{sep}{dest}')

# Lancia PyInstaller
PyInstaller.__main__.run(args)

# --- 3. PACCHETTIZZAZIONE (.TAR.GZ) ---
print("Creazione archivio tar.gz...")

# Percorso completo dell'eseguibile appena creato
exe_path = os.path.join(OUTPUT_DIR, APP_NAME)
if os.name == 'nt': exe_path += ".exe"

# Percorso dell'archivio finale
tar_path = os.path.join(OUTPUT_DIR, ARCHIVE_NAME)

if os.path.exists(exe_path):
    with tarfile.open(tar_path, "w:gz") as tar:
        # Aggiungi l'eseguibile
        print(f"   + Aggiungo: {APP_NAME}")
        tar.add(exe_path, arcname=APP_NAME)
        
        # Aggiungi il Readme (Versione TXT)
        if os.path.exists("README.txt"):
            print("   + Aggiungo: README.txt")
            tar.add("README.txt", arcname="README.txt")
        else:
            print("  ATTENZIONE: README.txt non trovato!")
            
        # Aggiungi la Licenza (se esiste)
        if os.path.exists("LICENSE"):
            print("   + Aggiungo: LICENSE")
            tar.add("LICENSE", arcname="LICENSE")
            
    print(f"\n SUCCESSO! Archivio pronto: {tar_path}")
    print("   Ora puoi caricare questo file sulle Release di GitHub.")
else:
    print("\n ERRORE: L'eseguibile non è stato creato correttamente.")
