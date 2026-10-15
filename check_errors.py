print("1. Controllo importazioni base...")
import sys
import os
print("2. Controllo PySide6...")
from PySide6.QtWidgets import QApplication
print("3. Controllo Matplotlib...")
import matplotlib
os.environ['QT_API'] = 'pyside6'
import matplotlib.pyplot as plt
print("4. Controllo Database Manager...")
try:
    from database_manager import AudioScanner
    print("   OK: database_manager trovato.")
except Exception as e:
    print(f"   ERRORE in database_manager: {e}")

print("5. Controllo sintassi app_desktop.py...")
try:
    import app_desktop
    print("   OK: Sintassi corretta.")
except Exception as e:
    print(f"   ERRORE SINTASSI: {e}")
    import traceback
    traceback.print_exc()
