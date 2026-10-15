from PySide6.QtWidgets import QApplication, QLabel
import sys
app = QApplication(sys.argv)
label = QLabel("Se vedi questa finestra, la grafica funziona!")
label.show()
sys.exit(app.exec())
