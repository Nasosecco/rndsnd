#!/bin/bash
read -p "Inserisci la descrizione o il numero della versione (es. v1.0): " versione

# Prende TUTTI i file modificati, aggiunti o eliminati
git add . 

git commit -m "Rilascio versione: $versione"
git push

echo "✅ Versione '$versione' inviata a GitHub."
echo "⏳ I server di GitHub stanno ora creando automaticamente le nuove build!"
