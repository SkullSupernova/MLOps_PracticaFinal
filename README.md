

# Modo entrenamiento (respeta checkpoint previo si existe)
python src/main.py train

# Modo entrenamiento forzando reentrenamiento desde cero
python src/main.py train --force-train

# Modo evaluación sobre el test set
python src/main.py evaluate --checkpoint models/ResNet_Final_Combined.pth

# Modo OOD
python src/main.py ood --checkpoint models/ResNet_Final_Combined.pth

# Verificar versiones del entorno
python src/main.py --version