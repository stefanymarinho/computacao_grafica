"""
Passo 2 — Detectar Contornos e Localizar a Placa
Aula 03 — Análise de Placas de Carros com OpenCV
"""

import os
import cv2
from placas_utils import encontrar_candidatos_placa, SAIDA_DIR

# ── Testar na imagem da placa ─────────────────────────────────
PLACA_PATH = os.path.join(SAIDA_DIR, "placa_Petros.png")
img = cv2.imread(PLACA_PATH)
if img is None:
    print(f"ERRO: execute passo1_preprocess.py primeiro para gerar {PLACA_PATH}")
    exit(1)

candidatos = encontrar_candidatos_placa(img, debug=True)

for i, (contorno, (x, y, w, h)) in enumerate(candidatos):
    print(f"Candidato {i}: posição=({x},{y}), tamanho={w}×{h}, proporção={w/h:.2f}")

print("\nPasso 2 concluído com sucesso!")