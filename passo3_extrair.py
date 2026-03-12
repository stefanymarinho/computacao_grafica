"""
Passo 3 — Extrair e Corrigir a Perspectiva da Placa
Aula 03 — Análise de Placas de Carros com OpenCV
"""

import os
import cv2
from placas_utils import encontrar_candidatos_placa, extrair_placa, corrigir_perspectiva, SAIDA_DIR

PLACA_PATH = os.path.join(SAIDA_DIR, "placa_corrigida.png")
img = cv2.imread(PLACA_PATH)
if img is None:
    print("ERRO: execute passo1_preprocess.py primeiro.")
    exit(1)

candidatos = encontrar_candidatos_placa(img, debug=False)

if candidatos:
    contorno, bbox = candidatos[0]

    # Opção 1: simples (apenas recorte)
    roi_simples = extrair_placa(img, bbox)
    p1 = os.path.join(SAIDA_DIR, "placa_roi_simples.png")
    cv2.imwrite(p1, roi_simples)

    # Opção 2: corrigir perspectiva (mais robusto)
    roi_persp = corrigir_perspectiva(img, contorno)
    p2 = os.path.join(SAIDA_DIR, "placa_roi_perspectiva.png")
    cv2.imwrite(p2, roi_persp)

    print(f"ROI simples:      {roi_simples.shape}  → {p1}")
    print(f"ROI perspectiva:  {roi_persp.shape}  → {p2}")
else:
    print("Nenhum candidato encontrado. Ajuste os parâmetros de filtro.")

print("\nPasso 3 concluído com sucesso!")