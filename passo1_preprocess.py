"""
Passo 1 — Criar Imagem de Teste e Pré-processar
Aula 03 — Análise de Placas de Carros com OpenCV
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')   # backend sem janela (compatível com ambientes headless)
import matplotlib.pyplot as plt

from placas_utils import criar_placa_sintetica, pre_processar, SAIDA_DIR

# ── Gerar e salvar a placa ────────────────────────────────────
placa = criar_placa_sintetica("Petros")
PLACA_PATH = os.path.join(SAIDA_DIR, "placa_Petros.png")
cv2.imwrite(PLACA_PATH, placa)
print(f"Placa criada: {PLACA_PATH}")

# ── Pré-processar ─────────────────────────────────────────────
img = cv2.imread(PLACA_PATH)
cinza, blur, bordas = pre_processar(img)

# ── Visualizar lado a lado ────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
titles = ["Original (BGR→RGB)", "Cinza", "Gaussian Blur", "Canny Edges"]
imgs   = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cinza, blur, bordas]
cmaps  = [None, 'gray', 'gray', 'gray']

for ax, titulo, im, cmap in zip(axes, titles, imgs, cmaps):
    ax.imshow(im, cmap=cmap)
    ax.set_title(titulo, fontsize=12)
    ax.axis('off')

plt.tight_layout()
fig_path = os.path.join(SAIDA_DIR, "pipeline_pre_proc.png")
plt.savefig(fig_path, dpi=120)
print(f"Figura salva: {fig_path}")

print("\nPasso 1 concluído com sucesso!")
