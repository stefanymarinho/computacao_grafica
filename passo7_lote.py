"""
Passo 7 — Processar Múltiplas Imagens em Lote
Aula 03 — Análise de Placas de Carros com OpenCV
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime

from placas_utils import criar_placa_sintetica, pipeline_leitura_placa, SAIDA_DIR, CARROS_DIR


def processar_lote(pasta_imagens, saida_json=None):
    if saida_json is None:
        saida_json = os.path.join(SAIDA_DIR, "registros.json")
    """
    Processa todas as imagens de uma pasta e retorna log das placas.
    """
    extensoes = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    arquivos  = [
        f for f in os.listdir(pasta_imagens)
        if f.lower().endswith(extensoes)
    ]

    print(f"Encontradas {len(arquivos)} imagens em '{pasta_imagens}'")

    registros = []

    for i, arquivo in enumerate(sorted(arquivos)):
        caminho = os.path.join(pasta_imagens, arquivo)
        print(f"\n[{i+1}/{len(arquivos)}] Processando: {arquivo}")

        res = pipeline_leitura_placa(caminho, debug=False)

        registro = {
            "id":        i + 1,
            "arquivo":   arquivo,
            "horario":   datetime.now().isoformat(),
            "placa":     res["placa"],
            "valido":    res["valido"],
            "formato":   res["formato"],
            "confianca": round(res["confianca"], 3),
        }
        registros.append(registro)

        status = "✓" if res["valido"] else "?"
        print(f"  {status} Placa: {res['placa']}  |  Confiança: {res['confianca']:.0%}")

    with open(saida_json, "w", encoding="utf-8") as f:
        json.dump(registros, f, ensure_ascii=False, indent=2)

    total     = len(registros)
    validos   = sum(1 for r in registros if r["valido"])
    invalidos = total - validos

    print("\n" + "="*50)
    print("RELATÓRIO DO LOTE")
    print("="*50)
    print(f"Total processado  : {total}")
    print(f"Placas válidas    : {validos} ({validos/total:.0%})")
    print(f"Não identificadas : {invalidos} ({invalidos/total:.0%})")
    print(f"Log salvo em      : {saida_json}")

    return registros


# ── Criar pasta de teste com placas sintéticas ────────────────
placas_teste = ["ABC1234", "BRA2E23", "XYZ9W88", "DEF5678", "GHI3J45"]
for placa in placas_teste:
    img = criar_placa_sintetica(placa, largura=500, altura=150)

    fundo = np.full((300, 700, 3), (80, 80, 80), dtype=np.uint8)
    fundo[100:250, 150:650] = img

    ruido = np.random.randint(0, 25, fundo.shape, dtype=np.uint8)
    fundo = cv2.add(fundo, ruido)

    cv2.imwrite(os.path.join(CARROS_DIR, f"carro_{placa}.png"), fundo)

print(f"Geradas {len(placas_teste)} imagens de carros sintéticos em {CARROS_DIR}/\n")

# ── Processar o lote ──────────────────────────────────────────
registros = processar_lote(CARROS_DIR)

print("\nPasso 7 concluído com sucesso!")