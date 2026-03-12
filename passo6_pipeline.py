"""
Passo 6 — Pipeline Completo: Foto → Texto
Aula 03 — Análise de Placas de Carros com OpenCV
"""

import os
from placas_utils import pipeline_leitura_placa, SAIDA_DIR

res = pipeline_leitura_placa(os.path.join(SAIDA_DIR, "placa_teste.png"), debug=True)

print("\n" + "="*50)
print("RESULTADO DO PIPELINE")
print("="*50)
print(f"Placa detectada : {res['placa']}")
print(f"Válida          : {'Sim ✓' if res['valido'] else 'Não ✗'}")
print(f"Formato         : {res['formato']}")
print(f"Confiança       : {res['confianca']:.1%}")
print(f"Localização     : {res['bbox']}")

print("\nPasso 6 concluído com sucesso!")