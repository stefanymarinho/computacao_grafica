"""
Passo 5 — Validar Formato de Placa Brasileira
Aula 03 — Análise de Placas de Carros com OpenCV
"""

from placas_utils import validar_placa, limpar_texto_ocr

placas_teste = [
    "ABC1234",    # antigo válido
    "BRA2E23",    # mercosul válido
    "ABC 1234",   # com espaço (deve ser corrigido)
    "BRA2E2",     # incompleto
    "BRAZE23",    # 4 letras (inválido)
    "0BC1234",    # OCR confundiu O→0 (deve corrigir)
    "BRA2E23\n",  # com newline do OCR
]

print("="*50)
print("VALIDAÇÃO DE PLACAS BRASILEIRAS")
print("="*50)
for placa_raw in placas_teste:
    limpo  = limpar_texto_ocr(placa_raw)
    valido, formato, final = validar_placa(limpo)
    status = "✓" if valido else "✗"
    print(f"{status}  '{placa_raw.strip()}' → '{final}'  [{formato}]")

print("\nPasso 5 concluído com sucesso!")