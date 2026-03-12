"""
Passo 4 — OCR: Ler os Caracteres da Placa
Aula 03 — Análise de Placas de Carros com OpenCV
"""

import os
import cv2
from placas_utils import criar_placa_sintetica, ocr_tesseract, ocr_easyocr, SAIDA_DIR

img_placa = cv2.imread(os.path.join(SAIDA_DIR, "placa_roi_simples.png"))

if img_placa is None:
    # Criar placa de teste inline caso passo3 não tenha rodado
    print("ROI não encontrado — usando placa sintética direta.")
    img_placa = criar_placa_sintetica("BRA2E23")

print("=== Tesseract OCR ===")
try:
    texto_tess, thresh = ocr_tesseract(img_placa)
    print(f"Resultado: '{texto_tess}'")
    thresh_path = os.path.join(SAIDA_DIR, "placa_thresh.png")
    cv2.imwrite(thresh_path, thresh)
    print(f"Threshold salvo: {thresh_path}")
except Exception as e:
    print(f"Tesseract não disponível: {e}")
    print("Instale: brew install tesseract")

print("\n=== EasyOCR ===")
try:
    texto_easy = ocr_easyocr(img_placa)
    print(f"Resultado: '{texto_easy}'")
except Exception as e:
    print(f"EasyOCR não disponível: {e}")

print("\nPasso 4 concluído!")