"""
placas_utils.py — Funções compartilhadas da Aula 03
Computação Gráfica — Prof. Petros Barreto
"""

import os
import cv2
import numpy as np
import re

# ─────────────────────────────────────────────────────────────
# Diretórios de saída (relativos ao projeto)
# ─────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
SAIDA_DIR  = os.path.join(_BASE, "saida")          # imagens geradas
CARROS_DIR = os.path.join(_BASE, "saida", "carros") # lote de carros
DEBUG_DIR  = os.path.join(_BASE, "saida", "debug")  # imagens de debug

for _d in (SAIDA_DIR, CARROS_DIR, DEBUG_DIR):
    os.makedirs(_d, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Gerador de placa sintética
# ─────────────────────────────────────────────────────────────

def criar_placa_sintetica(texto="ABC1234", largura=400, altura=120):
    """
    Gera uma imagem de placa brasileira sintética.
    Retorna: imagem BGR (numpy array)
    """
    img = np.ones((altura, largura, 3), dtype=np.uint8) * 240
    cv2.rectangle(img, (5, 5), (largura-5, altura-5), (180, 0, 0), 4)
    cv2.rectangle(img, (5, 5), (largura-5, 30), (180, 80, 20), -1)
    cv2.putText(img, "BRASIL", (largura//2 - 35, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_DUPLEX, 2.4, 4)
    x0 = (largura - tw) // 2
    cv2.putText(img, texto, (x0, altura - 25),
                cv2.FONT_HERSHEY_DUPLEX, 2.4, (10, 10, 10), 5)
    return img


# ─────────────────────────────────────────────────────────────
# Pré-processamento
# ─────────────────────────────────────────────────────────────

def pre_processar(img_bgr):
    """
    Converte BGR → Cinza → Blur → Canny.
    Retorna: (cinza, blur, bordas)
    """
    cinza  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(cinza, (5, 5), sigmaX=0)
    bordas = cv2.Canny(blur, threshold1=30, threshold2=90)
    return cinza, blur, bordas


# ─────────────────────────────────────────────────────────────
# Detecção de candidatos a placa
# ─────────────────────────────────────────────────────────────

def encontrar_candidatos_placa(img_bgr, debug=True):
    """
    Detecta candidatos à placa na imagem.
    Retorna lista de (contorno, bounding_rect).
    """
    cinza  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(cinza, (5, 5), 0)
    bordas = cv2.Canny(blur, 30, 90)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bordas = cv2.dilate(bordas, kernel, iterations=1)

    contornos, _ = cv2.findContours(
        bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    print(f"Total de contornos encontrados: {len(contornos)}")

    PROPORCAO_MIN = 2.0
    PROPORCAO_MAX = 6.5
    AREA_MIN      = 500

    candidatos = []
    img_debug  = img_bgr.copy()

    for contorno in contornos:
        perimetro = cv2.arcLength(contorno, closed=True)
        approx    = cv2.approxPolyDP(contorno, 0.02 * perimetro, closed=True)
        x, y, w, h = cv2.boundingRect(approx)
        area       = w * h
        proporcao  = w / h if h > 0 else 0

        if area < AREA_MIN:
            continue
        if not (PROPORCAO_MIN <= proporcao <= PROPORCAO_MAX):
            continue
        if w < img_bgr.shape[1] * 0.08:
            continue

        candidatos.append((contorno, (x, y, w, h)))

        if debug:
            cv2.rectangle(img_debug, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_debug, f"{proporcao:.1f}:1  {area}px",
                        (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 200, 0), 1)

    print(f"Candidatos com proporção de placa: {len(candidatos)}")

    if debug:
        path = os.path.join(DEBUG_DIR, "debug_contornos.png")
        cv2.imwrite(path, img_debug)
        print(f"Debug salvo: {path}")

    return candidatos


# ─────────────────────────────────────────────────────────────
# Extração e correção de perspectiva
# ─────────────────────────────────────────────────────────────

def extrair_placa(img_bgr, bbox):
    """
    Recorta a região da placa com margem.
    bbox: (x, y, w, h)
    """
    x, y, w, h = bbox
    margem = 5
    x1 = max(0, x - margem)
    y1 = max(0, y - margem)
    x2 = min(img_bgr.shape[1], x + w + margem)
    y2 = min(img_bgr.shape[0], y + h + margem)
    return img_bgr[y1:y2, x1:x2]


def corrigir_perspectiva(img_bgr, contorno):
    """
    Corrige a perspectiva da placa usando os 4 cantos do contorno.
    """
    perimetro = cv2.arcLength(contorno, True)
    approx    = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)

    if len(approx) != 4:
        print(f"Contorno tem {len(approx)} pontos (esperado 4). Usando bbox simples.")
        x, y, w, h = cv2.boundingRect(contorno)
        return img_bgr[y:y+h, x:x+w]

    pontos = approx.reshape(4, 2).astype(np.float32)
    soma   = pontos.sum(axis=1)
    diff   = np.diff(pontos, axis=1)

    rect = np.array([
        pontos[np.argmin(soma)],
        pontos[np.argmin(diff)],
        pontos[np.argmax(soma)],
        pontos[np.argmax(diff)],
    ], dtype=np.float32)

    W_SAIDA, H_SAIDA = 400, 130
    destino = np.array([
        [0,         0],
        [W_SAIDA-1, 0],
        [W_SAIDA-1, H_SAIDA-1],
        [0,         H_SAIDA-1],
    ], dtype=np.float32)

    M    = cv2.getPerspectiveTransform(rect, destino)
    warp = cv2.warpPerspective(img_bgr, M, (W_SAIDA, H_SAIDA))
    return warp


# ─────────────────────────────────────────────────────────────
# OCR
# ─────────────────────────────────────────────────────────────

def ocr_tesseract(img_placa):
    """
    Aplica OCR com Tesseract. Retorna (texto, imagem_thresh).
    """
    import pytesseract

    cinza = cv2.cvtColor(img_placa, cv2.COLOR_BGR2GRAY)
    cinza = cv2.resize(cinza, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    thresh = cv2.adaptiveThreshold(
        cinza, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11, C=2
    )
    config = (
        "--psm 7 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    texto = pytesseract.image_to_string(thresh, config=config)
    texto = texto.strip().replace(" ", "").replace("\n", "")
    return texto, thresh


def ocr_easyocr(img_placa):
    """
    Aplica OCR com EasyOCR. Retorna texto detectado.
    """
    import easyocr

    reader     = easyocr.Reader(['en'], gpu=False, verbose=False)
    resultados = reader.readtext(img_placa)

    texto_completo = ""
    for (bbox, texto, confianca) in resultados:
        print(f"  Detectado: '{texto}' (confiança: {confianca:.2%})")
        texto_completo += texto

    return texto_completo.strip().upper()


# ─────────────────────────────────────────────────────────────
# Validação de placa brasileira
# ─────────────────────────────────────────────────────────────

PADRAO_ANTIGO   = re.compile(r'^[A-Z]{3}\d{4}$')
PADRAO_MERCOSUL = re.compile(r'^[A-Z]{3}\d[A-Z]\d{2}$')


def validar_placa(texto):
    """
    Valida formato de placa brasileira.
    Retorna: (válido: bool, formato: str, texto_limpo: str)
    """
    limpo = re.sub(r'[^A-Z0-9]', '', texto.upper())

    if PADRAO_ANTIGO.match(limpo):
        return True, "Padrão Antigo (ABC1234)", limpo
    if PADRAO_MERCOSUL.match(limpo):
        return True, "Mercosul (BRA2E23)", limpo

    return False, "Formato inválido", limpo


def limpar_texto_ocr(texto_bruto):
    """
    Corrige erros comuns do OCR em leitura de placas.
    """
    texto = texto_bruto.upper().strip()
    texto = re.sub(r'[^A-Z0-9]', '', texto)

    if len(texto) < 7:
        return texto

    correcao_letra = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G'}
    prefixo = ''.join(correcao_letra.get(c, c) for c in texto[:3])

    correcao_num = {'O': '0', 'I': '1', 'S': '5', 'B': '8', 'G': '6',
                    'Z': '2', 'T': '7', 'L': '1', 'Q': '0'}
    sufixo = ''.join(correcao_num.get(c, c) for c in texto[3:])

    return prefixo + sufixo


# ─────────────────────────────────────────────────────────────
# Pipeline completo
# ─────────────────────────────────────────────────────────────

def pipeline_leitura_placa(caminho_imagem, debug=False):
    """
    Pipeline completo: imagem → texto da placa.
    Retorna dict com {placa, valido, formato, confianca, bbox}.
    """
    import os

    resultado = {
        "placa": None, "valido": False,
        "formato": "não detectado", "confianca": 0.0, "bbox": None
    }

    img = cv2.imread(caminho_imagem)
    if img is None:
        print(f"ERRO: imagem não encontrada: {caminho_imagem}")
        return resultado

    H, W = img.shape[:2]
    print(f"Imagem carregada: {W}×{H} pixels")

    cinza  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(cinza, (5, 5), 0)
    bordas = cv2.Canny(blur, 30, 90)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bordas = cv2.dilate(bordas, kernel, iterations=1)

    if debug:
        cv2.imwrite(os.path.join(DEBUG_DIR, "1_cinza.png"),  cinza)
        cv2.imwrite(os.path.join(DEBUG_DIR, "2_blur.png"),   blur)
        cv2.imwrite(os.path.join(DEBUG_DIR, "3_bordas.png"), bordas)

    contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:15]

    img_anotada  = img.copy()
    melhor_placa = None
    melhor_conf  = 0.0

    for rank, contorno in enumerate(contornos):
        x, y, w, h = cv2.boundingRect(contorno)
        proporcao   = w / h if h > 0 else 0
        area        = w * h

        if not (2.0 <= proporcao <= 6.5): continue
        if area < (W * H * 0.005):        continue

        roi = img[max(0,y-3):y+h+3, max(0,x-3):x+w+3]
        if roi.size == 0: continue

        cinza_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        cinza_roi = cv2.resize(cinza_roi, None, fx=2, fy=2,
                               interpolation=cv2.INTER_CUBIC)
        thresh    = cv2.adaptiveThreshold(cinza_roi, 255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 11, 2)

        if debug:
            cv2.imwrite(os.path.join(DEBUG_DIR, f"4_roi_{rank}.png"),    roi)
            cv2.imwrite(os.path.join(DEBUG_DIR, f"5_thresh_{rank}.png"), thresh)

        texto = ""
        # Tentar Tesseract primeiro
        try:
            import pytesseract
            config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            texto  = pytesseract.image_to_string(thresh, config=config).strip()
        except Exception:
            pass

        # Fallback: EasyOCR (não precisa de instalação de sistema)
        if not texto:
            try:
                import easyocr
                _reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                _candidatos = []
                for (_, _txt, _conf) in _reader.readtext(roi):
                    _limpo = re.sub(r'[^A-Z0-9]', '', _txt.upper())
                    _ok, _, _ = validar_placa(_limpo)
                    # priorizar textos que são placas válidas
                    _candidatos.append((_ok, _conf, _limpo))
                if _candidatos:
                    _candidatos.sort(key=lambda t: (t[0], t[1]), reverse=True)
                    texto = _candidatos[0][2]
            except Exception:
                texto = f"ROI_{rank}"

        texto = re.sub(r'[^A-Z0-9]', '', texto.upper())
        valido, formato, _ = validar_placa(texto)

        score_prop = 1.0 - abs(proporcao - 3.5) / 3.5
        score_val  = 1.0 if valido else 0.3
        score      = score_val * 0.6 + score_prop * 0.4

        cor_box = (0, 200, 0) if valido else (0, 100, 200)
        cv2.rectangle(img_anotada, (x,y), (x+w,y+h), cor_box, 2)
        cv2.putText(img_anotada, f"{texto} ({score:.2f})",
                    (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_box, 2)

        if score > melhor_conf:
            melhor_conf  = score
            melhor_placa = {
                "placa": texto, "valido": valido, "formato": formato,
                "confianca": score, "bbox": (x, y, w, h)
            }

    if melhor_placa:
        resultado.update(melhor_placa)

    path_resultado = os.path.join(SAIDA_DIR, "resultado_final.png")
    cv2.imwrite(path_resultado, img_anotada)
    print(f"Resultado final salvo: {path_resultado}")
    return resultado