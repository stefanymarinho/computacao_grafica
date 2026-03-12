"""
Passo 9 — Detecção de Placas em Videos do YouTube
Aula 03 — Análise de Placas de Carros com OpenCV

Uso:
    python passo9_youtube.py <url_do_youtube>
    

Dependências extras:
    pip install yt-dlp

Controles:
    q  — sair
    s  — salvar frame atual
    p  — pausar / retomar
"""

import sys
import subprocess
import cv2
import os
import numpy as np
import time
import re

from placas_utils import validar_placa, limpar_texto_ocr

# EasyOCR instanciado uma única vez para evitar lentidão por frame
_easyocr_reader = None

def _get_easyocr():
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            _easyocr_reader = easyocr.Reader(['en', 'pt'], gpu=False, verbose=False)
            print("EasyOCR carregado.")
        except Exception:
            pass
    return _easyocr_reader


def obter_url_stream(url_youtube: str) -> str:
    """
    Usa yt-dlp para extrair a URL direta do stream de vídeo.
    Prefere mp4/720p para equilibrar qualidade e velocidade.
    """
    # Usa o mesmo Python do venv para garantir que acha o yt-dlp instalado
    ytdlp_cmd = [sys.executable, "-m", "yt_dlp"]

    formatos = [
        "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]",
        "best[ext=mp4]/best",
    ]

    for fmt in formatos:
        try:
            resultado = subprocess.run(
                ytdlp_cmd + ["-g", "-f", fmt, url_youtube],
                capture_output=True, text=True, timeout=30
            )
            urls = resultado.stdout.strip().splitlines()
            if urls:
                # Quando há áudio + vídeo separados, yt-dlp retorna 2 URLs.
                # Para o OpenCV nos interessa apenas a URL de vídeo (primeira).
                print(f"Stream obtido ({fmt}).")
                return urls[0]
        except subprocess.TimeoutExpired:
            print("ERRO: timeout ao obter URL do YouTube.")
            sys.exit(1)

    print("ERRO: nao foi possivel obter stream do video.")
    sys.exit(1)


def analisar_frame(frame, historico: list, ultima_placa: str):
    """
    Detecta candidatos a placa no frame e tenta OCR.
    Retorna a ultima placa confirmada (pode ser a mesma de antes).
    """
    H, W = frame.shape[:2]
    cinza  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(cinza, (5, 5), 0)
    bordas = cv2.Canny(blur, 30, 90)

    # dilate fecha lacunas nos contornos, igual ao pipeline de imagem estática
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bordas = cv2.dilate(bordas, kernel, iterations=1)

    contornos, _ = cv2.findContours(
        bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contorno in sorted(contornos, key=cv2.contourArea, reverse=True)[:20]:
        x, y, w, h = cv2.boundingRect(contorno)
        proporcao   = w / h if h > 0 else 0

        if not (1.8 <= proporcao <= 7.0):
            continue
        if w * h < W * H * 0.001:
            continue

        # Desenha candidato em laranja
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 140, 255), 1)

        roi = frame[max(0, y-3):y+h+3, max(0, x-3):x+w+3]
        if roi.size == 0:
            continue

        texto = _ocr_roi(roi)
        if not texto:
            continue

        texto = limpar_texto_ocr(texto)
        valido, _, _ = validar_placa(texto)
        if valido and len(texto) >= 7:
            historico.append(texto)
            if len(historico) > 6:
                historico.pop(0)
            ultima_placa = max(set(historico), key=historico.count)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 220, 0), 3)
            cv2.putText(frame, ultima_placa,
                        (x, y - 10), cv2.FONT_HERSHEY_DUPLEX,
                        1.0, (0, 220, 0), 2)

    return ultima_placa


def _ocr_roi(roi):
    """Tenta Tesseract (threshold adaptativo); fallback para EasyOCR singleton."""
    cinza_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cinza_roi = cv2.resize(cinza_roi, None, fx=2, fy=2,
                           interpolation=cv2.INTER_CUBIC)
    # threshold adaptativo é mais robusto que Otsu em diferentes iluminações
    thresh = cv2.adaptiveThreshold(
        cinza_roi, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    try:
        import pytesseract
        config = (
            "--psm 7 "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        texto = pytesseract.image_to_string(thresh, config=config)
        resultado = re.sub(r'[^A-Z0-9]', '', texto.upper())
        if resultado:
            return resultado
    except Exception:
        pass

    # EasyOCR reutiliza a instância global (não recria a cada frame)
    reader = _get_easyocr()
    if reader:
        try:
            partes = []
            for (_, txt, conf) in reader.readtext(roi):
                if conf > 0.25:
                    partes.append(re.sub(r'[^A-Z0-9]', '', txt.upper()))
            return ''.join(partes)
        except Exception:
            pass

    return ""


def detectar_youtube(url_youtube: str):
    print(f"Obtendo stream de: {url_youtube}")
    stream_url = obter_url_stream(url_youtube)

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("ERRO: nao foi possivel abrir o stream de video.")
        return

    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps_video     = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"Video aberto — FPS: {fps_video:.1f}  Frames: {total_frames or 'desconhecido'}")
    print("Controles: q = sair | s = salvar frame | p = pausar/retomar")

    historico    = []
    ultima_placa = ""
    pausado      = False
    fps_anterior = time.time()
    placas_vistas = set()

    # Analisa 1 frame a cada N para performance
    INTERVALO = max(1, int(fps_video / 10))  # ~10 análises/s

    while True:
        if not pausado:
            ok, frame = cap.read()
            if not ok:
                print("Fim do video ou erro ao ler stream.")
                break

        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if not pausado and frame_num % INTERVALO == 0:
            ultima_placa = analisar_frame(frame, historico, ultima_placa)
            if ultima_placa and ultima_placa not in placas_vistas:
                placas_vistas.add(ultima_placa)
                print(f"[frame {frame_num:>6}] Nova placa detectada: {ultima_placa}")

        # HUD
        agora = time.time()
        fps   = 1.0 / max(agora - fps_anterior, 0.001)
        fps_anterior = agora

        status = "PAUSADO" if pausado else f"FPS: {fps:.0f}"
        cv2.putText(frame, status,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Ultima placa: {ultima_placa or '---'}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Total detectadas: {len(placas_vistas)}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

        cv2.imshow("Detector de Placas — YouTube | Q para sair", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            break
        elif tecla == ord('s'):
            path = f"/tmp/captura_yt_{frame_num}.png"
            cv2.imwrite(path, frame)
            print(f"Frame salvo: {path}")
        elif tecla == ord('p'):
            pausado = not pausado
            print("Pausado." if pausado else "Retomando.")

    cap.release()
    cv2.destroyAllWindows()

    print("\n--- Resumo da sessao ---")
    print(f"Total de placas unicas detectadas: {len(placas_vistas)}")
    for p in sorted(placas_vistas):
        print(f"  {p}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("Uso: python passo9_youtube.py <url_do_youtube>")
        sys.exit(1)

    detectar_youtube(sys.argv[1])