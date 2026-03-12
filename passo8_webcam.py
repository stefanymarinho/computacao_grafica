"""
Passo 8 — Detecção em Tempo Real (Webcam / Vídeo)
Aula 03 — Análise de Placas de Carros com OpenCV

Uso:
    python passo8_webcam.py            # webcam padrão
    python passo8_webcam.py video.mp4  # arquivo de vídeo

Controles:
    q  — sair
    s  — salvar frame atual
"""

import sys
import cv2
import numpy as np
import time
import re

from placas_utils import validar_placa


def detectar_ao_vivo(fonte=0, largura=1280, altura=720):
    """
    fonte: 0 = webcam, ou caminho de vídeo '/tmp/video.mp4'
    """
    cap = cv2.VideoCapture(fonte)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  largura)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)

    if not cap.isOpened():
        print(f"ERRO: não foi possível abrir a fonte: {fonte}")
        return

    ultima_placa = ""
    historico    = []
    fps_anterior = time.time()

    print("Câmera iniciada. Pressione 'q' para sair, 's' para salvar frame.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Fim do vídeo ou erro na câmera.")
            break

        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_num % 3 == 0:
            cinza  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur   = cv2.GaussianBlur(cinza, (5, 5), 0)
            bordas = cv2.Canny(blur, 30, 90)

            contornos, _ = cv2.findContours(
                bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            H, W = frame.shape[:2]

            for contorno in sorted(contornos, key=cv2.contourArea, reverse=True)[:8]:
                x, y, w, h = cv2.boundingRect(contorno)
                proporcao   = w / h if h > 0 else 0

                if not (2.0 <= proporcao <= 6.0): continue
                if w * h < W * H * 0.005:         continue

                cv2.rectangle(frame, (x,y), (x+w,y+h), (200, 100, 0), 1)

                roi = frame[max(0,y-3):y+h+3, max(0,x-3):x+w+3]
                if roi.size == 0: continue

                try:
                    import pytesseract
                    cinza_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    cinza_roi = cv2.resize(cinza_roi, None, fx=2, fy=2)
                    thresh    = cv2.threshold(cinza_roi, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    config    = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    texto     = pytesseract.image_to_string(thresh, config=config)
                    texto     = re.sub(r'[^A-Z0-9]', '', texto.upper())

                    valido, _, _ = validar_placa(texto)
                    if valido and len(texto) >= 7:
                        historico.append(texto)
                        if len(historico) > 5:
                            historico.pop(0)
                        ultima_placa = max(set(historico), key=historico.count)

                        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,220,0), 3)
                        cv2.putText(frame, ultima_placa,
                                    (x, y-10), cv2.FONT_HERSHEY_DUPLEX,
                                    1.0, (0, 220, 0), 2)
                except Exception:
                    pass

        agora = time.time()
        fps   = 1.0 / max(agora - fps_anterior, 0.001)
        fps_anterior = agora

        cv2.putText(frame, f"FPS: {fps:.0f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"Ultima placa: {ultima_placa or '---'}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("Detector de Placas — pressione Q para sair", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            break
        elif tecla == ord('s'):
            cv2.imwrite(f"/tmp/captura_{frame_num}.png", frame)
            print(f"Frame salvo: /tmp/captura_{frame_num}.png")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Última placa detectada na sessão: {ultima_placa}")


if __name__ == "__main__":
    fonte = sys.argv[1] if len(sys.argv) > 1 else 0
    # converter para int se for número (índice de câmera)
    try:
        fonte = int(fonte)
    except (ValueError, TypeError):
        pass
    detectar_ao_vivo(fonte=fonte)