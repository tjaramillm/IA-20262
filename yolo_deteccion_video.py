"""
Detección de objetos con YOLOv8 sobre un video cargado desde disco.

Este script:
  1. Carga un modelo de detección de YOLOv8 (preentrenado en COCO, 80 clases comunes:
     personas, autos, animales, objetos cotidianos, etc.).
  2. Abre un archivo de video (mp4, avi, mov, ...) indicado por el usuario.
  3. Procesa el video frame por frame, corriendo la detección de objetos en cada uno.
  4. Dibuja las cajas delimitadoras, clases y confianza de cada objeto detectado.
  5. Muestra el video anotado en una ventana en vivo (opcional) y/o lo guarda
     como un nuevo archivo de video con las detecciones dibujadas.

Requisitos (instalar una sola vez):
    pip install ultralytics opencv-python

Ejecución:
    python yolo_deteccion_video.py

Controles (si MOSTRAR_VENTANA = True):
    - Presiona 'q' en la ventana de video para detener el procesamiento antes de que termine.
"""

import os
import time
import cv2
from ultralytics import YOLO


# --------------------------------------------------------------------------
# Configuración — ajusta estos valores según tu caso
# --------------------------------------------------------------------------
RUTA_VIDEO_ENTRADA = "video.mp4"          # <-- ruta del video que quieres cargar
RUTA_VIDEO_SALIDA = "video_detectado.mp4"  # <-- ruta donde se guardará el video anotado
MODELO = "yolov8n.pt"                      # yolov8n/s/m/l/x.pt (n = más rápido, x = más preciso)
CONFIANZA_MINIMA = 0.25                    # umbral de confianza para mostrar una detección
CLASES_A_DETECTAR = None                   # None = todas las clases; o lista de ids, ej: [0, 2] (persona, auto)
MOSTRAR_VENTANA = True                     # True = muestra el video en vivo mientras procesa
GUARDAR_VIDEO = True                       # True = guarda el resultado anotado en RUTA_VIDEO_SALIDA


def main():
    if not os.path.exists(RUTA_VIDEO_ENTRADA):
        print(f"No se encontró el video de entrada: {RUTA_VIDEO_ENTRADA}")
        return

    # --- Cargar el modelo YOLOv8 (descarga automática la primera vez) ---
    print(f"Cargando modelo {MODELO} ...")
    modelo = YOLO(MODELO)

    # --- Abrir el video de entrada ---
    captura = cv2.VideoCapture(RUTA_VIDEO_ENTRADA)
    if not captura.isOpened():
        print(f"No se pudo abrir el video: {RUTA_VIDEO_ENTRADA}")
        return

    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = captura.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {ancho}x{alto} px, {fps_video:.1f} FPS, {total_frames} frames")

    # --- Preparar el escritor de video de salida (si aplica) ---
    escritor = None
    if GUARDAR_VIDEO:
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        escritor = cv2.VideoWriter(RUTA_VIDEO_SALIDA, codec, fps_video, (ancho, alto))

    contador_frames = 0
    tiempo_inicio = time.time()

    while True:
        ok, frame = captura.read()
        if not ok:
            break  # fin del video

        contador_frames += 1

        # --- Inferencia de YOLOv8 sobre el frame actual ---
        resultados = modelo.predict(
            source=frame,
            conf=CONFIANZA_MINIMA,
            classes=CLASES_A_DETECTAR,
            verbose=False,
        )
        resultado = resultados[0]

        # Dibuja cajas, clases y confianza sobre el frame (devuelve BGR, mismo formato que OpenCV)
        frame_anotado = resultado.plot()

        # --- Mostrar progreso en consola cada 30 frames ---
        if contador_frames % 30 == 0 or contador_frames == total_frames:
            n_objetos = len(resultado.boxes) if resultado.boxes is not None else 0
            print(f"Frame {contador_frames}/{total_frames} — objetos detectados: {n_objetos}")

        if GUARDAR_VIDEO:
            escritor.write(frame_anotado)

        if MOSTRAR_VENTANA:
            cv2.imshow("Detección de objetos con YOLOv8 (presiona 'q' para detener)", frame_anotado)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Procesamiento detenido por el usuario.")
                break

    # --- Liberar recursos ---
    tiempo_total = time.time() - tiempo_inicio
    captura.release()
    if escritor is not None:
        escritor.release()
    cv2.destroyAllWindows()

    print(f"\nProcesamiento terminado: {contador_frames} frames en {tiempo_total:.1f} s "
          f"({contador_frames / tiempo_total:.1f} FPS promedio).")
    if GUARDAR_VIDEO:
        print(f"Video anotado guardado en: {RUTA_VIDEO_SALIDA}")


if __name__ == "__main__":
    main()
