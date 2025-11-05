import cv2
import torch
import numpy as np
import pandas as pd
from typing import Tuple


# Настройки

CAMERA_INDEX = 0          
CONF_THRESHOLD = 0.4      
TARGET_CLASS_NAME = "cup" 


# Логика работы с моделью

def load_model(conf_threshold: float = CONF_THRESHOLD):
    model = torch.hub.load(
        'ultralytics/yolov5',
        'yolov5s',
        pretrained=True
    )
    model.conf = conf_threshold
    return model


# Векторная математика

def compute_geometry(
        frame_width: int,
        frame_height: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int
        ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], float]:
 
    cx_frame = frame_width / 2
    cy_frame = frame_height / 2

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    dx = cx - cx_frame
    dy = cy - cy_frame

    d = float(np.sqrt(dx**2 + dy**2))

    if d > 0:
        ux = dx / d
        uy = dy / d
    else:
        ux, uy = 0.0, 0.0

    return (cx, cy), (dx, dy), (ux, uy), d


# Отрисовка

def draw_overlay(
        frame,
        frame_center: Tuple[int, int],
        bbox: Tuple[int, int, int, int],
        cup_center: Tuple[int, int]
        ):
    cx_frame, cy_frame = frame_center
    x1, y1, x2, y2 = bbox
    cx, cy = cup_center

    cv2.circle(frame, (cx_frame, cy_frame), 5, (255, 0, 0), -1)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    cv2.arrowedLine(
        frame,
        (cx_frame, cy_frame),
        (cx, cy),
        (255, 0, 255),
        2,
        tipLength=0.1
    )


# Основной цикл

def run():
    model = load_model()

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось считать кадр")
            break

        h, w, _ = frame.shape
        cx_frame = w // 2
        cy_frame = h // 2

        results = model(frame)
        df: pd.DataFrame = results.pandas().xyxy[0]

        objects = df[df["name"] == TARGET_CLASS_NAME]

        cv2.circle(frame, (cx_frame, cy_frame), 5, (255, 0, 0), -1)

        for _, row in objects.iterrows():
            x1, y1 = int(row["xmin"]), int(row["ymin"])
            x2, y2 = int(row["xmax"]), int(row["ymax"])

            (cx, cy), (dx, dy), (ux, uy), d = compute_geometry(
                frame_width=w,
                frame_height=h,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2
            )

            print(f"{TARGET_CLASS_NAME} detectod:")
            print(f"  Bounding box: (x1={x1}, y1={y1}, x2={x2}, y2={y2})")
            print(f"  Center: (cx={cx:.1f}, cy={cy:.1f})")
            print(f"  Vector from frame center: (dx={dx:.1f}, dy={dy:.1f})")
            print(f"  Normalized: (ux={ux:.3f}, uy={uy:.3f})")
            print(f"  Distance: d={d:.1f}")
            print("-" * 40)

            draw_overlay(
                frame=frame,
                frame_center=(cx_frame, cy_frame),
                bbox=(x1, y1, x2, y2),
                cup_center=(int(cx), int(cy))
            )

        cv2.imshow("Cup detection", frame)

        key = cv2.waitKey(1)
        if key == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
