from pathlib  import Path

import cv2
import numpy

from ultralytics import YOLO
from PIL import Image


MODEL_PATH = Path(__file__).parent.joinpath('yolov11l-face.pt')
MARGIN = 10
FACE_CLASS_ID = 0 # ID класса для лица человека в yolov-face.pt это 0                  
CONFIDENCE_THRESHOLD = 0.6 # Минимальная уверенность для детекции
DEFULT_MARGIN = 0.3, 0.3, 0.3, 0.3

TARGET_WIDTH = 500
TARGET_HEIGHT = 500

try:
    model = YOLO(MODEL_PATH)
except Exception as error:
    raise Exception(f"Критическая ошибка: Не удалось загрузить модель YOLO из {MODEL_PATH}: {error}")


def process(img, margin=DEFULT_MARGIN):

    try:
        results = model.predict(source=img, 
                                conf=CONFIDENCE_THRESHOLD, 
                                classes=[FACE_CLASS_ID], 
                                verbose=False) # verbose=False убирает лишний вывод YOLO
    except Exception as error:
        raise Exception(f"Ошибка при выполнении model.predict {error}")

    # получаем bounding boxes для нужного класса
    # results[0].boxes содержит информацию о всех обнаруженных объектах на первом изображении
    boxes = results[0].boxes.xyxy.cpu().numpy() # координаты [x1, y1, x2, y2]

    num_faces = len(boxes)

    # проверка количества найденных лиц/объектов
    if num_faces == 0:
        raise Exception(f"Объект класса {FACE_CLASS_ID} не найден на изображении")
    elif num_faces > 1:
        raise Exception(f"Найдено несколько объектов {num_faces}")

    # кадрирование (с отступом)
    # берем координаты единственного найденного объекта
    x1, y1, x2, y2 = boxes[0].astype(int)

    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        raise Exception(f"Ошибка: Некорректные размеры рамки (w={w}, h={h})")

    pad_x = (int(w * margin[0]), int(w * margin[2])) # отступ сторон
    pad_y = (int(h * margin[1]), int(h * margin[3])) # отступ сверху/снизу

    # координаты кадрирования в исходном изображении
    # c отступом 
    crop_x1 = max((x1 - pad_x[0]), 0)
    crop_y1 = max((y1 - pad_y[0]), 0)
    crop_x2 = min((x2 + pad_x[1]), img.shape[1]-1)
    crop_y2 = min((y2 + pad_y[1]), img.shape[0]-1)

    # обрезка по bound box
    cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]  

    if cropped_img.size == 0:
        raise Exception(f"  Ошибка: Не удалось вырезать область (возможно, объект у края)")
    
    h, w = crop_x2-crop_x1, crop_y2-crop_y1
    if h>w:
        fx, fy = TARGET_HEIGHT/h, TARGET_HEIGHT/h
    else:
        fx, fy = TARGET_WIDTH/w, TARGET_WIDTH/w

    cropped_img = cv2.resize(cropped_img, dsize=None, fx=fx, fy=fy, interpolation= cv2.INTER_AREA)

    if cropped_img.shape[0] == TARGET_HEIGHT:
        shift = (int((TARGET_WIDTH - cropped_img.shape[1])/2), 0)
    else:
        shift = (0, int((TARGET_HEIGHT - cropped_img.shape[0])/2))

    pil_img  = Image.fromarray(cropped_img)
    background = Image.new(pil_img.mode, size=(TARGET_WIDTH, TARGET_HEIGHT))
    background.paste(pil_img, shift)
    background.convert('RGB')
    return numpy.array(background)