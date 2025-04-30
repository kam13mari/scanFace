from pathlib import Path

from ultralytics import YOLO


try:
    model_file = Path(__file__).parent.joinpath('models', 'weights', 'best.pt')
    MODEL = YOLO(model_file)
except Exception as error:
    print(f"Ошибка при загрузке модели {__name__}")
    MODEL = None