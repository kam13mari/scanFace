from pathlib import Path

def process(img, model):

    results = model.predict(img, conf=0.5, stream=False) #,save=True, save_txt=True, project= Path.cwd().joinpath('predict'))

    detections = [] 
    try:
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = model.names.get(class_id, f"Unknown ID: {class_id}")
                if class_id == 1: # Found
                    detection_data = {
                        'class_name': class_name,
                        'class_id': class_id,
                        'confidence': conf,
                        'bbox': xyxy
                    }
                    print(detection_data)
                    detections.append(detection_data)
    except AttributeError as error: print(error)

    return detections