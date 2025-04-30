from pathlib import Path

import cv2


#загружаем модели для определения пола и возраста
CWD = Path(__file__).parent
genderProto=CWD.joinpath('gender_deploy.prototxt')
genderModel=CWD.joinpath('gender_net.caffemodel')
ageProto=CWD.joinpath('age_deploy.prototxt')
ageModel=CWD.joinpath('age_net.caffemodel')

# настраиваем свет
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
# результаты работы нейросетей для пола и возраста
genderList=['Male ','Female']
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# запускаем нейросети по определению пола и возраста
genderNet=cv2.dnn.readNet(genderModel, genderProto)
ageNet=cv2.dnn.readNet(ageModel, ageProto)


def process(img):
    # получаем на этой основе новый бинарный пиксельный объект
    blob=cv2.dnn.blobFromImage(img, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
    # отправляем его в нейросеть для определения пола
    genderNet.setInput(blob)
    # получаем результат работы нейросети
    genderPreds=genderNet.forward()
    # выбираем пол на основе этого результата
    gender=genderList[genderPreds[0].argmax()]
    # делаем то же самое для возраста
    ageNet.setInput(blob)
    agePreds=ageNet.forward()
    age=ageList[agePreds[0].argmax()]

    return (age, gender)