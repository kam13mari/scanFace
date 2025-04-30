import csv
import argparse

from pathlib import Path
from enum import StrEnum

import cv2

import detect_face
import age_n_gender

from process import process
from glasses import MODEL as GLASSES_MDL
from hats import MODEL as HATS_MDL
from masks import MODEL as MASKS_MDL
from turn import MODEL as TURN_MDL


class ModelType(StrEnum):
    glasses = 'glasses'
    hats = 'hats'
    masks = 'masks'
    turn = 'turn'


MODELS = {
    ModelType.glasses: GLASSES_MDL,
    ModelType.hats: HATS_MDL,
    ModelType.masks: MASKS_MDL,
    ModelType.turn: TURN_MDL
}

HEADERS = ('image_name', 'detection', *(item.name for item in ModelType))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='Age and gender detector',
        description='Process images for age and gender detection')
    parser.add_argument('indir', type=str, help='Input directory for processing')
    parser.add_argument('outdir', type=str, help='Output directory for processing result')
    args = parser.parse_args()

    INPUT = Path(args.indir)
    OUTPUT = Path(args.outdir)
    OUTPUT.mkdir(exist_ok=True)
    FILTER = OUTPUT.joinpath('filter.csv')
    RESULT = OUTPUT.joinpath('result.csv')

    with open(FILTER, 'w', newline='', encoding='utf-8') as filter_scv, \
        open(RESULT, 'w', encoding='utf-8') as result_csv:
        csv_writer = csv.writer(filter_scv)
        csv_writer.writerow(HEADERS)

        for image in list(INPUT.glob('*.[jp][pn]g')):
            record = dict(image_name=image.name, detection=False)
            for item in ModelType:
                record[item.name] = False

            fits = True

            try:
                img = cv2.imread(image)
            except Exception as error:
                print(error)
                fits = False

            try:
                data = detect_face.process(img)
                record['detection'] = True
            except Exception as error:
                print(error)
                fits = False


            for name, model in MODELS.items():
                detections = process(data, model)

                if detections:
                    record[name] = True
                    fits = False

            if fits:
                age, gender = age_n_gender.process(img)
                result_csv.write(f'{image.name} {age} {gender}\n')

            csv_writer.writerow(list(record.values()))
