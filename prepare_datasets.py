from pathlib import Path

import detect_face


def make_annotations(src):
    value = '0' if src.name.startswith('without') else '1'
    for type_ in ('*.jpg', '*.png'):
        for label in src.glob(type_):
            content = value + ' 0.500000 0.500000 1.000000 1.000000'
            label.parent.joinpath(label.stem +'.txt').open('wt').write(content)


def fix(src):
    for file in src.iterdir():
        content = file.open('rt').read()
        content = '1'+content[1:]
        file.open('wt').write(content)


def make_split(src, limit=0.8):
    train_img = src.parent.joinpath('train', 'images')
    val_img = src.parent.joinpath('val', 'images')
    train_lbl = src.parent.joinpath('train', 'labels')
    val_lbl = src.parent.joinpath('val', 'labels')
    train_img.mkdir(parents=True, exist_ok=True)
    val_img.mkdir(parents=True, exist_ok=True)
    train_lbl.mkdir(exist_ok=True)
    val_lbl.mkdir(exist_ok=True)

    count, quantity = 0, len(list(src.glob('*.jpg')))
    for file in list(src.glob('*.jpg')):
        count +=1
        img, lbl = file.name, file.stem+'.txt'
        if count/quantity < limit:
            file.rename(train_img.joinpath(img))
            file.parent.joinpath(lbl).rename(train_lbl.joinpath(lbl))
        else:
            file.rename(val_img.joinpath(img))
            file.parent.joinpath(lbl).rename(val_lbl.joinpath(lbl))

    
if __name__ == '__main__':
    CWD = Path.cwd()


    to_process = (
        
        (CWD.joinpath('glasses', 'src','with_glasses'), CWD.joinpath('hats', 'learn', 'with_glasses')),
        (CWD.joinpath('glasses', 'src','without_glassses'), CWD.joinpath('hats', 'learn', 'without_glasses')),
    )

    for src, dst in to_process:
        detect_face.process(src, dst)
        make_annotations(dst)
        make_split(dst)

    detect_face.process(CWD.joinpath('hats', 'data', 'with_hat'), CWD.joinpath('hats','data','processed','with_hat'))
    detect_face.process(CWD.joinpath('hats', 'data', 'without_hat'), CWD.joinpath('hats','data','processed','without_hat'))
    fix(Path.home().joinpath(r'Рабочий стол', 'labels_1'))


    INPUT_DIR = CWD.parent.joinpath('filter_image') # папка JPG/JPEG/PNG изображениями
    
