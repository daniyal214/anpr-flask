from flask import Flask, request, jsonify
import numpy as np
import cv2
import imutils
import torch
from fuzzywuzzy import process
from common import DetectMultiBackend
from general import (check_img_size, non_max_suppression, scale_coords)
from torch_utils import select_device
from paddleocr import PaddleOCR

######################################################################
replace = ['ICT-ISLAMABAD', 'PUNJAB', 'SINDH', 'BALOCHISTAN', 'JNJAB'
           'GOVT OF SINDH', 'ETEN', 'PESHAWAR', 'KHYBER PAKHTUNKHWA',
           'LASBELA', 'ET&NC', 'CET', 'FISNC', 'GOVT.OF SINDH', 'PORA']
rep_char22 = ['皖', '·', '国','皖']
spec_char = '`~!@#$%^&*()_-+={[}}|",\':\',\';\',\',<,>.?/'

weights = 'D:\Daniyal\AppPyQt\last.pt'
imgsz = (640,640)
dnn = False
S = 10
device = select_device('cpu')
half = device.type != 'cpu'
#####################################################################


app = Flask(__name__)


def letterbox(img, new_shape=(512, 512), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

@app.route('/')

def index():
    return 'Hello World'

@app.route('/predict', methods=['POST'])
def predict():
    model = DetectMultiBackend(weights=weights, device=device, dnn=False)

    img = request.files.get("file")

    # fs = request.files.get('snap')
    if img:
        # print('FileStorage:', fs)
        # print('filename:', fs.filename)

        # https://stackoverflow.com/questions/27517688/can-an-uploaded-image-be-loaded-directly-by-cv2
        # https://stackoverflow.com/a/11017839/1832058
        image = cv2.imdecode(np.frombuffer(img.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # img = img.read()
    # print(img)
    # img = Image.open(io.BytesIO(img))
    # npimg=np.array(img)
    # image=npimg.copy()
    # image = img

    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, \
                                           model.engine
    imgsz = check_img_size(224, s=stride)

    # colors = (0, 0, 255)
    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    img = letterbox(image, new_shape=640)[0]

    # # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    # print(img)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    #
    # # Inference
    pred = model(img, augment=False)
    pred = non_max_suppression(pred, 0.2, 0.5, classes=None, agnostic=False, max_det=300)

    final_text = []
    crop_l = []
    count = 0
    for i, det in enumerate(pred):
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
            for *xyxy, conf, cls in reversed(det):
                count += 1
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                c = int(cls)  # integer class
                label = f'{names[c]} {conf:.2f}'
                crop = image[y1:y2, x1:x2]
                crop = imutils.resize(crop, 400)
                crop_l.append(crop)

                ocr = PaddleOCR(use_angle_cls=True, lang="ch")
                result = ocr.ocr(crop, cls=True)
                # print(result)
                # print(len(result))

                if len(result) != 0:

                    print('RESULT', result)

                    questionList = [line[1][0] if line[1][0] not in replace else '' for line in result]

                    for str2 in questionList:
                        Ratios = process.extract(str2, replace)

                        for i in Ratios:
                            if i[1] >= 75:
                                if str2 in questionList:
                                    questionList.remove(str2)

                    st_lst = []
                    yr_lst = []
                    for i in questionList:
                        if '-' in i:
                            orig = i
                            st = i.split('-')
                            if len(st[-1]) == 2:
                                st_lst.append(st[0])
                                yr_lst.append(st[-1])
                            else:
                                st_lst.append(orig)
                        else:
                            st_lst.append(i)

                    for ch in rep_char22:
                        st_lst = [x.replace(ch, '-') for x in st_lst]

                    if len((st_lst[0]).strip()) == 4:
                        st_lst.remove(st_lst[0])

                    st_lst2 = []
                    yr_lst2 = []
                    for i in st_lst:
                        if '-' in i:
                            orig = i
                            st = i.split('-')
                            if len(st[-1]) == 2:
                                st_lst2.append(st[0])
                                yr_lst2.append(st[-1])
                            else:
                                st_lst2.append(orig)
                        else:
                            st_lst2.append(i)

                    text = ""
                    #  Convert an array to a string
                    for str in st_lst2:
                        text += str

                    if len(yr_lst) != 0:
                        text += ', {}'.format(yr_lst[0])

                    if len(yr_lst2) != 0:
                        text += ', {}'.format(yr_lst2[0])

                    text = text.strip(spec_char)
                    final_text.append(text)
                    print('Paddle2', text)
                #
                # else:
                #     final_text.append('Not Found')

            if len(final_text) != 0:
                print(final_text[0])

    num = final_text[0]
    result = {'plate number': num}

    return result


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
