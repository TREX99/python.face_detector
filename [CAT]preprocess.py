###########################################################################################
#  ./data/CAT_** 폴더에 있는 jpg 고양이 이미지를 일정한 크기로 짤라
# Kaggle https://www.kaggle.com/crawford/cat-dataset 데이타 참조
# https://github.com/kairess/cat_hipsterizer code 참조
###########################################################################################
import random
import dlib, cv2, os
import pandas as pd
import numpy as np


IMG_SIZE = 224  # 이미지 재조정 크기 (정사각형)
dir_name = "CAT_00"
base_path = "./data/%s" % dir_name
file_list = sorted(os.listdir(base_path))

random.shuffle(file_list)

dataset = {
  'imgs': [],
  'lmks': [],
  'bbs': []
}

def resize_img(im):
    # im(이미지)는 3차원 배열의 이미지 높이, 폭, 배열 형태로 들어 있음 
    # 변경 전의 높이, 폭을 저장
    old_size = im.shape[:2] # old_size is in (height, width) format
    print(im.shape)
    print(type(im.shape))
    print(type(im))

    # 재조정할 이미지크기 224 를 높이 또는 폭 중에 큰 값으로 나누어 재조정 비율을 구한다.
    ratio = float(IMG_SIZE) / max(old_size)

    # 구해진 재조정 비율을 높이, 폭에 곱한다.
    new_size = tuple([int(x*ratio) for x in old_size])

    # resize() 는 width, height 형태로 넣어야 한다.
    im = cv2.resize(im, (new_size[1], new_size[0] ))

    # 고정크기(IMG_SIZE:224)에서 변경된 크기의 너비(width)를 뺀다.
    # 폭보다 너비가 큰 경우 이값은 0이다.
    delta_w = IMG_SIZE - new_size[1]

    # 고정크기(IMG_SIZE:224)에서 변경된 크기의 높이(height)를 뺀다.
    # 너비보다 폭이 큰 경우 이값은 0이다.
    delta_h = IMG_SIZE - new_size[0]

    # 고정크기(IMG_SIZE:224)에서 변경된 크기의 높이(height)를 뺀값을 2로 나눈 몫을 구한다.
    # 고정크기(IMG_SIZE:224)에서 변경된 크기의 높이(height)를 뺀값을 변경된 높이(height)로 뺀 값을 2로 나눈 몫을 구한다.
    # 폭이 넓인 경우 위와 아래 동일한 간격으로 이미지를 줄이기 위해 
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)

    # 고정크기(IMG_SIZE:224)에서 변경된 크기의 너비(width)를 뺀값을 2로 나눈 몫을 구한다.
    # 고정크기(IMG_SIZE:224)에서 변경된 크기의 너비(width)를 뺀값을 변경된 너비(width)로 뺀 값을 2로 나눈 몫을 구한다.
    # 너비가 넓인 경우 위와 아래 동일한 간격으로 이미지를 줄이기 위해 
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # 이미지 주변에 테두리를 생성하고 가운데 영역에 이미지를 넣는다. 
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return new_im, ratio, top, left


#########################################################################################
# .cat 파일에는 랜드마크 갯수(9개) 및 위치(x, y 좌표쌍으로 18개)가 기록되어 있다.
#  예 : 9 175 160 239 162 199 199 149 121 137 78 166 93 281 101 312 96 296 133 
for f in file_list:
    ## 파일명에 .cat 이 있는 경우만 처리하고 없으면 다음 파일을 검사
    if '.cat' not in f:
        continue

    # landmark 18개(9쌍)을 읽는다.
    pd_frame = pd.read_csv(os.path.join(base_path, f), sep = ' ', header = None)
    
    # dataframe 자료형을 numpy matrix로 변형한다.
    # 첫번재 숫자는 좌표를 갯수(9개)를 의미하기 때문에 포함시키지 않는다. 아울러 맨 끝의 NaN(결측치)도 제외되도록한다.
    # 그렇지는 않겠지만 만약을 대비하여 결측치가 중간에 발생하는 경우를 포함하여 완전하게 처리를 하기 위해서는 df.dropna(axis=1)로 결측치를 제거하는 것이 좋다.
    #     landmarks = pd_frame.dropna(axis=1)
    #     landmarks = landmarks.values[0][1:]
    landmarks = pd_frame.values[0][1 : -1]

    # 변형된 numpy matrix를 9행 2열로 변형한다. 
    landmarks = landmarks.reshape(-1, 2)
    original_landmarks = landmarks.copy().astype(np.int)

    # 좌표쌍이 들어있는 파일명에서 파일명과 확장자를 구분한다.
    # 파일명은 실제 존재하는 이미지 파일명이다.
    img_filename, ext = os.path.splitext(f)

    # 실제 존재하는 이미지 파일명을 이용하여 파일내용을 읽어온다.
    img = cv2.imread(os.path.join(base_path, img_filename))
    original_img = img.copy()

    # 이미지 크기를 조정하고 landmarks 를 재조정한다.
    img, ratio, top, left = resize_img(img)
    landmarks = ((landmarks * ratio) + np.array([left, top])).astype(np.int)
    print(landmarks)

    # 가장 높은 좌표와 가장 낮은 좌표를 기록한다. 랜드마크가 있는 범위 (바운딩 박스)
    bb = np.array([np.min(landmarks, axis=0), np.max(landmarks, axis=0)])
    print(bb)

    dataset['imgs'].append(img)  # 이미지
    dataset['lmks'].append(landmarks.flatten()) # 랜드마크
    dataset['bbs'].append(bb.flatten())  # 랜드마크 최대, 최소 위치 (바운딩 박스)

    for l in landmarks:
      cv2.circle(img, center=tuple(l), radius=1, color=(255, 255, 255), thickness=2)

    for l in original_landmarks:
      cv2.circle(original_img, center=tuple(l), radius=1, color=(255, 255, 255), thickness=2)

    for l in bb:
      cv2.circle(img, center=tuple(l), radius=5, color=(125, 34, 56), thickness=3)

    cv2.imshow('img', img)
    cv2.imshow('original_img', original_img)
    if cv2.waitKey(0) == ord('q'):
      break

np.save('./data/%s.npy' % dir_name, np.array(dataset))





