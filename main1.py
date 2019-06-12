import cv2, dlib, sys
import numpy as np

scaler = 0.3  # 동영상 윈도우 크기 축소 비율
detector = dlib.get_frontal_face_detector()

'''
머신러닝으로 학습된 안명인식 데이타
https://github.com/davisking/dlib-models 에서 다운로드 가능함.
'''
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


'''
cap = cv2.VideoCapture(0)
파일 이름대신에 0을 넣으면 웹캠이 작동함
'''
#cap = cv2.VideoCapture('single_face.mp4')
cap = cv2.VideoCapture('multi_face.mp4')
#cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break

    #화면크기 조정
    img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
    org = img.copy()

    # 안면인식
    faces = detector(img)

    # 인식된 얼굴이 없는 경우 인식될 때 까지 계속 조사    
    if len(faces) <= 0:
        continue
    
    # 얼굴이 인식되었으면 인식된 얼굴만큼 반복 처리
    for face in faces:

        # 얼굴의 특징점 찾기 = 68개
        dlib_shape = predictor(img, face)
        shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

        # 화면에 하얀색 네모로 인식된 얼굴영역 출력
        img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()), color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # 특징점 68개 출력
        for s in shape_2d:
            cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):  # 이걸해야 동영상이 제대로 보임, q 누르면 종료
        sys.exit(1)