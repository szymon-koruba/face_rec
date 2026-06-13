import cv2
from mtcnn import MTCNN
import numpy as np
import matplotlib.pyplot as plt

detector = MTCNN()
cap = cv2.VideoCapture(0)
maska = cv2.imread('48ce408a-728f-431d-b4b7-8087782d4068.jpg')
cv2.imshow("CircleFaceRing", maska)


l_e = [(870,243),(629,301),(380,302),(136,309)]
r_e = [(929,250),(711,301),(442,298),(212,298)]

r = 20

mask_end = np.zeros_like(maska, dtype=np.uint8)
for i in range(len(r_e)):
    cv2.circle(mask_end,r_e[i],r,255,-1)
    cv2.circle(mask_end,l_e[i],r,255,-1)

while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    h,w,_ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    for point in faces:
        keypoints = point['keypoints']
        eye_l = keypoints['left_eye']
        eye_r = keypoints['right_eye']

        patch_l = frame[eye_l[1] - r:eye_l[1]  + r, eye_l[0]  - r:eye_l[0]  + r,:].copy()
        patch_r = frame[eye_r[1] - r:eye_r[1] + r, eye_r[0] - r:eye_r[0] + r,:].copy()
        h, w, _ = patch_l.shape

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (w // 2, h // 2), r, 255, -1)

        patch_l = cv2.bitwise_and(patch_l, patch_l, mask=mask)
        patch_r = cv2.bitwise_and(patch_r, patch_r, mask=mask)

        for i in range(len(r_e)):
            if r_e[i][0] is not None and r_e[i][1] is not None:
                mask_end[(r_e[i][1]-r):(r_e[i][1]+r),(r_e[i][0]-r):(r_e[i][0]+r),:] = patch_r
                mask_end[(l_e[i][1] - r):(l_e[i][1] + r), (l_e[i][0] - r):(l_e[i][0] + r), :] = patch_l

        mask = np.all(mask_end == 0, axis=2)
        mask_end = mask_end.astype(np.uint8)
        mask_end[mask] = maska[mask]
        print(mask_end.shape)

        h, w = mask_end.shape[:2]

        cv2.namedWindow("CircleFaceRing", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("CircleFaceRing", mask_end)

    if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

