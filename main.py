import cv2
from mtcnn import MTCNN
import numpy as np


detector = MTCNN()
ring = cv2.imread("kolo.png", cv2.IMREAD_UNCHANGED)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    mask = np.zeros((1080,1920),dtype=np.uint8)
    ekran = np.stack((mask,mask,mask),axis=-1)
    for f in faces:
        x, y, w, h = f["box"]
        size = max(w, h)+100
        cx = x + w//2
        cy = y + h//2
        x1 = cx - size//2
        y1 = cy - size//2
        x1 = max(0, x1)
        y1 = max(0, y1)
        if y1+size > frame.shape[0] or x1+size > frame.shape[1]:
            continue

        ring_resized = cv2.resize(ring, (size, size))
        ring_rgb = ring_resized[:, :, :3].astype(float)
        ring_alpha = ring_resized[:, :, 3].astype(float) / 255.0

        white_mask = np.all(ring_rgb >= 240, axis=-1)
        ring_alpha[white_mask] = 0

        alpha_exp = ring_alpha[..., None]

        roi = frame[y1:y1 + size, x1:x1 + size].astype(float)

        out = (roi * (1 - alpha_exp) + ring_rgb * alpha_exp).astype(np.uint8)

        frame[y1:y1 + size, x1:x1 + size] = out

    H, W, _ = ekran.shape
    h1, w1, _ = frame.shape

    y1 = (H - h1) // 2
    y2 = y1 + h1
    x1 = (W - w1) // 2
    x2 = x1 + w1

    ekran[y1:y2, x1:x2] = frame
    cv2.imshow("CircleFaceRing", ekran)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()