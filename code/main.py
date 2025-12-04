import cv2
import matplotlib.pyplot as plt
import mediapipe as mp

img = cv2.imread("dataset/s01/30/IR/s01_30_IR_F_0001.png")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# 얼굴 bounding box 모델
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

# 눈 segmentation 그리기
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# cx, cy = 30, 70   # 학생이 잡은 center 값
# radius = 20         # 원하는 원 radius

# color = (0, 0, 255) # BGR 색 (여기서는 빨간색)
# thickness = 2         # 선 굵기

# cv2.circle(img, (cx, cy), radius, color, thickness)
# cv2.imwrite("result.png", img)