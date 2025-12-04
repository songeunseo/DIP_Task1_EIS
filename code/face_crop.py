# face_crop.py
import cv2
import mediapipe as mp
import numpy as np

class FaceCropper:
    def __init__(self):
        # Mediapipe 얼굴 검출기 초기화 [cite: 122]
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    def get_crop(self, img, file_name):
        """
        이미지와 파일명을 받아 얼굴 영역을 Crop하고,
        잘린 이미지와 원본 기준 오프셋(x, y)을 반환합니다.
        """
        # 1. 파일명에서 Viewpoint(F, L, R) 추출 [cite: 389]
        # 예: s01_30_IR_F_0001.png -> parts[3] == 'F'
        try:
            parts = file_name.split('_')
            viewpoint = parts[3] 
        except IndexError:
            viewpoint = 'F' # 예외 시 기본값

        # 2. 각도별 Padding 비율 설정 (튜닝 포인트!)
        # 측면(L, R)일수록 눈을 놓치지 않게 더 넓게 자르는 전략
        if viewpoint == 'F':
            padding_ratio = 0.1  # 상하좌우 10% 확장
        else: # L, R
            padding_ratio = 0.25 # 상하좌우 25% 확장 (측면은 박스가 불안정할 수 있음)

        # 3. Mediapipe 처리
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(img_rgb)

        if not results.detections:
            # 얼굴 못 찾으면 원본 그대로 반환 (offset 0, 0)
            return img, 0, 0

        # 첫 번째 감지된 얼굴 사용
        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box
        
        h, w, c = img.shape
        
        # 4. 좌표 계산 및 Padding 적용
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        # Padding 적용 (중심 기준 확장)
        pad_w = int(bw * padding_ratio)
        pad_h = int(bh * padding_ratio)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + bw + pad_w)
        y2 = min(h, y + bh + pad_h)

        # 5. 자르기 및 반환
        crop_img = img[y1:y2, x1:x2]
        
        # 오프셋(x1, y1)은 나중에 원본 좌표 복원할 때 필수!
        return crop_img, x1, y1