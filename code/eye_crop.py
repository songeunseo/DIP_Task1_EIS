# eye_crop.py
class EyeCropper:
    def __init__(self):
        # 얼굴 내에서 눈이 위치하는 대략적인 비율 설정
        # 상단 20% ~ 55% 사이, 가로로는 절반을 나눔
        self.y_start_ratio = 0.20
        self.y_end_ratio = 0.55

    def get_rois(self, face_img):
        """
        잘린 얼굴 이미지를 받아 왼쪽 눈(LEYE), 오른쪽 눈(REYE) 영역과
        해당 영역의 시작 좌표(offset)를 반환합니다.
        """
        if face_img is None or face_img.size == 0:
            return None, None, None, None

        h, w = face_img.shape[:2]
        
        # 1. Y축 범위 설정 (이마와 입을 제외)
        y_start = int(h * self.y_start_ratio)
        y_end = int(h * self.y_end_ratio)
        
        # 2. X축 반으로 가르기
        # 이미지상 왼쪽 절반 -> 피험자의 오른쪽 눈 (REYE)
        # 이미지상 오른쪽 절반 -> 피험자의 왼쪽 눈 (LEYE)
        center_x = w // 2
        
        # 3. 자르기 (Crop)
        # REYE (Image Left)
        reye_img = face_img[y_start:y_end, 0:center_x]
        reye_offset = (0, y_start) # (x, y)
        
        # LEYE (Image Right)
        leye_img = face_img[y_start:y_end, center_x:w]
        leye_offset = (center_x, y_start) # (x, y)
        
        return reye_img, reye_offset, leye_img, leye_offset