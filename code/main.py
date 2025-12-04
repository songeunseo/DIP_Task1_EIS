# main.py
import cv2
import os
import glob
from face_crop import FaceCropper # 분리한 코드 임포트

# 설정
DATASET_PATH = "./dataset"
SUBJECTS = ["s01", "s02", "s03", "s04", "t01", "t02"] [cite: 15, 1264]

# 크로퍼 객체 생성
cropper = FaceCropper()

def main():
    for sub in SUBJECTS:
        # IR 이미지만 찾기 (Type 1: 30, 50 / Type 2: VR)
        # glob 패턴: dataset/s01/*/IR/*.png
        search_path = os.path.join(DATASET_PATH, sub, "*", "IR", "*.png")
        file_list = sorted(glob.glob(search_path))

        print(f"[{sub}] 처리 시작... 총 {len(file_list)}장")

        for file_path in file_list:
            file_name = os.path.basename(file_path)
            img = cv2.imread(file_path)
            
            if img is None: continue

            # --- [1. 전처리 단계] ---
            if "VR" in file_path:
                # Type 2 (VR): 자르기 없음 [cite: 426]
                roi_img = img
                offset_x, offset_y = 0, 0
            else:
                # Type 1 (30, 50): 얼굴 자르기 (F/L/R 자동 적용) [cite: 72]
                roi_img, offset_x, offset_y = cropper.get_crop(img, file_name)

            # --- [2. Segmentation 단계 (여기에 핵심 알고리즘 작성)] ---
            # pupil_x, pupil_y = find_pupil(roi_img) ... (작성 예정)

            # --- [3. 좌표 복원] ---
            # global_x = pupil_x + offset_x
            # global_y = pupil_y + offset_y

            # --- [4. 결과 저장] ---
            # ...