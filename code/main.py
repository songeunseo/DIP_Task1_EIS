# main.py
import cv2
import os
import glob
import csv
import numpy as np

# 만든 모듈들 불러오기
from face_crop import FaceCropper
from eye_crop import EyeCropper
from find_pupil import find_pupil

# -----------------------------------------------------------
# 설정
# -----------------------------------------------------------
DATASET_PATH = "./dataset"
RESULT_PATH = "./Results"
CSV_PATH = "./Result_CSVs"
SUBJECTS = ["s01", "s02", "s03", "s04", "t01", "t02"]

# 객체 초기화
face_cropper = FaceCropper()
eye_cropper = EyeCropper()

if not os.path.exists(RESULT_PATH): os.makedirs(RESULT_PATH)
if not os.path.exists(CSV_PATH): os.makedirs(CSV_PATH)

def main():
    for sub in SUBJECTS:
        print(f"\nProcessing Subject: {sub} ==================")
        
        for env in ["30", "50", "VR"]:
            search_path = os.path.join(DATASET_PATH, sub, env, "IR", "*.png")
            file_list = sorted(glob.glob(search_path))
            if not file_list: continue
            
            # 저장 경로 설정
            save_dir = os.path.join(RESULT_PATH, sub, env, "IR")
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            
            csv_name = f"{sub}_{env}_IR.csv"
            csv_file = open(os.path.join(CSV_PATH, csv_name), 'w', newline='')
            writer = csv.writer(csv_file)
            writer.writerow(['FILENAME', 'LEYE_CENTER_X', 'LEYE_CENTER_Y', 'REYE_CENTER_X', 'REYE_CENTER_Y'])
            
            print(f"  Env: {env}, Files: {len(file_list)}")

            for idx, file_path in enumerate(file_list):
                file_name = os.path.basename(file_path)
                img = cv2.imread(file_path)
                if img is None: continue
                
                lx, ly, rx, ry = 0, 0, 0, 0
                vis_img = img.copy()

                # === [Type 1: 30cm, 50cm] ===
                if env in ["30", "50"]:
                    # 1단계: 얼굴 자르기
                    face_img, face_x, face_y = face_cropper.get_crop(img, file_name)
                    
                    if face_img.size != 0:
                        # 2단계: 눈 영역 자르기 (새로 만든 모듈 사용!)
                        reye_img, roff, leye_img, loff = eye_cropper.get_rois(face_img)
                        
                        # 3단계: 동공 찾기 & 좌표 복원
                        # (1) 왼쪽 눈 (LEYE)
                        l_res = find_pupil(leye_img)
                        if l_res:
                            lcx, lcy, lrad = l_res
                            # ★좌표 복원 공식★: 얼굴시작 + 눈ROI시작 + 동공로컬
                            lx = face_x + loff[0] + lcx
                            ly = face_y + loff[1] + lcy
                            cv2.circle(vis_img, (lx, ly), lrad, (0, 0, 255), 2)

                        # (2) 오른쪽 눈 (REYE)
                        r_res = find_pupil(reye_img)
                        if r_res:
                            rcx, rcy, rrad = r_res
                            rx = face_x + roff[0] + rcx
                            ry = face_y + roff[1] + rcy
                            cv2.circle(vis_img, (rx, ry), rrad, (0, 0, 255), 2)

                # === [Type 2: VR] ===
                else:
                    # VR은 이미 눈이므로 바로 Segmentation
                    res = find_pupil(img)
                    if res:
                        cx, cy, rad = res
                        cv2.circle(vis_img, (cx, cy), rad, (0, 0, 255), 2)
                        
                        if "LEYE" in file_name: lx, ly = cx, cy
                        elif "REYE" in file_name: rx, ry = cx, cy

                # === [결과 저장] ===
                writer.writerow([file_name, lx, ly, rx, ry])
                
                # 리포트용 이미지 샘플링 (10장마다 1장)
                if idx % 10 == 0:
                    cv2.imwrite(os.path.join(save_dir, file_name), vis_img)

            csv_file.close()

if __name__ == "__main__":
    main()