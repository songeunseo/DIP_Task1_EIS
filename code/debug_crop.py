import cv2
import os
import glob
import random
import shutil
from face_crop import FaceCropper

# -----------------------------------------------------------
# 설정 (Configuration)
# -----------------------------------------------------------
DATASET_PATH = "./dataset"
OUTPUT_PATH = "./debug_crops_viewpoint"  # 결과 저장할 폴더
SUBJECTS = ["s01", "s02", "s03", "s04"]  # 샘플 대상
DISTANCES = ["30", "50"]                 # 거리 조건
VIEWPOINTS = ["F", "L", "R"]             # 확인하고 싶은 각도들
SAMPLES_PER_VIEW = 2                     # 각 조건당 뽑을 장수

# -----------------------------------------------------------
# 초기화
# -----------------------------------------------------------
if os.path.exists(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)
os.makedirs(OUTPUT_PATH)

cropper = FaceCropper()

def run_debug():
    print(f"🚀 각도별 디버깅 시작! 결과는 '{OUTPUT_PATH}' 폴더에 저장됩니다.\n")

    count = 0
    
    for sub in SUBJECTS:
        for dist in DISTANCES:
            print(f"--- Checking {sub} / {dist}cm ---")
            
            for view in VIEWPOINTS:
                # 1. 파일명 필터링 (핵심!)
                # 예: dataset/s01/30/IR/*_F_*.png 패턴으로 검색
                search_pattern = f"*_{view}_*.png"
                search_path = os.path.join(DATASET_PATH, sub, dist, "IR", search_pattern)
                
                file_list = glob.glob(search_path)
                
                if not file_list:
                    print(f"  [Warning] {view} 타입의 파일이 없습니다.")
                    continue
                
                # 2. 랜덤 샘플링
                samples = random.sample(file_list, min(len(file_list), SAMPLES_PER_VIEW))
                
                for file_path in samples:
                    file_name = os.path.basename(file_path)
                    img = cv2.imread(file_path)
                    
                    if img is None: continue

                    # 3. Face Crop 실행 (face_crop.py 로직)
                    crop_img, off_x, off_y = cropper.get_crop(img, file_name)
                    
                    # 4. 결과 저장
                    # 저장명 예: s01_30_L_crop.png (알아보기 쉽게)
                    save_name = f"{sub}_{dist}_{view}_{os.path.basename(file_path)}"
                    save_path = os.path.join(OUTPUT_PATH, save_name)
                    
                    cv2.imwrite(save_path, crop_img)
                    count += 1
            
    print(f"\n✅ 총 {count}장의 크롭 이미지가 생성되었습니다.")
    print(f"📂 '{OUTPUT_PATH}' 폴더를 열어서 눈이 잘리지 않았는지 확인하세요!")

if __name__ == "__main__":
    run_debug()