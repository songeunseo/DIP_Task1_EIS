import os
import cv2
import glob
import mediapipe as mp
import numpy as np

# -----------------------------------------------------------
# 1. 설정 (Configuration)
# -----------------------------------------------------------
DATASET_ROOT = "dataset"  # 데이터셋 루트 폴더
SUBJECTS = ["s01", "s02", "s03", "s04", "t01", "t02"] # 전체 대상 [cite: 955, 957]
DISTANCES = ["30", "50"]  # Type 1 (Face Crop 필요)
VR_TYPE = "VR"            # Type 2 (Crop 불필요)
MODALITY = "IR"           # 우리는 IR만 사용 [cite: 1476]

# -----------------------------------------------------------
# 2. Mediapipe 얼굴 검출기 초기화 (Type 1용)
# -----------------------------------------------------------
# 과제 PDF Page 5 참고 
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def get_face_crop(img):
    """
    Type 1 이미지를 받아 얼굴 영역을 Crop하여 반환합니다.
    """
    # Mediapipe 처리를 위해 BGR -> RGB 변환
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if results.detections:
        # 첫 번째 감지된 얼굴만 사용
        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box
        
        h, w, c = img.shape
        
        # 상대 좌표를 절대 좌표(픽셀)로 변환 [cite: 130, 131, 134, 135]
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        
        # 이미지 범위 벗어나지 않게 클리핑 (IndexError 방지)
        x = max(0, x)
        y = max(0, y)
        bw = min(bw, w - x)
        bh = min(bh, h - y)

        # 얼굴 영역 Crop [cite: 136]
        face_img = img[y:y+bh, x:x+bw]
        return face_img
    else:
        # 얼굴을 못 찾은 경우 (예외 처리), 원본 반환하거나 에러 처리
        print("  [Warning] No face detected. Using original image.")
        return img

# -----------------------------------------------------------
# 3. 메인 순회 로직
# -----------------------------------------------------------
def process_all_data():
    for sub in SUBJECTS:
        print(f"Processing Subject: {sub}...")
        
        # 처리해야 할 폴더 목록 (30, 50, VR)
        target_types = DISTANCES + [VR_TYPE] 
        
        for dtype in target_types:
            # 이미지 경로 패턴 생성 (예: dataset/s01/30/IR/*.png) [cite: 75, 76]
            # IR 이미지는 png 포맷입니다.
            search_path = os.path.join(DATASET_ROOT, sub, dtype, MODALITY, "*.png")
            img_files = sorted(glob.glob(search_path))
            
            if not img_files:
                print(f"  No files found in {search_path}")
                continue

            print(f"  Type: {dtype}, Count: {len(img_files)} frames")

            # 각 이미지 파일 순회
            for file_path in img_files:
                # 이미지 로드
                img = cv2.imread(file_path)
                if img is None: continue

                processed_img = None

                # --- [분기 처리] ---
                if dtype in DISTANCES: 
                    # Type 1 (30, 50): 얼굴 자르기 수행 [cite: 1369]
                    processed_img = get_face_crop(img)
                else:
                    # Type 2 (VR): 원본 그대로 사용 [cite: 1382]
                    processed_img = img
                
                # -------------------------------------------------------
                # [TODO]: 여기에 'IR Otsu 알고리즘'을 적용하면 됩니다.
                # 예: pupil_center = find_pupil(processed_img)
                # -------------------------------------------------------
                
                # 테스트용: 잘 잘렸나 확인하려면 아래 주석 해제 후 1장만 보고 break
                # cv2.imshow("Result", processed_img)
                # cv2.waitKey(0)
                # break 
            
            # (테스트용) 폴더 하나만 보고 싶으면 break
            # break 

if __name__ == "__main__":
    process_all_data()