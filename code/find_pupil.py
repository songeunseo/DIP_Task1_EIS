# Otsu Thresholding & Mopology to find the pupil area

import cv2
import numpy as np

def find_pupil(roi_img):
    """
    잘린 눈 이미지(roi_img)를 받아 동공의 중심(cx, cy)과 반지름(radius)을 반환합니다.
    (못 찾으면 None 반환)
    """
    if roi_img is None: return None
    
    # 1. Grayscale 변환 (이미 흑백일 수 있지만 안전하게)
    if len(roi_img.shape) == 3:
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_img

    # 2. 전처리: 가우시안 블러로 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. 이진화 (핵심): Otsu 알고리즘 사용 [cite: 5]
    # IR 영상에서 동공은 매우 어둡기 때문에, 임계값보다 낮은 부분을 잡습니다.
    # cv2.THRESH_BINARY_INV를 써서 동공(어두운 부분)을 흰색(255)으로 만듭니다.
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. 모폴로지 연산: 노이즈 제거 (Opening) & 구멍 메우기 (Closing) [cite: 107, 109]
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. 컨투어(덩어리) 검출 [cite: 125, 129]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # 6. 동공 선별 (Selection Strategy)
    # 가장 '동공스러운' 덩어리 찾기 (보통 크기가 적당하고 원형에 가까운 것)
    best_pupil = None
    max_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # (1) 면적 필터링: 너무 작거나(노이즈) 너무 큰(머리카락 등) 것 제외 [cite: 251]
        if area < 50 or area > 5000:  # 값은 데이터 보면서 튜닝 필요
            continue
        
        # (2) 원형성(Circularity) 검사: 1에 가까울수록 완벽한 원 [cite: 71, 250]
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        # 원형성이 0.6 이상인 것 중 가장 큰 것을 동공으로 가정
        if circularity > 0.6: 
            if area > max_score:
                max_score = area
                best_pupil = cnt

    # 7. 결과 반환
    if best_pupil is not None:
        # 최소 외접원(Minimum Enclosing Circle) 구하기
        (cx, cy), radius = cv2.minEnclosingCircle(best_pupil)
        return int(cx), int(cy), int(radius)
    
    return None