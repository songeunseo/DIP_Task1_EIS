# Otsu Thresholding & Mopology to find the pupil area

import cv2
import numpy as np

def find_pupil(roi_img, env_mode="normal"):
    """
    roi_img: 입력 이미지 (Gray or BGR)
    env_mode: "normal" (30/50cm용) 또는 "vr" (VR용)
    """
    if roi_img is None: return None
    
    # 1. Grayscale 변환
    if len(roi_img.shape) == 3:
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_img

    # 이미지 크기 확인
    h, w = gray.shape
    img_area = h * w
    
    # 2. 전처리 & 이진화 (공통)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ==========================================
    # [차별화 포인트 1] 마스킹 (VR 전용)
    # ==========================================
    if env_mode == "vr":
        mask = np.zeros((h, w), dtype=np.uint8)
        # VR은 중앙 50%만 집중적으로 봄 (안경테 제거)
        roi_x = int(w * 0.25)
        roi_y = int(h * 0.25)
        roi_w = int(w * 0.50)
        roi_h = int(h * 0.50)
        mask[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = 255
        binary = cv2.bitwise_and(binary, binary, mask=mask)
        
    # 3. 모폴로지 연산 (노이즈 제거)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4. 컨투어 검출
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None

    # 5. 최적의 동공 찾기
    best_pupil = None
    min_dist_from_center = float('inf')
    img_cx, img_cy = w // 2, h // 2

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # ==========================================
        # [차별화 포인트 2] 동적 면적 필터링
        # ==========================================
        # 30/50 이미지는 작고(60x60), VR은 큼(640x480). 고정값(50)을 쓰면 위험함.
        # 이미지 전체 면적의 0.5% ~ 30% 사이만 인정
        if area < (img_area * 0.005) or area > (img_area * 0.3):
            continue
        
        # 원형성 검사
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        # VR은 찌그러진 노이즈가 많으므로 기준을 좀 더 높임
        threshold_circ = 0.7 if env_mode == "vr" else 0.5
        if circularity < threshold_circ: continue

        # 종횡비 검사 (길쭉한 노이즈 제거)
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect_ratio = float(cw) / ch
        if not (0.6 < aspect_ratio < 1.6): continue

        # ==========================================
        # [차별화 포인트 3] 우선순위 결정
        # ==========================================
        # VR: "중앙에 있는 놈"이 진짜일 확률이 높음 (안경테가 외곽에 있으므로)
        # Normal: 그냥 원형성 좋고 큰 놈이 장땡일 수 있음
        
        M = cv2.moments(cnt)
        if M['m00'] == 0: continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        dist = np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2)
        
        if env_mode == "vr":
            # VR은 중앙 거리 우선
            if dist < min_dist_from_center:
                min_dist_from_center = dist
                best_pupil = cnt
        else:
            # 30/50은 그냥 적당히 중앙에 있고(너무 구석이 아니고) 큰 것
            # 여기서는 로직을 단순화해서 VR과 동일하게 가되,
            # 이미 EyeCropper가 눈을 중앙에 뒀으므로 중앙 거리 방식도 유효함
            if dist < min_dist_from_center:
                min_dist_from_center = dist
                best_pupil = cnt

    # 7. 결과 반환
    if best_pupil is not None:
        # 최소 외접원(Minimum Enclosing Circle) 구하기
        (cx, cy), radius = cv2.minEnclosingCircle(best_pupil)
        return int(cx), int(cy), int(radius)
    
    return None