import pandas as pd
import numpy as np
import os
import glob

# 설정
PRED_PATH = "./Result_CSVs"
LABEL_PATH = "./dataset/label"
SUBJECTS = ["s01", "s02", "s03", "s04"] # 정답이 있는 샘플만 평가

def calculate_error():
    total_error = []
    
    print(f"{'Subject':<10} {'Type':<10} {'Modality':<10} {'Mean Error (px)':<15}")
    print("-" * 50)

    for sub in SUBJECTS:
        # 각 환경별 순회 (30, 50, VR)
        for env in ["30", "50", "VR"]:
            # 1. 정답 파일 로드
            # 라벨 파일명 규칙: s01_30_IR.csv
            label_file = os.path.join(LABEL_PATH, f"{sub}_{env}_IR.csv")
            if not os.path.exists(label_file): continue
            
            df_label = pd.read_csv(label_file)
            
            # 2. 예측 파일 로드
            pred_file = os.path.join(PRED_PATH, f"{sub}_{env}_IR.csv")
            if not os.path.exists(pred_file): 
                print(f"{sub} {env} 예측 파일 없음")
                continue
            
            df_pred = pd.read_csv(pred_file)

            # 파일명(FILENAME) 기준으로 병합 (순서 꼬임 방지)
            merged = pd.merge(df_label, df_pred, on="FILENAME", suffixes=('_GT', '_PRED'))
            
            # 3. 에러 계산 (Left Eye, Right Eye)
            errors = []
            
            for side in ["LEYE", "REYE"]:
                # GT와 Pred 좌표 가져오기
                gt_x = merged[f"{side}_CENTER_X_GT"]
                gt_y = merged[f"{side}_CENTER_Y_GT"]
                pred_x = merged[f"{side}_CENTER_X_PRED"]
                pred_y = merged[f"{side}_CENTER_Y_PRED"]
                
                # 유효한 데이터만 필터링 (GT가 0이 아니고, Pred도 0이 아닌 경우)
                # 눈을 감았거나(GT=0), 못 찾은 경우(Pred=0) 제외하고 정확도 측정
                valid_mask = (gt_x != 0) & (gt_y != 0) & (pred_x != 0) & (pred_y != 0)
                
                if valid_mask.sum() == 0: continue

                diff_x = gt_x[valid_mask] - pred_x[valid_mask]
                diff_y = gt_y[valid_mask] - pred_y[valid_mask]
                
                # 유클리드 거리 계산
                dist = np.sqrt(diff_x**2 + diff_y**2)
                errors.extend(dist.tolist())

            if errors:
                mean_error = np.mean(errors)
                total_error.extend(errors)
                print(f"{sub:<10} {env:<10} IR         {mean_error:.4f}")
            else:
                print(f"{sub:<10} {env:<10} IR         N/A (No valid samples)")

    print("-" * 50)
    print(f"Total Average Error: {np.mean(total_error):.4f} px")

if __name__ == "__main__":
    calculate_error()