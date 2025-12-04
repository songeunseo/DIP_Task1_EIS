import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# 설정
PRED_PATH = "./Result_CSVs"
LABEL_PATH = "./dataset/label"
SUBJECTS = ["s01", "s02", "s03", "s04"] 

def visualize_error_distribution(error_data):
    """
    에러 분포를 시각화하는 함수
    (0,0)이 정답(GT) 위치이고, 점들이 예측(Pred)이 떨어진 상대적 위치입니다.
    """
    plt.figure(figsize=(10, 8))
    
    # 30, 50cm 데이터 (파란색)
    normal_x = [e['x'] for e in error_data if e['type'] != 'VR']
    normal_y = [e['y'] for e in error_data if e['type'] != 'VR']
    plt.scatter(normal_x, normal_y, c='blue', alpha=0.3, s=5, label='Normal (30/50cm)')

    # VR 데이터 (빨간색) - 문제의 원흉을 따로 표시
    vr_x = [e['x'] for e in error_data if e['type'] == 'VR']
    vr_y = [e['y'] for e in error_data if e['type'] == 'VR']
    plt.scatter(vr_x, vr_y, c='red', alpha=0.3, s=5, label='VR')

    # 기준선 (0,0)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    
    plt.title(f"Error Distribution (Center=GT, Points=Prediction Offset)")
    plt.xlabel("X Error (px) [Pred - GT]")
    plt.ylabel("Y Error (px) [Pred - GT]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')  # X, Y 축 비율 맞춤
    
    # 그래프 저장 또는 보여주기
    plt.savefig("error_distribution_analysis.png")
    print("\n[Info] 'error_distribution_analysis.png' 그래프가 저장되었습니다.")
    plt.show()

def calculate_error():
    total_error_dist = [] # 거리 오차 저장용
    all_error_vectors = [] # 방향 분석용 데이터 저장 리스트
    
    print(f"{'Subject':<8} {'Type':<6} {'Modality':<5} | {'Mean Dist':<10} | {'Mean X-Bias':<12} {'Mean Y-Bias':<12}")
    print("-" * 75)

    for sub in SUBJECTS:
        for env in ["30", "50", "VR"]:
            label_file = os.path.join(LABEL_PATH, f"{sub}_{env}_IR.csv")
            if not os.path.exists(label_file): continue
            
            df_label = pd.read_csv(label_file)
            pred_file = os.path.join(PRED_PATH, f"{sub}_{env}_IR.csv")
            if not os.path.exists(pred_file): continue
            
            df_pred = pd.read_csv(pred_file)
            merged = pd.merge(df_label, df_pred, on="FILENAME", suffixes=('_GT', '_PRED'))
            
            diff_x_list = []
            diff_y_list = []
            dist_list = []
            
            for side in ["LEYE", "REYE"]:
                gt_x = merged[f"{side}_CENTER_X_GT"]
                gt_y = merged[f"{side}_CENTER_Y_GT"]
                pred_x = merged[f"{side}_CENTER_X_PRED"]
                pred_y = merged[f"{side}_CENTER_Y_PRED"]
                
                valid_mask = (gt_x != 0) & (gt_y != 0) & (pred_x != 0) & (pred_y != 0)
                if valid_mask.sum() == 0: continue

                # 방향성 에러: (예측값 - 정답값)
                # +값이면 정답보다 오른쪽/아래, -값이면 정답보다 왼쪽/위
                d_x = pred_x[valid_mask] - gt_x[valid_mask]
                d_y = pred_y[valid_mask] - gt_y[valid_mask]
                
                diff_x_list.extend(d_x.tolist())
                diff_y_list.extend(d_y.tolist())
                
                # 유클리드 거리 (기존 에러)
                dist = np.sqrt(d_x**2 + d_y**2)
                dist_list.extend(dist.tolist())

                # 시각화를 위해 데이터 수집
                for x, y in zip(d_x, d_y):
                    all_error_vectors.append({'x': x, 'y': y, 'type': env, 'subject': sub})

            if dist_list:
                mean_dist = np.mean(dist_list)
                mean_bias_x = np.mean(diff_x_list)
                mean_bias_y = np.mean(diff_y_list)
                
                total_error_dist.extend(dist_list)
                
                # 결과 출력 (Bias 추가)
                # Bias가 0에 가까울수록 중앙 정렬 잘됨. 크면 한쪽으로 쏠림.
                print(f"{sub:<8} {env:<6} IR    | {mean_dist:<10.4f} | {mean_bias_x:<12.4f} {mean_bias_y:<12.4f}")
            else:
                print(f"{sub:<8} {env:<6} IR    | N/A")

    print("-" * 75)
    print(f"Total Average Distance Error: {np.mean(total_error_dist):.4f} px")
    
    # 시각화 실행
    visualize_error_distribution(all_error_vectors)

if __name__ == "__main__":
    calculate_error()