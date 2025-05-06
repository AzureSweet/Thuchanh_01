import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from TienXuLy import preprocess_data

df_friday, data = preprocess_data()

# Kiểm tra dữ liệu
if data is None or len(data) == 0:
    print("Không thể chạy mô hình vì dữ liệu không hợp lệ!")
else:
    # Mô hình Seasonal (chu kỳ 5)
    n_seasons = 5
    transition_matrix = np.zeros((n_seasons, n_seasons))
    for i in range(n_seasons-1):
        transition_matrix[i, i+1] = 1
    transition_matrix[n_seasons-1, 0] = 1

    kf = KalmanFilter(
        transition_matrices=transition_matrix,      # Ma trận chuyển trạng thái cho chu kỳ
        observation_matrices=np.ones((1, n_seasons))  # Quan sát tổng các trạng thái chu kỳ
    )
    kf = kf.em(data, n_iter=5) 
    state_means, state_covs = kf.filter(data)

    # Tính RMSE
    rmse = np.sqrt(mean_squared_error(data, state_means[:, 0]))
    print(f"RMSE Seasonal: {rmse}")

    plt.figure(figsize=(10, 5))
    plt.plot(data, label='Observed')
    plt.plot(state_means[:, 0], label='Filtered')
    plt.title('Seasonal Model - Friday')
    plt.xlabel('Time')
    plt.ylabel('Listening Time (minutes)')
    plt.legend()
    plt.show()