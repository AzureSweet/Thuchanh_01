import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from TienXuLy import preprocess_data

# Lấy dữ liệu
df_friday, data = preprocess_data()

if data is None or len(data) == 0:
    print("Không thể chạy mô hình vì dữ liệu không hợp lệ!")
else:
    # Mô hình Linear Trend
    kf = KalmanFilter(
        transition_matrices=[[1, 1], [0, 1]],
        observation_matrices=[[1, 0]],
        initial_state_mean=[0, 0]
    )
    kf = kf.em(data, n_iter=5)
    state_means, state_covs = kf.filter(data)
    rmse = np.sqrt(mean_squared_error(data, state_means[:, 0]))
    print(f"RMSE Linear Trend: {rmse}")

    plt.figure(figsize=(10, 5))
    plt.plot(data, label='Observed')
    plt.plot(state_means[:, 0], label='Filtered')
    plt.title('Linear Trend Model - Friday')
    plt.xlabel('Time')
    plt.ylabel('Listening Time (minutes)')
    plt.legend()
    plt.show()