import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from TienXuLy import preprocess_data

# Lấy dữ liệu từ TienXuLy.py
df_friday, data = preprocess_data()

# Kiểm tra dữ liệu
if data is None or len(data) == 0:
    print("Không thể chạy mô hình vì dữ liệu không hợp lệ!")
else:
    # Mô hình Local Level
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    kf = kf.em(data, n_iter=5)  
    state_means, state_covs = kf.filter(data)

    rmse = np.sqrt(mean_squared_error(data, state_means))
    print(f"RMSE Local Level: {rmse}")

    plt.figure(figsize=(10, 5))
    plt.plot(data, label='Observed')
    plt.plot(state_means, label='Filtered')
    plt.title('Local Level Model - Friday')
    plt.xlabel('Time')
    plt.ylabel('Listening Time (minutes)')
    plt.legend()
    plt.show()