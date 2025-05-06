import pandas as pd

def preprocess_data():
    try:
        df = pd.read_csv('podcast_dataset.csv')
        print("Đọc file CSV thành công!")
        print("Các cột trong dữ liệu:", df.columns.tolist())

        # Lọc dữ liệu cho Friday
        df_friday = df[df['Publication_Day'] == 'Friday']
        if df_friday.empty:
            raise ValueError("Không có dữ liệu nào cho ngày Friday! Kiểm tra giá trị trong cột Publication_Day.")

        # Xử lý giá trị thiếu
        df_friday = df_friday.dropna(subset=['Listening_Time_minutes'])
        print(f"Số lượng dòng dữ liệu cho Friday: {len(df_friday)}")

        # Tạo chuỗi thời gian
        data = df_friday['Listening_Time_minutes'].values
        print("Chuỗi thời gian:", data)

        return df_friday, data

    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'podcast_dataset.csv'. Vui lòng kiểm tra đường dẫn!")
        return None, None
    except ValueError as e:
        print(f"Lỗi: {e}")
        return None, None
    except Exception as e:
        print(f"Lỗi không xác định: {e}")
        return None, None

# Chạy hàm tiền xử lý
if __name__ == "__main__":
    df_friday, data = preprocess_data()