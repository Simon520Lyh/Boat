import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 加载数据
print("加载数据...")
train_path = 'D:/南华大学/深度学习/222/train.csv'
test_path = 'D:/南华大学/深度学习/222/test.csv'
submission_path = 'D:/南华大学/深度学习/222/submission.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
submission_data = pd.read_csv(submission_path)

print("数据加载完毕。")

# 数据预处理
print("数据预处理...")


def preprocess_data(data):
    # 处理缺失值
    data = data.ffill().bfill()

    # 特征工程
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['year'] = data['timestamp'].dt.year
    data['month'] = data['timestamp'].dt.month
    data['day'] = data['timestamp'].dt.day
    data['hour'] = data['timestamp'].dt.hour
    data['minute'] = data['timestamp'].dt.minute
    data['second'] = data['timestamp'].dt.second

    return data


train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

print("数据预处理完成。")

# 数据标准化
print("数据标准化...")
scaler_lat = MinMaxScaler()
scaler_lon = MinMaxScaler()
scaler_sog_cog = MinMaxScaler()

train_data['lat_scaled'] = scaler_lat.fit_transform(train_data[['lat']])
train_data['lon_scaled'] = scaler_lon.fit_transform(train_data[['lon']])
train_data[['Sog_scaled', 'Cog_scaled']] = scaler_sog_cog.fit_transform(train_data[['Sog', 'Cog']])

train_features = train_data[['lat_scaled', 'lon_scaled', 'Sog_scaled', 'Cog_scaled']]
train_labels = train_data[['lat', 'lon']]

# 划分训练集和验证集
print("划分训练集和验证集...")
X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

# 重塑数据以适应LSTM输入格式
X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val = X_val.values.reshape((X_val.shape[0], 1, X_val.shape[1]))

# 构建LSTM模型
print("构建LSTM模型...")
model = Sequential()
model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(2))  # 输出纬度和经度

model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')

# 训练模型
print("训练模型...")
history = model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_val, y_val), verbose=1)

print("模型训练完成。")

# 预测
print("进行预测...")
test_data['lat_scaled'] = scaler_lat.transform(test_data[['lat']])
test_data['lon_scaled'] = scaler_lon.transform(test_data[['lon']])
test_data[['Sog_scaled', 'Cog_scaled']] = scaler_sog_cog.transform(test_data[['Sog', 'Cog']])

test_features = test_data[['lat_scaled', 'lon_scaled', 'Sog_scaled', 'Cog_scaled']]
X_test = test_features.values.reshape((test_features.shape[0], 1, test_features.shape[1]))

predictions = model.predict(X_test)

# 反标准化
predicted_lat = scaler_lat.inverse_transform(predictions[:, 0].reshape(-1, 1))
predicted_lon = scaler_lon.inverse_transform(predictions[:, 1].reshape(-1, 1))

# 评估
mse = np.mean((predictions - test_data[['lat_scaled', 'lon_scaled']].values) ** 2)
score = 1 / (mse + 1)
print(f"模型评估完成，MSE: {mse}, Score: {score}")

# 生成提交文件
print("生成提交文件...")

# 创建包含预测结果的 DataFrame，并根据 test_data 的索引进行对齐
submission_df = pd.DataFrame({
    'lat': predicted_lat.flatten(),
    'lon': predicted_lon.flatten()
}, index=test_data.index)

# 将预测结果写入 submission_data 中
submission_data['lat'] = submission_df['lat']
submission_data['lon'] = submission_df['lon']

# 将结果保存为 CSV 文件
submission_data.to_csv('D:/南华大学/深度学习/222/submission/submission.csv', index=False)

print("提交文件已生成。")
