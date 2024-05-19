import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler
from keras.api.models import Sequential
from keras.api.layers import LSTM
from keras.api.layers import Dense, Dropout
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams["font.sans-serif"] = ["SimHei"]  # 设置显示中文字体
mpl.rcParams["axes.unicode_minus"] = False  # 设置正常显示符号

# csv文件为谷歌的股价数据，其中第1列是日期，最后1列是收盘价格，第2到5列是其他相关信息。本代码利用t-12到t时段的第2~5列数据来预测t+1时刻的收盘价格。
# 读取数据
df = pd.read_csv("data.csv")
DATE = df.values[:, 0]
data = df.values[:, 1:]

# 数据归一化处理
scaler = MinMaxScaler(feature_range=(-1, 1))
data_all_scaled = scaler.fit_transform(data)

# 划分训练集和测试集
test_split = round(len(df) * 0.80)
data_for_training = data_all_scaled[:test_split, :]
data_for_testing = data_all_scaled[test_split:, :]


# 定义函数将数据拆分为X和Y
def createXY(dataset, n_past, n_future):
    dataX, dataY = [], []
    for j in range(n_past, len(dataset) - n_future):
        dataX.append(dataset[j - n_past:j])
        dataY.append(dataset[j + n_future, -1])
    return np.array(dataX), np.array(dataY)


# 数据划分
# past_time_steps是指利用过去几个时刻的数据来预测下一时刻的数据
# future_time_steps=预测未来时长-1，比如想预测t+3时刻的值，future_time_steps=2。future_time_steps值越大，预测效果越差，默认future_time_steps=0
past_time_steps = 12
future_time_steps = 2
trainX, trainY = createXY(data_for_training, past_time_steps, future_time_steps)
testX, testY = createXY(data_for_testing, past_time_steps, future_time_steps)
trainY = trainY.reshape(-1, 1)
testY = testY.reshape(-1, 1)

# 搭建模型
model = Sequential()
model.add(LSTM(64, activation='tanh', return_sequences=True, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
history = model.fit(trainX, trainY, batch_size=32, epochs=50, verbose=1)
print(model.summary())

# 绘制 loss(mse) 图
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('loss(mse)')
plt.xlabel('Epoch')
plt.savefig("图1-训练集loss图.png", dpi=600)
plt.show()

# 利用测试集来评估模型性能
prediction_scaled = model.predict(testX)
# 对数据进行逆缩放
prediction_scaled_copies_array = np.repeat(prediction_scaled, data.shape[1], axis=1)
prediction = scaler.inverse_transform(prediction_scaled_copies_array)[:, -1]
original_scaled_copies_array = np.repeat(np.reshape(testY, (len(testY), 1)), data.shape[1], axis=1)
original = scaler.inverse_transform(original_scaled_copies_array)[:, -1]

# 计算各种评价指标
MSE = metrics.mean_squared_error(prediction, original)
RMSE = metrics.mean_squared_error(prediction, original) ** 0.5
MAE = metrics.mean_absolute_error(prediction, original)
MAPE = metrics.mean_absolute_percentage_error(prediction, original)
R2 = r2_score(prediction, original)
print("MSE-- ", MSE)
print("RMSE-- ", RMSE)
print("MAE-- ", MAE)
print("MAPE-- ", MAPE)
print("R2-- ", R2)

# 绘制一个图来对比预测数据和原始数据
X = DATE[test_split + past_time_steps + future_time_steps:]
plt.figure()
plt.plot(X, data[test_split + past_time_steps + future_time_steps:, -1], color='red', label='真实值')
plt.plot(X, prediction, color='blue', label='预测值')
plt.xticks(range(1, len(prediction), 240))
plt.title('LSTM预测')
plt.xlabel('日期时间')
plt.ylabel('股票价格')
plt.legend()
plt.savefig("图2-测试集折线对比图.png", dpi=600)
plt.show(block=True)

# 绘制预测值与实际值之间的散点密度图
xy_max = max(max(original), max(prediction))
xy_min = min(min(original), min(prediction))
surplus = (xy_max - xy_min) / 10
xy = np.vstack([prediction, original])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = prediction[idx], original[idx], z[idx]
fig, ax = plt.subplots(figsize=(8, 7), dpi=100)
scatter = ax.scatter(x, y, marker='o', c=z, s=15, edgecolors=None, label='LST', cmap='Spectral_r')
ax.plot([xy_min - surplus, xy_max + surplus], [xy_min - surplus, xy_max + surplus], 'k--', lw=1)  # 画的1:1线
divider = make_axes_locatable(ax)
plt.xlabel("预测值")
plt.ylabel("实际值")
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(scatter, cax=cax, label='密度')
plt.savefig("图3-测试集散点密度图.png", dpi=600)
plt.show(block=True)

# 输出保存各类数据
# 将数组转换为DataFrame
df_loss = pd.DataFrame({'loss': history.history['loss']})
df_eval = pd.DataFrame({'MSE': [MSE], 'RMSE': [RMSE], 'MAE': [MAE], 'MAPE': [MAPE], 'R2': [R2]})
df_pre = pd.DataFrame({'original': original, 'prediction': prediction})
# 将DataFrame保存为Excel文件
df_loss.to_excel('输出1-loss.xlsx', index=False)
df_eval.to_excel('输出2-eval.xlsx', index=False)
df_pre.to_excel('输出3-pre.xlsx', index=False)
