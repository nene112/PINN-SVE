import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 选择GPU或CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 从文件加载已经训练完成的模型
model_loaded = torch.load('saint_venant_model_dambreak.pth', map_location=device)
model_loaded.eval()  # 设置模型为evaluation状态

# 生成时空网格
h = 0.01  # 空间步长
k = 0.01  # 时间步长
x = torch.arange(-1, 1, h)
t = torch.arange(0, 10, k)
X = torch.stack(torch.meshgrid(x, t)).reshape(2, -1).T
X = X.to(device)

# 计算该时空网格对应的预测值
with torch.no_grad():
    # 输出 Saint-Venant 方程的两个变量 h 和 u
    U_pred = model_loaded(X)  # 模型预测 (h, u) 的值
    h_pred = U_pred[:, 0].reshape(len(x), len(t)).cpu().numpy()  # 水深 h 的预测
    u_pred = U_pred[:, 1].reshape(len(x), len(t)).cpu().numpy()  # 流速 u 的预测

# 绘制水深 h 的计算结果
plt.figure(figsize=(5, 3), dpi=300)
xnumpy = x.numpy()
plt.plot(xnumpy, h_pred[:, 0], '-', markersize=1, label='t=0')
plt.plot(xnumpy, h_pred[:, 20], '-', markersize=1, label='t=0.2')
plt.plot(xnumpy, h_pred[:, 40], '-', markersize=1, label='t=0.4')
plt.plot(xnumpy, h_pred[:, 60], '-', markersize=1, label='t=0.6')
plt.plot(xnumpy, h_pred[:, 80], '-', markersize=1, label='t=0.8')
plt.plot(xnumpy, h_pred[:, 100], '-', markersize=1, label='t=1.0')
plt.xlabel('x')
plt.ylabel('Water Depth h')
plt.title('Water Depth (h) over time at different time steps')
plt.legend()

# 绘制水深 h 的热力图
plt.figure(figsize=(5, 3), dpi=300)
sns.heatmap(h_pred, cmap='jet')
plt.title('Heatmap of Water Depth h')

# 绘制流速 u 的计算结果
plt.figure(figsize=(5, 3), dpi=300)
plt.plot(xnumpy, u_pred[:, 0], 'o', markersize=1, label='t=0')
plt.plot(xnumpy, u_pred[:, 20], 'o', markersize=1, label='t=0.2')
plt.plot(xnumpy, u_pred[:, 40], 'o', markersize=1, label='t=0.4')
plt.plot(xnumpy, u_pred[:, 60], 'o', markersize=1, label='t=0.6')
plt.plot(xnumpy, u_pred[:, 80], 'o', markersize=1, label='t=0.8')
plt.plot(xnumpy, u_pred[:, 100], 'o', markersize=1, label='t=1.0')
plt.xlabel('x')
plt.ylabel('Flow Velocity u')
plt.title('Flow Velocity (u) over time at different time steps')
plt.legend()

# 绘制流速 u 的热力图
plt.figure(figsize=(5, 3), dpi=300)
sns.heatmap(u_pred, cmap='jet')
plt.title('Heatmap of Flow Velocity u')

# 显示所有绘图
plt.show()

# 将水深 h 和流速 u 的预测结果保存到CSV文件
df_h = pd.DataFrame(h_pred, columns=[f't={i*k:.2f}' for i in range(len(t))])
df_u = pd.DataFrame(u_pred, columns=[f't={i*k:.2f}' for i in range(len(t))])
df_h.to_csv('water_depth_h.csv', index=False)
df_u.to_csv('flow_velocity_u.csv', index=False)
