import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# 读取CSV文件
df = pd.read_csv('test.csv')
# 转置 DataFrame
# df = df.T
# 重置行索引
df.reset_index(drop=True, inplace=True)
# 绘制热图
plt.figure(figsize=(14, 8))
ax = sns.heatmap(df, cmap='jet')

# 获取色棒,默认的cbar没有首尾标签，这里手动修改
cbar = ax.collections[0].colorbar
# 手动计算色棒的标签
cbar_step=4
vmin, vmax = df.values.min(), df.values.max()# 计算色棒的最小值和最大值
step = (vmax - vmin) / cbar_step# 计算色棒刻度的步长
ticks = [vmin + i * step for i in range(5)]# 设置色棒的刻度位置，包括最小值和最大值
cbar.set_ticks(ticks)
cbar.set_ticklabels(['{:.2f}'.format(tick) for tick in ticks])# 设置色棒的刻度标签



plt.show()