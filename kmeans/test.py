from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D




times = pd.read_csv(r'C:\Users\DaBiao\Downloads\PyCharmProject\data_test\0_times.csv',usecols=[1])
a = pd.read_csv(r'C:\Users\DaBiao\Downloads\PyCharmProject\data_test\acceleration_mean.csv',usecols=[1])
v = pd.read_csv(r'C:\Users\DaBiao\Downloads\PyCharmProject\data_test\v_mena.csv',usecols=[1])
df1 = pd.read_csv(r'C:\Users\DaBiao\Downloads\PyCharmProject\data_test\cluster_data.csv', usecols=[1,2,3])
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

center=[[83.31855,0.37218,0.02455],[25.10711,0.26598,0.12537],[55.09941,0.29078,0.02667],[8.69720,0.14736,0.35035]]



# Generate sample data
n_samples = 259
n_components = 4

X, y_true = make_blobs(n_samples=n_samples,
                       centers=n_components,
                       cluster_std=0.60,
                       random_state=0)
X = X[:, ::-1]

# Calculate seeds from kmeans++
centers_init, indices = kmeans_plusplus(X, n_clusters=4,
                                        random_state=0)

# Plot init seeds along side sample data
plt.figure(dpi=500)
colors = ['#4EACC5', '#4E9A06', 'm','#FF9C34']
ax=plt.subplot(projection='3d')
x=v.values[:, 0],
y=a.values[:, 0],
z=times.values[:, 0]












# plt.title("K-Means++ Initialization")
# ax.set_xlabel('平均速度')
# ax.set_xticks([0,10,20,30,40,50,60,70,80,90,100,110])
# ax.set_ylabel('平均加速度')
# ax.set_yticks([0,0.5,1.0,1.5,2.0,2.5,3.0])
# ax.set_zlabel('怠速时间比')
# ax.set_zticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
plt.show()