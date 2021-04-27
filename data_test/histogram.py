import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import MultipleLocator
df1 = pd.read_excel(r'C:\Users\DaBiao\Desktop\buayes_v.xlsx',usecols="F:H")

data1=[]
# print(df1.values[:,0])
# times_city=df1.values[:,0]
# times_uban=df1[0].values[:,1]
times_hightway=df1.values[:,2]
listbins=[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]
listlabel=[0,'10','20','30','40','50','60','70','80','90','100','110','120','130','140']
test=pd.cut(times_hightway,bins=listbins,include_lowest=True,right=False,labels=listlabel)
# print(pd.value_counts(test).sort_index())
s=pd.value_counts(test)
for t in s:
    c1=t/38*100
    data1.append(c1)


# categories=test.categories.values[0:,]  #城市横坐标
# categories=[30,60,40,70,20,50,10,80,90,100,120,110,130,140,150]  #郊区横坐标
categories=[60,80,70,50,20,40,10,30,90,100,110,120,130,140,150,]   #高速横坐标




dataframe_new={'section':categories,'frequency':data1}
e=pd.DataFrame(dataframe_new)
ax=plt.figure(figsize=(30,20))
sns.barplot(x='section',y='frequency',data=e,palette='Set3')
plt.ylim(0,60)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #
plt.ylabel("累计百分比 / %")
plt.xlabel("高速公路平均速度 v /(Km/h)")
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)

# plt.savefig('绘制高速的频率分布图.jpg',dpi=500,bbox_inches='tight')



