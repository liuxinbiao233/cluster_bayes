import pandas as pd
import numpy as np



df=pd.ExcelFile(r'C:\Users\DaBiao\Downloads\vichel\data.xlsx')
sheet_name=df.sheet_names
data=[]
split=[]
v_mean=[]
a_mean_part=[]
a_mean_all=[]
temp1=[]


##进行数据第一次分类
# for i in sheet_name:
#     df1 = pd.read_excel(r'C:\Users\DaBiao\Downloads\data.xlsx', sheet_name=[i], nrows=1000, usecols="A:B")
#     # print(i)
#     a=df1[i].values[:,1]
#     # print(a)
#     i = pd.DataFrame({i+"_speed": [a]})
#     c1 = i.to_dict('list')
#     data.append(c1)
#     print(c1)
# pd.DataFrame(data).to_csv("opera_data.csv")
# print("csv第一次保存完毕")

#尝试数据分类


def split_list_by_n(list_collection,n):
    for i in range(0,len(list_collection),n):
        yield list_collection[i:i+n]




if __name__ == '__main__':

    #数据切分和求平均速度

    # for i in sheet_name:
    #         df1 = pd.read_excel(r'C:\Users\DaBiao\Downloads\vichel\data.xlsx', sheet_name=[i], usecols="A:B")
    #         # print(i)
    #         a = df1[i].values[:, 1] #读取excel数据
    #         b=np.set_printoptions(precision=3)
    #         print(b)
    #         temp=split_list_by_n(a,100)
    #         #for循环是进行切割和求平均值
    #         for t in temp:
    #             split.append(t)
    #         # pd.DataFrame(split).to_csv("test_split.csv")  #进行切片
    #
    #             #求平均值
    #             # solo_data = pd.DataFrame({i+"_v_mean":t}).mean()
    #             # v_mean.append(solo_data)
    #
    # # print(v_mean)
    #         # pd.DataFrame(v_mean).to_csv("v_mean_new.csv")

    #求平均加速度
        df1 = pd.read_excel(r'C:\Users\DaBiao\Desktop\test11.xlsx', sheet_name=[0], usecols="A:B")
        a = df1[0].values[:, 1]
        for t in range(len(a) - 1):
            list_change = [a[t + 1] - a[t]]
            a_mean_part.append(list_change)
        temp1.extend([x[0] for x in a_mean_part])
        splits = split_list_by_n(temp1, 100)
        for temp in splits:
            c1=pd.DataFrame({"_v_mean":temp}).sum()/100
            a_mean_all.append(c1)
        print(a_mean_all)
        pd.DataFrame(a_mean_all).to_csv(r'C:\Users\DaBiao\Desktop\acceleration_mean.csv',mode='a',header=False)



    #求怠速时间

    # c=273
    # for i in range(c):
    #     df1 = pd.read_csv(r'C:\Users\DaBiao\Downloads\PyCharmProject\data_test\data_split.csv' )
    #     a=df1.values[i]
    #     c2 = len(a)
    #     c=np.sum(a == 0)
    #     print(c)
    #     c1=c/c2
    #     temp1.append(c1)
    # pd.DataFrame(temp1).to_csv('0_times.csv', header=False)







    #把不同的csv进行合并

    # times = pd.read_csv(r'C:\Users\DaBiao\Downloads\PyCharmProject\data_test\0_times.csv',usecols=[1])
    # a = pd.read_csv(r'C:\Users\DaBiao\Downloads\PyCharmProject\data_test\acceleration_mean.csv',usecols=[1])
    # v = pd.read_csv(r'C:\Users\DaBiao\Downloads\PyCharmProject\data_test\v_mena.csv',usecols=[1])
    # test=pd.concat([v,a,times],axis=1)
    # pd.DataFrame(test).to_csv('v_a_time.csv')


    #把三列合并为一列数据
    # c = 272
    # for t in range(c):
    #     df1=pd.read_csv(r'C:\Users\DaBiao\Downloads\PyCharmProject\data_test\v_a_time.csv',usecols=[1,2,3])
    #     a = df1.values[t].__array__()
    #     for i in range(3):
    #         z2 = a[i]
    #         temp1.append(z2)
    # temp = split_list_by_n(temp1, 3)
    # for t in temp:
    #     i = pd.DataFrame({"_speed": [t]})
    #     c1 = i.to_dict('list')
    #
    #     split.append(c1)
    # pd.DataFrame(split).to_csv('cluster.csv')


    #DataFrame控制小数点后几位
    # df1 = pd.read_csv(r'C:\Users\DaBiao\Downloads\PyCharmProject\data_test\cluster_data.csv', usecols=[1, 2, 3])
    # df2=df1.__float__('%.2f'%df1)






    #测试数据能否在kmeans++使用
    # df1 = pd.read_csv(r'C:\Users\DaBiao\Downloads\PyCharmProject\data_test\cluster.csv', usecols=[1])
    # data=df1.values



    # df1 = pd.read_excel(r'C:\Users\DaBiao\Downloads\PyCharmProject\data_test\聚类数据.xlsx', usecols=[1,2,3])









    # print("第二次保存完毕")











