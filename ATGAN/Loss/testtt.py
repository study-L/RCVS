import pickle

# 打开Pickle文件
with open(r'H:\three\matchfonew_final\matchformerraw\ATGAN\Loss\sbt_data_700_fw20_yz35_dt1981_2019_6m.pkl', 'rb') as f:
    # 从文件中读取内容
    data = pickle.load(f)

# 输出内容
print(data)