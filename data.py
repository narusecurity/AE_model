import os
import numpy as np
import torch
from utils import filter_search
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from itertools import cycle


class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, ip_list, file_list, window_size, test_mode=False):
        self.ip_list = ip_list # 특정 서버 카테고리가 식별된 ip를 가지고 학습에 이용하도록 한다(정상상태라고 가정)
        self.file_list = file_list
        self.usecols = ['#ts','src_ip','dst_ip','dst_port','duration','sbytes','spkts','dbytes','dpkts']
        self.window_size = window_size
        self.df_filename = f'df_{"_".join(ip_list)}.csv'
        self.test_mode = test_mode
        self.df = self.parse_file(self.file_list)

    def __iter__(self):
        if not self.test_mode:
            return self.stream(self.df)
        else : return self.stream(self.df)
    def stream(self, df):
        for ip in self.ip_list:
            df_temp = df[df.dst_ip==ip].reset_index()
            offset = self.window_size-1
            if len(df_temp)<5: continue
            for idx in range(0, len(df_temp)-offset): # 80퍼센트 이상 데이터가 존재하면 분석 실시, 연결시도 관련 통신은 통신량 0으로 기록되어 있는데 이거는 그대로 반영
                if (df_temp.ts[idx+offset] - df_temp.ts[idx])/pd.Timedelta(hours=1)+1 <= self.window_size*1.2:
                    yield torch.tensor(df_temp[['bcf','bkf','pcr']].loc[idx:idx+offset].values, dtype=torch.float)
        
    def parse_file(self, file_list):
        if os.path.isfile(self.df_filename):
            df = pd.read_csv(self.df_filename, parse_dates=['ts'])
        else:
            df_list=[]
            for file in file_list:
                df_list.append(self.preprocessing(file)) # 필요한 ip에대한 데이터 뽑고, bcf,bkf,pcr 값 계산
            df = pd.concat(df_list)
            # df.to_csv(self.df_filename)
        self.__len__ = (len(df)-self.window_size+1) * len(self.ip_list)
        return df

    def preprocessing(self, file):
        df = pd.read_csv(file, sep='\t',dtype={'dst_port':str}, parse_dates=['#ts'], usecols=self.usecols) # 고용량일경우 pandas 대신 Dask 패키지를 이용해보는것이 좋을듯
        df.rename(columns={'#ts': 'ts'}, inplace=True) 
        df = df[df.dst_ip.isin(self.ip_list)] # ip_list에 있는 것만 필터링

        df = df.groupby(['ts','dst_ip'], as_index=False).sum()
        df = df[df['dpkts']!=0]
        df['bcf']=(df['sbytes']+df['dbytes'])/(df['spkts']+df['dpkts']+0.1)/1500
        df['bkf']=df['duration']/(df['duration']+df['spkts']+df['dpkts']+0.1)
        df['pcr']=(df['sbytes']-df['dbytes'])/(df['sbytes']+df['dbytes']+0.1)

        return df.drop(['duration','sbytes','spkts','dbytes','dpkts'],axis=1)   

    



if __name__ == "__main__":
    import time
    s = time.time()

    dataset = Dataset(['10.200.2.184'], '../서버식별/data/',5)
    dataloader = DataLoader(dataset, batch_size=40) # pin_memory=True # worker, 
    for i in dataloader:
        tensor = i.to(device)

    print(time.time()-s)

