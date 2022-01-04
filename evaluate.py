from model import *
from data import DataLoader, Dataset
from tqdm import tqdm
from model import LSTMAutoEncoder
from utils import filter_search, network_cfg, islocal
import parameter as para
import torch
import numpy as np
import pandas as pd
def get_loss_list(device, model, test_loader):
    
    loss_list = []
    
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
                
            batch_data = batch_data.to(device)
            predict_values = model(batch_data)
            
            ## MAE(Mean Absolute Error)로 계산
            loss = F.l1_loss(predict_values[0], predict_values[1], reduction='none')
            loss = loss.mean().item()
            # loss = loss.sum(dim=2).mean().item()
            loss_list.append(loss)
    if loss_list:
        loss_list = np.mean(loss_list)
    else : return None

    return loss_list

server_category = 'nfs'
data_path = '../서버식별/data'
file_list = filter_search(data_path, 'conn_summ.csv')
print(file_list[-1])
df = pd.read_csv(file_list[-1], sep='\t',dtype={'dst_port':str}, parse_dates=['#ts'], usecols=['#ts','src_ip','dst_ip','dst_port','duration','sbytes','spkts','dbytes','dpkts'])
df = df[df['dpkts']!=0]
local = network_cfg()
ip_list = [ip for ip in df.dst_ip.unique() if islocal(ip,local)]
print(ip_list[:10])
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = LSTMAutoEncoder(input_dim=para.input_size, latent_dim=para.latent_size, window_size=para.window_size, num_layers=para.num_layers, teacher_forcing=False)
model.load_state_dict(torch.load(f'{server_category}_model_1223.pt'))
import pytorch_model_summary

model.to(device)

ip_iter = tqdm(enumerate(ip_list), total=len(ip_list), desc="testing")
res=[]
for i, ip in ip_iter:
    dataset = Dataset(ip_list=[ip], file_list=[file_list[-1]], window_size=5)
    dataloader = DataLoader(dataset, batch_size=50)

    res.append(get_loss_list(device, model, dataloader))

# TODO
# 모든 내부망 ip 돌면서 loss 계산 해보기
# 데이터 부족한거는 부족하다고 체크

res_df = pd.DataFrame(ip_list, res)
res_df.to_csv(f'result_{server_category}.csv')
