# from parameter import *
from model import *
from data import *
from tqdm import tqdm
from datetime import datetime
from utils import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
# from ignite.engine import Engine, Events
# from ignite.contrib.handlers import ProgressBar

# def update(engine, batch):
#     print(batch)

def run(model, train_loader, test_loader=None, learning_rate=0.001, epoch=1, device=torch.device('cpu'), early_stop=None):
   
    # optimizer 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ## 반복 횟수 Setting
    epochs = tqdm(range(epoch))
    
    ## 학습하기
    count = 0
    # best_loss = 100000000
    for epoch in epochs:
 
        model.train()
        optimizer.zero_grad()
        train_loss = 0
        for i, batch_data in enumerate(train_loader):
                    
            batch_data = batch_data.to(device)
            predict_values = model(batch_data)
            loss = model.loss_function(*predict_values)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss.mean().item()

        writer.add_scalar('Loss/train', train_loss/(i+1), epoch)
        epochs.set_postfix({
                "train Score": float(train_loss)/(i+1),
            })
        model.eval()
        eval_loss = 0
        if test_loader!=None: # validation dataset 이 있을경우 eval_loss 를 계산하고 early_stop을 사용할 수 있음
            with torch.no_grad():
                for i, batch_data in enumerate(test_loader):
                    
                    batch_data = batch_data.to(device)
                    predict_values = model(batch_data)
                    loss = model.loss_function(*predict_values)

                    eval_loss += loss.mean().item()

            eval_loss = eval_loss / (i+1)
            writer.add_scalar('Loss/test', eval_loss, epoch)
            epochs.set_postfix({
                "Evaluation Score": float(eval_loss),
            })

            if early_stop != None:
                early_stop(eval_loss, model)
                if early_stop.early_stop:
                    print("Early stop")
                    break
        
    return model




if __name__ == "__main__":
    # parameter 
    today = datetime.today()
    server_category = "nfs"
    ip_list = ["10.200.2.184", "10.200.2.199", "10.200.1.60"] # 
    data_path = '../서버식별/data'
    file_list = filter_search(data_path, 'conn_summ.csv')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model parameter
    window_size = 5
    batch_size = 50
    
    input_size = 3
    num_layers = 2
    latent_size = 5

    # train parameter
    epoch = 1000
    learning_rate = 0.001
    early_stop = EarlyStopping(patience=20, path=f'{server_category}_model_{today.month}{today.day}.pt')


    # load data
    load_model = None
    # load_model = f'{server_category}_model_{today.month}{today.day}.pt'
    train_dataset = Dataset(ip_list=ip_list, file_list=file_list[:-1], window_size=window_size)
    test_dataset = Dataset(ip_list=ip_list, file_list=file_list[-1:], window_size=window_size, test_mode=True)



    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)


    model = LSTMAutoEncoder(input_dim=input_size, latent_dim=latent_size, window_size=window_size, num_layers=num_layers, teacher_forcing=True)

    if load_model != None:
        model.load_state_dict(torch.load(load_model))

    model.to(device)
    model = run(model, train_loader, test_loader,
                device= device,
                epoch = epoch,
                early_stop=early_stop)
    
    torch.save(model.state_dict(), f'{server_category}_model_{today.month}{today.day}.pt')


    # db_server_list = ["10.200.2.168", "10.200.1.51","222.122.213.215", "52.78.162.129"]
    # nfs_server_list = ["10.200.2.184", "10.200.2.199", "10.200.1.60"]
    # mail_server_list = ['10.200.2.150', '121.163.121.81', '121.163.121.82', '121.163.121.67']
    # dns_server_list = ["8.8.8.8","8.8.4.4","168.126.63.1","10.200.1.21","10.200.1.20"]