import os, re
import path
import ipaddress
import numpy as np
import torch
class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def filter_search(dir, filter):
    '''
    dir : 탐색하고자 하는 폴터 경로
    filter : 정규표현식으로 필요한 파일만 필터링 할수 있도록함
    '''

    file_list = []

    def search(dir):
        files = os.listdir(dir)

        for file in files:
            fullFilename = os.path.join(dir, file)
            if os.path.isdir(fullFilename):
                search(fullFilename)
            else:
                file_list.append(fullFilename)

        return file_list

    file_list = search(dir)

    file_list_ = []

    for i in file_list:
        if re.findall(filter, i):
            file_list_.append(i)

    return file_list_

def islocal(ip, local):
    if ip =='-':
        return False
    for network in local:
        if ipaddress.ip_address(ip) in network:
            return True

def network_cfg(file_path=path.NETWORK_CFG):
    local = []
    with open(file_path) as f:
        for line in f.readlines():
            if line[0] =="#" or line == '\n':
                continue
            local.append(ipaddress.ip_network(line.split()[0].strip()))
    return local