import os
import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=0.1, bidirectional=False)

    def forward(self, x): # x: [batch_size, window_size, field 개수]
        outputs, (hidden, cell) = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        return (hidden, cell)


class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=0.1, bidirectional=False)

        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden):
        output, (hidden, cell) = self.lstm(x, hidden)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        prediction = self.fc(output)

        return prediction, (hidden, cell)


class LSTMAutoEncoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 window_size: int=1,
                 **kwargs) -> None:
        """
        :param input_dim: 변수 Tag 갯수
        :param latent_dim: 최종 압축할 차원 크기
        :param window_size: 길이
        :param kwargs:
        """

        super(LSTMAutoEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.window_size = window_size
        
        if "num_layers" in kwargs:
            num_layers = kwargs.pop("num_layers")
        else:
            num_layers = 1

        if "teacher_forcing" in kwargs:
            self.teacher_forcing = kwargs.pop("teacher_forcing")
        else : self.teacher_forcing = False
        
        self.encoder = Encoder(
            input_size=input_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
        )
        self.reconstruct_decoder = Decoder(
            input_size=input_dim,
            output_size=input_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
        )

    def forward(self, src:torch.Tensor, **kwargs):
        batch_size, sequence_length, var_length = src.size() # batch_size, window_size, field의 갯수

        ## Encoder 넣기
        encoder_hidden = self.encoder(src)
        
        
        inv_idx = torch.arange(sequence_length - 1, -1, -1).long() # 숫자거꾸로 생성 (n, n-1, ... ,2, 1, 0)
        reconstruct_output = []
        temp_input = torch.zeros((batch_size, 1, var_length), dtype=torch.float).to(src.device) # decoder 초기값
        hidden = encoder_hidden # hidden[0] : [num_layer, batch_size, hidden_size]
        for idx in inv_idx:
            temp_input, hidden = self.reconstruct_decoder(temp_input, hidden) # temp_input:[batch, 1, var_length]
            reconstruct_output.append(temp_input)

            if self.teacher_forcing:
                temp_input = src[:,[idx], :] # temp_input:[batch, 1, var_length]
        reconstruct_output = torch.cat(reconstruct_output, dim=1)[:, inv_idx, :] # 합치기, 다시 뒤집기
        
        return [reconstruct_output, src]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        
        ## MSE loss(Mean squared Error)
        loss =F.mse_loss(recons, input)
        return loss