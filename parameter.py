# model parameter
window_size = 5 ## sequence Length
batch_size = 50 ## 배치 사이즈 설정

input_size = 3 # 입력 차원 설정
num_layers = 2 # LSTM layer 갯수 설정
latent_size = 5 # Hidden 차원 설정

# train parameter
epoch = 100
learning_rate = 0.001
early_stop = False