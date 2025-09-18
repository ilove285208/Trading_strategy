import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import torch

class ResBlock(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size,padding,dilation,dropout=0.2):
        super(ResBlock,self).__init__()
        self.padding = padding
        self.conv1 = nn.Conv1d(in_channels = input_channels,\
                            out_channels = output_channels,\
                            kernel_size = kernel_size,\
                            stride = 1,\
                            padding = padding,\
                            dilation = dilation)
        self.conv1 = weight_norm(self.conv1)
        

        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(in_channels = output_channels,\
                            out_channels = output_channels,\
                            kernel_size = kernel_size,\
                            stride = 1,\
                            padding = padding,\
                            dilation = dilation)
        self.conv2 = weight_norm(self.conv2)
        
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.shortcut = None
        if input_channels != output_channels:
            self.shortcut = nn.Conv1d(input_channels, output_channels, kernel_size=1)

        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        res = self.conv1(x)
        res = res[:,:,0:-self.padding].contiguous()
        res = self.relu1(res)
        res = self.dropout1(res)

        res = self.conv2(res)
        res = res[:,:,0:-self.padding].contiguous()
        res = self.relu2(res)
        res = self.dropout2(res)

        if self.shortcut is not None:
            x = self.shortcut(x)
        
        y = self.relu(x + res)
        return y
    #
    def init_weight(self):
        self.conv1.weight.data.normal_(0, 1.0)
        
        self.conv2.weight.data.normal_(0, 1.0)
        
        if self.shortcut is not None:
            self.shortcut.weight.data.normal_(0, 1.0)

    
# TCN model
class TCN(nn.Module):
    def __init__(self, input_channels, layer_channels, kernel_size):
        super(TCN,self).__init__()

        n_layer = len(layer_channels)
        layers = []

        layer_input_ch = input_channels
        for layer in range(n_layer):
            layer_output_ch = layer_channels[layer]

            dilation = 2 ** layer
            padding = (kernel_size - 1) * dilation

            res = ResBlock(layer_input_ch,layer_output_ch,kernel_size,padding,dilation)
            layers.append(res)

            layer_input_ch = layer_output_ch

        self.res_block = nn.Sequential(*layers)

    
    def forward(self, x):
        features = self.res_block(x)
        return features


# main program
if __name__ == '__main__':
    # define the data
    oil_price = [84.83, 84.84, 85.14, 84.4, 85.53, 87.27, 86.63, 86.29, 84.96, 84.86, 82.82, 80.48, 81.51]
    
    # a TCN for prediction the trend labels: [down, flat, up] (output channel = 3)
    tcn_model = TCN(1, [8, 8, 8], 3)
    
    # covert the data to the torch tensor
    x = torch.FloatTensor(oil_price)
    x = x.unsqueeze(dim=0) # extend the channel = 1
    x = x.unsqueeze(dim=0) # extend the batch_size = 1
    
    print(f'x: {x.shape}')
    
    # pass through the tcn_model
    y = tcn_model(x) 
    print(f'y: {y.shape}')    
    y = y[:,:,-1]
    print(f'y: {y.shape}')    
    print(f'y: {y}')    

