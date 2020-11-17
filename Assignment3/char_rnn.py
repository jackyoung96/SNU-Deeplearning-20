import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        ##### please implment here
        self.hidden_size = hidden_size
        
        self.input2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.input2output = nn.Linear(input_size + hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, d, hidden):
        
        ##### please implment here
        for i in range(d.shape[0]):
            state = torch.cat((d[i],hidden),dim=1)
            hidden = self.input2hidden(state)
            output = self.input2output(state)
        
        output = self.softmax(output)
        return output

