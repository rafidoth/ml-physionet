import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class MyRNN(nn.Module):
	def __init__(self):
		super(MyRNN, self).__init__()
		self.gru = nn.GRU(input_size=1, hidden_size=68, num_layers=2, batch_first=True)
		self.fc = nn.Linear(in_features=68, out_features=5)
        
	def forward(self, x):
		x, _ = self.gru(x)
		x = self.fc(x[:, -1, :])
		return x
    
    