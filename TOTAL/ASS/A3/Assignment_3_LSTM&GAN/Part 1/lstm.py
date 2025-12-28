from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy
import torch.nn.functional as F

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(LSTM, self).__init__()
        
        # Initialization here ...
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        
        # Activation function
        self.tanh = torch.tanh
        self.sig = torch.sigmoid
        
        # Linear Layers
        self.gx = nn.Linear(input_dim, hidden_dim)
        self.gh = nn.Linear(hidden_dim, hidden_dim)
        self.ix = nn.Linear(input_dim, hidden_dim)
        self.ih = nn.Linear(hidden_dim, hidden_dim)
        self.fx = nn.Linear(input_dim, hidden_dim)
        self.fh = nn.Linear(hidden_dim, hidden_dim)
        self.ox = nn.Linear(input_dim, hidden_dim)
        self.oh = nn.Linear(hidden_dim, hidden_dim)
        self.ph = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Implementation here ...
        # x shape: (batch_size, seq_length, input_dim)
        batch_size = x.size(0)
        seq_length = x.size(1)  # Use actual sequence length from input
        
        # Initialize hidden state and cell state
        h = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        c = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        
        # use for loop to forward through time
        for t in range(seq_length):  # Use actual seq_length from input
            x_t = x[:, t, :]  # Get input at time t: (batch_size, input_dim)
            
            g = self.tanh(self.gx(x_t) + self.gh(h))
            i = self.sig(self.ix(x_t) + self.ih(h))
            f = self.sig(self.fx(x_t) + self.fh(h))
            o = self.sig(self.ox(x_t) + self.oh(h))
            
            c = g * i + c * f
            h = self.tanh(c) * o
    
        p = self.ph(h)
        out = F.softmax(p, dim=1)
        return out

    # add more methods here if needed