from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.batch_size = batch_size
        
        # LSTM(input_size, hidden_size, num_layers)
        self.rnn = nn.LSTM(input_dim, hidden_dim, 1)
        self.outlayer = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Implementation here ...
        # x shape: (batch_size, seq_length, input_dim)
        batch_size = x.size(0)
        
        # Initialize hidden states dynamically based on batch size
        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
        
        # LSTM expects input shape: (seq_length, batch_size, input_dim)
        # So we need to permute from (batch_size, seq_length, input_dim)
        x = x.permute(1, 0, 2)
        
        # Forward pass through LSTM
        output, _ = self.rnn(x, (h0, c0))
        
        # output shape: (seq_length, batch_size, hidden_dim)
        # Get the last time step output
        output = output[-1, :, :]  # (batch_size, hidden_dim)
        
        # Pass through output layer
        outputs = self.outlayer(output)
        return outputs