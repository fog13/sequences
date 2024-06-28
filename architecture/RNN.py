import torch
import torch.nn as nn


#  Class for a single RNN Cell

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        super(RNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

        self.layer_norm_hidden = nn.LayerNorm(hidden_size)
        self.layer_norm_output = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tensor, prev_hidden_state):
        inner_tensor = self.hidden_layer(prev_hidden_state) + self.input_layer(input_tensor)
        hidden_tensor = torch.tanh(self.layer_norm_hidden(inner_tensor))
        # hidden_tensor = torch.tanh(inner_tensor)
        hidden_tensor = self.dropout(hidden_tensor)
        output_tensor = self.output_layer(hidden_tensor)
        # output_tensor = self.dropout(output_tensor)

        return output_tensor, hidden_tensor


# Class for RNN model composed of one or more RNN cells
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, output_layer_dim=None, hidden_size_dim=None, device='cpu'):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.device = device

        if num_layers != 1:
            assert (output_layer_dim is not None and len(output_layer_dim) == num_layers and
                    len(hidden_size_dim) == num_layers)
            input_dim = [input_size]+ output_layer_dim[:-1]
        else:
            input_dim = [input_size]
            output_layer_dim = [output_size]
            hidden_size_dim = [hidden_size]

        print(input_dim, output_layer_dim)
        self.rnn_cells = nn.ModuleList([RNNCell(input_dim[i], hidden_size_dim[i], output_layer_dim[i]) for i in
                                        range(self.num_layers)])

        self.to(self.device)

    def forward(self, input_sequence):

        batch_size = input_sequence.size(0)
        sequence_size = input_sequence.size(1)
        # Different initialization techniques can be tried. For now, I am sticking to zeros
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        outputs = torch.zeros(sequence_size, batch_size, self.output_size).to(self.device)
        # iterate over each time step
        for seq in range(sequence_size):
            token_tensor = input_sequence[:, seq, :]
            new_hidden_state = []
            # iterate over each layer in rnn
            for i, rnn_cell in enumerate(self.rnn_cells):
                y_out, h_out = rnn_cell(token_tensor, hidden_state[i])
                token_tensor = y_out

                # update hidden state of rnn cells of layer
                new_hidden_state.append(h_out)
            hidden_state = torch.stack(new_hidden_state)
            outputs[seq] = token_tensor

        return outputs.view(outputs.shape[1], outputs.shape[0], outputs.shape[2])