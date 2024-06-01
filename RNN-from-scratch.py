import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Initialise device

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Primary device set to GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Primary device set to CPU.")


#  Class for a single RNN Cell
class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input_tensor, prev_hidden_state):
        inner_tensor = self.hidden_layer(prev_hidden_state) + self.input_layer(input_tensor)
        hidden_tensor = torch.tanh(inner_tensor)
        output_tensor = self.output_layer(hidden_tensor)

        return output_tensor, hidden_tensor


# Class for RNN model composed of one or more RNN cells
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, device='cpu'):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.device = device

        self.rnn_cells = nn.ModuleList([RNNCell(input_size, hidden_size, output_size) for _ in range(self.num_layers)])
        self.to(self.device)

    def forward(self, input_sequence):

        batch_size = input_sequence.size(0)
        # Different initialization techniques can be tried. For now, I am sticking to zeros
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        outputs = torch.zeros(input_sequence.size(1), batch_size, self.output_size).to(self.device)

        # iterate over each time step
        for seq in range(input_sequence.shape[1]):
            token_tensor = input_sequence[:, seq, :]
            # iterate over each layer in rnn
            for i, rnn_cell in enumerate(self.rnn_cells):
                y_out, h_out = rnn_cell(token_tensor, hidden_state[i])
                token_tensor = y_out

                # update hidden state of rnn cells of layer
                hidden_state[i] = h_out

            outputs[seq] = token_tensor

        return outputs.view(outputs.shape[1], outputs.shape[0], outputs.shape[2]), hidden_state

def sequence_generator(input_sequence, window_size, batch_size = 20000):
    start = 0

input_size = 1
hidden_size = 4
output_size = 1
num_layers = 2

model = RNN(input_size, hidden_size, output_size, num_layers, device.__str__()).to(device)

df = pd.read_csv('data/daily-minimum-temperatures-in-me.csv')
print(df.dtypes)
df['Daily minimum temperatures'] = pd.to_numeric(df['Daily minimum temperatures'], errors='coerce').astype('float32')
arr = df['Daily minimum temperatures'].tolist()

window_size = 14
sequence = [arr[i:i + window_size + 1] for i in range(len(arr) - window_size)]

assert len(sequence) + window_size == len(arr)

input = torch.Tensor(sequence[0:3]).to(device)
input = input.view(3, input.shape[1], 1)
model_out = model(input)
print('output: ', model_out[0])
print('hidden state: ', model_out[1])
