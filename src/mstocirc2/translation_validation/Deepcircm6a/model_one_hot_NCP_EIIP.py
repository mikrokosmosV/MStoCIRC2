from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM_Attention(nn.Module):
    def __init__(self, input_size: int, hidden_num: int, layer_num: int, rnn_dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size=hidden_num,
            num_layers=layer_num,
            bidirectional=True,
            dropout=rnn_dropout,
        )

    def attention_net(self, lstm_output: torch.Tensor, final_state: torch.Tensor) -> torch.Tensor:
        hidden_num = 128
        hidden = final_state.view(-1, hidden_num * 2, 3)
        hidden = torch.mean(hidden, 2).unsqueeze(2)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, (final_hidden_state, _final_cell_state) = self.lstm(x)
        output = output.permute(1, 0, 2)
        attn_output = self.attention_net(output, final_hidden_state)
        return attn_output


class CNN51_RNN(nn.Module):
    def __init__(self, hidden_num: int, layer_num: int, fc_dropout: float, rnn_dropout: float, _cell: str) -> None:
        super().__init__()
        self.basicconv0a = torch.nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=64, kernel_size=(1, 12), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.basicconv0b = torch.nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 6), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.rnn = BiLSTM_Attention(32, hidden_num, layer_num, rnn_dropout)
        self.fc1 = nn.Linear(hidden_num * 2, 10)
        self.fc2 = nn.Linear(10, 2)
        self.dropout = nn.Dropout(fc_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(3).permute(0, 2, 3, 1)
        x = self.basicconv0a(x)
        x = self.basicconv0b(x)
        x = x.squeeze(2).permute(2, 0, 1)
        x = self.rnn(x)
        out = self.fc1(x)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
