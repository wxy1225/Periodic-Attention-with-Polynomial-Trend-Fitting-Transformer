import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Identity

class Expert(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1, activation="relu"):
        super(Expert, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=input_size, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
 
    def forward(self, x):
        y = x
        x = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        x = self.dropout(self.conv2(x).transpose(-1, 1))
        x = x + y
        return x

class ResidualExpert(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1, activation="relu", kernel_size=3, stride=1, dilation=1):
        super(ResidualExpert, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size, 
                               stride=stride, dilation=dilation, padding=(kernel_size - 1) * dilation // 2, bias=False) 
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, 
                               padding=(kernel_size - 1) * dilation // 2, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.linear = nn.Linear(hidden_size, input_size)
        if input_size == hidden_size:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(nn.Conv1d(input_size, hidden_size, 1),
                                          nn.BatchNorm1d(hidden_size))
                                    
    def forward(self, x):
        identity = x.transpose(-1, 1)
        y = self.conv1(x.transpose(-1, 1))
        y = self.bn1(y)
        y = self.activation(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y += self.shortcut(identity)
        y = self.activation(y).transpose(-1, 1)
        y = self.linear(y)
        return y

class GatingNetwork(nn.Module):
    def __init__(self, input_size, num_experts):
        super(GatingNetwork, self).__init__()
        self.linear = nn.Linear(input_size, num_experts)

    def forward(self, x):
        logits = self.linear(x)
        weights = F.softmax(logits, dim=-1)
        return weights

class TransformNet(nn.Module):
    def __init__(self, input_len, output_len, d_model, dropout=0.1, activation="relu"):
        super(TransformNet, self).__init__()
        self.trans_conv = nn.ConvTranspose1d(input_len, output_len, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        x = self.trans_conv(x)
        out = self.dropout(self.activation(x))
        return out

class MoElayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_experts, dropout=0.1, activation="relu"):
        super(MoElayer, self).__init__()
        self.experts = nn.ModuleList([Expert(input_size, hidden_size, dropout, activation) for _ in range(num_experts)])
        self.gate = GatingNetwork(input_size, num_experts)
    def forward(self, x):
        batch_size, seq_length, input_size = x.size()
        y = x.view(-1, input_size)
        weights = self.gate(y)
        outputs = []
        for i in range(len(self.experts)):
            expert_output = self.experts[i](x)
            expert_output = expert_output.reshape(-1, input_size)
            outputs.append(expert_output * weights[:, i].unsqueeze(1))
        combined_output = torch.sum(torch.stack(outputs, dim=0), dim=0)
        combiner_output = combined_output.view(batch_size, seq_length, input_size)
        return combiner_output

class attentionMoElayer(nn.Module):
    def __init__(self, input_size, num_experts):
        super(attentionMoElayer, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.gate = GatingNetwork(input_size, num_experts)
    def forward(self, input, output):
        x = input
        input_len = input.size(1)
        d_model = input.size(2)
        batch_size, seq_length, input_size = output[0].size()
        model = TransformNet(input_len, seq_length, d_model).to(x.device)
        x = model(x)
        y = x.reshape(-1, self.input_size)
        self.gate = self.gate.to(x.device)
        weights = self.gate(y)
        outputs = []
        for i in range(self.num_experts):
            expert_output = output[i]
            expert_output = expert_output.reshape(-1, input_size)
            outputs.append(expert_output * weights[:, i].unsqueeze(1))
        combined_output = torch.sum(torch.stack(outputs, dim=0), dim=0)
        combiner_output = combined_output.view(batch_size, seq_length, input_size)
        return combiner_output

    