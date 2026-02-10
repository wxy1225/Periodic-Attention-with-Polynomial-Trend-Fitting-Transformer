import torch
import torch.nn as nn
import math

class PAPE(nn.Module):
    def __init__(self, seasonal_inf, d_model, head_dim=8):
        super(PAPE, self).__init__()
        self.d_model = d_model
        self.seasonal_inf = seasonal_inf
        self.head_dim = head_dim
    
    def rope(self, seq_len):
        D = self.d_model
        position = torch.arange(0, seq_len, dtype=torch.float)
        freqs = 1.0 / 10000 ** torch.arange(0, D, 2).float() / D
        freqs = torch.outer(position, freqs)
        freqs = torch.polar(torch.ones_like(freqs), freqs).unsqueeze(0).unsqueeze(2)
        return freqs

    def seasonalPAPE(self, seq_len):
        seasonal_inf = self.seasonal_inf
        position = torch.arange(0, seq_len, dtype=torch.float)
        D = self.d_model
        seasonal_inf = seasonal_inf.tolist()
        n = len(seasonal_inf)
        freqs = 2 * math.pi/seasonal_inf[0] * torch.arange(0, D/n, 2).float()
        if n > 1:
            for i in range(1, n):
                freqs_1 = 2 * math.pi/seasonal_inf[i] * torch.arange(0, D/n, 2).float()
                freqs = torch.cat([freqs, freqs_1], dim = 0)
        freqs = freqs[:int(D/2)]
        freqs = torch.outer(position, freqs)
        freqs = torch.polar(torch.ones_like(freqs), freqs).unsqueeze(0).unsqueeze(2)
        return freqs
    
    def headPAPE(self, seq_len):
        D = self.d_model
        seasonal_inf = self.seasonal_inf
        head_dim = self.head_dim
        position = torch.arange(0, seq_len, dtype=torch.float)
        freqs = 2 * math.pi/seasonal_inf[0] * torch.arange(0, D, 2).float()
        freq = 2 * math.pi/seasonal_inf[0] * torch.arange(0, D, 2).float()
        for i in range(int(head_dim/len(seasonal_inf))-1):
            freqs = torch.cat([freqs, freq], dim = 0)
        for i in range(1, len(seasonal_inf)):
            freq = 2 * math.pi/seasonal_inf[i] * torch.arange(0, D, 2).float()
            for i in range(int(head_dim/len(seasonal_inf))):
                freqs = torch.cat([freqs, freq], dim = 0)
        freqs = freqs[:int(D/2)]
        freqs = torch.outer(position, freqs)
        length = freqs.shape[1]
        freqs = freqs.reshape(seq_len, head_dim, int(length/head_dim)).unsqueeze(0)
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs
    
    def forward(self, queries, keys, type='default'): # B L H D
        type_dict = {
            'default': self.rope,
            'seasonal': self.seasonalPAPE,
            "head": self.headPAPE,
        }
        q = torch.view_as_complex(queries.float().reshape(*queries.shape[:-1], -1, 2))
        k = torch.view_as_complex(keys.float().reshape(*keys.shape[:-1], -1, 2))

        B, L, H, E = queries.shape
        _, S, _, D = keys.shape
  
        freqs_q = type_dict[type](L).to(queries.device)

        freqs_k = type_dict[type](S).to(keys.device)
        q_out = torch.view_as_real(q * freqs_q).flatten(3)
        k_out = torch.view_as_real(k * freqs_k).flatten(3)
        return q_out.float(), k_out.float()