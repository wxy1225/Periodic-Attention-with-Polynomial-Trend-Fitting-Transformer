import torch
import torch.nn as nn
import torch.nn.functional as F

class ptm(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, k=4, forecast_dim=96):
        super(ptm, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k  
        self.forecast_dim = forecast_dim

        self.mu_generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, k) 
        )
        
        self.sigma_generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, k) 
        )

        self.polynomial_fitters = nn.ModuleList([
            SignedPolynomialFitter(degree=1, seq_len=input_dim, forecast_dim=forecast_dim, positive_coeffs=True), 
            SignedPolynomialFitter(degree=2, seq_len=input_dim, forecast_dim=forecast_dim, positive_coeffs=True),  
            SignedPolynomialFitter(degree=3, seq_len=input_dim, forecast_dim=forecast_dim, positive_coeffs=True),  
            SignedPolynomialFitter(degree=1, seq_len=input_dim, forecast_dim=forecast_dim, positive_coeffs=False),   
            SignedPolynomialFitter(degree=2, seq_len=input_dim, forecast_dim=forecast_dim, positive_coeffs=False),
            SignedPolynomialFitter(degree=3, seq_len=input_dim, forecast_dim=forecast_dim, positive_coeffs=False), 
        ])

        self.weight_generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) 
        )
        
        self.ortho_step_count = 0
        self.ortho_freq = 5

    def forward(self, x):

        B, N, T = x.shape
        x = x.permute(0, 2, 1)  # [B, T, N]
        temporal_features = []
        for channel in range(N):
            x_channel = x[:, :, channel]  # [B, T]
            
            mu = self.mu_generator(x_channel).unsqueeze(-1)  
            sigma = self.sigma_generator(x_channel).unsqueeze(-1)  

            epsilon = torch.randn(B, self.k, T, device=x.device)
            z = F.softplus(mu) + epsilon * F.softplus(sigma) 
            weights = self.weight_generator(z)
            weights = F.softmax(weights, dim=1)  # [B, k, 1]

            poly_outputs = []
            time_indices = torch.linspace(0, 1, self.forecast_dim, device=x.device).unsqueeze(0).expand(B, -1)

            for i in range(self.k):
                feat = self.polynomial_fitters[i](z[:, i, :], time_indices)  # [B, forecast_dim]
                poly_outputs.append(feat.unsqueeze(1))  # [B, 1, forecast_dim]
            
            combined_features = torch.cat(poly_outputs, dim=1)  # [B, k, forecast_dim]
            weighted_features = torch.sum(weights * combined_features, dim=1)  # [B, forecast_dim]
            temporal_features.append(weighted_features.unsqueeze(1))
        
        result = torch.cat(temporal_features, dim=1).permute(0, 2, 1)
        
        if self.training and self.ortho_step_count % self.ortho_freq == 0:
            self.apply_orthogonal_constraint()
        self.ortho_step_count += 1
        
        return result
    
    def apply_orthogonal_constraint(self):
        
        degree_groups = {} 
        
        for i, fitter in enumerate(self.polynomial_fitters):
            degree = fitter.degree
            last_linear = None
            for module in reversed(list(fitter.coeff_generator.modules())):
                if isinstance(module, nn.Linear):
                    last_linear = module
                    break
            
            if last_linear is not None:
                weight = last_linear.weight 
                bias = last_linear.bias      
                highest_coeff_idx = weight.shape[0] - 1
                if degree not in degree_groups:
                    degree_groups[degree] = []
                degree_groups[degree].append((weight, bias, highest_coeff_idx, i))

        with torch.no_grad():
            for degree, group in degree_groups.items():
                if len(group) < 2:  
                    continue

                weights_to_orthogonalize = []
                highest_coeff_indices = []
                original_to_group_idx = {}
                
                for group_idx, (weight, bias, idx, fitter_idx) in enumerate(group):
                    weights_to_orthogonalize.append((weight, bias))
                    highest_coeff_indices.append((idx, group_idx))  
                    original_to_group_idx[fitter_idx] = group_idx

                original_highest_coeffs = {}
                for idx, group_idx in highest_coeff_indices:
                    weight, _ = weights_to_orthogonalize[group_idx]
                    original_highest_coeffs[(idx, group_idx)] = weight[idx].clone()

                vectors = []
                for weight, bias in weights_to_orthogonalize:
                    vector = weight.flatten()
                    if bias is not None:
                        vector = torch.cat([vector, bias])
                    vectors.append(vector)
                
                orthogonal_vectors = []
                for i, v in enumerate(vectors):
                    u = v.clone()
                    for j in range(i):
                        proj = torch.dot(u, orthogonal_vectors[j]) / torch.dot(orthogonal_vectors[j], orthogonal_vectors[j])
                        u = u - proj * orthogonal_vectors[j]

                    u = u / (torch.norm(u) + 1e-8)
                    orthogonal_vectors.append(u)

                new_weights = []
                new_biases = []
                
                for i, (weight, bias) in enumerate(weights_to_orthogonalize):
                    flat_vector = orthogonal_vectors[i]
                    
                    weight_size = weight.numel()
                    weight_flat = flat_vector[:weight_size]
                    new_weight = weight_flat.reshape(weight.shape)

                    new_bias = None
                    if bias is not None:
                        bias_size = bias.numel()
                        bias_flat = flat_vector[weight_size:weight_size+bias_size]
                        new_bias = bias_flat.reshape(bias.shape)
                    
                    new_weights.append(new_weight)
                    new_biases.append(new_bias)

                for i, (weight, bias) in enumerate(weights_to_orthogonalize):
                    weight.data.copy_(new_weights[i])
                    if bias is not None:
                        bias.data.copy_(new_biases[i])

                for (idx, group_idx), value in original_highest_coeffs.items():
                    weight, _ = weights_to_orthogonalize[group_idx]
                    weight[idx].data.copy_(value)

class SignedPolynomialFitter(nn.Module):
    def __init__(self, degree, seq_len=96, forecast_dim=512, positive_coeffs=True):
        super().__init__()
        self.seq_len = seq_len
        self.forecast_dim = forecast_dim
        self.degree = degree
        self.positive_coeffs = positive_coeffs
        
        # a0 + a1*t + a2*t^2 + a3*t^3
        self.coeff_generator = nn.Sequential(
            nn.Linear(self.seq_len, self.forecast_dim),  
            nn.ReLU(),
            nn.Linear(self.forecast_dim, degree)  
        )
        
        self._init_coeffs()

    def _init_coeffs(self):
        for m in self.coeff_generator.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, sequence, time_indices):

        B, _ = sequence.shape
        seq_stats = sequence
        lower_coeffs = self.coeff_generator(seq_stats)  # [B, degree]
        pred = torch.zeros_like(time_indices)  # [B, forecast_dim]
        for d in range(self.degree):
            pred += lower_coeffs[:, d].unsqueeze(1) * (time_indices ** d)
        
        highest_coeff = 0.5 if self.positive_coeffs else -0.5
        pred += highest_coeff * (time_indices ** self.degree)  

        return pred