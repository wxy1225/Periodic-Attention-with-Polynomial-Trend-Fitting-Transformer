import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

plt.switch_backend('agg')

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter +=1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
def visual(true, preds=None, name='./pic/test.pdf'):
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2, color='orange')
    plt.plot(true, label='GroundTruth', linewidth=2, color='blue')
    plt.legend()
    plt.savefig(name, format='png', bbox_inches='tight')

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2).sum(0) * ((pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, corr

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

def attention_vision(input_seq, attention, output_seq, layer, step, path):
    plt.figure(figsize=(12, 4), dpi=300)
    input_color = (
        attention['cross_attn'][layer][0].mean(dim=0)
        [step + 48][:-1]
        .squeeze()
        .cpu()
    )
    input_norm = plt.Normalize(input_color.min(), input_color.max())
    input_color = cm.Reds(input_norm(input_color))
    output_color = []
    for i in range(step):
        output_color.append(0.2)
    output_color.append(0.9)
    output_color = torch.tensor(output_color).cpu()
    output_color = cm.Blues(output_color)
    input_l = len(input_seq)
    for i in range(len(input_seq) - 1):
        plt.plot(np.arange(i, i + 2), input_seq[i : i + 2], color=input_color[i])
    for i in range(step + 1):
        plt.plot(np.arange(i + input_l, i + 2 + input_l), output_seq[i : i + 2], color=output_color[i])
    plt.savefig(path, bbox_inches='tight')

def plot_attention_matrix(attention_matrix, tokens=None, title="Attention Matrix", figsize=(12, 10), save_path=None):
    seq_len = attention_matrix.shape[0] 
    
    colors = [
        (0.8, 0.9, 1.0),   
        (0.1, 0.2, 0.6),   
        (0.05, 0.1, 0.3)     
    ]
    cmap = LinearSegmentedColormap.from_list("deep_blue", colors, N=200)
    
    plt.figure(figsize=figsize)
    
    ax = sns.heatmap(
        attention_matrix,
        cmap=cmap,       
        vmin=0,         
        vmax=attention_matrix.max(),  
        annot=False,
        linewidths=0.5,
        square=True,
        xticklabels=False,
        yticklabels=False
    )
    
    ax.set_xticks(np.arange(seq_len) + 0.5)
    ax.set_yticks(np.arange(seq_len) + 0.5)
    
    if tokens is not None and len(tokens) == seq_len:
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(tokens, rotation=0, fontsize=8)
    else:
        ax.set_xticklabels([f"Pos {i}" for i in range(seq_len)], rotation=45, ha="right")
        ax.set_yticklabels([f"Pos {i}" for i in range(seq_len)], rotation=0)
    
    plt.title(title, fontsize=14)
    plt.xlabel("Key Position", fontsize=12)
    plt.ylabel("Query Position", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches="tight")
        plt.savefig(save_path.replace('.svg', '.png'), format='png', bbox_inches="tight")
    plt.show()
