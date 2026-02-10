import torch
import torch.nn as nn
import numpy as np
from PTNformer.layers.PAPT_enc_dec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.attention import Fullattention, AttentionLayer
from layers.moe import MoElayer
from layers.Embed import DataEmbedding, DataEmbedding_inverted, MixedEmbedding, TokenEmbedding
from PTNformer.layers.TGM import tgm
from PTNformer.layers.PTM import ptm

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.gate = 6 
        self.batch_size = configs.batch_size
        self.output_attention = configs.output_attention
        
        self.enc_embedding_2 = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.enc_embedding_1 = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.dec_embedding_2 = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding_1 = DataEmbedding_inverted(configs.label_len + configs.pred_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Fullattention(False, attention_dropout=configs.dropout), configs.d_model, configs.n_heads),
                    MoElayer(configs.d_model, configs.d_ff, configs.num_experts, configs.dropout, configs.activation),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Fullattention(True, attention_dropout=configs.dropout),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        Fullattention(False, attention_dropout=configs.dropout),
                        configs.d_model, configs.n_heads),
                    MoElayer(configs.d_model, configs.d_ff, configs.num_experts, configs.dropout, configs.activation),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        self.hierarchical_vae = tgm(configs.d_model,  configs.latent_dim, configs.latent_dim)
        self.enc_projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        self.dec_projection = nn.Linear(configs.d_model, configs.label_len + configs.pred_len, bias=True)

        self.trendlayer = ptm(configs.seq_len, configs.d_model, self.gate, configs.seq_len)
        self.token_embedding = TokenEmbedding(configs.enc_in, configs.d_model)
        self.projection1 = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, attn_embed=False, seasonal_inf=None, type='default', 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        B, T, N = x_enc.shape
        window = 25
        trend_ma = np.zeros((B, T, N))
        pad_width = window // 2
        data = x_enc.cpu()
        for b in range(B):
            for n in range(N):
                x = data[b, :, n]
                x_padded = np.pad(x, pad_width, mode='reflect')
                ma = np.convolve(x_padded, np.ones(window)/window, mode='valid')
                trend_ma[b, :, n] = ma
        trend_data = torch.from_numpy(trend_ma).float().to(x_enc.device)
        trend = self.trendlayer(trend_data.permute(0, 2, 1))
        trend_emb = self.token_embedding(trend)

        enc_out = MixedEmbedding(x_enc, x_mark_enc, self.enc_embedding_1, self.enc_embedding_2, self.enc_projection)
        enc_out = enc_out + trend_emb
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, embed=attn_embed, seasonal_inf=seasonal_inf, type='seasonal')

        enc_out = self.hierarchical_vae(enc_out)
        dec_out = MixedEmbedding(x_dec, x_mark_dec, self.dec_embedding_1, self.dec_embedding_2, self.dec_projection)
        dec_out, self_attns, cross_attns = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask
        , embed=attn_embed, seasonal_inf=seasonal_inf, type='seasonal')

        dec_out = dec_out[:, -self.pred_len:, :] 
        attention = {
            'enc_attn': attns,
            'dec_attn': self_attns,
            'cross_attn': cross_attns
        }

        if self.output_attention:
            return dec_out, attention   
            
        else:
            return dec_out # size: [B, L, D]
      