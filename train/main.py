from Data.data_load import data_provider
from PTNformer.model import PAPT 
from train.tool import EarlyStopping, visual, metric, plot_attention_matrix
from sklearn.preprocessing import StandardScaler
import logging
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings

warnings.filterwarnings('ignore')

class exp_main(nn.Module):
    def __init__(self, args):
        super(exp_main, self).__init__()
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model()
        self.model_type = self.args.model
        self.batch_size = self.args.batch_size
        self.scaler1 = []
        self.scaler2 = []
        for i in range(self.batch_size):
            scaler = StandardScaler()
            self.scaler1.append(scaler)  

    def _build_model(self):
        model_dict = {
            'PAPT': PAPT, 
        }
        model = model_dict[self.args.model].Model(self.args).float().to(self.device)
        return model

    def batch_scaler(self, batch_x, batch_y):
        batch_x = batch_x.numpy()
        batch_y = batch_y.numpy()
        for i in range(batch_x.shape[0]):
            self.scaler1[i].fit(batch_x[i])
            data = self.scaler1[i].transform(batch_x[i])
            batch_x[i] = data

            self.scaler1[i].fit(batch_y[i])
            data = self.scaler1[i].transform(batch_y[i])
            batch_y[i] = data
            
        batch_x = torch.from_numpy(batch_x)
        batch_y = torch.from_numpy(batch_y)
        return batch_x, batch_y

    def batch_inverse_scaler(self, batch_y, outputs):
        batch_y = batch_y.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()  
        for i in range(batch_y.shape[0]):
            scaler = StandardScaler()
            scaler.mean_ = self.scaler1[i].mean_[-1].tolist()
            scaler.scale_ = self.scaler1[i].scale_[-1].tolist()
            data = scaler.inverse_transform(batch_y[i])
            batch_y[i] = data
            data = scaler.inverse_transform(outputs[i])
            outputs[i] = data
            
        return batch_y, outputs
    
    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def _predict(self, batch_x, batch_y, batch_x_mark, batch_y_mark, attn_embed=False):

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
        if(self.model_type == 'PAPT'):
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, attn_embed=attn_embed, seasonal_inf=self.args.seasonal_inf)  
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)    

        if self.args.output_attention:
            output = outputs[0]
            attention = outputs[1]

        else:
            output = outputs
            attention = None

        f_dim = -1 if self.args.features == 'MS' else 0
        output = output[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return output, attention, batch_y
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

                batch_x, batch_y = self.batch_scaler(batch_x, batch_y)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, _, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark, attn_embed=True)  

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true) 
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                batch_x, batch_y = self.batch_scaler(batch_x, batch_y)

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, _, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark, attn_embed=True)

                loss = criterion(outputs, batch_y) 

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(model_path))
        return
    
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        preds = []
        trues = []
        folder_path = './PAPT/result/' + setting + '/'
        if not(os.path.exists(folder_path)):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                inputs = batch_x.detach().cpu().numpy().copy()

                batch_x, batch_y = self.batch_scaler(batch_x, batch_y)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, attention, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark, attn_embed=True)

                batch_y, outputs = self.batch_inverse_scaler(batch_y, outputs)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 10 == 0:
                    gt = np.concatenate((inputs[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((inputs[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.png'))
        #tokens = None
        #attention = attention['enc_attn'][-1].mean(dim=0)[-1].cpu().numpy() 
        #plot_attention_matrix(attention, tokens=tokens, title="Example Self-Attention Matrix",figsize=(12, 10), save_path=os.path.join(folder_path + '.svg'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f = open("./PAPT/result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()
        predic = preds[:, 0, -1]
        real = trues[:, 0, -1]
        visual(real, predic, os.path.join(folder_path, "result_1" + '.png'))
        return

    def pred(self, setting, test_loader):
        path = os.path.join('./PAPT/checkpoints', setting)
        model_path = path + '/' + 'checkpoint.pth'
        logging.info(model_path)
        self.model.load_state_dict(torch.load(model_path))

        preds = []
        trues = []
        input = []
        attentions = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                inputs = batch_x.detach().cpu().numpy().copy()
                batch_x, batch_y = self.batch_scaler(batch_x, batch_y)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, attention, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark, attn_embed=True)

                batch_y, outputs = self.batch_inverse_scaler(batch_y, outputs)

                pred = outputs
                true = batch_y
                
                attentions.append(attention)
                input.append(inputs)
                preds.append(pred)
                trues.append(true)
        return input, attentions, preds, trues
    