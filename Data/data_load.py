import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader
from statsmodels.tsa.seasonal import STL
from Data.data_tool import series_decomp, get_cycle
import pywt
import statsmodels.api as sm
import numpy as np
import warnings

warnings.filterwarnings('ignore')

class Dataset_(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='container1.csv', target='load'):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        length = len(df_raw)
        len_1 = int(length * 0.7)
        len_2 = int(length * 0.8)
        len_3 = int(length)
        border1s = [0, len_1 - self.seq_len, len_2 - self.seq_len]
        border2s = [len_1, len_2, len_3]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        train_data = df_data[border1s[0]: border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
        data_stamp = df_stamp.drop(['date'], axis=1).values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
 
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def data_provider(args, flag):
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    
    data_set = Dataset_(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target
    )

    data_loader = DataLoader(
        data_set,
        batch_size = batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    return data_set, data_loader