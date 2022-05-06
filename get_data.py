import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

def get_data(batch_size, test_user, args=None):
    adapt_ratio = 0.5
    path = args.dataset_path + args.dataset + '/processed_data'

    if args.dataset == 'PAMAP2':
        window = args.window_P
        N_channels = args.N_channels_P
        N_users = args.N_users_P
    elif args.dataset == 'OPPORTUNITY':
        window = args.window_O
        N_channels = args.N_channels_O
        N_users = args.N_users_O
    elif args.dataset == 'realWorld':
        window = args.window_R
        N_channels = args.N_channels_R
        N_users = args.N_users_R

    x_S = np.empty([0, window, N_channels], dtype=np.float)
    y_S = np.empty([0], dtype=np.int)

    for user_idx in range( N_users ):
        if user_idx == test_user:
            x_T = np.load(path+'/sub{}_features.npy'.format(test_user))
            y_T = np.load(path+'/sub{}_labels.npy'.format(test_user))
        else:
            x_S = np.concatenate((x_S, np.load(path+'/sub{}_features.npy'.format(user_idx)) ), axis=0)
            y_S = np.concatenate((y_S, np.load(path+'/sub{}_labels.npy'.format(user_idx)) ), axis=0)

    train_x_T, test_x_T, train_y_T, test_y_T = train_test_split(x_T, y_T, train_size = adapt_ratio, random_state = 0)
    train_x_S = x_S
    train_y_S = y_S

    train_dataset_S = TensorDataset(torch.from_numpy(train_x_S.astype(np.float32)), torch.from_numpy(train_y_S))   
    train_loader_S = DataLoader(train_dataset_S, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    train_dataset_T = TensorDataset(torch.from_numpy(train_x_T.astype(np.float32)), torch.from_numpy(train_y_T))
    train_loader_T = DataLoader(train_dataset_T, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    test_dataset = TensorDataset(torch.from_numpy(test_x_T.astype(np.float32)), torch.from_numpy(test_y_T))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    return train_loader_S, train_loader_T, test_loader