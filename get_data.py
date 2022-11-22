import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

def get_data_SBHAR(batch_size, test_user, args=None):
    assert args.dataset == 'SBHAR'

    adapt_ratio = 0.5
    path = '/home/rox/data/' + args.dataset + '/processed_data'

    window = args.window_S
    N_channels = args.N_channels_S
    N_users = args.N_users_S
    users_S = np.arange(0, 15, 1)

    x_S = np.empty([0, window, N_channels], dtype=np.float)
    y_S = np.empty([0], dtype=np.int)

    for user_idx in range( N_users ):
        if user_idx == test_user: # load all data for test user
            x_T = np.load(path+'/sub{}_features.npy'.format(test_user))
            y_T = np.load(path+'/sub{}_labels.npy'.format(test_user))
        elif user_idx in users_S: # only load ADL data for source users (activity 0-5)
            x_S = np.concatenate((x_S, np.load(path+'/sub{}_features.npy'.format(user_idx)) ), axis=0)
            y_S = np.concatenate((y_S, np.load(path+'/sub{}_labels.npy'.format(user_idx)) ), axis=0)
            idx = np.where(y_S < 6)[0]
            x_S = x_S[idx]
            y_S = y_S[idx]

    idx = np.where(y_T > 5)[0]
    x_T_trans = x_T[idx]
    y_T_trans = y_T[idx]
    idx = np.where(y_T < 6)[0]
    x_T = x_T[idx]
    y_T = y_T[idx]
    train_x_T, test_x_T, train_y_T, test_y_T = train_test_split(x_T, y_T, train_size = adapt_ratio, random_state = 0)
    train_x_T = np.concatenate((train_x_T, x_T_trans), axis=0)
    train_y_T = np.concatenate((train_y_T, y_T_trans), axis=0)

    train_x_S = x_S
    train_y_S = y_S

    train_dataset_S = TensorDataset(torch.from_numpy(train_x_S.astype(np.float32)), torch.from_numpy(train_y_S))   
    train_loader_S = DataLoader(train_dataset_S, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    train_dataset_T = TensorDataset(torch.from_numpy(train_x_T.astype(np.float32)), torch.from_numpy(train_y_T))
    train_loader_T = DataLoader(train_dataset_T, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    test_dataset = TensorDataset(torch.from_numpy(test_x_T.astype(np.float32)), torch.from_numpy(test_y_T))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0) # to use torchmetrics all batches must have the same size

    return train_loader_S, train_loader_T, test_loader

def get_data_OPPORTUNITY(batch_size, test_user, args=None):
    assert args.dataset == 'OPPORTUNITY'

    adapt_ratio = 0.5
    path = args.dataset_path + args.dataset + '/processed_data'

    window = args.window_O
    N_channels = args.N_channels_O
    N_users = args.N_users_O

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

def get_data_realWorld(batch_size, test_user, args=None):
    assert args.dataset == 'realWorld'

    adapt_ratio = 0.5
    path = args.dataset_path + args.dataset + '/processed_data'

    window = args.window_R
    N_channels = args.N_channels_R
    N_users = args.N_users_R

    user_age = np.array([52, 26, 27, 26, 62, 24, 26, 36, 26, 26, 48, 16, 27, 26, 30])
    users_S = np.where(user_age < 30)[0] # source users idx, total users 0-14

    x_S = np.empty([0, window, N_channels], dtype=np.float)
    y_S = np.empty([0], dtype=np.int)

    for user_idx in range( N_users ):
        if user_idx == test_user:
            x_T = np.load(path+'/sub{}_features.npy'.format(test_user))
            y_T = np.load(path+'/sub{}_labels.npy'.format(test_user))
        elif user_idx in users_S:
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
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0) # to use torchmetrics all batches must have the same size

    return train_loader_S, train_loader_T, test_loader