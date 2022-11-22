import os
import shutil
import numpy as np
from scipy import stats
from pandas import Series
from sliding_window import sliding_window
import pandas as pd
from scipy import interpolate
from scipy import signal

def preprocess_SBHAR(dataset_path, window, stride):
    
    load_dataset_path = dataset_path + 'RawData/'
    channel_num=3

    label_data = np.array(pd.read_csv(load_dataset_path + 'labels.txt', delim_whitespace=True, header=None))
    act_record = {}
    for exp_num in range(1,62):
        ind = np.where(label_data[:,0] == exp_num)
        act_record[exp_num] = label_data[ind,2:][0]
    
    user_data = {}
    user_label = {}
    for usr_idx in range(30):
        user_data[usr_idx] = np.empty([0, 128, 3], dtype=np.float)
        user_label[usr_idx] = np.empty([0], dtype=np.int)

    file_list = os.listdir(load_dataset_path)
    for file in file_list:
        if 'acc' in file:
            exp_num = int(file.split('_')[1][3:])
            usr_idx = int(file.split('_')[2][4:-4]) - 1 # 0-29 users
            data = np.array(pd.read_csv(load_dataset_path + file, delim_whitespace=True, header=None))
            
            # filtering
            for i in range(channel_num):
                data[:,i]=signal.medfilt(data[:,i], 3)# median filter
            sos = signal.butter(N=3, Wn=2*20/50, btype='lowpass', output='sos')# 3rd order low-pass Butter-worth filter with a 20 Hz cutoff frequency
            for i in range(channel_num):
                data[:,i] = signal.sosfilt(sos, data[:,i])
            sos = signal.butter(N=4, Wn=2*0.3/50, btype='highpass', output='sos')# separating gravity, low-pass Butterworth filter with a 0.3 Hz corner frequency
            for i in range(channel_num):
                data[:,i] = signal.sosfilt(sos, data[:,i])

            # normalization
            lower_bound = np.array([-0.5323077036833478, -0.4800314262209822, -0.4063855491288771])
            upper_bound = np.array([0.7359294642946127, 0.35672401151151384, 0.3462854467071975])
            diff = upper_bound - lower_bound
            data = 2 * (data - lower_bound) / diff - 1

            # generate labels
            label = np.ones(len(data)) * -1
            for act_seg in act_record[exp_num]:
                label[int(act_seg[1]-1):int(act_seg[2])] = act_seg[0] - 1 # label 0-11

            # sliding window
            data    = sliding_window( data, (window, channel_num), (stride, 1) )
            label   = sliding_window( label, window, stride )
            label   = stats.mode( label, axis=1 )[0][:,0]# choose the most common value as the label of the window

            invalid_idx = np.nonzero( label < 0 )[0]# remove invalid time windows (label==-1)
            data        = np.delete( data, invalid_idx, axis=0 )
            label       = np.delete( label, invalid_idx, axis=0 )

            user_data[usr_idx] = np.concatenate((user_data[usr_idx], data), axis=0)
            user_label[usr_idx] = np.concatenate((user_label[usr_idx], label), axis=0)
            print( "exp{} finished".format( exp_num) )

    for usr_idx in range(30):
        np.save( dataset_path + 'processed_data/' + 'sub{}_features'.format( usr_idx ), user_data[usr_idx] )
        np.save( dataset_path + 'processed_data/' + 'sub{}_labels'.format( usr_idx ), user_label[usr_idx] )  

def preprocess_opportunity(dataset_path, window, stride):

    channel_num     = 3

    file_list = [   ['S1-Drill.dat',
                    'S1-ADL1.dat',
                    'S1-ADL2.dat',
                    'S1-ADL3.dat',
                    'S1-ADL4.dat',
                    'S1-ADL5.dat'] ,
                    ['S2-Drill.dat',
                    'S2-ADL1.dat',
                    'S2-ADL2.dat',
                    'S2-ADL3.dat',
                    'S2-ADL4.dat',
                    'S2-ADL5.dat'] ,
                    ['S3-Drill.dat',
                    'S3-ADL1.dat',
                    'S3-ADL2.dat',
                    'S3-ADL3.dat',
                    'S3-ADL4.dat',
                    'S3-ADL5.dat'] ,
                    ['S4-Drill.dat',
                    'S4-ADL1.dat',
                    'S4-ADL2.dat',
                    'S4-ADL3.dat',
                    'S4-ADL4.dat',
                    'S4-ADL5.dat'] ]

    upper_bound = np.array([975.0, 1348.0, 1203.0])

    lower_bound = np.array([-1484.0, -493.0, -803.0])

    if os.path.exists( dataset_path + 'processed_data/' ):
        shutil.rmtree( dataset_path + 'processed_data/' )
    os.mkdir( dataset_path + 'processed_data/' )

    for usr_idx in range( 4 ):
        
        print( "process data... user{}".format( usr_idx ) )
        time_windows    = np.empty( [0, window, channel_num], dtype=np.float )
        act_labels      = np.empty( [0], dtype=np.int )

        for file_idx in range( 5 ):

            filename = file_list[ usr_idx ][ file_idx ]

            file    = dataset_path + filename
            signal  = np.loadtxt( file )

            index = [i for i in range(63, 66)]
            data = signal.take(index, axis=1)
            label = signal[:, 243].astype( np.int )

            label[ label == 0 ] = 0 # Null
            label[ label == 1 ] = 1 # Stand
            label[ label == 2 ] = 2 # Walk
            label[ label == 4 ] = 3 # Sit
            label[ label == 5 ] = 4 # Lie

            # fill missing values using Linear Interpolation
            data    = np.array( [Series(i).interpolate(method='linear') for i in data.T] ).T
            data[ np.isnan( data ) ] = 0.

            # normalization
            diff = upper_bound - lower_bound
            data = 2 * (data - lower_bound) / diff - 1

            data[ data > 1 ] = 1.0
            data[ data < -1 ] = -1.0

            #sliding window
            data    = sliding_window( data, (window, channel_num), (stride, 1) )
            label   = sliding_window( label, window, stride )
            label   = stats.mode( label, axis=1 )[0][:,0]

            time_windows    = np.concatenate( (time_windows, data), axis=0 )
            act_labels      = np.concatenate( (act_labels, label), axis=0 )

        np.save( dataset_path + 'processed_data/' + 'sub{}_features'.format( usr_idx ), time_windows )
        np.save( dataset_path + 'processed_data/' + 'sub{}_labels'.format( usr_idx ), act_labels )                
        print( "sub{} finished".format( usr_idx) )  
    
def preprocess_realworld(dataset_path, window, overlap):
    
    if os.path.exists( dataset_path + 'processed_data/' ):
        shutil.rmtree( dataset_path + 'processed_data/' )
    os.mkdir( dataset_path + 'processed_data/' )

    lowerBound = np.array( [-13.93775515,  -7.077859,     -14.2442133] )
    upperBound = np.array( [11.016919,     19.608511,     9.479243] )

    window_ms = window * 1000
    label_list = ['climbingdown', 'jumping', 'lying', 'running', 'sitting', 'standing', 'walking', 'climbingup']
    pos_list = ['chest']
    mod_list = ['acc']
    sen_list = []

    for usr_idx in range(15):
        data = []
        label = []
        path = os.path.join(dataset_path, 'proband' + str(usr_idx+1))
        for i in range(len(label_list)):
            act = label_list[i]
            sen_readers = []
            for pos in pos_list:
                for mod in mod_list:
                    if i == 0 and usr_idx == 0:
                        sen_list.append(pos + '_' + mod)
                    sen_readers.append(pd.read_csv(os.path.join(path, mod + '_' + act + '_' + pos + '.csv')))
            t_min = max([sen_readers[i]['attr_time'].min() for i in range(len(sen_readers))])
            t_max = min([sen_readers[i]['attr_time'].max() for i in range(len(sen_readers))])
            window_range = int((t_max - t_min - window_ms) / (window_ms * (1 - overlap)))
            min_size = window * 50 - 5          
            for idx in range(window_range):
                tmpx = []
                start_idx = t_min + idx * window_ms * (1 - overlap)
                stop_idx = start_idx + window_ms
                windows_list = [ sen_readers[i][(sen_readers[i]['attr_time'] >= (start_idx-30)) & (sen_readers[i]['attr_time'] <= (stop_idx+10))] for i in range(len(sen_readers)) ]
                windows_size = [ windows_list[i]['id'].size for i in range(len(sen_readers)) ]
                windows_mint = [ windows_list[i]['attr_time'].min() for i in range(len(sen_readers)) ]
                windows_maxt = [ windows_list[i]['attr_time'].max() for i in range(len(sen_readers)) ]
                if (min(windows_size) >= min_size) and (max(windows_mint) <= start_idx) and (min(windows_maxt) >= stop_idx-20):
                    # resample to even stepsize at 50Hz
                    new_time_steps = start_idx + 20 * np.arange(window * 50)
                    tmpx = np.hstack( [ resample(windows_list[i].iloc[:, 1].values, windows_list[i].iloc[:, 2:].values, new_time_steps) for i in range(len(sen_readers)) ] )
                    if tmpx is not None:
                        data.append(tmpx)
                        label.extend([i])                    
                else:
                    continue

        # normalization
        diff = upperBound - lowerBound
        data = 2 * (data - lowerBound) / diff - 1

        data[ data > 1 ]    = 1.0
        data[ data < -1 ]   = -1.0

        np.save( dataset_path + 'processed_data/' + 'sub{}_features'.format( usr_idx ), data )
        np.save( dataset_path + 'processed_data/' + 'sub{}_labels'.format( usr_idx ), label )
        print( "sub{} finished".format( usr_idx) )  

def resample(x, y, xnew):
    f = interpolate.interp1d(x, y, kind='linear', axis=0)
    ynew = f(xnew)
    return ynew


    
    




