import os
import shutil
import numpy as np
from scipy import stats
from pandas import Series
from sliding_window import sliding_window
import pandas as pd
from scipy import interpolate

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

def preprocess_pamap2(dataset_path, window, stride):
    channel_num     = 3
    
    if os.path.exists( dataset_path + 'processed_data/' ):
        shutil.rmtree( dataset_path + 'processed_data/' )
    os.mkdir( dataset_path + 'processed_data/' )

    lowerBound = np.array( [-4.0144814,     -2.169227,      -10.6296] )

    upperBound = np.array( [4.8361714,      21.236957,      9.54976] )

    file_list = [
        'subject101.dat', 'subject102.dat', 'subject103.dat', 'subject104.dat',
        'subject105.dat', 'subject106.dat', 'subject107.dat', 'subject108.dat'  ]

    for usr_idx in range(len(file_list)):

        file    = dataset_path + file_list[usr_idx]
        data    = np.loadtxt( file )

        label   = data[:,1].astype( int )
        label[label == 0]   = -1
        label[label == 1]   = 0         # lying
        label[label == 2]   = 1         # sitting
        label[label == 3]   = 2         # standing
        label[label == 4]   = 3         # walking
        label[label == 5]   = 4         # running
        label[label == 6]   = 5         # cycling
        label[label == 7]   = 6         # nordic walking
        label[label == 12]  = 7         # ascending stairs
        label[label == 13]  = 8         # descending stairs
        label[label == 16]  = 9         # vacuum cleaning
        label[label == 17]  = 10        # ironing
        label[label == 24]  = 11        # rope jumping

        # fill missing values
        valid_idx   = np.arange(21, 24)
        data        = data[ :, valid_idx ]
        data        = np.array( [Series(i).interpolate() for i in data.T] ).T
        
        # min-max normalization
        diff = upperBound - lowerBound
        data = 2 * (data - lowerBound) / diff - 1

        data[ data > 1 ]    = 1.0
        data[ data < -1 ]   = -1.0

        # sliding window
        data    = sliding_window( data, (window, channel_num), (stride, 1) )
        label   = sliding_window( label, window, stride )
        label   = stats.mode( label, axis=1 )[0][:,0]

        invalid_idx = np.nonzero( label < 0 )[0]
        data        = np.delete( data, invalid_idx, axis=0 )
        label       = np.delete( label, invalid_idx, axis=0 )
        
        np.save( dataset_path + 'processed_data/' + 'sub{}_features'.format( usr_idx ), data )
        np.save( dataset_path + 'processed_data/' + 'sub{}_labels'.format( usr_idx ), label )
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
                    # all resample to even stepsize at 50Hz
                    new_time_steps = start_idx + 20 * np.arange(window * 50)
                    tmpx = np.hstack( [ resample(windows_list[i].iloc[:, 1].values, windows_list[i].iloc[:, 2:].values, new_time_steps) for i in range(len(sen_readers)) ] )
                    if tmpx is not None:
                        data.append(tmpx)
                        label.extend([i])                    
                else:
                    continue

        # min-max normalization
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


    
    




