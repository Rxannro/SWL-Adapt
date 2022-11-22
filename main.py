from __future__ import print_function
import argparse
import SWL_Adapt
import preprocess
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def main(args):
    if args.dataset =='SBHAR':
        args.WA_N_hid = 3
        args.w_c_T = 0.001
        users_T = np.arange(15, 30, 1) 
        N_users = len(users_T)  
        preprocess.preprocess_SBHAR(args.dataset_path + 'SBHAR/', args.window_S, args.overlap_S)
    elif args.dataset =='OPPORTUNITY':
        args.WA_N_hid = 5
        args.w_c_T = 1
        users_T = np.arange(0, args.N_users_O, 1) 
        N_users = args.N_users_O   
        preprocess.preprocess_opportunity(args.dataset_path + 'OPPORTUNITY/', args.window_O, args.overlap_O)
    elif args.dataset =='realWorld':
        args.WA_N_hid = 7
        args.w_c_T = 1
        user_age = np.array([52, 26, 27, 26, 62, 24, 26, 36, 26, 26, 48, 16, 27, 26, 30])
        users_T = np.where(user_age >= 30)[0] # source users idx, total users 0-14
        N_users = len(users_T)
        preprocess.preprocess_realworld(args.dataset_path + 'realWorld/', args.window_R_s, args.overlap_R_r)

    acc, f1 = [np.empty([N_users], dtype=np.float) for _ in range(2)]
    for s in range(N_users):
        args.test_user = users_T[s]
        model = SWL_Adapt.Solver(args)
        print('\n=== ' + args.dataset + '_SWL-Adapt_sub' + str(args.test_user) + ' ===')
        test_acc, test_f1 = model.train()
        acc[s] = test_acc
        f1[s] = test_f1

    print('\n=== ' + args.dataset + '_SWL-Adapt ===')
    print("FINAL VALUE: \nacc: ", np.around(acc, 3), "\nf1: ", np.around(f1, 3))
    print("FINAL AVERAGE: \nacc: ", np.around(np.mean(acc), 3), "\nf1: ", np.around(np.mean(f1), 3))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Implementation for SWL-Adapt')
    
    parser.add_argument('--window_S', type=int, default=128, help='the number of readings in a time window in UCI HAPT, this dataset is already partitioned into 2.56 s windows')
    parser.add_argument('--overlap_S', type=int, default=64, help='the overlap ratio between time windows in UCI HAPT')
    parser.add_argument('--N_classes_S', type=int, default=6, help='the number of activity classes in UCI HAPT')
    parser.add_argument('--N_channels_S', type=int, default=3, help='the total number of channels in UCI HAPT')
    parser.add_argument('--N_users_S', type=int, default=30, help='the number of users in UCI HAPT')
    parser.add_argument('--window_O', type=int, default=90, help='the length of a time window for OPPORTUNITY (number of readings)')
    parser.add_argument('--overlap_O', type=int, default=45, help='the overlap between time windows for OPPORTUNITY (number of readings)')
    parser.add_argument('--N_classes_O', type=int, default=5, help='the number of activity classes for OPPORTUNITY')
    parser.add_argument('--N_channels_O', type=int, default=3, help='the number of channels in total for OPPORTUNITY')
    parser.add_argument('--N_users_O', type=int, default=4, help='the number of users for OPPORTUNITY')
    parser.add_argument('--window_R', type=int, default=150, help='the length of a time window for realWorld (number of readings)')
    parser.add_argument('--window_R_s', type=int, default=3, help='the length of a time window for realWorld (seconds)')
    parser.add_argument('--overlap_R_r', type=float, default=0.5, help='the overlap between time windows for realWorld (ratio: overlap / window length)')
    parser.add_argument('--N_classes_R', type=int, default=8, help='the number of activity classes for realWorld')
    parser.add_argument('--N_channels_R', type=int, default=3, help='the number of channels in total for realWorld')
    parser.add_argument('--N_users_R', type=int, default=15, help='the number of users for realWorld')
    
    parser.add_argument('--dataset_path', type=str, default='data/', help='the path of the downloaded datasets')
    parser.add_argument('--dataset', type=str, help='the evaluated dataset, this can be set to SBHAR, OPPORTUNITY or realWorld')
    parser.add_argument('--test_user', type=int, default=0, help='the new user')
    parser.add_argument('--seed', type=int, default=1, help='random seed, this is set to 1 to 5 for the 5 repeats in our experiments')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for the subnetworks excluding weight allocator (Adam)')
    parser.add_argument('--WA_lr', type=float, default=0.001, help='learning rate for weight allocator (Adam)')
    parser.add_argument('--N_steps', type=int, default=1000, help='the total number of steps')
    parser.add_argument('--N_steps_eval', type=int, default=10, help='the number of steps between adjacent evaluations')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--confidence_rate', type=float, default=0.7, help='threshold of the classification confidence for the selection of pseudo labeled target samples, this is set to 0.7 for all datasets')
    parser.add_argument('--WA_N_hid', type=int, help='the number of neurons in the hidden layer of weight allocator, this is set to 3 for SBHAR, 5 for OPPORTUNITY, and 7 for RealWorld')
    parser.add_argument('--w_c_T', type=float, help='balancing parameter')

    args = parser.parse_args()
    main(args)