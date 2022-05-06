from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchmetrics
from get_data import get_data
from model_opp import FeatureExtracter
from model_opp import Discriminator
from model_opp import ActivityClassifier
from model_opp import WeightAllocator
from model_opp import ReverseLayerF
import higher

class Solver(object):
    def __init__(self, args):
        self.seed = args.seed
        self.N_steps = args.N_steps
        self.N_steps_eval = args.N_steps_eval
        self.N_eval = int(args.N_steps/args.N_steps_eval)
        self.test_user = args.test_user

        self.batch_size = args.batch_size
        self.lr = args.lr
        self.confidence_rate = args.confidence_rate

        self.dataset = args.dataset
        self.tag = args.dataset + '_SWL-Adapt_user' + str(args.test_user)

        if args.dataset == 'PAMAP2':
            self.N_channels = args.N_channels_P
            self.N_sensors = args.N_sensors_P
            self.N_classes = args.N_classes_P
        if args.dataset == 'OPPORTUNITY':
            self.N_channels = args.N_channels_O
            self.N_sensors = args.N_sensors_O
            self.N_classes = args.N_classes_O
        if args.dataset == 'realWorld':
            self.N_channels = args.N_channels_R
            self.N_sensors = args.N_sensors_R
            self.N_classes = args.N_classes_R
        
        self.train_loader_S, self.train_loader_T, self.test_loader = get_data(self.batch_size, args.test_user, args)
    
        self.FE = FeatureExtracter(self.N_channels)
        self.D = Discriminator()
        self.AC = ActivityClassifier(self.N_classes)
        self.WA = WeightAllocator(args.WA_N_hid)
        
        self.FE.cuda()
        self.D.cuda()
        self.AC.cuda()
        self.WA.cuda()
        
        self.opt_fe = optim.Adam(self.FE.parameters(), lr=self.lr)
        self.opt_d = optim.Adam(self.D.parameters(), lr=self.lr)
        self.opt_ac = optim.Adam(self.AC.parameters(), lr=self.lr)
        self.opt_wa = optim.Adam(self.WA.parameters(), lr=args.WA_lr)

        self.scheduler_fe = optim.lr_scheduler.CosineAnnealingLR(self.opt_fe, self.N_eval)
        self.scheduler_d = optim.lr_scheduler.CosineAnnealingLR(self.opt_d, self.N_eval)
        self.scheduler_ac = optim.lr_scheduler.CosineAnnealingLR(self.opt_ac, self.N_eval)
        self.scheduler_wa = optim.lr_scheduler.CosineAnnealingLR(self.opt_wa, self.N_eval)

    def reset_grad(self):
        self.opt_fe.zero_grad()
        self.opt_d.zero_grad()
        self.opt_ac.zero_grad()
        self.opt_wa.zero_grad()

    def forward_pass(self, inputs, out_type=None):
        fused_feature = self.FE(inputs)
        disc = None
        activity_clsf = None
        if out_type != 'C':
            reverse_feature = ReverseLayerF.apply(fused_feature, 1)
            disc = self.D(reverse_feature)
        if out_type != 'D':
            activity_clsf = self.AC(fused_feature)
        return disc, activity_clsf

    def ld_weight(self, logits_d, logits_ac, y=None, yd=None):
        with torch.no_grad():
            criterion_d = nn.BCEWithLogitsLoss(reduction='none').cuda()
            loss_d = criterion_d(logits_d, yd)
            criterion = nn.CrossEntropyLoss(reduce=False).cuda()
            if y is None:
                y = logits_ac.max(1)[1]
            loss_c = criterion(logits_ac, y)
        
        d_w = self.WA(align_G=loss_d, clsf=loss_c)

        scale = d_w.sum(dim=0)
        if scale == 0:
            scale = scale + 0.05
            print('zero weights!')
        d_w = d_w * self.batch_size / scale.repeat(128,1)

        return d_w.reshape(-1)

    def get_oll(self, logits_ac_T):
        pseudo_y_T = logits_ac_T.max(1)[1]
        certainty_y_T = logits_ac_T.softmax(dim=1).max(1)[0]
        mask_T = certainty_y_T > self.confidence_rate
        loss_c = torch.sum(F.cross_entropy(logits_ac_T, pseudo_y_T, reduction='none') * mask_T.float().detach())
        return loss_c

    def train(self):
        print('\n>>> Start Training ...')
        test_acc, test_f1 = 0, 0
        
        criterion_c = nn.CrossEntropyLoss().cuda()
        criterion_ld = nn.BCEWithLogitsLoss(reduction='none').cuda()

        train_c_acc_S = torchmetrics.Accuracy().cuda()
        train_c_f1_S = torchmetrics.F1(num_classes=self.N_classes, average='macro').cuda()
        
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        step = 0
        self.train_loader_S_iter = iter(self.train_loader_S)
        self.train_loader_T_iter = iter(self.train_loader_T)
        for n_eval in range(self.N_eval):

            self.FE.train()
            self.D.train()
            self.AC.train()
            self.WA.train()

            Loss_c = 0
            Loss_d = 0
            for batch_idx in range(self.N_steps_eval):
                step += 1

                x_T, y_T = None, None
                try:
                    x_T, y_T = next(self.train_loader_T_iter)
                except StopIteration:
                    self.train_loader_T_iter = iter(self.train_loader_T)
                    x_T, y_T = next(self.train_loader_T_iter)

                x_S, y_S = None, None
                try:
                    x_S, y_S = next(self.train_loader_S_iter)
                except StopIteration:
                    self.train_loader_S_iter = iter(self.train_loader_S)
                    x_S, y_S = next(self.train_loader_S_iter)

                x_S = Variable(x_S.cuda())
                y_S = Variable(y_S.long().cuda())
                x_T = Variable(x_T.cuda())
                yd_S = torch.zeros(self.batch_size)
                yd_T = torch.ones(self.batch_size)                
                yd_S = Variable(yd_S.cuda())
                yd_T = Variable(yd_T.cuda())

                self.reset_grad()
                
                """ step 1: update feature extractor and classifier w.r.t. classification loss """
                _, logits_ac_S = self.forward_pass(x_S, 'C')
                loss_c_S = criterion_c(logits_ac_S, y_S)
                
                _, logits_ac_T = self.forward_pass(x_T)
                with torch.no_grad():
                    pseudo_y_T = logits_ac_T.max(1)[1]
                    certainty_y_T = logits_ac_T.softmax(dim=1).max(1)[0]
                    mask_C = (certainty_y_T > self.confidence_rate).float()
                if mask_C.sum() > 0:
                    loss_c_T = torch.sum(F.cross_entropy(logits_ac_T, pseudo_y_T, reduction='none') * mask_C)/mask_C.sum()
                else:
                    loss_c_T = 0
                loss_c = loss_c_S + loss_c_T

                loss_c.backward()
                self.opt_fe.step()
                self.opt_ac.step()
                self.reset_grad()

                # track training losses and metrics
                Loss_c += loss_c.item()
                train_c_acc_S(logits_ac_S.softmax(dim=-1), y_S)
                train_c_f1_S(logits_ac_S.softmax(dim=-1), y_S)

                """ step 2: update weight allocator w.r.t. meta-classification loss (first optimize feature extractor w.r.t. weighted domain alignment loss) """
                torch.cuda.empty_cache()

                with higher.innerloop_ctx(self.FE, self.opt_fe) as (fmodel, diffopt): # make a copy of feature extractor and its optimizer
                    # forward pass to domain alignment loss with the copied feature extractor
                    fused_feature = fmodel(x_S)
                    reverse_feature = ReverseLayerF.apply(fused_feature, 1)
                    logits_d_S = self.D(reverse_feature)
                    logits_ac_S = self.AC(fused_feature)
                    d_w_S = self.ld_weight(logits_d_S, logits_ac_S, y=y_S, yd=yd_S)
                    loss_d_S = criterion_ld(logits_d_S, yd_S).mul(d_w_S)
                    loss_d_S = loss_d_S.mean()

                    fused_feature = fmodel(x_T)
                    reverse_feature = ReverseLayerF.apply(fused_feature, 1)
                    logits_d_T = self.D(reverse_feature)
                    logits_ac_T = self.AC(fused_feature)
                    d_w_T = self.ld_weight(logits_d_T, logits_ac_T, yd=yd_T)
                    loss_d_T = criterion_ld(logits_d_T, yd_T).mul(d_w_T)
                    loss_d_T = loss_d_T.mean()

                    loss_d = loss_d_S + loss_d_T

                    # update the copied feature extractor and leave the original feature extractor unchanged
                    diffopt.step(loss_d)

                    # forward pass to meta-classification loss
                    fused_feature = fmodel(x_T)
                    logits_ac_T = self.AC(fused_feature)
                    loss_c = self.get_oll(logits_ac_T)

                    # update weight allocator
                    loss_c.backward()
                    self.opt_wa.step()
                    self.reset_grad()

                """ step 3: update feature extractor and domain discriminator w.r.t. weighted domain alignment loss """
                logits_d_S, logits_ac_S = self.forward_pass(x_S)
                d_w_S = self.ld_weight(logits_d_S, logits_ac_S, y=y_S, yd=yd_S)
                loss_d_S = criterion_ld(logits_d_S, yd_S).mul(d_w_S)
                loss_d_S = loss_d_S.mean()

                logits_d_T, logits_ac_T = self.forward_pass(x_T)
                d_w_T = self.ld_weight(logits_d_T, logits_ac_T, yd=yd_T)
                loss_d_T = criterion_ld(logits_d_T, yd_T).mul(d_w_T)
                loss_d_T = loss_d_T.mean()

                loss_d = loss_d_S + loss_d_T

                loss_d.backward()
                self.opt_fe.step()
                self.opt_d.step()
                self.reset_grad()            

                # track training losses and metrics after optimization
                Loss_d += loss_d.item()

            test_Loss_c, test_c_acc_T, test_c_f1_T = self.eval()

            print('Train Eval {}: Train: c_acc_S:{:.6f} c_f1_S:{:.6f} Loss_c:{:.6f} Loss_d:{:.6f}'.format(
                n_eval, train_c_acc_S.compute().item(), train_c_f1_S.compute().item(), Loss_c, Loss_d))
            print('               Test:  c_acc_T:{:.6f} c_f1_T:{:.6f} Loss_c_T:{:.6f}'.format(
                test_c_acc_T, test_c_f1_T, test_Loss_c))  

            if n_eval == self.N_eval-1:
                test_acc = test_c_acc_T
                test_f1 = test_c_f1_T              

            self.scheduler_fe.step()
            self.scheduler_d.step()
            self.scheduler_ac.step()
            self.scheduler_wa.step()

            train_c_acc_S.reset()
            train_c_f1_S.reset()             

        print('>>> Training Finished!')
        return test_acc, test_f1
        
    def eval(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        criterion_c = nn.CrossEntropyLoss().cuda()

        test_c_acc_T = torchmetrics.Accuracy().cuda()
        test_c_f1_T = torchmetrics.F1(num_classes=self.N_classes, average='macro').cuda()

        self.FE.eval()
        self.AC.eval()

        Loss_c = 0
        with torch.no_grad():
            for _, (x_T, y_T) in enumerate(self.test_loader):

                x_T = Variable(x_T.cuda())
                y_T = Variable(y_T.long().cuda())
                        
                _, logits_ac_T = self.forward_pass(x_T, 'C')
                loss_c = criterion_c(logits_ac_T, y_T)

                # track training losses and metrics 
                Loss_c += loss_c.item()
                test_c_acc_T(logits_ac_T.softmax(dim=-1), y_T)
                test_c_f1_T(logits_ac_T.softmax(dim=-1), y_T)
        
        self.FE.train()
        self.AC.train()

        return Loss_c, test_c_acc_T.compute().item(), test_c_f1_T.compute().item()