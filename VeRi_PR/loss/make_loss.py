import torch.nn as nn
import torch.nn.functional as F
import logging
import math
import config 
import torch
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss
from .kl_loss import KL, Fitnet_loss
from .MKD import MKD


class Make_Loss(nn.Module):
    def __init__(self, cfg, num_classes):
        super(Make_Loss, self).__init__()
        logger = logging.getLogger("reid_baseline.train")
        if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
            if cfg.MODEL.NO_MARGIN:
                self.triplet = TripletLoss(mining_method='batch_soft')
                logger.info("using soft margin triplet loss for training, mining_method: batch_soft")
            else:
                self.triplet = TripletLoss(cfg.SOLVER.MARGIN, mining_method=cfg.SOLVER.HARD_EXAMPLE_MINING_METHOD)
                logger.info("using Triplet Loss for training with margin:{}, mining_method:{}".format(cfg.SOLVER.MARGIN, cfg.SOLVER.HARD_EXAMPLE_MINING_METHOD))
        if cfg.MODEL.IF_LABELSMOOTH == 'on':
            self.id_loss = CrossEntropyLabelSmooth(num_classes=num_classes)
            logger.info("label smooth on, numclasses:{}".format(num_classes))
        else:
            self.id_loss = F.cross_entropy

        self.KD_METHOD = cfg.MODEL.KD_METHOD
        self.KL_Loss = KL(cfg.MODEL.KL_T)
        self.DL = Fitnet_loss

        if self.KD_METHOD == 'kl':
            pass
        
        # elif self.KD_METHOD == 'apkd':
        #     self.KD_Loss = APKD(cfg.SOLVER.IMS_PER_BATCH, 
        #     int(cfg.SOLVER.IMS_PER_BATCH / cfg.DATALOADER.NUM_INSTANCE), 2048).cuda()
        
        # elif self.KD_METHOD == 'apkda':
        #     self.KD_Loss = APKDA(cfg.SOLVER.IMS_PER_BATCH, 
        #     int(cfg.SOLVER.IMS_PER_BATCH / cfg.DATALOADER.NUM_INSTANCE), 2048).cuda()

        elif self.KD_METHOD == 'mkd':
            self.KD_Loss = MKD(cfg.MODEL.P, cfg.MODEL.MKD_ALPHA, cfg.MODEL.MKD_BETA, cfg.MODEL.MKD_MODE)
            logger.info('using MKD alpha weight is {}, MKD beta weight is {}'.format(cfg.MODEL.MKD_ALPHA, cfg.MODEL.MKD_BETA))
        
        # elif self.KD_METHOD == 'rank':
        #     self.KD_Loss = RKD(cfg.SOLVER.IMS_PER_BATCH, int(cfg.SOLVER.IMS_PER_BATCH / cfg.DATALOADER.NUM_INSTANCE), 
        #     262144, cfg.MODEL.RANK_ALPHA, cfg.MODEL.RANK_BETA).cuda()
        #     logger.info('using rank alpha weight is {}, rank beta weight is {}'.format(cfg.MODEL.RANK_ALPHA, cfg.MODEL.RANK_BETA))

        else:
            raise NotImplementedError(self.KD_METHOD)

        logger.info('using {} method as KD Loss and KD weight is {}'.format(self.KD_METHOD, cfg.MODEL.KD_WEIGHT))

        self.id_loss_weight = cfg.MODEL.ID_LOSS_WEIGHT
        self.tri_loss_weight = cfg.MODEL.TRIPLET_LOSS_WEIGHT
        self.kl_loss_weight = cfg.MODEL.KL_WEIGHT
        self.dl_loss_weight = cfg.MODEL.DL_WEIGHT
        self.kd_loss_weight = cfg.MODEL.KD_WEIGHT
        self.lasso1_weight = cfg.MODEL.LASSO_CONV1_WEIGHT
        self.lasso2_weight = cfg.MODEL.LASSO_CONV2_WEIGHT


    def forward(self, score, tri_feat, lasso_conv1, lasso_conv2, feat_map, target): 
        ce_loss = self.id_loss_weight * self.id_loss(score[0], target)
        triplet_loss = self.tri_loss_weight * self.triplet(tri_feat, target)
        lasso_conv1_loss = self.lasso1_weight * sum(lasso_conv1)
        lasso_conv2_loss = self.lasso2_weight * sum(lasso_conv2)
        kl_loss = self.kl_loss_weight * self.KL_Loss(score[0], score[1])
        dl_loss = self.dl_loss_weight * Fitnet_loss(feat_map[0])

        s_feat = feat_map[1][0]
        t_feat = feat_map[1][1]
        

        if self.KD_METHOD == 'kl':
            kd_loss = torch.Tensor([0.0]).cuda()

        # elif self.KD_METHOD == 'apkd':
        #     kd_loss = self.KD_Loss(feat_map[1][0], feat_map[1][1])
        
        # elif self.KD_METHOD == 'apkda':
        #     kd_loss = self.KD_Loss(feat_map[1][0], feat_map[1][1])

        elif self.KD_METHOD == 'mkd':
            kd_loss = self.KD_Loss(s_feat, t_feat)
        
        # elif self.KD_METHOD == 'rank':
        #     kd_loss = self.KD_Loss(feat_map[1][0], feat_map[1][1])
        
        kd_loss = self.kd_loss_weight * kd_loss
        total_loss = ce_loss + triplet_loss + kl_loss + lasso_conv1_loss + lasso_conv2_loss + dl_loss + kd_loss
        return total_loss, ce_loss, triplet_loss, kl_loss, lasso_conv1_loss, lasso_conv2_loss, dl_loss, kd_loss
