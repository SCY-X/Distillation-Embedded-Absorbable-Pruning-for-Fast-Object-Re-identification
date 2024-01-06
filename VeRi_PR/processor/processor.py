import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast, GradScaler
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
import copy
from .prune_model import prune_model

def do_train(cfg,
             model,
             train_loader,
             optimizer,
             scheduler,
             loss_fn,
             ):
    #log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info('start training')

    if device:
        model.to(device)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

    loss_meter = AverageMeter()
    ce_meter = AverageMeter()
    tri_meter = AverageMeter()
    kl_meter = AverageMeter()
    acc_meter = AverageMeter()
    lasso_conv1_meter = AverageMeter()
    lasso_conv2_meter = AverageMeter()
    dl_meter = AverageMeter()
    kd_meter = AverageMeter()

    # train
    if cfg.SOLVER.MIXED_PRECISION:
        scaler = GradScaler()

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        ce_meter.reset()
        tri_meter.reset()
        kl_meter.reset()
        acc_meter.reset()
        lasso_conv1_meter.reset()
        lasso_conv2_meter.reset()
        dl_meter.reset()
        kd_meter.reset()

        model.train()
        n_iter = 0
        for n_iter, (img, vid) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            if cfg.SOLVER.MIXED_PRECISION:
                with autocast():
                    score, feat, lasso_conv1, lasso_conv2, feat_map = model(img, target)
                    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet' and cfg.MODEL.USE_KD:
                        loss, ce_loss, tri_loss, kl_loss, conv1_loss, conv2_loss, dl_loss, kd_loss = loss_fn(score, feat, lasso_conv1, lasso_conv2, feat_map, target)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                score, feat = model(img, target)
                loss, ce_loss, tri_loss = loss_fn(score, feat, target)
                loss.backward()
                optimizer.step()

            acc = (score[0].max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
            ce_meter.update(ce_loss.item(), img.shape[0])
            tri_meter.update(tri_loss.item(), img.shape[0])
            kl_meter.update(kl_loss.item(), img.shape[0])
            acc_meter.update(acc, 1)
            lasso_conv1_meter.update(conv1_loss.item(), 1)
            lasso_conv2_meter.update(conv2_loss.item(), 1)
            dl_meter.update(dl_loss.item(), 1)
            kd_meter.update(kd_loss.item(), 1)

        logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, CE: {:.3f}, TRI: {:.3f}, KL: {:.3f}, DL: {:.3f}, KD: {:.3f}, Acc: {:.3f}, conv1:{:.3f}, conv2:{:.3f}, Base Lr: {:.2e}"
            .format(epoch, (n_iter + 1), len(train_loader),
                    loss_meter.avg, ce_meter.avg, tri_meter.avg, kl_meter.avg, dl_meter.avg, kd_meter.avg, acc_meter.avg, lasso_conv1_meter.avg, lasso_conv2_meter.avg, scheduler.get_lr()[0]))
        scheduler.step()
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if torch.cuda.device_count() > 1:
                model = model.module
            stu_model = copy.deepcopy(model.state_dict())
            for name in list(stu_model.keys()):
                weihts_name = name.split(".")[0]
                if "Teacher" in weihts_name:
                    stu_model.pop(name)
            torch.save(stu_model, os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_original_{}.pth'.format(epoch)))

            prune_model(model)

            for m in model.modules():
                m.requires_grad_(False)
                if hasattr(m, 'switch_to_deploy'):
                    m.switch_to_deploy()

            stu_model = copy.deepcopy(model.state_dict())
            for name in list(stu_model.keys()):
                weihts_name = name.split(".")[0]
                if "Teacher" in weihts_name:
                    stu_model.pop(name)
            torch.save(stu_model, os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))


def do_inference(
        cfg,
        model,
        val_loader,
        num_query,
):
    device = "cuda"
    logger = logging.getLogger("reid_baseline.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)

    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    for n_iter, (img, pid, camid) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.TEST.FLIP_FEATS == 'on':
                feat = torch.FloatTensor(img.size(0), 2048).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                    f = model(img)
                    feat = feat + f
            else:
                feat = model(img)

            evaluator.update((feat, pid, camid))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.2%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))

    with open('%s/test_acc.txt' % cfg.OUTPUT_DIR, 'a') as test_file:
        if cfg.TEST.RE_RANKING:
            test_file.write('[With Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank5: {:.4f} rank10: {:.4f}\n'
                            .format(mAP, cmc[0], cmc[4], cmc[9]))


        #########################no re rank##########################
        else:
            test_file.write('[Without Re-Ranking]mAP: {:.4f} rank1: {:.4f} rank5: {:.4f} rank10: {:.4f}\n'
                            .format(mAP, cmc[0], cmc[4], cmc[9]))

