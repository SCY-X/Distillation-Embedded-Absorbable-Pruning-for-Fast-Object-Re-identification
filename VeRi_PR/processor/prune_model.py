import os
import torch
import numpy as np
from config import cfg


def prune_model(model):
    conv1_idx = []
    conv1_channel = []
    conv2_idx = []
    conv2_channel = []
    model_state_dict = model.state_dict()
    conv1_thresh = cfg.MODEL.CONV1_THRESH
    conv2_thresh = cfg.MODEL.CONV2_THRESH
    #print(model_state_dict.keys())

    for name in list(model_state_dict.keys()):
        if "Teacher" in name:
            model_state_dict.pop(name)

    for name in list(model_state_dict.keys()):
        if 'stage0' in name:
            continue
        if 'compactor1.weight' in name:
            param_data = model_state_dict[name]
            prune_data = param_data.detach().cpu().numpy()
            l2_prune_data = np.sqrt(np.sum(prune_data ** 2, axis=(0, 2, 3)))
            out_filter_thresh = np.where(l2_prune_data < conv1_thresh)[0]
            length = len(prune_data) - len(out_filter_thresh)
            if length == 0:
                l2_max = l2_prune_data.max()
                out_filter_thresh = np.where(l2_prune_data < l2_max)[0]
                conv1_channel.append(len(prune_data) - len(out_filter_thresh))
                conv1_idx.append(out_filter_thresh)

            elif length > 0:
                conv1_channel.append(length)
                conv1_idx.append(out_filter_thresh)

            prune_data = np.delete(prune_data, out_filter_thresh, axis=1)
            prune_data = torch.from_numpy(prune_data)
            model_state_dict[name] = prune_data

        elif 'compactor2.weight' in name:
            param_data = model_state_dict[name]
            prune_data = param_data.detach().cpu().numpy()
            l2_prune_data = np.sqrt(np.sum(prune_data ** 2, axis=(1, 2, 3)))
            out_filter_thresh = np.where(l2_prune_data < conv2_thresh)[0]
            length = len(prune_data) - len(out_filter_thresh)
            if length == 0:
                l2_max = l2_prune_data.max()
                out_filter_thresh = np.where(l2_prune_data < l2_max)[0]
                conv2_channel.append(len(prune_data) - len(out_filter_thresh))
                conv2_idx.append(out_filter_thresh)

            elif length > 0:
                conv2_channel.append(length)
                conv2_idx.append(out_filter_thresh)
            prune_data = np.delete(prune_data, out_filter_thresh, axis=0)
            prune_data = torch.from_numpy(prune_data)
            model_state_dict[name] = prune_data

    conv1_count = 0
    conv2_count = 0
    print(conv1_channel)
    print(conv2_channel)
    for name in list(model_state_dict.keys()):
        if 'num_batches_tracked' in name:
            model_state_dict.pop(name)
        elif 'stage0' in name:
            continue

        elif 'conv1.conv.weight' in name:
            param_data = model_state_dict[name]
            prune_data = param_data.detach().cpu().numpy()
            in_filter_thresh = conv1_idx[conv1_count]
            prune_data = np.delete(prune_data, in_filter_thresh, axis=0)
            prune_data = torch.from_numpy(prune_data)
            model_state_dict[name] = prune_data


        elif 'conv1.bn' in name:
            param_data = model_state_dict[name]
            prune_data = param_data.detach().cpu().numpy()
            in_filter_thresh = conv1_idx[conv1_count]
            prune_data = np.delete(prune_data, in_filter_thresh, axis=0)
            prune_data = torch.from_numpy(prune_data)
            model_state_dict[name] = prune_data

            if 'running_var' in name:
                conv1_count += 1

        elif 'conv3.conv.weight' in name:
            param_data = model_state_dict[name]
            prune_data = param_data.detach().cpu().numpy()
            in_filter_thresh = conv2_idx[conv2_count]
           
            prune_data = np.delete(prune_data, in_filter_thresh, axis=1)
            prune_data = torch.from_numpy(prune_data)
            model_state_dict[name] = prune_data
            conv2_count += 1

    prune_channel = []
    for i in range(len(conv1_channel)):
        prune_channel.append(conv1_channel[i])
        prune_channel.append(conv2_channel[i])
    print(prune_channel)
    torch.save(model_state_dict, os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_prune_{}.pth'.format(cfg.TEST.WEIGHT)))
    with open('./log/prune_channel.txt', 'w') as f:
        f.write(str(prune_channel))
        f.close()