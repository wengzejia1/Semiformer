"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
from contextlib import suppress

def margin_triplet_loss(anchor_view, fit_view, margin, dis_type='cos'):
    assert anchor_view.size() == fit_view.size()

    if dis_type == 'cos':
        view1_norm = anchor_view / anchor_view.norm(dim=1)[:, None]
        view2_norm = fit_view / fit_view.norm(dim=1)[:, None]
        similarity = torch.mm(view1_norm, view2_norm.transpose(0,1))

        # (batch,batch)
        pos = similarity.diag().view(anchor_view.shape[0], 1)
        d1 = pos.expand_as(similarity)
        # d2 = pos.t().expand_as(similarity)
        cost1 = (margin + similarity - d1).clamp(min=0)
        # cost2 = (margin + similarity - d2).clamp(min=0)
        mask = torch.eye(similarity.shape[0], device=anchor_view.device) > 0.5
        cost1 = cost1.masked_fill_(mask, 0)
        # cost2 = cost2.masked_fill_(mask, 0)
        # loss = cost1.mean() + cost2.mean()
    elif dis_type == 'l2':
        size=anchor_view.shape[0]
        # anchor_view = anchor_view.to(torch.float32)
        # fit_view = fit_view.to(torch.float32)
        A = (anchor_view * anchor_view).sum(-1).view((size, 1)).expand((size, size))
        B = (fit_view * fit_view).sum(-1).view((1, size)).expand((size, size))
        L2_square = A + B - 2 * torch.mm(anchor_view, fit_view.transpose(0,1))
        L2_dis = L2_square ** 0.5
        
        pos = L2_dis.diag().view(anchor_view.shape[0], 1)
        d1 = pos.expand_as(L2_dis)
        cost1 = (margin - L2_dis + d1).clamp(min=0)
        mask = torch.eye(size, device=anchor_view.device) > 0.5
        cost1 = cost1.masked_fill_(mask, 0)

    loss = cost1.mean()
    return loss 

def infonce_loss(anchor_view, fit_view, T=1.):
    assert anchor_view.size() == fit_view.size()
    device = anchor_view.device
    view1_norm = anchor_view / anchor_view.norm(dim=1)[:, None]
    view2_norm = fit_view / fit_view.norm(dim=1)[:, None]
    logits = torch.mm(view1_norm, view2_norm.transpose(0,1)) / T
    label = torch.range(0, anchor_view.shape[0]-1).long().to(device, non_blocking=True)
    return torch.nn.functional.cross_entropy(logits, label) 

def train_one_epoch(model: torch.nn.Module, criterion_x: torch.nn.Module, criterion_u: torch.nn.Module,
                    data_loader_x: Iterable, data_loader_u: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, mask_cnn_ratio=1.0, mask_transformer_ratio=1.0,
                    mask_cnn_ratio_u=1.0, mask_transformer_ratio_u=1.0,
                    amp_autocast=suppress, temperature=1.0, threshold=0.7, semi_lambda=10.,
                    semi_start_epoch=0,
                    pseudo_type='cnn',
                    dual_consistency=False,
                    consistency_start_epoch=0.,
                    consistency_type='triplet',
                    consistency_lambda=0.5,
                    infonce_T=1.,
                    margin=0.5,
                    anchor_type='weak',
                    dis_type='cos',
                    ):
    # TODO fix this for finetuning
    model.train(set_training_mode)
    criterion_x.train()
    criterion_u.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if consistency_start_epoch == -1:
        consistency_start_epoch = semi_start_epoch

    for samples, targets in metric_logger.log_every(data_loader_x, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
         
        with amp_autocast():
        # with torch.cuda.amp.autocast():
            label_bs = samples.shape[0]
            if epoch >= semi_start_epoch:
                try:
                    strong_samples_u, weak_samples_u, _ = next(u_iterator)
                except:
                    u_iterator = iter(data_loader_u)
                    strong_samples_u, weak_samples_u, _ = next(u_iterator)
                
                strong_samples_u = strong_samples_u.to(device, non_blocking=True)
                weak_samples_u = weak_samples_u.to(device, non_blocking=True)
                
                unlabel_s_bs = strong_samples_u.shape[0]
                unlabel_w_bs = weak_samples_u.shape[0]
                
                inputs = torch.cat((samples, strong_samples_u), 0)
                inputs2 = weak_samples_u 
            else:
                inputs = samples 
            
            out_dict = model(inputs, ret_feat=True)
            out_logits = out_dict['output']
            out_feat = out_dict['feat']

            # label data output
            if isinstance(out_logits, list):
                outputs = [out_logits[0][:label_bs], out_logits[1][:label_bs]]
            else:
                outputs = out_logits[:label_bs]
               
            # loss calculation
            if isinstance(outputs, list):
                loss_list = [criterion_x(o, targets) / len(outputs) for o in outputs]    
                # loss = sum(loss_list)
                # TODO: check and make sure: loss_list[0] -> cnn; loss_list[1] -> transformer;
                loss = mask_cnn_ratio * loss_list[0] + mask_transformer_ratio * loss_list[1]
            else:
                loss = criterion_x(outputs, targets)
            
            # unlabeled part
            if epoch >= semi_start_epoch:
                
                if isinstance(outputs, list):
                    # unlabel data output 
                    logits_u_s_out = [out_logits[0][label_bs: label_bs + unlabel_s_bs], out_logits[1][label_bs: label_bs + unlabel_s_bs]]
                    feat_u_s_out = [out_feat[0][label_bs: label_bs + unlabel_s_bs], out_feat[1][label_bs: label_bs + unlabel_s_bs]]

                    model.eval()
                    with torch.no_grad():
                        out_dict2 = model(inputs2, ret_feat=True)
                        logits_u_w_out = out_dict2['output']
                        feat_u_w_out = out_dict2['feat']

                    model.train()
                    # logits_u_w_out = [out_logits[0][label_bs: label_bs+unlabel_w_bs], out_logits[1][label_bs: label_bs+unlabel_w_bs]]
                
                    # TODO: unlable data branch
                    if pseudo_type == 'cnn':
                        pseudo_logits_u_w = logits_u_w_out[0]
                        pseudo_label = torch.softmax(pseudo_logits_u_w.detach()/temperature, dim=-1)
                        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                        mask = max_probs.ge(threshold).float()
                        
                    elif pseudo_type == 'transformer':
                        pseudo_logits_u_w = logits_u_w_out[1]
                        pseudo_label = torch.softmax(pseudo_logits_u_w.detach()/temperature, dim=-1)
                        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                        mask = max_probs.ge(threshold).float()
                        
                    elif pseudo_type == 'fuse':
                        pseudo_logits_u_w = (logits_u_w_out[0] + logits_u_w_out[1]) / 2 
                        pseudo_label = torch.softmax(pseudo_logits_u_w.detach()/temperature, dim=-1)
                        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                        mask = max_probs.ge(threshold).float()
                     
                    elif pseudo_type == 'softmax_fuse':
                        a = torch.softmax(logits_u_w_out[0].detach()/temperature, dim=-1)
                        b = torch.softmax(logits_u_w_out[1].detach()/temperature, dim=-1)
                        pseudo_label = (a+b) / 2
                        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                        mask = max_probs.ge(threshold).float()
   
                    elif pseudo_type == 'Vote_equal' or pseudo_type == 'Vote_cnnBias' or pseudo_type == 'Vote_larger':
                        cnn_label = torch.softmax(logits_u_w_out[0].detach()/temperature, dim=-1)
                        cnn_max_probs, cnn_targets_u = torch.max(cnn_label, dim=-1)
                        cnn_mask = cnn_max_probs.ge(threshold).float()

                        trans_label = torch.softmax(logits_u_w_out[1].detach()/temperature, dim=-1)
                        trans_max_probs, trans_targets_u = torch.max(trans_label, dim=-1)
                        trans_mask = trans_max_probs.ge(threshold).float()
                        
                        targets_u = torch.zeros(cnn_targets_u.shape).to(device, non_blocking=True).long()
                        mask = torch.zeros(cnn_mask.shape).to(device, non_blocking=True).float()
                        
                        for i in range(cnn_max_probs.shape[0]):
                            if cnn_mask[i] == 1. and trans_mask[i] == 0.:
                                targets_u[i] = cnn_targets_u[i]
                                mask[i] = 1.
                            elif cnn_mask[i] == 0. and trans_mask[i] == 1.:
                                targets_u[i] = trans_targets_u[i]
                                mask[i] = 1.
                            elif cnn_mask[i] == 1. and trans_mask[i] == 1.:
                                if pseudo_type == 'Vote_equal':
                                    if cnn_targets_u[i] == trans_targets_u[i]:
                                        targets_u[i] = cnn_targets_u[i]
                                        mask[i] = 1.
                                    else:
                                        mask[i] = 0.

                                elif pseudo_type == 'Vote_cnnBias':
                                    targets_u[i] = cnn_targets_u[i]
                                    mask[i] = 1.

                                elif pseudo_type == 'Vote_larger':
                                    if cnn_max_probs[i] > trans_max_probs[i]:
                                        targets_u[i] = cnn_targets_u[i]
                                        mask[i] = 1.
                                    else:
                                        targets_u[i] = trans_targets_u[i]
                                        mask[i] = 1.
                                
                            elif cnn_mask[i] == 0. and trans_mask[i] == 0.:
                                mask[i] = 0.
                    
                    elif pseudo_type == 'Vote_larger2' or pseudo_type == 'Vote_equal2':
                        cnn_label = torch.softmax(logits_u_w_out[0].detach()/temperature, dim=-1)
                        cnn_max_probs, cnn_targets_u = torch.max(cnn_label, dim=-1)
                        cnn_mask = cnn_max_probs.ge(threshold).float()

                        trans_label = torch.softmax(logits_u_w_out[1].detach()/temperature, dim=-1)
                        trans_max_probs, trans_targets_u = torch.max(trans_label, dim=-1)
                        trans_mask = trans_max_probs.ge(threshold).float()
                        
                        targets_u = torch.zeros(cnn_targets_u.shape).to(device, non_blocking=True).long()
                        mask = torch.zeros(cnn_mask.shape).to(device, non_blocking=True).float()
                        
                        for i in range(cnn_max_probs.shape[0]):
                            if cnn_mask[i] == 1. and trans_mask[i] == 0.:
                                targets_u[i] = cnn_targets_u[i]
                                mask[i] = 1.
                            elif cnn_mask[i] == 0. and trans_mask[i] == 1.:
                                mask[i] = 0.
                            elif cnn_mask[i] == 1. and trans_mask[i] == 1.:
                                if pseudo_type == 'Vote_larger2':
                                    if cnn_max_probs[i] > trans_max_probs[i]:
                                        targets_u[i] = cnn_targets_u[i]
                                        mask[i] = 1.
                                    else:
                                        targets_u[i] = trans_targets_u[i]
                                        mask[i] = 1.
                                
                                elif pseudo_type == 'Vote_equal2':
                                    if cnn_targets_u[i] == trans_targets_u[i]:
                                        targets_u[i] = cnn_targets_u[i]
                                        mask[i] = 1.
                                    else:
                                        mask[i] = 0.

                            elif cnn_mask[i] == 0. and trans_mask[i] == 0.:
                                mask[i] = 0.
                    
                    elif pseudo_type == 'weight_v1' or pseudo_type == 'weight_v2' or pseudo_type == 'weight_v3' or pseudo_type == 'weight_v4':
                        cnn_label = torch.softmax(logits_u_w_out[0].detach()/temperature, dim=-1)
                        cnn_max_probs, cnn_targets_u = torch.max(cnn_label, dim=-1)
                        cnn_mask = cnn_max_probs.ge(threshold).float()

                        trans_label = torch.softmax(logits_u_w_out[1].detach()/temperature, dim=-1)
                        trans_max_probs, trans_targets_u = torch.max(trans_label, dim=-1)
                        trans_mask = trans_max_probs.ge(threshold).float()
                        
                        targets_u = torch.zeros(cnn_targets_u.shape).to(device, non_blocking=True).long()
                        mask = torch.zeros(cnn_mask.shape).to(device, non_blocking=True).float()
                        
                        for i in range(cnn_max_probs.shape[0]):
                            if cnn_mask[i] == 1. and trans_mask[i] == 0.:
                                targets_u[i] = cnn_targets_u[i]
                                if pseudo_type == 'weight_v3' or pseudo_type == 'weight_v4':
                                    mask[i] = 0.5
                                else:
                                    mask[i] = 1.
                            elif cnn_mask[i] == 0. and trans_mask[i] == 1.:
                                if pseudo_type == 'weight_v2' or pseudo_type == 'weight_v4':
                                    targets_u[i] = trans_targets_u[i]
                                    mask[i] = 0.5
                                else:
                                    mask[i] = 0.
                            elif cnn_mask[i] == 1. and trans_mask[i] == 1.:
                                if cnn_targets_u[i] == trans_targets_u[i]:
                                    targets_u[i] = cnn_targets_u[i]
                                    mask[i] = 1.
                                else:
                                    if cnn_max_probs[i] > trans_max_probs[i]:
                                        targets_u[i] = cnn_targets_u[i]
                                        mask[i] = 0.5
                                    else:
                                        targets_u[i] = trans_targets_u[i]
                                        mask[i] = 0.5
                            elif cnn_mask[i] == 0. and trans_mask[i] == 0.:
                                mask[i] = 0.
                    
                    elif pseudo_type == 'weight_v5':
                        cnn_label = torch.softmax(logits_u_w_out[0].detach()/temperature, dim=-1)
                        cnn_max_probs, cnn_targets_u = torch.max(cnn_label, dim=-1)
                        cnn_mask = cnn_max_probs.ge(threshold).float()

                        trans_label = torch.softmax(logits_u_w_out[1].detach()/temperature, dim=-1)
                        trans_max_probs, trans_targets_u = torch.max(trans_label, dim=-1)
                        trans_mask = trans_max_probs.ge(threshold).float()
                        
                        targets_u = torch.zeros(cnn_targets_u.shape).to(device, non_blocking=True).long()
                        mask = torch.zeros(cnn_mask.shape).to(device, non_blocking=True).float()
                        
                        for i in range(cnn_max_probs.shape[0]):
                            if cnn_mask[i] == 1. and trans_mask[i] == 0.:
                                targets_u[i] = cnn_targets_u[i]
                                mask[i] = 0.5
                            elif cnn_mask[i] == 0. and trans_mask[i] == 1.:
                                mask[i] = 0.
                            elif cnn_mask[i] == 1. and trans_mask[i] == 1.:
                                if cnn_targets_u[i] == trans_targets_u[i]:
                                    targets_u[i] = cnn_targets_u[i]
                                    mask[i] = 1.
                                else:
                                    targets_u[i] = cnn_targets_u[i]
                                    mask[i] = 0.5
                            elif cnn_mask[i] == 0. and trans_mask[i] == 0.:
                                mask[i] = 0.
                    
                    elif pseudo_type == 'weight_v6':
                        cnn_label = torch.softmax(logits_u_w_out[0].detach()/temperature, dim=-1)
                        cnn_max_probs, cnn_targets_u = torch.max(cnn_label, dim=-1)
                        cnn_mask = cnn_max_probs.ge(threshold).float()

                        trans_label = torch.softmax(logits_u_w_out[1].detach()/temperature, dim=-1)
                        trans_max_probs, trans_targets_u = torch.max(trans_label, dim=-1)
                        trans_mask = trans_max_probs.ge(threshold).float()
                        
                        targets_u = torch.zeros(cnn_targets_u.shape).to(device, non_blocking=True).long()
                        mask = torch.zeros(cnn_mask.shape).to(device, non_blocking=True).float()
                        
                        for i in range(cnn_max_probs.shape[0]):
                            if cnn_mask[i] == 1. and trans_mask[i] == 0.:
                                targets_u[i] = cnn_targets_u[i]
                                mask[i] = 1.
                            elif cnn_mask[i] == 0. and trans_mask[i] == 1.:
                                mask[i] = 0.
                            elif cnn_mask[i] == 1. and trans_mask[i] == 1.:
                                if cnn_targets_u[i] == trans_targets_u[i]:
                                    targets_u[i] = cnn_targets_u[i]
                                    mask[i] = 1.
                                else:
                                    targets_u[i] = cnn_targets_u[i]
                                    mask[i] = 0.5
                            elif cnn_mask[i] == 0. and trans_mask[i] == 0.:
                                mask[i] = 0.

                else:
                    logits_u_s_out = out_logits[label_bs: label_bs + unlabel_s_bs]
                    model.eval()
                    logits_u_w_out = model(inputs2)
                    model.train()
                    pseudo_logits_u_w = logits_u_w_out
                
                    pseudo_label = torch.softmax(pseudo_logits_u_w.detach()/temperature, dim=-1)
                    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                    mask = max_probs.ge(threshold).float()

                if isinstance(logits_u_s_out, list):
                    # fix here, add '/len(logits_u_s_out)'
                    Lu_list = [(criterion_u(o, targets_u) * mask).mean() / len(logits_u_s_out) for o in logits_u_s_out]
                    Lu = mask_cnn_ratio_u * Lu_list[0] + mask_transformer_ratio_u * Lu_list[1]
                
                else:
                    Lu = (torch.nn.functional.cross_entropy(logits_u_s_out, targets_u, reduction='none') * mask).mean()
                
                loss = loss + semi_lambda * Lu

                if dual_consistency and isinstance(outputs, list) and epoch >= consistency_start_epoch:
                    
                    if consistency_type == 'triplet':
                        Lc_list = []
                        if anchor_type == 'weak' or anchor_type == 'both':
                            Lc_list.append(margin_triplet_loss(
                                anchor_view=feat_u_w_out[0], 
                                fit_view=feat_u_s_out[0], 
                                margin=margin,
                                dis_type=dis_type))
                            Lc_list.append(margin_triplet_loss(
                                anchor_view=feat_u_w_out[1], 
                                fit_view=feat_u_s_out[1], 
                                margin=margin,
                                dis_type=dis_type))
                        
                        if anchor_type == 'strong' or anchor_type == 'both':
                            Lc_list.append(margin_triplet_loss(
                                anchor_view=feat_u_s_out[0], 
                                fit_view=feat_u_w_out[0], 
                                margin=margin,
                                dis_type=dis_type))
                            Lc_list.append(margin_triplet_loss(
                                anchor_view=feat_u_s_out[1], 
                                fit_view=feat_u_w_out[1], 
                                margin=margin,
                                dis_type=dis_type))

                        Lc = sum(Lc_list)
                        loss = loss + consistency_lambda * Lc
                    
                    elif consistency_type == 'infonce':
                        Lc_list = []
                        if anchor_type == 'weak' or anchor_type == 'both':
                            Lc_list.append(infonce_loss(
                                anchor_view=feat_u_w_out[0], 
                                fit_view=feat_u_s_out[0], 
                                T=infonce_T,))
                            Lc_list.append(infonce_loss(
                                anchor_view=feat_u_w_out[1], 
                                fit_view=feat_u_s_out[1], 
                                T=infonce_T))
                        
                        if anchor_type == 'strong' or anchor_type == 'both':
                            Lc_list.append(infonce_loss(
                                anchor_view=feat_u_s_out[0], 
                                fit_view=feat_u_w_out[0], 
                                T=infonce_T,))
                            Lc_list.append(infonce_loss(
                                anchor_view=feat_u_s_out[1], 
                                fit_view=feat_u_w_out[1], 
                                T=infonce_T))

                        Lc = sum(Lc_list)
                        loss = loss + consistency_lambda * Lc 


        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            if isinstance(outputs, list):
                print("Loss1 is {}".format(loss_list[0]))
                print("Loss2 is {}".format(loss_list[1]))
                print(outputs[0])
                print(outputs[1])
                
            sys.exit(1)

        optimizer.zero_grad()

        if loss_scaler is not None:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss.backward(create_graph=is_second_order)
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        if isinstance(outputs, list):
            metric_logger.update(loss_0=loss_list[0].item())
            metric_logger.update(loss_1=loss_list[1].item())
            if epoch >= semi_start_epoch:
                metric_logger.update(loss_0_u=Lu_list[0].item())
                metric_logger.update(loss_1_u=Lu_list[1].item())
                if dual_consistency and epoch >= consistency_start_epoch:
                    metric_logger.update(loss_consistency=Lc.item())

            metric_logger.update(loss=loss_value)
        else:
            if epoch >= semi_start_epoch:
                metric_logger.update(loss_u=Lu.item())
            metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, amp_autocast=suppress,):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with amp_autocast(): 
        # with torch.cuda.amp.autocast():
            output = model(images)
            # Conformer
            if isinstance(output, list):
                loss_list = [criterion(o, target) / len(output)  for o in output]
                loss = sum(loss_list)
            # others
            else:
                loss = criterion(output, target)
        if isinstance(output, list):
            # Conformer
            acc1_head1, acc5_head1 = accuracy(output[0], target, topk=(1, 5))
            acc1_head2, acc5_head2 = accuracy(output[1], target, topk=(1, 5))
            acc1_total, acc5_total = accuracy(output[0] + output[1], target, topk=(1, 5))
            acc1_softmax, acc5_softmax = accuracy(torch.softmax(output[0], -1) + torch.softmax(output[1], -1), target, topk=(1, 5))
        else:
            # others
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        if isinstance(output, list):
            metric_logger.update(loss=loss.item())
            metric_logger.update(loss_0=loss_list[0].item())
            metric_logger.update(loss_1=loss_list[1].item())
            metric_logger.meters['acc1'].update(acc1_total.item(), n=batch_size)
            metric_logger.meters['acc1_head1'].update(acc1_head1.item(), n=batch_size)
            metric_logger.meters['acc1_head2'].update(acc1_head2.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5_total.item(), n=batch_size)
            metric_logger.meters['acc5_head1'].update(acc5_head1.item(), n=batch_size)
            metric_logger.meters['acc5_head2'].update(acc5_head2.item(), n=batch_size)
            metric_logger.meters['acc1_softmax'].update(acc1_softmax.item(), n=batch_size)
            metric_logger.meters['acc5_softmax'].update(acc5_softmax.item(), n=batch_size)
        else:
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    if isinstance(output, list):
        print('* Acc@heads_top1 {heads_top1.global_avg:.3f} Acc@head_1 {head1_top1.global_avg:.3f} Acc@head_2 {head2_top1.global_avg:.3f} '
              '* Acc@heads_top5 {heads_top5.global_avg:.3f} Acc@head_1_top5 {head1_top5.global_avg:.3f} Acc@head_2_top5 {head2_top5.global_avg:.3f} '
              'loss@total {losses.global_avg:.3f} loss@1 {loss_0.global_avg:.3f} loss@2 {loss_1.global_avg:.3f} '
              .format(heads_top1=metric_logger.acc1, head1_top1=metric_logger.acc1_head1, head2_top1=metric_logger.acc1_head2,
                      heads_top5=metric_logger.acc5, head1_top5=metric_logger.acc5_head1, head2_top5=metric_logger.acc5_head2,
                      losses=metric_logger.loss, loss_0=metric_logger.loss_0, loss_1=metric_logger.loss_1))
    else:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
