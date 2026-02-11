from argparse import Namespace
from collections import OrderedDict
import os
import pickle

from lifelines.utils import concordance_index
import numpy as np
from sksurv.metrics import concordance_index_censored

import torch

from models.model_coattn import MCAT_Surv
from utils import *

def train(datasets: tuple, cur: int, args: Namespace):
    """
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    train_split, val_split = datasets

    print('\nInit loss function...', end=' ')
    loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    reg_fn = None

    print('Done!')

    print('\nInit Model...', end=' ')
    assert args.fusion in ['concat', 'bilinear'], f"Unsupported fusion: {args.fusion}"
    assert args.transformer_mode in ['separate', 'shared'], f"Unsupported transformer_mode: {args.transformer_mode}"
    assert args.pooling in ['gap', 'attn'], f"Unsupported pooling: {args.pooling}"

    dropout = 0.25 if args.drop_out else 0.0
    model_dict = {
        'fusion': args.fusion,
        'n_classes': args.n_classes,
        'dropout': dropout,
        'transformer_mode': args.transformer_mode,
        'pooling': args.pooling,
    }
    model = MCAT_Surv(**model_dict).to(device)

    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, weighted=args.weighted_sample, mode=args.mode, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split, mode=args.mode, batch_size=args.batch_size)
    print('Done!')

    print('Done!')
    val_cindex_all = []
    for epoch in range(args.max_epochs):
        if args.task_type == 'survival':
            train_loop_survival_coattn(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn,
                                       reg_fn, args.lambda_reg, args.gc)
            val_cindex_i = validate_survival_coattn(cur, epoch, model, val_loader, args.n_classes, writer, loss_fn,
                                                    reg_fn, args.lambda_reg, args.results_dir)
            val_cindex_all.append(val_cindex_i)

    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    results_val_dict, val_cindex = summary_survival_coattn(model, val_loader, args.n_classes)
    print('Val c-Index: {:.4f}'.format(val_cindex))
    if writer:
        writer.close()
    return results_val_dict, val_cindex_all


def train_loop_survival_coattn(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None,
                               reg_fn=None, lambda_reg=0., gc=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    optimizer.zero_grad()  # Zero gradients once at start
    for batch_idx, (data_WSI, codex_feature, label, event_time, c) in enumerate(loader):

        data_WSI = data_WSI.to(device)
        codex_feature = codex_feature.to(device)
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)

        hazards, S, Y_hat, A = model(x_path=data_WSI, x_codex=codex_feature)
        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size:'.format(batch_idx,
                                                                                                          loss_value + loss_reg,
                                                                                                          label.item(),
                                                                                                          float(
                                                                                                              event_time),
                                                                                                          float(risk)))
        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0 or (batch_idx + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    c_index = \
    concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv,
                                                                                                 train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)


def validate_survival_coattn(cur, epoch, model, loader, n_classes,
                             writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, codex_feature, label, event_time, c) in enumerate(loader):

        data_WSI = data_WSI.to(device)
        codex_feature = codex_feature.to(device)
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)

        with torch.no_grad():
            hazards, S, Y_hat, A = model(x_path=data_WSI, x_codex=codex_feature)

        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = \
    concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    return c_index


def summary_survival_coattn(model, loader, n_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, codex_feature, label, event_time, c) in enumerate(loader):
        data_WSI = data_WSI.to(device)
        codex_feature = codex_feature.to(device)
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)
        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            hazards, survival, Y_hat, A = model(x_path=data_WSI, x_codex=codex_feature)

        risk = float(-torch.sum(survival, dim=1).cpu().numpy())
        event_time = float(event_time)
        c = float(c)
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(),
                                           'survival': event_time, 'censorship': c, 'A': A}})

    c_index = \
    concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    return patient_results, c_index
