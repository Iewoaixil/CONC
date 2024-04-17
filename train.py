import os
import sys
import pathlib
import time
import datetime
import argparse
import shutil
import torch
import torch.nn as nn 
from utils.core import accuracy, evaluate
from utils.builder import *
from utils.utils import *
from utils.meter import AverageMeter
from utils.logger import Logger, print_to_logfile, print_to_console
from utils.loss import *
from utils.module import MLPHead
from utils.plotter import plot_results
import numpy as np
import matplotlib.pyplot as plt
LOG_FREQ = 1

from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import CitationFull
from torch_geometric.datasets import Coauthor
import copy
from model.conc import CONC
from torch_geometric.utils import degree
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch_geometric.nn import MessagePassing

class MessagePassing(MessagePassing):
    def __init__(self):
        super(MessagePassing, self).__init__(aggr='mean')

    def forward(self, x, edge_index):
        def message_func(edge_index, x_i):
            return x_i

        aggregated = self.propagate(edge_index, x=x, message_func=message_func)
        return aggregated
    


def evaluate_test(data, model, y_true, indices, ind_indices, ood_indices, ground_truth_degree_matrix):
    model.eval()
    with torch.no_grad():
        output = model(data, ground_truth_degree_matrix, dev)
        logits = output['class_logits']
        scores, pred = logits.softmax(dim=1).max(dim=1)
        pred = pred.to('cpu').numpy()
        scores = scores.to('cpu').numpy()
        max_acc = 0
        for i in range(0, 10):
            t = i * 0.1
            scores_temp, pred_temp = logits.softmax(dim=1).max(dim=1)
            index = scores_temp < t
            pred_temp[index == True] = -1
            pred_temp = pred_temp.to('cpu').numpy()
            acc = accuracy_score(y_true[indices], pred_temp[indices])
            if acc > max_acc:
                max_acc = acc
                pred = pred_temp
        test_acc = accuracy_score(y_true[indices], pred[indices])
        test_f1 = f1_score(y_true[indices], pred[indices], average = 'macro')
        test_ind_acc = accuracy_score(y_true[ind_indices], pred[ind_indices])
        test_ood_acc = accuracy_score(y_true[ood_indices], pred[ood_indices])
        labels_all = copy.deepcopy(y_true)
        labels_all[ind_indices] = 1
        labels_all[ood_indices] = 0
        test_auc = roc_auc_score(labels_all[indices], scores[indices]) 
    return {'test_acc': test_acc, 'test_f1': test_f1, 'test_ind_acc': test_ind_acc,'test_ood_acc': test_ood_acc, 'test_auc': test_auc, }



def save_current_script(log_dir):
    current_script_path = __file__
    shutil.copy(current_script_path, log_dir)


def record_network_arch(result_dir, net):
    with open(f'{result_dir}/network.txt', 'w') as f:
        f.writelines(net.__repr__())


def get_smoothed_label_distribution(labels, num_class, epsilon):
    smoothed_label = torch.full(size=(labels.size(0), num_class), fill_value=epsilon / (num_class - 1))
    smoothed_label.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1).cpu(), value=1 - epsilon)
    return smoothed_label.to(labels.device)


def build_logger(params):
    logger_root = f'Results/{params.synthetic_data}'
    if not os.path.isdir(logger_root):
        os.makedirs(logger_root, exist_ok=True)
    percentile = int(params.closeset_ratio * 100)
    noise_condition = f'symm_{percentile:2d}' if params.noise_type == 'symmetric' else f'asym_{percentile:2d}'
    logtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(logger_root, noise_condition, params.project, f'{params.log}-{logtime}')
    logger = Logger(logging_dir=result_dir, DEBUG=True)
    print(result_dir)
    logger.set_logfile(logfile_name='log.txt')
    print(params)
    save_config(params, f'{result_dir}/params.cfg')
    save_params(params, f'{result_dir}/params.json', json_format=True)
    save_current_script(result_dir)
    logger.msg(f'Result Path: {result_dir}')
    return logger, result_dir


def build_model_optim_scheduler(params, device, build_scheduler=True):
    n_classes = params.n_classes
    dim_feats = params.dim_feats
    lambda_loss = params.lambda_loss

    net = CONC(dim_feats, 512, 128, n_classes, lambda_loss, device=device)

    if params.opt == 'sgd':
        optimizer = build_sgd_optimizer(net.parameters(), params.lr, params.weight_decay, nesterov=True)
    elif params.opt == 'adam':
        optimizer = build_adam_optimizer(net.parameters(), params.lr)
    else:
        raise AssertionError(f'{params.opt} optimizer is not supported yet.')
    if build_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True, threshold=1e-4)
    else:
        scheduler = None
    return net.to(device), optimizer, scheduler, n_classes

    
    

def wrapup_training(result_dir, best_accuracy):
    stats = get_stats(f'{result_dir}/log.txt')
    with open(f'{result_dir}/result_stats.txt', 'w') as f:
        f.write(f"valid epochs: {stats['valid_epoch']}\n")
        if 'mean' in stats.keys():
            f.write(f"mean: {stats['mean']:.4f}, std: {stats['std']:.4f}\n")
        else:
            f.write(f"mean1: {stats['mean1']:.4f}, std2: {stats['std1']:.4f}\n")
            f.write(f"mean2: {stats['mean2']:.4f}, std2: {stats['std2']:.4f}\n")
    os.rename(result_dir, f'{result_dir}-bestAcc_{best_accuracy:.4f}')

def emptify_graph(train_graph, idx, device):
    train_X_features = train_graph.x
    no_train_x_features = torch.zeros(train_X_features[0].shape[0]).to(device)
    train_X_features[idx] = no_train_x_features

def main(cfg, device):
    init_seeds(0)
    logger, result_dir = build_logger(cfg)

    #load model ---------------------------------------------------------------------------------------------------------------------------------------
    graph = Planetoid(root='./datasets/PYG', name='Citeseer')  
    g = graph[0]  # Graph
    cfg.dim_feats = g.num_node_features
    data_idx = np.load('./idx/' + cfg.dataset + '_5_idx.npz')
    train_indices = data_idx['train_indices']
    valid_indices = data_idx['valid_indices']
    test_indices = data_idx['test_indices']
    y_true = data_idx['y_true']
    y_real_true = data_idx['y_real_true']
    y_real_true_train = y_real_true[train_indices]
    y_true_train = y_true[train_indices]
    cfg.n_classes = np.max(y_true) + 1
    y = torch.from_numpy(y_true).to(device)
    y_train = y[train_indices]
    y_test = y[test_indices]
    y_valid = y[valid_indices]
    
    
    in_nodes, out_nodes = g.edge_index[0], g.edge_index[1]
    ground_truth_degree_matrix = degree(g.edge_index[0]).to(device)


    net, optimizer, scheduler, n_classes = build_model_optim_scheduler(cfg, device, build_scheduler=False)
    
    record_network_arch(result_dir, net)

    if cfg.loss_func_nois == 's-mae':
        nois_loss_func = F.smooth_l1_loss
    elif cfg.loss_func_nois == 'mae':
        nois_loss_func = F.l1_loss
    elif cfg.loss_func_nois == 'mse':
        nois_loss_func = F.mse_loss
    else:
        raise AssertionError(f'{cfg.loss_func_aux} loss is not supported for auxiliary loss yet.')

    # meters -----------------------------------------------------------------------------------------------------------------------------------------
    best_accuracy, best_epoch = 0.0, None

    g_train = copy.deepcopy(g)
    g, g_train= g.to(device), g_train.to(device)
    our_non_train_idx = np.concatenate((test_indices, valid_indices))
    emptify_graph(g_train, our_non_train_idx, device)
    
    test_f1_list = []
    test_acc_list = []
    test_auc_list = []
    test_ind_acc_list = []
    test_ood_acc_list = []

    
    
    # training ---------------------------------------------------------------------------------------------------------------------------------------
    for epoch in range(0, cfg.epochs):
        start_time = time.time()
        net.train()
        optimizer.zero_grad()
        train_loss = 0
        train_accuracy = 0
        # train this epoch
        s = time.time()
        output = net(g_train, ground_truth_degree_matrix, device=device, isTrain=True)
        loss_rec = output['loss_rec']
        logits = output['class_logits'][train_indices]
        probs = logits.softmax(dim=1)
        logits_reconstructed = output['class_logits_reconstructed'][train_indices]
        probs_reconstructed = logits_reconstructed.softmax(dim=1)
        all_logits_reconstructed = output['class_logits_reconstructed']
        all_probs_reconstructed = all_logits_reconstructed.softmax(dim=1)
        train_acc = accuracy(logits, y_train, topk=(1,))
        print(f"train_acc:{train_acc}-------------------------------")

        trust_prob = output['trust_prob'][train_indices]
        trust_x = trust_prob[:, 0]
        trust_y = trust_prob[:, 1]

        print(f'TrainAcc: {train_accuracy:3.2f}%; TrainLoss: {train_loss:3.2f}')

        with torch.no_grad():
            sharpened_target_s = (probs_reconstructed / cfg.temperature).softmax(dim=1)
            flattened_target_s = (probs_reconstructed * cfg.temperature).softmax(dim=1)
            
        loss_cls = 0
        clean_indices = torch.where((trust_x > cfg.tau1) & (trust_y > cfg.tau2))[0]  
        ind_indices = torch.where((trust_x > cfg.tau1) & (trust_y <= cfg.tau2))[0]
        ood_indices = torch.where(trust_x <= cfg.tau1)[0] 
    
        given_labels = F.one_hot(y_train, n_classes).to(torch.float64)
        if epoch < cfg.warmup_epochs:
            print(f'WARMUP TRAINING (lr={cfg.lr:.3e})')
            loss = cross_entropy(logits, given_labels, reduction='mean') 
        else:
            print(f'ROBUST TRAINING (lr={cfg.lr:.3e})')
            if clean_indices.shape[0] > 0:
                y_presuo_clean = given_labels[clean_indices] 
                loss_clean = cross_entropy(logits[clean_indices], y_presuo_clean, reduction='none') + 0.01 * cross_entropy(probs[clean_indices], probs_reconstructed[clean_indices], reduction='none') +  0.1 * entropy_loss(logits[clean_indices], reduction='none')
                loss_cls = loss_cls + (loss_clean * trust_x[clean_indices] * trust_y[clean_indices]).mean() 

            if ind_indices.shape[0] > 0:
                y_presuo_ind = given_labels[ind_indices]
                y_presuo_ind = y_presuo_ind * trust_y[ind_indices][:, None]
                loss_idn = cross_entropy(y_presuo_ind, sharpened_target_s[ind_indices], reduction='none') + 0.01 * cross_entropy(probs[ind_indices], sharpened_target_s[ind_indices], reduction='none') + 0.1 * entropy_loss(logits[ind_indices], reduction='none') 
                loss_cls = loss_cls + (loss_idn * trust_x[ind_indices] * (1 - trust_y[ind_indices])).mean() 
                
            if ood_indices.shape[0] > 0:
                y_presuo_ood = torch.ones(len(ood_indices), n_classes).to(device)
                y_presuo_ood = y_presuo_ood * trust_y[ood_indices][:, None] / n_classes
                loss_ood =  cross_entropy(y_presuo_ood, flattened_target_s[ood_indices], reduction='none') + 0.1 * cross_entropy(probs[ood_indices], flattened_target_s[ood_indices], reduction='none')
                loss_cls = loss_cls + (loss_ood * (1 - trust_x[ood_indices])).mean()

            all_labels = torch.zeros(y.shape[0], n_classes)
            all_labels = all_labels.to(torch.float64)
            all_labels = all_labels.to(device)
            all_labels[train_indices] = given_labels
            

            aggregated = MessagePassing()
            all_labels = aggregated(all_labels, g.edge_index)
            loss_nois = nois_loss_func(given_labels * trust_y[:, None], all_labels[train_indices])
            loss = loss_cls + cfg.lambda_loss_nois * loss_nois + cfg.lambda_loss_rec * loss_rec
            print("----------------------------------")

        loss.backward()
        optimizer.step()

        train_accuracy = train_acc[0]
        train_loss = loss.item()

        test_ind_indices = np.array(test_indices[np.where(y_true[test_indices] != -1)])
        test_ood_indices = np.array(test_indices[np.where(y_true[test_indices] == -1)])
        test_eval_result = evaluate_test(g, net, y_true, test_indices, test_ind_indices, test_ood_indices, ground_truth_degree_matrix)
        test_acc = test_eval_result['test_acc']
        test_f1 = test_eval_result['test_f1']
        test_ind_acc = test_eval_result['test_ind_acc']
        test_ood_acc = test_eval_result['test_ood_acc']
        test_auc = test_eval_result['test_auc']
        test_acc_list.append(test_acc)
        test_f1_list.append(test_f1)
        test_ind_acc_list.append(test_ind_acc)
        test_ood_acc_list.append(test_ood_acc)
        test_auc_list.append(test_auc)
        logger.info(f'test_acc : {test_acc}, test_f1 : {test_f1}')
        logger.info(f'test_ind_acc : {test_ind_acc}, test_ood_acc : {test_ood_acc}')
        logger.info(f'test_auc : {test_auc}')
        test_accuracy = test_eval_result['test_acc']
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            if cfg.save_model:
                torch.save(net.state_dict(), f'{result_dir}/best_epoch.pth')
                torch.save(net, f'{result_dir}/best_model.pth')

        runtime = time.time() - start_time
        logger.info(f'epoch: {epoch + 1:>3d} | '
                    f'train loss: {train_loss:>6.4f} | '
                    f'train accuracy: {train_accuracy:>6.3f} | '
                    f'test loss: {0:>6.4f} | '
                    f'test accuracy: {test_accuracy:>6.3f} | '
                    f'epoch runtime: {runtime:6.2f} sec | '
                    f'best accuracy: {best_accuracy:6.3f} @ epoch: {best_epoch:03d}')
        
    test_epoch = test_acc_list.index(np.max(test_acc_list))
    logger.info(f'max_test_acc_epoch : {test_epoch}')
    logger.info(f'max_test_ind_acc_by_test : {test_ind_acc_list[test_epoch]}')
    logger.info(f'max_test_ood_acc_by_test : {test_ood_acc_list[test_epoch]}')
    logger.info(f'max_test_acc_by_test : {test_acc_list[test_epoch]}')
    logger.info(f'max_test_f1_by_test : {test_f1_list[test_epoch]}')
    logger.info(f'max_test_auc_by_test : {test_auc_list[test_epoch]}')


#cora
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/citeseer.cfg')
    parser.add_argument('--synthetic-data', type=str, default='citeseer')
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--noise-type', type=str, default='symmetric')
    parser.add_argument('--closeset-ratio', type=float, default='0.05')
    parser.add_argument('--gpu', type=str, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--project', type=str, default='')
    parser.add_argument('--log', type=str, default='RONG')
    
    parser.add_argument('--lambda_loss_rec', type=float, default=1)
    parser.add_argument('--lambda_loss', type=float, default=1)
    parser.add_argument('--lambda_loss_nois', type=float, default=1)
    
    parser.add_argument('--tau1', type=float, default=0.05)
    parser.add_argument('--tau2', type=float, default=0.63)


    parser.add_argument('--loss-func-nois', type=str, default='mae')
    parser.add_argument('--neg-cons', action='store_true')
    parser.add_argument('--ablation', action='store_true')



    
    args = parser.parse_args()

    config = load_from_cfg(args.config)
    override_config_items = [k for k, v in args.__dict__.items() if k != 'config' and v is not None]
    for item in override_config_items:
        config.set_item(item, args.__dict__[item])
    
    assert config.noise_type in ['symmetric', 'asymmetric']
    if config.ablation:
        config.project = f'ablation/{config.project}'
    config.log_freq = LOG_FREQ
    print(config)
    return config



if __name__ == '__main__':
    params = parse_args()
    dev = set_device(params.gpu)
    script_start_time = time.time()
    main(params, dev)                    
    script_runtime = time.time() - script_start_time
    print(f'Runtime of this script {str(pathlib.Path(__file__))} : {script_runtime:.1f} seconds ({script_runtime/3600:.3f} hours)')
