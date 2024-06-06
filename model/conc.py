from model.layers import MLP,FNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.multiprocessing as mp
import random
import math
from torch_geometric.nn import MessagePassing

class MessagePassing(MessagePassing):
    def __init__(self): 
        super(MessagePassing, self).__init__(aggr='sum')

    def forward(self, x, edge_index):
        def message_func(edge_index, x_i):
            return x_i

        aggregated = self.propagate(edge_index, x=x, message_func=message_func)
        return aggregated


class CONC(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, out_dim, lambda_loss, device):
        super(CONC, self).__init__()
        self.lambda_loss = lambda_loss
        
        self.encoderconv1 = GCNConv(in_dim, hidden_dim2)
        self.encoderconv2 = GCNConv(hidden_dim2, hidden_dim2)
        self.encoderconv3 = GCNConv(hidden_dim2, hidden_dim2)
        self.encoderconv4 = GCNConv(hidden_dim2, hidden_dim2)

        self.degree_decoder = FNN(hidden_dim2, hidden_dim2, 1, 4)
        self.feature_decoder = FNN(hidden_dim2, hidden_dim2, in_dim, 3)
        self.degree_loss_func = nn.MSELoss()
        self.feature_loss_func = nn.MSELoss()

        self.conv1 = GCNConv(in_dim, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.classifiation = nn.Linear(hidden_dim2, out_dim)
        self.trust_worthiness = nn.Linear(hidden_dim2, 2)



    def forward_decoder(self, x, h, edge_index,ground_truth_degree_matrix, device):
        degree_logits = self.degree_decoding(h)
        ground_truth_degree_matrix = torch.unsqueeze(ground_truth_degree_matrix, dim=1)
        degree_loss = self.degree_loss_func(degree_logits, ground_truth_degree_matrix.float())
        x_rec = self.feature_decoder(h)
        feature_loss = self.feature_loss_func(x, x_rec)
        loss = degree_loss  + self.lambda_loss * feature_loss
        return loss, x_rec

    def degree_decoding(self, node_embeddings):
        degree_logits = F.relu(self.degree_decoder(node_embeddings))
        return degree_logits

    def forward(self, g, ground_truth_degree_matrix, device, isTrain=False):
        x, edge_index = g.x, g.edge_index
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index)
        class_logits = self.classifiation(h)
        h = self.encoderconv1(x, edge_index)
        h = self.encoderconv2(h, edge_index)
        h = self.encoderconv3(h, edge_index)
        h = self.encoderconv4(h, edge_index)
        trust_logits = self.trust_worthiness(h)
        trust_prob = trust_logits.sigmoid()
        
        if isTrain == False:
            return {'class_logits': class_logits}
        else:
            loss, x_rec= self.forward_decoder(x, h, edge_index, ground_truth_degree_matrix, device)
            h = self.conv1(x_rec, edge_index).relu()
            h = self.conv2(h, edge_index)
            class_logits_reconstructed = self.classifiation(h)
            return {'class_logits': class_logits, 'class_logits_reconstructed' : class_logits_reconstructed, 'trust_prob': trust_prob, 'loss_rec': loss}
