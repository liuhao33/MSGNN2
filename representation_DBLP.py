'''
this script is for producing representations of all nodes.

'''
import time
# import argparse
from utils.data import load_DBLP_data2
from utils.tools_re import representations_generator
import torch
import torch.nn.functional as F
from model.msgnn_re import MSGNN
import numpy as np

num_epochs = 200
batch_size = 64
patience = 10

nlayers = 2
global_readout = 'avg'
pooling = 'max'
graph_readout = 'max'

hidden_dim = 1024
out_dim = 256
dropout_rate = 0.3
alpha = 0.01 # leakyrelu slope
beta_1 = 0.1 
nheads_1 = 4 

save_postfix = 'dblp'
rpt = 0

data_path = r'./data/DBLP_processed'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

features, subgraphs, subgraphs_train_val_test, labels, adj, type_mask = load_DBLP_data2(prefix = data_path)

subgraphs_train = subgraphs_train_val_test['subgraphs_train']
subgraphs_val = subgraphs_train_val_test['subgraphs_val']
subgraphs_test = subgraphs_train_val_test['subgraphs_test']

in_dims = [feature.shape[1] for feature in features]
features = [torch.FloatTensor(feature).to(device) for feature in features]
ntype = len(in_dims)

t_all_start = time.time()

net = MSGNN(ntype, in_dims, hidden_dim, out_dim, nheads_1, dropout_rate, alpha, beta_1, device, nlayers, global_readout, pooling, graph_readout)
net.to(device)

net.load_state_dict(torch.load(r'./ckpt/checkpoint_{}_{}.pt'.format(save_postfix, rpt)))


def representations_generating_v2(net, subgraphs, features, type_mask, device, save_flag, max_size=10000, name = 'dblp'):
    # from utils.tools import representations_generator
    print('——__——__——__——__——__——__——__——__——__——__')
    print('generating representations')
    t_g = time.time()
    repre_g = representations_generator(subgraphs, features, type_mask, device, max_size=max_size) # 2048
    net.eval()
    with torch.no_grad():
        while repre_g.node_left() > 0:
            _, batch_new, batch_features, feature_idx = repre_g.next()
            
            batch_representations, __, ___ = net((batch_features, batch_new, None))
            
            if repre_g.num_iter() == 0:
                node_representations = batch_representations[feature_idx[0]:feature_idx[1]]
            else: node_representations = torch.cat((node_representations,batch_representations[feature_idx[0]:feature_idx[1]]),dim=0)
    print('finish!')
    if save_flag:
        print('saving...')           
        np.save(data_path + '/representations/'+ name + '_node_representations.npy', node_representations.cpu().numpy())
    print('done! time costs:', time.time() - t_g)
    return node_representations.cpu().numpy()

if __name__ == '__main__':
    node_representations = representations_generating_v2(net, subgraphs, features, type_mask, device, save_flag = True, max_size=10000, name = 'dblp')
    # node_representations = np.load(data_path + '/representations/' + 'dblp' + '_node_representations.npy')
    