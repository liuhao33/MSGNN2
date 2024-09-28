import time, sys
from utils.data import load_DBLP_data2
from utils.tools_re import batch_generator, Logger
from utils.pytorchtools_re import EarlyStopping, FocalLoss, Schema_perturbation, link_prediction
import torch
import torch.nn.functional as F
from model.msgnn_re import MSGNN
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, auc, precision_recall_curve

lr = 0.0003 # learning rate
weight_decay = 0.001
repeat = 1
num_epochs = 200
batch_size = 64 
patience = 10
perturb_ratio = 0.6
nlayers = 2 # number of layers

gamma =  0.8454 

global_readout = 'avg'
pooling = 'max'
graph_readout = 'max' 
sp_mode = 'normal' 

out_dim = 256
hidden_dim = 1024
dropout_rate = 0.3 # dropout rate
alpha = 0.01 # leaky slope
beta_1 = 0.1 
nheads_1 = 4 # number of attention heads

save_postfix = 'msgnn_dblp'
sys.stdout = Logger(r'./train_log/' + save_postfix +'.log', sys.stdout)
data_path = r'./data/DBLP_processed'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

features, subgraphs, subgraphs_train_val_test, labels, adj, type_mask = load_DBLP_data2(prefix = data_path)

subgraphs_train = subgraphs_train_val_test['subgraphs_train']
subgraphs_val = subgraphs_train_val_test['subgraphs_val']
subgraphs_test = subgraphs_train_val_test['subgraphs_test']

in_dims = [feature.shape[1] for feature in features]
features = [torch.FloatTensor(feature).to(device) for feature in features]
ntype = len(in_dims)

auc_lp_list = []
pr_auc_lp_list = []
f1_lp_list = []

t_all_start = time.time()

FLloss = FocalLoss(alpha=0.64)
print('name:', save_postfix)
for rpt in range(repeat):
    rpt_start = time.time()
    net = MSGNN(ntype, in_dims, hidden_dim, out_dim, nheads_1, dropout_rate, alpha, beta_1, device, nlayers, global_readout, pooling, graph_readout)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=patience, verbose=True, save_path=r'./ckpt/checkpoint_{}_{}.pt'.format(save_postfix, rpt))

    dur1_opd = []
    dur2_opd = []
    dur3_opd = []
    
    batch_g = batch_generator(subgraphs_train, features, device, batch_size)
    val_g = batch_generator(subgraphs_val, features, device, 2048, shuffle=False)
    sp = Schema_perturbation(subgraphs, perturb_ratio, device, mode = sp_mode)
    
    for epoch in range(num_epochs):
        t_start = time.time()
        ## training
        net.train()  
        for iteration in range(batch_g.num_iterations()):
            # forward
            t0 = time.time()
            batch, batch_features, _ = batch_g.next()
            perturbation, node_label, label = sp.get_perturbation(batch)
            features_train, y_pred,y_pred_graph = net((batch_features, batch, perturbation))

            t1 = time.time()
            dur1_opd.append(t1 - t0)
            label = torch.from_numpy(label).float().to(device)
            node_label = node_label.float().to(device)
            node_label = node_label.reshape(-1,1)
            y_pred = y_pred.reshape(-1,ntype).reshape(-1,1)
            OPD_loss = gamma * FLloss(y_pred, node_label) + F.binary_cross_entropy_with_logits(y_pred_graph, label)

            t2 = time.time()
            dur2_opd.append(t2 - t1)
            
            # autograd
            optimizer.zero_grad()

            OPD_loss.backward()
            
            optimizer.step()

            t3 = time.time()
            dur3_opd.append(t3 - t2)

            # print training info
            if iteration % 50 == 0:
                print(
                    'Epoch {:03d} | Iteration {:04d} | Train_Loss {:.4f} | Time1(s) {:.4f}/{:.3f} | Time2(s) {:.4f}/{:.3f} | Time3(s) {:.4f}/{:.3f}'.format(
                        epoch, iteration, OPD_loss.item(), np.mean(dur1_opd), np.mean(dur1_opd)*50, np.mean(dur2_opd),np.mean(dur2_opd)*50, np.mean(dur3_opd),np.mean(dur3_opd)*50))

        # eval
        net.eval()
        val_loss = []
        with torch.no_grad():
            for iteration in range(val_g.num_iterations()):
                # forward
                batch, batch_features, _ = val_g.next()
                perturbation, node_label, label = sp.get_perturbation(batch)
                features_train, y_pred,y_pred_graph = net((batch_features, batch, perturbation))

                t1 = time.time()
                dur1_opd.append(t1 - t0)
                label = torch.from_numpy(label).float().to(device)
                node_label = node_label.float().to(device)
                zero_rows=torch.all(node_label==0,dim=1).nonzero().squeeze()
                node_label = node_label[~zero_rows].reshape(-1,1)
                y_pred = y_pred.reshape(-1,ntype)[~zero_rows].reshape(-1,1)
                OPD_loss = gamma * FLloss(y_pred, node_label) + F.binary_cross_entropy_with_logits(y_pred_graph, label)

                val_loss.append(OPD_loss.item()) 

        t_end = time.time()
        # print validation info
        print('Epoch {:03d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
            epoch, np.mean(val_loss), t_end - t_start))   
        print('lr = {}, weight_decay = {}, batch_size = {}'.format(lr, weight_decay, batch_size))
        early_stopping(np.mean(val_loss), net)
        if early_stopping.early_stop:
            print('Early stopping!')
            break

    print('——__——__——__——__——__——__——__——__——__——__')
    print('epoch iter ends, test starts!')
    t_test = time.time()
    test_g = batch_generator(subgraphs_train, features, device, 2048, shuffle = False)

    net.load_state_dict(torch.load(r'./ckpt/checkpoint_{}_{}.pt'.format(save_postfix, rpt)))
    net.eval()

    auc_lp_batch_list = []
    f1_lp_batch_list = []
    pr_auc_lp_batch_list = []
    OPD_loss_list = []
    
    with torch.no_grad():
        for iteration in range(test_g.num_iterations()):
            # forward
            batch, batch_features, batch_adj = test_g.next()
            
            perturbation = None

            features_test, y_pred,y_pred_graph = net((batch_features, batch, perturbation))
            
            pos_out, neg_out, y_true_test = link_prediction(batch_adj.cpu(), adj, features_test)
            
            pos_proba = torch.sigmoid(pos_out.flatten())
            neg_proba = torch.sigmoid(neg_out.flatten())
            
            y_proba_test = torch.cat([pos_proba, neg_proba])
            y_proba_test = y_proba_test.cpu()
            
            y_proba_test_label = torch.where(y_proba_test > sorted(y_proba_test)[len(y_proba_test)//2], torch.ones_like(y_proba_test), torch.zeros_like(y_proba_test)).int()

            auc_lp = roc_auc_score(y_true_test, y_proba_test)
            f1_lp = f1_score(y_true_test, y_proba_test_label)
            precision_lp, recall_lp, _ = precision_recall_curve(y_true_test, y_proba_test)
            pr_auc_lp = auc(recall_lp, precision_lp)
        
            auc_lp_batch_list.append(auc_lp)
            f1_lp_batch_list.append(f1_lp)
            pr_auc_lp_batch_list.append(pr_auc_lp)
            
        dur_test = time.time() - t_test    
    
    print('\n')
    print('Schema Masking Learning')
    print('\n')
    print('repeat {} with totally {} epoches'.format((rpt + 1), (epoch + 1)))

    print('\n')
    print('AUC_lp = {}'.format(np.mean(auc_lp_batch_list)))
    print('PR_AUC_lp = {}'.format(np.mean(pr_auc_lp_batch_list)))
    print('F1_lp = {}'.format(np.mean(f1_lp_batch_list)))
    print('\n')

    print('test costs {:.4f} seconds'.format(dur_test))
    print('rpt costs {:.4f} mins'.format((time.time()-rpt_start)/60))
    print('——__——__——__——__——__——__——__——__——__——__')
    print('\n')
    
    auc_lp_list.append(np.mean(auc_lp_batch_list))
    f1_lp_list.append(np.mean(f1_lp_batch_list))
    pr_auc_lp_list.append(np.mean(pr_auc_lp_batch_list))

print('----------------------------------------------------------------')
print('Schema Masking Learning Summary')
print('\n')
print('name:', save_postfix)
print('\n')
print('total time costs {:.2f} hours'.format((time.time()-t_all_start)/3600))
print('\n')
print('AUC_lp_mean = {:.5f}, AUC_std = {:.5f}'.format(np.mean(auc_lp_list), np.std(auc_lp_list)))
print('PR_AUC_lp_mean = {:.5f}, AP_std = {:.5f}'.format(np.mean(pr_auc_lp_list), np.std(pr_auc_lp_list)))
print('F1_lp_mean = {:.5f}, AP_std = {:.5f}'.format(np.mean(f1_lp_list), np.std(f1_lp_list)))

print('\n')
print('AUC_lp history = {}'.format(auc_lp_list))
print('PR_AUC_lp history = {}'.format(pr_auc_lp_list))
print('F1_lp history = {}'.format(f1_lp_list))