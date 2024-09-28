import numpy as np
import torch     
import torch.nn as nn
import torch.nn.functional as F

class Fssn_layers(nn.Module):
    def __init__(self, ntype, alpha, device, pooling = 'max', concat=True):
        super(Fssn_layers, self).__init__()
        self.concat = concat
        self.ntype = ntype
        self.device = device
        self.pooling = pooling 
        self.leakyrelu = nn.LeakyReLU(alpha)
    
    def aggregator(self, batch, att_feat): 
        nodes = torch.unique(batch)
        batch_nodes = batch.T.flatten()
        
        if self.pooling == 'avg':
            node_aggregation = torch.zeros((len(nodes),att_feat.shape[0]), dtype = torch.float32) 
            for node in nodes:
                mean = 1/(batch_nodes == nodes).sum()
                node_aggregation[node][(nodes == node)] = mean
                
            aggregated = torch.matmul(node_aggregation.to(self.device), att_feat.transpose(0,1)) 
            
        if self.pooling == 'max':
            aggregated = torch.stack([torch.max(att_feat[batch_nodes == node], dim = 0)[0] for node in nodes])
            aggregated = aggregated.transpose(0,1) # 4heads x unique_nodes x dims
            
        return aggregated


    def forward(self, batch, batch_features, att_weights):
        filters = ~(torch.eye(self.ntype).bool())
        feat = torch.cat([F.embedding(batch[:,filter], batch_features) for filter in filters])
        att_feat = torch.matmul(att_weights,feat)
             
        aggregated = self.aggregator(batch, att_feat) # pooling
        batch_features = batch_features + aggregated # skip-connection
        batch_features = batch_features.transpose(0,1)
            
        if self.concat:
            batch_features = self.leakyrelu(batch_features) # activation
            batch_features = batch_features.reshape(batch_features.shape[0],-1)
        else: 
            batch_features = torch.mean(batch_features,dim=1)

        return batch_features
    

class AttentionLayer_wg(nn.Module):  # with global
    def __init__(self, ntype, hidden_dim, dropout_rate, alpha, beta, nheads, device, readout = 'avg'): 
        super(AttentionLayer_wg, self).__init__()
        self.ntype = ntype
        self.beta = beta  
        self.nheads = nheads  # multi-heads
        self.device = device
        self.readout = readout # global readout
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout_rate)
        self.a = nn.ParameterList([nn.Parameter(torch.empty(size=(nheads, 3 * hidden_dim), device=device))
                                   for i in range(ntype)]) # attention vector
        for i in range(ntype):
            nn.init.xavier_uniform_(self.a[i], gain=1.414)

    def forward(self, feats, subgraphs, attentions):
        batch_size = len(subgraphs)

        subgraphs_features = torch.stack([feats[p] for p in subgraphs])
        a_input = self._prepare_attentional_mechanism_input(feats, subgraphs_features, subgraphs) # concatenation
        att_temp = torch.cat([torch.matmul(a_input[i*batch_size:(i+1)*batch_size], self.a[i].T) for i in range(self.ntype)])
        att_temp = att_temp.transpose(1,2)
        att_temp = self.leakyrelu(att_temp) # activation

        att_temp = F.softmax(att_temp, dim=2)
        att_temp = self.dropout(att_temp)

        if len(attentions) != 0: # skip-connection
            attentions = (1 - self.beta) * att_temp + self.beta * attentions.detach()
        else:
            attentions = att_temp

        return attentions

    def _prepare_attentional_mechanism_input(self, feat, subgraphs_features, subgraphs):
        N = self.ntype
        batch_size = len(subgraphs)
        all_combinations_matrix = []
        
        for i in range(N):
            type_selector = torch.zeros(N).bool()
            type_selector[i] = True
            subgraph_repeated_in_chunks = subgraphs_features[:, type_selector]
            subgraph_repeated_in_chunks = subgraph_repeated_in_chunks.repeat_interleave(N-1, dim=0).reshape(batch_size, N-1, -1)
            subgraph_repeated_in_chunks = torch.cat([subgraph_repeated_in_chunks, subgraphs_features[:, ~type_selector]], dim=2)
            all_combinations_matrix.append(subgraph_repeated_in_chunks)
        all_combinations_matrix = torch.cat(all_combinations_matrix)

        global_read_out = self.global_read_out(feat)  # calculating global embedding
        global_read_out = global_read_out.repeat(all_combinations_matrix.shape[0] * all_combinations_matrix.shape[1]).reshape(all_combinations_matrix.shape[0],all_combinations_matrix.shape[1],global_read_out.shape[0])
        all_combinations_matrix_with_global = torch.cat([all_combinations_matrix, global_read_out], dim=2) # concatenate global embedding

        return all_combinations_matrix_with_global
    
    def global_read_out(self, feat):
        if self.readout == 'max':
            read_out = torch.max(feat, dim = 0)[0]
                
        if self.readout == 'sum':
            read_out = torch.sum(feat, dim = 0)
        
        if self.readout == 'avg':
            read_out = torch.mean(feat, dim = 0)

        return read_out


class Dscrmnt_v2(nn.Module):
    def __init__(self, ntype, out_dim, alpha, device, readout):
        super(Dscrmnt_v2, self).__init__()
        self.ntype = ntype
        self.device = device
        self.out_dim = out_dim

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(0.5)
        self.fc_graph = nn.Linear(out_dim,out_dim)
        self.fc_node = nn.ModuleList([nn.Linear(out_dim, out_dim) for _ in range(self.ntype)])
        self.OPD_out_node = nn.Linear(out_dim, 1)
        self.OPD_out_graph = nn.Linear(out_dim, 1)
        self.readout = readout # graph readout
    
    def graph_read_out(self, feats, perturbation):
        read_out = []
        if self.readout == 'max':
            for graph in perturbation:
                graph_embedd = feats[graph]
                read_out.append(torch.max(graph_embedd, dim = 0)[0])
            read_out = torch.stack(read_out)
                
        if self.readout == 'sum':
            graph_aggregation = torch.zeros((len(perturbation),feats.shape[0]), dtype = torch.float32)
            for graph_id in range(len(perturbation)):
                graph_aggregation[graph_id][perturbation[graph_id]] = 1
            read_out = torch.matmul(graph_aggregation.to(self.device), feats)
        
        if self.readout == 'avg':
            graph_aggregation = torch.zeros((len(perturbation),feats.shape[0]), dtype = torch.float32)
            for graph_id in range(len(perturbation)):
                graph_aggregation[graph_id][perturbation[graph_id]] = 1/self.ntype
            read_out = torch.matmul(graph_aggregation.to(self.device), feats)

        if self.readout == 'max_zero':
            for graph in perturbation:
                zero_id = graph==-1
                graph_embedd = feats[graph[~zero_id]]
                for _ in graph[zero_id]:
                    graph_embedd = torch.cat([graph_embedd,torch.zeros((1,graph_embedd.shape[1]),device = graph_embedd.device)])
                read_out.append(torch.max(graph_embedd, dim = 0)[0])
            read_out = torch.stack(read_out)
            
            pass
        
        
        return read_out
    
    def forward(self, feats, perturbation):
        
        read_out = self.graph_read_out(feats, perturbation)
        read_out_graph = self.fc_graph(read_out)
        read_out_node = torch.stack([p(read_out_graph) for p in self.fc_node])
        read_out_node = read_out_node.transpose(0,1).reshape(-1,self.out_dim)
       
        logits_node = self.OPD_out_node(read_out_node)
        logits_graph = self.OPD_out_graph(read_out_graph)
        return logits_node,logits_graph 