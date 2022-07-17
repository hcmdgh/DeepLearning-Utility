from ..imports import * 
from ogb.nodeproppred import DglNodePropPredDataset

__all__ = ['load_ogb_dataset']


def load_ogb_dataset(dataset_name: str) -> dgl.DGLGraph:
    if dataset_name in ['ogbn-products', 'ogbn-arxiv', 'ogbn-papers100M', 'ogbn-mag']:
        dataset = DglNodePropPredDataset(name=dataset_name, root='/home/Dataset/OGB')
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]
        
        if dataset_name == 'ogbn-mag':
            num_nodes = graph.num_nodes('paper') 
            
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            train_mask[train_idx['paper']] = True 
            val_mask[valid_idx['paper']] = True 
            test_mask[test_idx['paper']] = True 
            
            graph.nodes['paper'].data['train_mask'] = train_mask
            graph.nodes['paper'].data['val_mask'] = val_mask
            graph.nodes['paper'].data['test_mask'] = test_mask

            graph.nodes['paper'].data['label'] = label['paper'].squeeze()
        else:
            num_nodes = graph.num_nodes() 
            
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            train_mask[train_idx] = True 
            val_mask[valid_idx] = True 
            test_mask[test_idx] = True 
            
            graph.ndata['train_mask'] = train_mask
            graph.ndata['val_mask'] = val_mask
            graph.ndata['test_mask'] = test_mask

            graph.ndata['label'] = label.squeeze()
            
        return graph 
    else:
        raise AssertionError 
