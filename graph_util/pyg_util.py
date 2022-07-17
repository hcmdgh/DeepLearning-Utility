from ..imports import * 
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon

__all__ = ['load_pyg_dataset']


def load_pyg_dataset(dataset_name: str) -> dgl.DGLGraph:
    dataset_name = dataset_name.lower().strip() 
    
    if dataset_name == 'amazon-computers':
        dataset = Amazon(root='/home/Dataset/PyG/Amazon-Computers', name='computers')
        _graph = dataset[0]
        feat = _graph.x 
        edge_index = tuple(_graph.edge_index)
        label = _graph.y 
        
        graph = dgl.graph(edge_index, num_nodes=len(feat))
        graph.ndata['feat'] = feat 
        graph.ndata['label'] = label 
        
        return graph 
    else:
        raise AssertionError 
