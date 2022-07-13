from ..imports import * 
from .bean import * 

from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

__all__ = [
    'load_dgl_dataset',
    'save_dgl_graph',
    'load_dgl_graph', 
]


def load_dgl_dataset(dataset_name: str) -> dgl.DGLGraph:
    dataset_name = dataset_name.lower() 
    
    if dataset_name == 'cora':
        dataset = CoraGraphDataset()
        graph = dataset[0]
        
        return graph 
    
    elif dataset_name == 'citeseer':
        dataset = CiteseerGraphDataset()
        graph = dataset[0]
        
        return graph 
    
    elif dataset_name == 'pubmed':
        dataset = PubmedGraphDataset()
        graph = dataset[0]

        return graph 
    
    else:
        raise AssertionError 


def save_dgl_graph(graph: dgl.DGLGraph,
                   file_path: str):
    if graph.is_homogeneous:
        HomoGraph.from_dgl(graph).save_to_file(file_path)
    else:
        HeteroGraph.from_dgl(graph).save_to_file(file_path)
        
        
def load_dgl_graph(file_path: str) -> dgl.DGLGraph:
    try:
        return HomoGraph.load_from_file(file_path).to_dgl()
    except TypeError:
        return HeteroGraph.load_from_file(file_path).to_dgl()
