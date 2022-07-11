from .imports import * 

__all__ = [
    'to_torch_sparse',
    'to_coo_mat', 
]


def to_coo_mat(val) -> sp.coo_matrix:
    if isinstance(val, sp.coo_matrix):
        pass 
    elif isinstance(val, (sp.csc_matrix, sp.csr_matrix, sp.lil_matrix, sp.dia_matrix, sp.dok_matrix, sp.bsr_matrix)):
        val = val.tocoo() 
    elif isinstance(val, np.ndarray):
        val = sp.coo_matrix(val)
    elif isinstance(val, torch.Tensor):
        if not val.is_sparse:
            val = val.detach().cpu().numpy() 
            val = sp.coo_matrix(val)
        else:
            if not val.is_coalesced():
                val = val.coalesce()
                
            indices = val.indices() 
            values = val.values() 
            shape = tuple(val.shape) 
            
            val = sp.coo_matrix((values, (indices[0], indices[1])), shape=shape) 
    elif isinstance(val, dict):  # 邻接表
        edge_list = [] 
        num_nodes = 0 
        
        for src_nid in val:
            for dest_nid in val[src_nid]:
                num_nodes = max(num_nodes, src_nid + 1, dest_nid + 1)
                edge_list.append((src_nid, dest_nid))
                
        edge_index = np.array(edge_list, dtype=np.int64).T 
        values = np.ones(len(edge_list), dtype=np.float32)
        
        val = sp.coo_matrix((values, (edge_index[0], edge_index[1])), shape=[num_nodes, num_nodes])
    else:
        raise NotImplementedError
    
    val = val.astype(np.float32)
    
    return val 


def to_torch_sparse(val) -> SparseTensor:
    def coo_mat2torch_sparse(coo_mat: sp.coo_matrix) -> SparseTensor:
        indices = torch.from_numpy(np.vstack([coo_mat.row, coo_mat.col])).to(torch.int64)
        values = torch.from_numpy(coo_mat.data).to(torch.float32)
        shape = torch.Size(coo_mat.shape)
        
        return torch.sparse_coo_tensor(indices, values, shape)
    
    coo_mat = to_coo_mat(val)
    
    return coo_mat2torch_sparse(coo_mat)        
