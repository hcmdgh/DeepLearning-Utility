from .imports import * 

import gzip 

__all__ = [
    'pickle_dump',
    'pickle_load',
    'torch_dump',
    'torch_load',
]


def pickle_dump(obj: Any,
                file_path: str):
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'wb') as fp:
            pickle.dump(obj, fp) 
    elif file_path.endswith('.pkl'):
        with open(file_path, 'wb') as fp:
            pickle.dump(obj, fp) 
    else:
        raise AssertionError 


def pickle_load(file_path: str) -> Any:
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rb') as fp:
            return pickle.load(fp)
    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as fp:
            return pickle.load(fp)
    else:
        raise AssertionError 
    
    
def torch_dump(obj: Any,
               file_path: str):
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'wb') as fp:
            torch.save(obj, fp) 
    elif file_path.endswith('.pt'):
        torch.save(obj, file_path) 
    else:
        raise AssertionError 


def torch_load(file_path: str) -> Any:
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rb') as fp:
            return torch.load(fp)
    elif file_path.endswith('.pt'):
        return torch.load(file_path)
    else:
        raise AssertionError 
