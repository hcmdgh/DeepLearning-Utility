from typing import Any 

__all__ = [
    'set_config',
    'get_config',
    'clear_config',
]

_config_dict = dict() 


def set_config(k: str, v: Any):
    _config_dict[k] = v 
    
    
def get_config(k: str) -> Any:
    return _config_dict.get(k)


def clear_config():
    _config_dict.clear()
