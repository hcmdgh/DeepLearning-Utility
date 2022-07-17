from .imports import * 


def set_cwd(path: str):
    os.chdir(os.path.dirname(path))


def init_log(log_path: Optional[str] = './log.log',
             stdout: bool = True):
    handlers = []
             
    if log_path:
        handlers.append(logging.FileHandler(log_path, 'w', encoding='utf-8'))
    
    if stdout:
        handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        format = '%(asctime)s [%(levelname)s] %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S',
        handlers = handlers,
        level = logging.INFO,
    )
    
    
def log_multi(
    wandb_log: bool = False,
    **kwargs,
):
    seps = []
    
    for key, value in kwargs.items():
        if isinstance(value, float):
            seps.append(f"{key}: {value:.4f}") 
        else:
            seps.append(f"{key}: {value}")
            
    text = ', '.join(seps)
    
    logging.info(text)
    
    if wandb_log:
        if 'epoch' in kwargs:
            epoch = kwargs.pop('epoch')
        else:
            epoch = None 
        
        wandb.log(
            kwargs, 
            step = epoch,
        )
            

def get_set_mapping(_set: set[Any]) -> tuple[list[Any], dict[Any, int]]:
    idx2val = list(_set)
    val2idx = { v: i for i, v in enumerate(idx2val) }
    
    return idx2val, val2idx


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def clone_module(module: nn.Module,
                 cnt: int) -> nn.ModuleList:
    module_list = nn.ModuleList() 
    
    for _ in range(cnt):
        module.reset_parameters()
        module_list.append(module)
        
        module = copy.deepcopy(module)
        
    return module_list


def load_yaml(file_path: str) -> DotDict:
    with open(file_path, 'r', encoding='utf-8') as fp:
        obj = yaml.safe_load(fp)
        
    assert isinstance(obj, dict)
    
    def to_dotdict(_dict: dict) -> DotDict:
        for key, value in _dict.items():
            if isinstance(value, dict):
                _dict[key] = to_dotdict(value)
                
        return DotDict(_dict)
    
    return to_dotdict(obj)
