from .imports import * 

__all__ = [
    'seed_all',
    'auto_set_device',
    'set_device',
    'get_device',
    'MetricRecorder',
]

_device = torch.device('cpu')


def seed_all(seed: Optional[int]):
    if not seed:
        return 
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    dgl.seed(seed)
    dgl.random.seed(seed)


def auto_set_device(use_gpu: bool = True) -> torch.device:
    global _device 

    if not use_gpu:
        _device = torch.device('cpu')
        return _device
    
    exe_res = os.popen('gpustat --json').read() 
    
    state_dict = json.loads(exe_res)
    
    gpu_infos = [] 
    
    for gpu_entry in state_dict['gpus']:
        gpu_id = int(gpu_entry['index'])
        used_mem = int(gpu_entry['memory.used'])

        gpu_infos.append((used_mem, gpu_id))
    
    gpu_infos.sort()
    
    _device = torch.device(f'cuda:{gpu_infos[0][1]}')
    
    return _device 


def set_device(device_name: str):
    global _device
    _device = torch.device(device_name)
    
    
def get_device() -> torch.device:
    return _device 


class MetricRecorder:
    def __init__(self):
        # 第i个元素表示第i个epoch的记录
        self.record_list: list[dict[str, Any]] = [] 
        
    def record(self, 
               *,
               epoch: int,
               log: bool = True,
               wandb_log: bool = False,
               **metrics): 
        metrics = dict(metrics)
        
        for k, v in metrics.items():
            if isinstance(v, (FloatTensor, FloatArray)):
                metrics[k] = float(v)
                
        while len(self.record_list) < epoch:
            self.record_list.append(dict())
        
        assert len(self.record_list) == epoch
        self.record_list.append(metrics)
        
        # [BEGIN] log 
        if log:
            seps = [f"epoch: {epoch}"]
            
            for k, v in metrics.items():
                if isinstance(v, float):
                    seps.append(f"{k}: {v:.4f}") 
                else:
                    seps.append(f"{k}: {v}")
                    
            text = ', '.join(seps)
            
            logging.info(text)
        
        if wandb_log:
            wandb.log(
                metrics, 
                step = epoch,
            )
        # [END]
        
    def check_early_stopping(self,
                             field_name: str,
                             expected_trend: Literal['asc', 'desc'],
                             tolerance: int) -> bool:
        if len(self.record_list) < tolerance + 5:
            return False 
        
        val_list = [] 
        
        for entry in self.record_list[-tolerance:]:
            if field_name in entry:
                val_list.append(entry[field_name])
                
        if not val_list:
            return False 
        
        if expected_trend == 'asc':
            return max(val_list) == val_list[0]
        elif expected_trend == 'desc':
            return min(val_list) == val_list[0]
        else:
            raise AssertionError 

    def best_record(self,
                    field_name: str,
                    log: bool = True,
                    wandb_log: bool = False, 
                    min_max: Literal['min', 'max'] = 'max',) -> tuple[int, float]:
        min_val = +1e9
        max_val = -1e9
        min_epoch = max_epoch = -1 
        
        for epoch, entry in enumerate(self.record_list):
            if field_name in entry:
                val = entry[field_name]
                
                if val < min_val:
                    min_val = val 
                    min_epoch = epoch 
                    
                if val > max_val:
                    max_val = val 
                    max_epoch = epoch 
                    
        if min_max == 'min':
            best_epoch = min_epoch
            best_val = min_val 
        elif min_max == 'max':
            best_epoch = max_epoch
            best_val = max_val
        else:
            raise AssertionError 
 
        if log:
            logging.info(f"Best {field_name}: {best_val:.4f} in epoch {best_epoch}")
            
        if wandb_log:
            wandb.summary['best_epoch'] = best_epoch 
            wandb.summary[f'best_{field_name}'] = best_val  
        
        return best_epoch, best_val 
