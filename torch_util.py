from .imports import * 
from .metric import * 

from io import BytesIO

__all__ = [
    'seed_all',
    'auto_set_device',
    'set_device',
    'get_device',
    'MetricRecorder',
    'ClassificationRecorder', 
    'is_on_cpu',
    'is_on_gpu',
]

_device = torch.device('cpu')


def is_on_cpu(obj: Any) -> bool:
    device_type = obj.device.type 
    
    if device_type == 'cpu':
        return True 
    elif device_type == 'cuda':
        return False 
    else:
        raise AssertionError 
    
    
def is_on_gpu(obj: Any) -> bool:
    return not is_on_cpu(obj)


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


class ClassificationRecorder:
    def __init__(self,
                 log: bool = True,
                 wandb_log: bool = True,
                 model: Optional[nn.Module] = None):
        self.use_log = log 
        self.use_wandb_log = wandb_log 
        self.model = model 
        
        self.best_val_acc = 0.
        self.best_val_epoch = -1 
        self.best_val_model_state: bytes = bytes() 
        
    def train(self,
              epoch: int,
              loss: Any):
        loss = float(loss)
        
        if self.use_log:
            logging.info(f"epoch: {epoch}, loss: {loss:.4f}")
            
        if self.use_wandb_log:
            wandb.log(
                { 'loss': loss }, 
                step = epoch,
            )
    
    def validate(self,
                 epoch: int, 
                 val_acc: Optional[float] = None,
                 y_pred: Optional[FloatArrayTensor] = None,
                 y_true: Optional[IntArrayTensor] = None):
        if val_acc is not None and y_pred is None and y_true is None:
            pass 
        elif val_acc is None and y_pred is not None and y_true is not None:
            val_acc = calc_acc(y_pred=y_pred, y_true=y_true)
        else:
            raise AssertionError 
        
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc 
            self.best_val_epoch = epoch 
            
            if self.model is not None:
                bio = BytesIO()
                torch.save(self.model.state_dict(), bio)
                self.best_val_model_state = bio.getvalue()
        
        if self.use_log:
            logging.info(f"epoch: {epoch}, val_acc: {val_acc:.4f} (best: {self.best_val_acc:.4f} in epoch {self.best_val_epoch})")
            
        if self.use_wandb_log:
            wandb.log(
                { 'val_acc': val_acc }, 
                step = epoch,
            )

    def load_best_model_state(self):
        assert self.model is not None 
        
        bio = BytesIO(self.best_val_model_state)
        self.model.load_state_dict(torch.load(bio))
                
    def test(self,
             test_acc: Optional[float] = None,
             y_pred: Optional[FloatArrayTensor] = None,
             y_true: Optional[IntArrayTensor] = None):
        if self.use_log:
            logging.info(f"Use the best model (epoch: {self.best_val_epoch}, best_val_acc: {self.best_val_acc:.4f}) for testing......")
            
        if test_acc is not None and y_pred is None and y_true is None:
            pass 
        elif test_acc is None and y_pred is not None and y_true is not None:
            test_acc = calc_acc(y_pred=y_pred, y_true=y_true)
        else:
            raise AssertionError
        
        if self.use_log:
            logging.info(f"test_acc: {test_acc:.4f}")
            
        if self.use_wandb_log:
            wandb.summary['best_val_acc'] = self.best_val_acc 
            wandb.summary['best_val_epoch'] = self.best_val_epoch
            wandb.summary['test_acc'] = test_acc  


class MetricRecorder:
    def __init__(self,
                 log: bool = True,
                 wandb_log: bool = True):
        # 第i个元素表示第i个epoch的记录
        self.record_list: list[dict[str, Any]] = [] 

        self.use_log = log 
        self.use_wandb_log = wandb_log
        
    def record(self, 
               epoch: int,
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
        if self.use_log:
            seps = [f"epoch: {epoch}"]
            
            for k, v in metrics.items():
                if isinstance(v, float):
                    seps.append(f"{k}: {v:.4f}") 
                else:
                    seps.append(f"{k}: {v}")
                    
            text = ', '.join(seps)
            
            logging.info(text)
        
        if self.use_wandb_log:
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
 
        if self.use_log:
            logging.info(f"Best {field_name}: {best_val:.4f} in epoch {best_epoch}")
            
        if self.use_wandb_log:
            wandb.summary['best_epoch'] = best_epoch 
            wandb.summary[f'best_{field_name}'] = best_val  
        
        return best_epoch, best_val 
