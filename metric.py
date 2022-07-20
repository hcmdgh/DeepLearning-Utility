from .imports import * 
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

__all__ = [
    'calc_f1_micro',
    'calc_f1_macro',
    'calc_roc_auc_score',
    'calc_acc',
    'calc_cosine_similarity',
]


def convert_y_true_pred(y_true: Union[IntArray, IntTensor],
                        y_pred: Union[IntArray, IntTensor, FloatArray, FloatTensor]) -> tuple[IntArray, IntArray]:
    if isinstance(y_true, Tensor):
        y_true = y_true.detach().cpu().numpy() 
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.detach().cpu().numpy() 

    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=-1)
        
    return y_true, y_pred 


def calc_f1_micro(y_true: Union[IntArray, IntTensor],
                  y_pred: Union[IntArray, IntTensor, FloatArray, FloatTensor]) -> float:
    y_true, y_pred = convert_y_true_pred(y_true, y_pred)
                  
    return f1_score(y_true=y_true, y_pred=y_pred, average='micro')


def calc_f1_macro(y_true: Union[IntArray, IntTensor],
                  y_pred: Union[IntArray, IntTensor, FloatArray, FloatTensor]) -> float:
    y_true, y_pred = convert_y_true_pred(y_true, y_pred)
    
    return f1_score(y_true=y_true, y_pred=y_pred, average='macro')


def calc_roc_auc_score(y_true: IntArray,
                       y_pred: FloatArray) -> float:
    raise NotImplementedError
    return roc_auc_score(y_true=y_true, y_score=y_pred)


def calc_acc(y_true: IntArrayTensor,
             y_pred: Union[IntArrayTensor, FloatArrayTensor]) -> float:
    y_true, y_pred = convert_y_true_pred(y_true, y_pred)
    
    return accuracy_score(y_true=y_true, y_pred=y_pred)


def calc_cosine_similarity(h1: FloatTensor, 
                           h2: FloatTensor) -> FloatTensor:
    assert h1.shape == h2.shape and h1.ndim == 2 
                           
    h1 = F.normalize(h1, p=2, dim=-1)
    h2 = F.normalize(h2, p=2, dim=-1)

    out = torch.mm(h1, h2.T)
    
    return out 
