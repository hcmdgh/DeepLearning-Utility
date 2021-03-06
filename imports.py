import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd 
from torch import Tensor
from numpy import ndarray
import scipy 
import scipy.io as sio 
import scipy.sparse as sp 
import random
import logging
from typing import Optional, Union, List, Dict, Tuple, Any, Callable, Iterable, Literal, Iterator
import itertools
import functools
import math
from datetime import datetime, date, timedelta
from collections import defaultdict, namedtuple, deque 
from tqdm import tqdm
from pprint import pprint
import pickle
import os
import dataclasses
from dataclasses import dataclass, asdict 
import pymongo
import argparse 
import json
import yaml 
import copy 
import csv 
import matplotlib.pyplot as plt 
import networkx as nx 
import re 
import xgboost as xgb
import requests
import wandb 

# ========== DGL ==========
import dgl
import dgl.function as dglfn
import dgl.nn.pytorch as dglnn
import dgl.nn.functional as dglF
import dgl.data.utils as dglutil
# =========================

# ========== PyG ==========
import torch_geometric as pyg
import torch_geometric.data as pygdata 
import torch_geometric.nn as pygnn 
import torch_geometric.nn.conv as pygconv 
import torch_geometric.loader as pygloader 
import torch_geometric.utils as pygutil
# =========================

IntTensor = FloatTensor = BoolTensor = FloatScalarTensor = SparseTensor = Tensor
IntArray = FloatArray = BoolArray = ndarray
IntArrayTensor = FloatArrayTensor = BoolArrayTensor = Union[Tensor, ndarray]
NodeType = str 
EdgeType = tuple[str, str, str]
