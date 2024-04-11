import time
import logging
import os
import sys
import torch
from train_util import extract_param
from FL_loading import load_partition_data
from training import train_gnn
from trainer.fed_AML_trainer import FedClsTrainer
from trainer.fed_AML_heter_trainer import FedClsTrainer_Heter

from torch_geometric.utils import degree
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero
from models import RGCN, PNA
#from inference import infer_gnn
import wandb

def logger_setup():
    # Setup logging
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_directory, "logs.log")),     ## log to local log file
            logging.StreamHandler(sys.stdout)          ## log also to stdout (i.e., print to screen)
        ]
    )

logger_setup()

class Args():
    def __init__(self, ports, tds, model, data, reverse_mp, ego, emlps,
                 epochs, batch_size, num_neighs, tqdm, wandb, client_number,
                 global_path, local_path):
        self.ports = ports
        self.tds = tds
        self.model = model
        self.data = data
        self.reverse_mp = reverse_mp
        self.ego = ego
        self.emlps = emlps
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_neighs = num_neighs
        self.tqdm = tqdm
        self.wandb = wandb

        self.client_number = client_number
        self.n_gnn_layers = None,
        self.n_hidden = None,
        self.dropout = None,
        self.final_dropout = None,
        self.w_ce1 = None
        self.w_ce2 = None
        self.lr = None

        self.global_path = global_path
        self.local_path = local_path

args = Args(ports=False, tds=False, model='pna', data=None, reverse_mp=True, ego=False, emlps=False,
            epochs=100, batch_size=512, num_neighs=[10,10], tqdm=True, wandb=True, client_number=5,
            global_path='raw/formatted_transactions.csv', local_path='raw/uniform1/client',)
args.n_gnn_layers = extract_param("n_gnn_layers", args)
args.dropout = extract_param("dropout", args)
args.final_dropout = extract_param("final_dropout", args)
args.n_hidden = extract_param("n_hidden", args)
args.w_ce1 = extract_param("w_ce1", args)
args.w_ce2 = extract_param("w_ce2", args)
args.lr = extract_param("lr", args)

#get data
print("Retrieving data")
t1 = time.perf_counter()

#tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args)

def create_model(args, model_name, sample_batch):
    logging.info("create_model. model_name = %s" % (model_name))
    if model_name == "rgcn":
        model = RGCN(
            num_features=1, edge_dim=4, num_relations=8, num_gnn_layers=round(args.n_gnn_layers),
            n_classes=2, n_hidden=round(args.n_hidden),
            edge_update=args.emlps, dropout=args.dropout, final_dropout=args.final_dropout, n_bases=None  # (maybe)
        )
    elif model_name == 'pna':
        if not isinstance(sample_batch, HeteroData):
            d = degree(sample_batch.edge_index[1], dtype=torch.long)
        else:
            index = torch.cat((sample_batch['node', 'to', 'node'].edge_index[1], sample_batch['node', 'rev_to', 'node'].edge_index[1]), 0)
            d = degree(index, dtype=torch.long)
        deg = torch.bincount(d, minlength=1)
        model = PNA(
            num_features=1, num_gnn_layers=round(args.n_gnn_layers), n_classes=2,
            n_hidden=round(args.n_hidden), edge_updates=args.emlps, edge_dim=4,
            dropout=args.dropout, deg=deg, final_dropout=args.final_dropout
            )
    else:
        raise Exception("such model does not exist !")
    return model

#get data
def load_data(args):
    print("Retrieving data")
    (
        train_data_num,
        val_data_num,
        test_data_num,
        train_data_global,
        val_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        val_data_local_dict,
        test_data_local_dict
    ) = load_partition_data(args)

    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict
    ]
    return dataset

t2 = time.perf_counter()
print(f"Retrieved data in {t2-t1:.2f}s")


print(f"Running Training")

dataset = load_data(args)
sample_batch = next(iter(dataset[2][0]))
model = create_model(args, args.model, sample_batch)

if args.reverse_mp:
    model = to_hetero(model, sample_batch.metadata(), aggr='mean')

if args.reverse_mp:
    trainer = FedClsTrainer_Heter(model, args)
else:
    trainer = FedClsTrainer(model, args)

if not hasattr(trainer, 'test_data'):
    setattr(trainer, 'test_data', dataset[3])
if not hasattr(trainer, 'Wandb'):
    setattr(trainer, 'Wandb', True)
    # 🐝 1️⃣ Start a new run to track this script
    wandb.init(
        # Set the project where this run will be logged
        project="AML",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"pna_reverse_central",
    )

trainer.train(dataset[2], 'cuda:0', args)
wandb.finish()
#f1, roc, recall, precision = trainer.test(dataset[2][0], 'cuda:0', args)
#print(f1, roc, recall, precision)
# train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args)

