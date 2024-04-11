import time
import fedml
import logging
import torch
from data_loading import get_data
from train_util import extract_param
from FL_loading import load_partition_data
from fedml import FedMLRunner
from models import RGCN, PNA

from trainer.fed_AML_trainer import FedClsTrainer
from trainer.fed_AML_heter_trainer import FedClsTrainer_Heter
from trainer.fed_AML_aggregator import FedSubgraphLPAggregator

from torch_geometric.utils import degree
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero

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
    t1 = time.perf_counter()
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
    t2 = time.perf_counter()
    print(f"Retrieved data in {t2-t1:.2f}s")

    return dataset

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)


    # load data
    dataset = load_data(args)

    # load model
    args.n_gnn_layers = extract_param("n_gnn_layers", args)
    args.dropout = extract_param("dropout", args)
    args.final_dropout = extract_param("final_dropout", args)
    args.n_hidden = extract_param("n_hidden", args)

    sample_batch = next(iter(dataset[2][0]))
    model = create_model(args, args.model, sample_batch)
    if args.reverse_mp:
        model = to_hetero(model, sample_batch.metadata(), aggr='mean')

    # load trainer
    args.w_ce1 = extract_param("w_ce1", args)
    args.w_ce2 = extract_param("w_ce2", args)
    args.lr = extract_param("lr", args)
    if args.reverse_mp:
        trainer = FedClsTrainer_Heter(model, args)
    else:
        trainer = FedClsTrainer(model, args)
    aggregator = FedSubgraphLPAggregator(model, args)
    #
    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    fedml_runner.run()
