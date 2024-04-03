import time
import fedml
import logging
from data_loading import get_data
from train_util import extract_param
from FL_loading import load_partition_data
from fedml import FedMLRunner
from models import RGCN, PNA

from trainer.fed_AML_trainer import FedClsTrainer
from trainer.fed_AML_aggregator import FedSubgraphLPAggregator

# def logger_setup():
#     # Setup logging
#     log_directory = "logs"
#     if not os.path.exists(log_directory):
#         os.makedirs(log_directory)
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s [%(levelname)-5.5s] %(message)s",
#         handlers=[
#             logging.FileHandler(os.path.join(log_directory, "logs.log")),     ## log to local log file
#             logging.StreamHandler(sys.stdout)          ## log also to stdout (i.e., print to screen)
#         ]
#     )
#
# logger_setup()

class Args():
    def __init__(self, ports, tds, model, data, reverse_mp, ego, emlps, n_epochs, batch_size, num_neighs, tqdm, wandb):
        self.ports = ports
        self.tds = tds
        self.model = model
        self.data = data
        self.reverse_mp = reverse_mp
        self.ego = ego
        self.emlps = emlps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.num_neighs = num_neighs
        self.tqdm = tqdm
        self.wandb = wandb


#args = Args(ports=False, tds=False, model='pna', data=None, reverse_mp=False, ego=False, emlps=False,\
            #n_epochs=100, batch_size=8192, num_neighs=[100,100], tqdm=True, wandb=False)

def create_model(args, model_name):
    logging.info("create_model. model_name = %s" % (model_name))
    if model_name == "rgcn":
        model = RGCN(
            num_features=1, edge_dim=3, num_relations=8, num_gnn_layers=round(args.n_gnn_layers),
            n_classes=2, n_hidden=round(args.n_hidden),
            edge_update=args.emlps, dropout=args.dropout, final_dropout=args.final_dropout, n_bases=None  # (maybe)
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
    ) = load_partition_data(args, 'raw/formatted_transactions.csv', 'raw/bank/bank',\
                            970)

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
    model = create_model(args, args.model)

    # load trainer
    args.w_ce1 = extract_param("w_ce1", args)
    args.w_ce2 = extract_param("w_ce2", args)
    trainer = FedClsTrainer(model, args)
    aggregator = FedSubgraphLPAggregator(model, args)
    #
    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    fedml_runner.run()
