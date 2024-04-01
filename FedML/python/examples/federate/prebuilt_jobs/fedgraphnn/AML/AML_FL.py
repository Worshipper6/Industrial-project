import time
import logging
import os
import sys
from data_loading import get_data
from training import train_gnn
from FL_loading import load_partition_data
# from fedml import FedMLRunner
# from trainer.fed_AML_trainer import FedSubgraphLPTrainer
# from trainer.fed_AML_aggregator import FedSubgraphLPAggregator

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


args = Args(ports=False, tds=False, model='pna', data=None, reverse_mp=False, ego=False, emlps=False,\
            n_epochs=100, batch_size=8192, num_neighs=[100,100], tqdm=True, wandb=False)


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
    # # init FedML framework
    # args = fedml.init()
    #
    # # init device
    # device = fedml.device.get_device(args)


    # load data
    dataset = load_data(args)

    # # load model
    # model = create_model(args, args.model, dataset[7])
    #
    # trainer = FedSubgraphLPTrainer(model, args)
    # aggregator = FedSubgraphLPAggregator(model, args)
    #
    # # start training
    # fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    # fedml_runner.run()
