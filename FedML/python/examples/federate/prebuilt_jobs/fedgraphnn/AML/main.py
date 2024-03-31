import time
import logging
import os
import sys
from data_loading import get_data
from training import train_gnn
#from inference import infer_gnn

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
print("Retrieving data")
t1 = time.perf_counter()

tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args)

t2 = time.perf_counter()
print(f"Retrieved data in {t2-t1:.2f}s")


print(f"Running Training")
train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args)

