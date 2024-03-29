import time

from data_loading import get_data
# from training import train_gnn
# from inference import infer_gnn

class Args():
    def __init__(self, ports, tds, model, reverse_mp):
        self.ports = ports
        self.tds = tds
        self.model = model
        self.reverse_mp = reverse_mp



args = Args(ports=False, tds=False, model='rgcn', reverse_mp=False)


#get data
print("Retrieving data")
t1 = time.perf_counter()

tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args)

t2 = time.perf_counter()
print(f"Retrieved data in {t2-t1:.2f}s")

# if not args.inference:
#     #Training
#     print(f"Running Training")
#     train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config)
# else:
#     #Inference
#     print(f"Running Inference")
#     infer_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config)

