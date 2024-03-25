from data.data_loader import *

from model.gcn_link import GCNLinkPred
from model.gat_link import GATLinkPred
from model.sage_link import SAGELinkPred

from trainer.fed_subgraph_lp_trainer import FedSubgraphLPTrainer

import wandb

class Args():
    def __init__(self, dataset, data_cache_dir, model, client_num_in_total,
                 hidden_size, node_embedding_dim,
                 client_optimizer, learning_rate, weight_decay, epochs, num_heads):
        self.dataset = dataset
        self.data_cache_dir = data_cache_dir
        self.model = model
        self.client_num_in_total = client_num_in_total
        self.hidden_size = hidden_size
        self.node_embedding_dim = node_embedding_dim

        self.client_optimizer=client_optimizer
        self.learning_rate = learning_rate
        self.weight_decay=weight_decay
        self.epochs = epochs
        self.num_heads = num_heads
        

def load_data(args):
    if args.dataset not in ["ciao", "epinions"]:
        raise Exception("no such dataset!")

    args.pred_task = "link_prediction"

    args.metric = "MSE"

    if args.model == "gcn":
        args.normalize_features = True
        args.normalize_adjacency = True

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
        test_data_local_dict,
        feature_dim,
    ) = load_partition_data(args, args.data_cache_dir, args.client_num_in_total)

    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        feature_dim,
    ]

    return dataset


def create_model(args, model_name, feature_dim):
    print("create_model. model_name = %s" % (model_name))
    if model_name == "gcn":
        model = GCNLinkPred(feature_dim, args.hidden_size, args.node_embedding_dim)
    elif model_name == "gat":
        model = GATLinkPred(
            feature_dim, args.hidden_size, args.node_embedding_dim, args.num_heads
        )
    elif model_name == "sage":
        model = SAGELinkPred(feature_dim, args.hidden_size, args.node_embedding_dim)
    else:
        raise Exception("such model does not exist !")
    return model


if __name__ == '__main__':
    # args
    args = Args('epinions', data_cache_dir='./data', model='gcn', client_num_in_total=4, hidden_size=32, node_embedding_dim=32, 
                client_optimizer='adam', learning_rate=0.001, weight_decay=0.001, epochs=100, num_heads=2)
    # load data
    dataset = load_data(args)

    # load model
    model = create_model(args, args.model, dataset[7])
    # # train
    # trainer = FedSubgraphLPTrainer(model, args)
    # # Add test_set
    # if not hasattr(trainer, 'test_data'):
    #     setattr(trainer, 'test_data', dataset[3])
    # if not hasattr(trainer, 'Wandb'):
    #     setattr(trainer, 'Wandb', True)
    #     # 🐝 1️⃣ Start a new run to track this script
    #     wandb.init(
    #         # Set the project where this run will be logged
    #         project="Epinions", 
    #         # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    #         name=f"gcn_experiment_central", 
    #     )
    
    # trainer.train(dataset[2], 'cpu', args)
    # wandb.finish()

    # # Local train
    # train_local_dict = dataset[5]
    # for i in range(len(train_local_dict)):
    #     # load model
    #     model = create_model(args, args.model, dataset[7])
    #     # train
    #     trainer = FedSubgraphLPTrainer(model, args)
    #     # Add test_set
    #     if not hasattr(trainer, 'test_data'):
    #         setattr(trainer, 'test_data', dataset[3])
    #     if not hasattr(trainer, 'Wandb'):
    #         setattr(trainer, 'Wandb', True)
    #         # 🐝 1️⃣ Start a new run to track this script
    #         wandb.init(
    #             # Set the project where this run will be logged
    #             project="Epinions", 
    #             # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    #             name=f"gcn_experiment_local{i}", 
    #         )
        
    #     trainer.train(train_local_dict[i], 'cpu', args)
    #     wandb.finish()

    
