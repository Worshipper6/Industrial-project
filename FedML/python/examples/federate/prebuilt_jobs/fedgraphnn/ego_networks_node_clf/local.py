
from data.data_loader import load_partition_data, get_data

from model.gat import GATNodeCLF
from model.gcn import GCNNodeCLF
from model.sage import SAGENodeCLF
from model.sgc import SGCNodeCLF
from trainer.federated_nc_trainer import FedNodeClfTrainer

import wandb

class Args():
    def __init__(self, dataset, data_cache_dir, model, partition_method, client_num_in_total,
                 batch_size, hidden_size, n_layers, dropout,
                 client_optimizer, learning_rate, weight_decay, epochs,
                 normalize_features, normalize_adjacency):
        self.dataset = dataset
        self.data_cache_dir = data_cache_dir
        self.model = model
        self.partition_method = partition_method
        self.client_num_in_total = client_num_in_total
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.client_optimizer = client_optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs

        self.normalize_features = normalize_features
        self.normalize_adjacency = normalize_adjacency
        



def load_data(args):
    num_cats, feat_dim = 0, 0
    if args.dataset not in ["CS", "Physics", "cora", "citeseer", "DBLP", "PubMed"]:
        raise Exception("no such dataset!")
    elif args.dataset in ["CS", "Physics"]:
        args.type_network = "coauthor"
    else:
        args.type_network = "citation"

    compact = args.model == "graphsage"

    unif = True if args.partition_method == "homo" else False

    if args.model == "gcn":
        args.normalize_features = True
        args.normalize_adjacency = True

    _, _, feat_dim, num_cats = get_data(args.data_cache_dir, args.dataset)

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
    ) = load_partition_data(
        args,
        args.data_cache_dir,
        args.client_num_in_total,
        uniform=unif,
        compact=compact,
        normalize_features=args.normalize_features,
        normalize_adj=args.normalize_adjacency,
    )

    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        num_cats,
    ]

    return dataset, num_cats, feat_dim


def create_model(args, feat_dim, num_cats, output_dim=None):
    print("create_model. model_name = %s, output_dim = %s" % (args.model, num_cats))
    if args.model == "gcn":
        model = GCNNodeCLF(
            nfeat=feat_dim, nhid=args.hidden_size, nclass=num_cats, nlayer=args.n_layers, dropout=args.dropout,
        )
    elif args.model == "sgc":
        model = SGCNodeCLF(in_dim=feat_dim, num_classes=num_cats, K=args.n_layers)
    elif args.model == "sage":
        model = SAGENodeCLF(
            nfeat=feat_dim, nhid=args.hidden_size, nclass=num_cats, nlayer=args.n_layers, dropout=args.dropout,
        )
    elif args.model == "gat":
        model = GATNodeCLF(in_channels=feat_dim, out_channels=num_cats, dropout=args.dropout,)
    else:
        # MORE MODELS
        raise Exception("such model does not exist !")
    trainer = FedNodeClfTrainer(model, args)
    print("Model and Trainer  - done")
    return model, trainer


if __name__ == '__main__':
    args = Args('CS', '/Users/dujiaxing/FedML/python/examples/federate/prebuilt_jobs/fedgraphnn/ego_networks_node_clf/ego-networks/', 'sage', partition_method = 'homo',
                client_num_in_total=10, batch_size=64, hidden_size=32, n_layers=5, dropout=0.3,
                client_optimizer = 'AdamW', learning_rate=0.001, weight_decay=0.001, epochs=100,
                normalize_features=False, normalize_adjacency=False)
    dataset, num_cats, feat_dim = load_data(args)
    model, trainer = create_model(args, feat_dim, num_cats)

    # Add test_set
    if not hasattr(trainer, 'test_data'):
        setattr(trainer, 'test_data', dataset[3])
    if not hasattr(trainer, 'Wandb'):
        setattr(trainer, 'Wandb', True)

    # Train the model
    # 🐝 1️⃣ Start a new run to track this script
    wandb.init(
        # Set the project where this run will be logged
        project="CS", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"sage_experiment_central", 
    )
    trainer.train(dataset[2], 'cpu', args)
    micro_F1, model = trainer.test(dataset[3], 'cpu')
    print(micro_F1)
    
    wandb.finish()

    # Local train
    train_local_dict = dataset[5]
    for i in range(len(train_local_dict)):
        model, trainer = create_model(args, feat_dim, num_cats)
        # Add test_set
        if not hasattr(trainer, 'test_data'):
            setattr(trainer, 'test_data', dataset[3])
        if not hasattr(trainer, 'Wandb'):
            setattr(trainer, 'Wandb', True)

        # Record
        wandb.init(
            # Set the project where this run will be logged
            project="CS", 
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"sage_experiment_local{i}", 
        )

        trainer.train(train_local_dict[i], 'cpu', args)
        micro_F1, model = trainer.test(dataset[3], 'cpu')
        print(micro_F1)

        wandb.finish()



    