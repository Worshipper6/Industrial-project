

from data.data_loader import load_partition_data, get_data
from model.gat_readout import GatMoleculeNet
from model.gcn_readout import GcnMoleculeNet
from model.sage_readout import SageMoleculeNet

from trainer.gat_readout_trainer import GatMoleculeNetTrainer
from trainer.gcn_readout_trainer import GcnMoleculeNetTrainer
from trainer.sage_readout_trainer import SageMoleculeNetTrainer

# from fedml.centralized.centralized_trainer import CentralizedTrainer
import wandb

class Args():
    def __init__(self, dataset, data_cache_dir, model, metric, partition_method, client_num_in_total, normalize_features, normalize_adjacency,
                 hidden_size, node_embedding_dim, dropout, readout_hidden_dim, graph_embedding_dim,
                 alpha, num_heads, sparse_adjacency,
                 client_optimizer, learning_rate, epochs, frequency_of_the_test, data_parallel):
        self.dataset = dataset
        self.data_cache_dir = data_cache_dir
        self.model = model
        self.metric = metric
        self.partition_method = partition_method
        self.normalize_features= normalize_features
        self.normalize_adjacency = normalize_adjacency
        self.client_num_in_total = client_num_in_total

        self.hidden_size = hidden_size
        self.node_embedding_dim = node_embedding_dim
        self.dropout = dropout
        self.readout_hidden_dim = readout_hidden_dim
        self.graph_embedding_dim = graph_embedding_dim
        self.alpha = alpha
        self.num_heads = num_heads
        self.sparse_adjacency = sparse_adjacency

        self.client_optimizer = client_optimizer
        self.learning_rate= learning_rate
        self.epochs = epochs
        self.frequency_of_the_test = frequency_of_the_test
        self.data_parallel = data_parallel




# load data
def load_data(args, dataset_name):
    if (
        (args.dataset != "sider")
        and (args.dataset != "clintox")
        and (args.dataset != "bbbp")
        and (args.dataset != "bace")
        and (args.dataset != "pcba")
        and (args.dataset != "tox21")
        and (args.dataset != "toxcast")
        and (args.dataset != "muv")
        and (args.dataset != "hiv")
    ):
        raise Exception("no such dataset!")

    compact = args.model == "graphsage"

    print("load_data. dataset_name = %s" % dataset_name)
    _, feature_matrices, labels = get_data(args.data_cache_dir + args.dataset)
    unif = True if args.partition_method == "homo" else False
    if args.model == "gcn":
        args.normalize_features = True
        args.normalize_adjacency = True

    if args.dataset == "pcba":
        args.metric = "prc-auc"
    else:
        args.metric = "roc-auc"

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
        args.data_cache_dir + args.dataset,
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
        labels[0].shape[0],
    ]

    return dataset, feature_matrices[0].shape[1], labels[0].shape[0]



def create_model(args, model_name, feat_dim, num_cats, output_dim):
    print("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    if model_name == "graphsage":
        model = SageMoleculeNet(
            feat_dim,
            args.hidden_size,
            args.node_embedding_dim,
            args.dropout,
            args.readout_hidden_dim,
            args.graph_embedding_dim,
            num_cats,
        )
        trainer = SageMoleculeNetTrainer(model, args)
        # aggregator = SageMoleculeNetAggregator(model, args)
    elif model_name == "gat":
        model = GatMoleculeNet(
            feat_dim,
            args.hidden_size,
            args.node_embedding_dim,
            args.dropout,
            args.alpha,
            args.num_heads,
            args.readout_hidden_dim,
            args.graph_embedding_dim,
            num_cats,
        )
        trainer = GatMoleculeNetTrainer(model, args)
        # aggregator = GatMoleculeNetAggregator(model, args)

    elif model_name == "gcn":
        model = GcnMoleculeNet(
            feat_dim,
            args.hidden_size,
            args.node_embedding_dim,
            args.dropout,
            args.readout_hidden_dim,
            args.graph_embedding_dim,
            num_cats,
            sparse_adj=args.sparse_adjacency,
        )
        trainer = GcnMoleculeNetTrainer(model, args)
        # aggregator = GcnMoleculeNetAggregator(model, args)
    else:
        raise Exception("such model does not exist !")
    return model, trainer


args = Args('clintox', '/Users/dujiaxing/fedgraphnn_data/', 'gat', metric='roc-auc', partition_method='homo', 
            client_num_in_total=4, normalize_features=True, normalize_adjacency=True, hidden_size=32, node_embedding_dim=32, dropout=0.5, readout_hidden_dim=64, graph_embedding_dim=64, alpha=0.2, num_heads=2, sparse_adjacency=False,
            client_optimizer='adam', learning_rate=0.0015, epochs=100, frequency_of_the_test=5000, data_parallel=2)


dataset, feat_dim, num_cats = load_data(args, args.dataset)
model, trainer = create_model(args, args.model, feat_dim, num_cats, output_dim=None)


# Add test_set
if not hasattr(trainer, 'test_data'):
    setattr(trainer, 'test_data', dataset[3])
if not hasattr(trainer, 'Wandb'):
    setattr(trainer, 'Wandb', True)

# 🐝 1️⃣ Start a new run to track this script
wandb.init(
    # Set the project where this run will be logged
    project="Clintox", 
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=f"gat_experiment_central", 
)

trainer.train(dataset[2], 'cpu', args)
score, model = trainer.test(dataset[3], 'cpu', args)
wandb.finish()

train_local_dict = dataset[5]



# Local train
for i in range(len(train_local_dict)):
    model, trainer = create_model(args, args.model, feat_dim, num_cats, output_dim=None)
    # Add test_set
    if not hasattr(trainer, 'test_data'):
        setattr(trainer, 'test_data', dataset[3])
    if not hasattr(trainer, 'Wandb'):
        setattr(trainer, 'Wandb', True)

    # 🐝 1️⃣ Start a new run to track this script
    wandb.init(
        # Set the project where this run will be logged
        project="Clintox", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"gat_experiment_local{i}", 
    )

    trainer.train(train_local_dict[i], 'cpu', args)
    score, model = trainer.test(dataset[3], 'cpu', args)
    
    wandb.finish()