import copy
import logging
import os
import pickle
import random

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx
from torch_geometric.utils import to_networkx, degree



def _read_mapping(path, data, filename):
    mapping = {}
    with open(os.path.join(path, data, filename)) as f:
        for line in f:
            s = line.strip().split()
            mapping[int(s[0])] = int(s[1])

    return mapping


def _build_nxGraph(path, data, filename, mapping_user, mapping_item):
    G = nx.Graph()
    with open(os.path.join(path, data, filename)) as f:
        f.readline()  # skip the first line
        for line in f:
            s = line.strip().split(',')
            s = [int(float(i)) for i in s]
            G.add_edge(mapping_user[s[0]], mapping_item[s[1]], edge_label=s[2])
    dic = {}
    for node in G.nodes:
        dic[node] = node
    nx.set_node_attributes(G, dic, "index_orig")
    return G


def _subgraphing(g, mapping_user):
    nodelist = []
    for k,v in mapping_user.items():
        user = [v]
        user_neighbor = []
        user += list(g.neighbors(v))   #append items
        for i in user[1:]:  # item
            user_neighbor += list(g.neighbors(i)) # append neighbor users
        partition = user + list(set(user_neighbor))
        nodelist.append(partition)
    graphs = []
    for nodes in nodelist:
        if len(nodes) < 2:
            continue
        graph = nx.subgraph(g, nodes)
        graphs.append(from_networkx(graph))
        
    return graphs

def _convert_to_nodeDegreeFeatures(graphs):
    graph_infos = []
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree
        graph_infos.append(
            (graph, g.degree, graph.num_nodes)
        )  # (graph, node_degrees, num_nodes)

    new_graphs = []
    for i, tpl in enumerate(graph_infos):
        # idx, x = tpl[0].edge_index[0], tpl[0].x
        idx, x = torch.tensor(range(tpl[0].num_nodes)), tpl[0].x
        deg = degree(idx, tpl[2], dtype=torch.long)
        deg = F.one_hot(deg, num_classes=maxdegree + 1).to(torch.float)

        new_graph = tpl[0].clone()
        new_graph.__setitem__("x", deg)
        new_graphs.append(new_graph)

    return new_graphs

def split_graph(graph, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert train_ratio + val_ratio + test_ratio == 1
    graph.edge_label = graph.edge_label.float()
    edge_size = graph.edge_label.size()[0]
    train_num = int(edge_size * train_ratio)
    val_num = max(1, int(edge_size * val_ratio))
    test_num = max(1, int(edge_size * test_ratio))
    train_num = edge_size - val_num - test_num

    [train_split, val_split, test_split] = torch.utils.data.random_split(
        range(edge_size), [train_num, val_num, test_num]
    )
    train_split = torch.tensor(train_split)
    val_split = torch.tensor(val_split)
    test_split = torch.tensor(test_split)
    graph.edge_train = graph.edge_index[:, train_split]
    graph.label_train = graph.edge_label[train_split]
    graph.edge_val = graph.edge_index[:, val_split]
    graph.label_val = graph.edge_label[val_split]
    graph.edge_test = graph.edge_index[:, test_split]
    graph.label_test = graph.edge_label[test_split]

    return graph

def get_data_category(args, path, data):
    logging.info("read mapping")
    mapping_user = _read_mapping(path, data, "user.dict")
    mapping_item = _read_mapping(path, data, "item.dict")
    logging.info("build networkx graph")
    graph = _build_nxGraph(path, data, "ratings.csv", mapping_user, mapping_item)
    logging.info("subgraphing")
    graphs = _subgraphing(graph, mapping_user)

    logging.info("converting to node degree")
    graphs = _convert_to_nodeDegreeFeatures(graphs)
    graphs_split = []
    logging.info("spliting into trian val and test")
    for g in graphs:
        graphs_split.append(split_graph(g))
    return graphs_split

def partition_data_by_category(args, path, compact=True):
    graphs_split = get_data_category(args, path, args.dataset)

    client_number = len(graphs_split)

    partition_dicts = [None] * client_number

    for client in range(client_number):
        partition_dict = {"graph": graphs_split[client]}
        partition_dicts[client] = partition_dict

    global_data_dict = {
        "graphs": graphs_split,
    }

    return global_data_dict, partition_dicts


# Single process sequential
def load_partition_data(args, path, client_number):
    global_data_dict, partition_dicts = partition_data_by_category(args, path)
    feature_dim = global_data_dict["graphs"][0].x.shape[1]

    data_local_num_dict = {}
    train_data_local_dict = {}

    # This is a PyG Dataloader
    train_data_global = DataLoader(
        global_data_dict["graphs"], batch_size=1, shuffle=True, pin_memory=True
    )

    train_data_num = len(global_data_dict["graphs"])

    for client in range(client_number):
        train_dataset_client = partition_dicts[client]["graph"]

        data_local_num_dict[client] = 1
        train_data_local_dict[client] = DataLoader(
            [train_dataset_client], batch_size=1, shuffle=True, pin_memory=True
        )
        logging.info(
            "Client idx = {}, local sample number = {}".format(
                client, len(train_dataset_client)
            )
        )

    val_data_num = test_data_num = train_data_num
    val_data_local_dict = test_data_local_dict = train_data_local_dict
    val_data_global = test_data_global = train_data_global
    return (
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
    )


# class Args:
#     def __init__(self, dataset):
#         self.dataset = dataset

# args = Args(dataset='ml-small')
# global_data_dict, partition_dicts = partition_data_by_category(args, './data')

