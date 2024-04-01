import pandas as pd
import numpy as np
import torch
import logging
import itertools
from data_util import GraphData, HeteroData, z_norm, create_hetero_obj

from torch_geometric.loader import LinkNeighborLoader

def get_data(path, args):
    '''Loads the AML transaction data.
    
    1. The data is loaded from the csv and the necessary features are chosen.
    2. The data is split into training, validation and test data.
    3. PyG Data objects are created with the respective data splits.
    '''

    #replace this with your path to the respective AML data objects
    df_edges = pd.read_csv(path)

    logging.info(f'Available Edge Features: {df_edges.columns.tolist()}')

    df_edges['Timestamp'] = df_edges['Timestamp'] - df_edges['Timestamp'].min()

    max_n_id = df_edges.loc[:, ['from_id', 'to_id']].to_numpy().max() + 1
    df_nodes = pd.DataFrame({'NodeID': np.arange(max_n_id), 'Feature': np.ones(max_n_id)})
    timestamps = torch.Tensor(df_edges['Timestamp'].to_numpy())
    y = torch.LongTensor(df_edges['Is Laundering'].to_numpy())

    logging.info(f"Illicit ratio = {sum(y)} / {len(y)} = {sum(y) / len(y) * 100:.2f}%")
    logging.info(f"Number of nodes (holdings doing transcations) = {df_nodes.shape[0]}")
    logging.info(f"Number of transactions = {df_edges.shape[0]}")

    edge_features = ['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format']
    node_features = ['Feature']

    logging.info(f'Edge features being used: {edge_features}')
    logging.info(f'Node features being used: {node_features} ("Feature" is a placeholder feature of all 1s)')

    x = torch.tensor(df_nodes.loc[:, node_features].to_numpy()).float()
    edge_index = torch.LongTensor(df_edges.loc[:, ['from_id', 'to_id']].to_numpy().T)
    edge_attr = torch.tensor(df_edges.loc[:, edge_features].to_numpy()).float()

    n_days = int(timestamps.max() / (3600 * 24) + 1)
    n_samples = y.shape[0]
    logging.info(f'number of days and transactions in the data: {n_days} days, {n_samples} transactions')

    #data splitting
    daily_irs, weighted_daily_irs, daily_inds, daily_trans = [], [], [], [] #irs = illicit ratios, inds = indices, trans = transactions
    for day in range(n_days):
        l = day * 24 * 3600
        r = (day + 1) * 24 * 3600
        day_inds = torch.where((timestamps >= l) & (timestamps < r))[0]
        daily_irs.append(y[day_inds].float().mean())
        weighted_daily_irs.append(y[day_inds].float().mean() * day_inds.shape[0] / n_samples)
        daily_inds.append(day_inds)
        daily_trans.append(day_inds.shape[0])
    
    split_per = [0.6, 0.2, 0.2]
    daily_totals = np.array(daily_trans)
    d_ts = daily_totals
    I = list(range(len(d_ts)))
    split_scores = dict()
    for i,j in itertools.combinations(I, 2):
        if j >= i:
            split_totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
            split_totals_sum = np.sum(split_totals)
            split_props = [v/split_totals_sum for v in split_totals]
            split_error = [abs(v-t)/t for v,t in zip(split_props, split_per)]
            score = max(split_error) #- (split_totals_sum/total) + 1
            split_scores[(i,j)] = score
        else:
            continue

    i,j = min(split_scores, key=split_scores.get)
    #split contains a list for each split (train, validation and test) and each list contains the days that are part of the respective split
    split = [list(range(i)), list(range(i, j)), list(range(j, len(daily_totals)))]
    logging.info(f'Calculate split: {split}')

    #Now, we seperate the transactions based on their indices in the timestamp array
    split_inds = {k: [] for k in range(3)}
    for i in range(3):
        for day in split[i]:
            split_inds[i].append(daily_inds[day]) #split_inds contains a list for each split (tr,val,te) which contains the indices of each day seperately

    tr_inds = torch.cat(split_inds[0])
    val_inds = torch.cat(split_inds[1])
    te_inds = torch.cat(split_inds[2])

    logging.info(f"Total train samples: {tr_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
            f"{y[tr_inds].float().mean() * 100 :.2f}% || Train days: {split[0][:5]}")
    logging.info(f"Total val samples: {val_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
        f"{y[val_inds].float().mean() * 100:.2f}% || Val days: {split[1][:5]}")
    logging.info(f"Total test samples: {te_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
        f"{y[te_inds].float().mean() * 100:.2f}% || Test days: {split[2][:5]}")
    
    #Creating the final data objects
    tr_x, val_x, te_x = x, x, x
    e_tr = tr_inds.numpy()
    e_val = np.concatenate([tr_inds, val_inds])

    tr_edge_index,  tr_edge_attr,  tr_y,  tr_edge_times  = edge_index[:,e_tr],  edge_attr[e_tr],  y[e_tr],  timestamps[e_tr]
    val_edge_index, val_edge_attr, val_y, val_edge_times = edge_index[:,e_val], edge_attr[e_val], y[e_val], timestamps[e_val]
    te_edge_index,  te_edge_attr,  te_y,  te_edge_times  = edge_index,          edge_attr,        y,        timestamps

    tr_data = GraphData (x=tr_x,  y=tr_y,  edge_index=tr_edge_index,  edge_attr=tr_edge_attr,  timestamps=tr_edge_times, tr_inds=tr_inds)
    val_data = GraphData(x=val_x, y=val_y, edge_index=val_edge_index, edge_attr=val_edge_attr, timestamps=val_edge_times, val_inds=val_inds)
    te_data = GraphData (x=te_x,  y=te_y,  edge_index=te_edge_index,  edge_attr=te_edge_attr,  timestamps=te_edge_times, te_inds=te_inds)

    #Adding ports and time-deltas if applicable
    if args.ports:
        logging.info(f"Start: adding ports")
        tr_data.add_ports()
        val_data.add_ports()
        te_data.add_ports()
        logging.info(f"Done: adding ports")
    if args.tds:
        logging.info(f"Start: adding time-deltas")
        tr_data.add_time_deltas()
        val_data.add_time_deltas()
        te_data.add_time_deltas()
        logging.info(f"Done: adding time-deltas")
    
    #Normalize data
    tr_data.x = val_data.x = te_data.x = z_norm(tr_data.x)
    if not args.model == 'rgcn':
        tr_data.edge_attr, val_data.edge_attr, te_data.edge_attr = z_norm(tr_data.edge_attr), z_norm(val_data.edge_attr), z_norm(te_data.edge_attr)
    else:
        tr_data.edge_attr[:, :-1], val_data.edge_attr[:, :-1], te_data.edge_attr[:, :-1] = z_norm(tr_data.edge_attr[:, :-1]), z_norm(val_data.edge_attr[:, :-1]), z_norm(te_data.edge_attr[:, :-1])

    #Create heterogenous if reverese MP is enabled
    #TODO: if I observe wierd behaviour, maybe add .detach.clone() to all torch tensors, but I don't think they're attached to any computation graph just yet
    if args.reverse_mp:
        tr_data = create_hetero_obj(tr_data.x,  tr_data.y,  tr_data.edge_index,  tr_data.edge_attr, tr_data.timestamps, args)
        val_data = create_hetero_obj(val_data.x,  val_data.y,  val_data.edge_index,  val_data.edge_attr, val_data.timestamps, args)
        te_data = create_hetero_obj(te_data.x,  te_data.y,  te_data.edge_index,  te_data.edge_attr, te_data.timestamps, args)
    
    logging.info(f'train data object: {tr_data}')
    logging.info(f'validation data object: {val_data}')
    logging.info(f'test data object: {te_data}')

    return tr_data, val_data, te_data, tr_inds, val_inds, te_inds


def get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args):
    if isinstance(tr_data, HeteroData):
        tr_edge_label_index = tr_data['node', 'to', 'node'].edge_index
        tr_edge_label = tr_data['node', 'to', 'node'].y

        tr_loader = LinkNeighborLoader(tr_data, num_neighbors=args.num_neighs,
                                       edge_label_index=(('node', 'to', 'node'), tr_edge_label_index),
                                       edge_label=tr_edge_label, batch_size=args.batch_size, shuffle=True)

        val_edge_label_index = val_data['node', 'to', 'node'].edge_index[:, val_inds]
        val_edge_label = val_data['node', 'to', 'node'].y[val_inds]

        val_loader = LinkNeighborLoader(val_data, num_neighbors=args.num_neighs,
                                        edge_label_index=(('node', 'to', 'node'), val_edge_label_index),
                                        edge_label=val_edge_label, batch_size=args.batch_size, shuffle=False)

        te_edge_label_index = te_data['node', 'to', 'node'].edge_index[:, te_inds]
        te_edge_label = te_data['node', 'to', 'node'].y[te_inds]

        te_loader = LinkNeighborLoader(te_data, num_neighbors=args.num_neighs,
                                       edge_label_index=(('node', 'to', 'node'), te_edge_label_index),
                                       edge_label=te_edge_label, batch_size=args.batch_size, shuffle=False)
    else:
        tr_loader = LinkNeighborLoader(tr_data, num_neighbors=args.num_neighs, batch_size=args.batch_size, shuffle=True)
        val_loader = LinkNeighborLoader(val_data, num_neighbors=args.num_neighs,
                                        edge_label_index=val_data.edge_index[:, val_inds],
                                        edge_label=val_data.y[val_inds], batch_size=args.batch_size, shuffle=False)
        te_loader = LinkNeighborLoader(te_data, num_neighbors=args.num_neighs,
                                       edge_label_index=te_data.edge_index[:, te_inds],
                                       edge_label=te_data.y[te_inds], batch_size=args.batch_size, shuffle=False)

    return tr_loader, val_loader, te_loader

def load_partition_data(args, global_path, local_path, client_number):
    # Get global data
    global_tr_data, global_val_data, global_te_data, global_tr_inds, global_val_inds, global_te_inds = get_data(global_path, args)
    # Convert to dataloader
    global_tr_loader, global_val_loader, global_te_loader = \
        get_loaders(global_tr_data, global_val_data, global_te_data, global_tr_inds, global_val_inds, global_te_inds, args)
    train_data_num = global_tr_data.x.shape[0]
    val_data_num = global_val_data.x.shape[0]
    test_data_num = global_te_data.x.shape[0]

    # Clients
    train_data_local_dict = {}
    val_data_local_dict = {}
    test_data_local_dict = {}
    data_local_num_dict = {}
    for client in range(client_number):
        # Get global data
        path = local_path + str(client)
        client_tr_data, client_val_data, client_te_data, client_tr_inds, client_val_inds, client_te_inds = get_data(
            path, args)
        # Convert to dataloader
        train_data_local_dict[client], val_data_local_dict[client], test_data_local_dict[client] = \
            get_loaders(client_tr_data, client_val_data, client_te_data, client_tr_inds, client_val_inds, client_te_inds, args)
        data_local_num_dict[client] = client_tr_data.x.shape[0] + client_val_data.x.shape[0] + client_te_data.x.shape[0]

    return (
        train_data_num,
        val_data_num,
        test_data_num,
        global_tr_loader,
        global_val_loader,
        global_te_loader,
        data_local_num_dict,
        train_data_local_dict,
        val_data_local_dict,
        test_data_local_dict
    )