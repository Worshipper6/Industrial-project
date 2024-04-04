import logging

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    recall_score,
    precision_score
)

from fedml.core.alg_frame.client_trainer import ClientTrainer


class FedClsTrainer(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        test_data = None
        try:
            test_data = self.test_data
        except:
            pass

        Wandb = None
        try:
            Wandb = self.Wandb
        except:
            pass

       
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )

        max_test_score = 10000
        best_model_params = {}
        for epoch in range(args.epochs):
            total_loss = total_examples = 0
            preds = []
            ground_truths = []
            for idx_batch, batch in enumerate(train_data[0]):
                optimizer.zero_grad()
                # select the seed edges from which the batch was created
                inds = train_data[1].detach().cpu()
                batch_edge_inds = inds[batch.input_id.detach().cpu()]
                batch_edge_ids = train_data[0].data.edge_attr.detach().cpu()[batch_edge_inds, 0]
                mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

                # remove the unique edge id from the edge features, as it's no longer needed
                batch.edge_attr = batch.edge_attr[:, 1:]

                batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr)
                pred = out[mask]
                ground_truth = batch.y[mask]
                preds.append(pred.argmax(dim=-1))
                ground_truths.append(ground_truth)
                loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([args.w_ce1, args.w_ce2]).to(device))
                loss = loss_fn(pred, ground_truth)

                loss.backward()
                optimizer.step()

                total_loss += float(loss) * pred.numel()
                total_examples += pred.numel()

            if test_data is not None:
                test_score, _, _, _= self.test(
                    test_data, device)
                print(
                    "Epoch = {}, Iter = {}/{}: Test score = {}".format(
                        epoch, idx_batch + 1, len(train_data[0]), test_score
                    )
                )
                if test_score < max_test_score:
                    max_test_score = test_score
                    best_model_params = {
                        k: v.cpu() for k, v in model.state_dict().items()
                    }
                print("Current best = {}".format(max_test_score))

                # Record
                if Wandb is not None:
                    wandb.log({"f1/test":test_score})

        return max_test_score, best_model_params

    def test(self, test_data, device, args):
        logits = []
        preds = []
        ground_truths = []
        for batch in test_data[0]:
            # select the seed edges from which the batch was created
            inds = test_data[1].detach().cpu()
            batch_edge_inds = inds[batch.input_id.detach().cpu()]
            batch_edge_ids = test_data[0].data.edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)
            # remove the unique edge id from the edge features, as it's no longer needed
            batch.edge_attr = batch.edge_attr[:, 1:]

            with torch.no_grad():
                #batch.to(device)
                batch.x.to(device)
                batch.edge_index.to(device)
                batch.edge_attr.to(device)
                out = self.model(batch.x, batch.edge_index, batch.edge_attr)
                out = out[mask]
                pred = out.argmax(dim=-1)
                logits.append(out[:, 1])
                preds.append(pred)
                ground_truths.append(batch.y[mask])
        pred = torch.cat(preds, dim=0).cpu().numpy()
        logit = torch.cat(logits, dim=0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
        f1 = f1_score(ground_truth, pred)
        if len(np.unique(ground_truth)) == 2:
            roc = roc_auc_score(ground_truth, logit)
        else:
            # ROC is not defined
            roc = None
        recall = recall_score(ground_truth, pred)
        precision = precision_score(ground_truth, pred)

        return f1, roc, recall, precision



    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        logging.info("----------test_on_the_server--------")

        f1_list, roc_list, recall_list, precision_list = [], [], [], []
        for client_idx in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_idx]
            f1, roc, recall, precision = self.test(test_data, device)

            # for idx in range(len(model_list)):
            #     self._compare_models(model, model_list[idx])
            #model_list.append(model)
            f1_list.append(f1)
            roc_list.append(roc)
            recall_list.append(recall)
            precision_list.append(precision)

            logging.info(
                "Client {}, Test {} = {}, ROC = {}, Recall = {}, Precision = {}".format(
                    client_idx, args.metric, f1, roc, recall, precision
                )
            )
            if args.enable_wandb:
                wandb.log(
                    {
                        "Client {} Test/{}".format(client_idx, args.metric): f1,
                        "ROC, Recall, Precision": [roc, recall, precision],
                    }
                )

        f1_mean = np.mean(np.array(f1_list))
        roc_mean = np.mean(np.array(roc_list))
        recall_mean = np.mean(np.array(recall_list))
        precision_mean = np.mean(np.array(precision_list))

        logging.info(
            "Test {} = {}, ROC = {}, Recall = {}, Precision = {}".format(
                args.metric, f1_mean, roc_mean, recall_mean, precision_mean
            )
        )
        if args.enable_wandb:
            wandb.log(
                {
                    "Client {} Test/{}".format(client_idx, args.metric): f1_mean,
                    "ROC, Recall, Precision = ": [roc_mean, recall_mean, precision_mean],
                }
            )

        return True
