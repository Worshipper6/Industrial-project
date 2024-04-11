import copy
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


class FedClsTrainer_Heter(ClientTrainer):
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

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
                batch_edge_inds = inds[batch['node', 'to', 'node'].input_id.detach().cpu()]
                batch_edge_ids = train_data[0].data['node', 'to', 'node'].edge_attr.detach().cpu()[batch_edge_inds, 0]
                mask = torch.isin(batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu(), batch_edge_ids)
                # remove the unique edge id from the edge features, as it's no longer needed
                batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
                batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]


                batch.to(device)
                out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
                out = out[('node', 'to', 'node')]
                pred = out[mask]
                ground_truth = batch['node', 'to', 'node'].y[mask]
                preds.append(pred.argmax(dim=-1))
                ground_truths.append(batch['node', 'to', 'node'].y[mask])
                loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([args.w_ce1, args.w_ce2]).to(device))
                loss = loss_fn(pred, ground_truth)

                loss.backward()
                optimizer.step()

                total_loss += float(loss) * pred.numel()
                total_examples += pred.numel()
            if Wandb is not None:
                wandb.log({"loss/train": total_loss})

            if (test_data is not None) & (epoch % 5 == 0):
                test_score, roc, recall, precision = self.test(
                    test_data, device, args)
                print(
                    "Epoch = {}, Iter = {}/{}: Test score = {}, ROC = {}, Recall = {}, Precision = {}".format(
                        epoch, idx_batch + 1, len(train_data[0]), test_score, roc, recall, precision
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
                    wandb.log({"roc/test": roc})
                    wandb.log({"recall/test": recall})
                    wandb.log({"precision/test": precision})


        return max_test_score, best_model_params

    def test(self, test_data, device, args):
        logits = []
        preds = []
        ground_truths = []
        #test_ = copy.deepcopy(test_data[0])
        for batch in test_data[0]:
            # select the seed edges from which the batch was created
            inds = test_data[1].detach().cpu()
            batch_edge_inds = inds[batch['node', 'to', 'node'].input_id.detach().cpu()]
            batch_edge_ids = test_data[0].data['node', 'to', 'node'].edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu(), batch_edge_ids)

            # remove the unique edge id from the edge features, as it's no longer needed
            batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
            batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]


            with torch.no_grad():
                #batch.to(device)
                # batch.x_dict.to(device)
                # batch.edge_index_dict.to(device)
                # batch.edge_attr_dict.to(device)
                batch['node'].x.to(device)
                batch['node', 'to', 'node'].edge_index.to(device)
                batch['node', 'rev_to', 'node'].edge_index.to(device)
                batch['node', 'to', 'node'].edge_attr.to(device)
                batch['node', 'rev_to', 'node'].edge_attr.to(device)

                out = self.model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
                out = out[('node', 'to', 'node')]
                out = out[mask]
                pred = out.argmax(dim=-1)
                logits.append(out[:, 1])
                preds.append(pred)
                ground_truths.append(batch['node', 'to', 'node'].y[mask])

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
