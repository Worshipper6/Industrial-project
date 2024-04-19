# Industrial-project
## Recommendation System
- [Epinions](FedML/python/examples/federate/prebuilt_jobs/fedgraphnn/recsys_subgraph_link_pred)
- [MovieLens](FedML/python/examples/federate/prebuilt_jobs/fedgraphnn/movie)
### Federated learning
```
cd recsys
sh run_fed_subgraph_link_pred.sh 1 gcn fedavg
```
```
cd movie
python movie_prediction.py --cf config_fedavg/gcn_config.yaml
```
### Centralized and local training
```
python movie/local.py
```
## Anti-Money Laundering (AML) Detection
- [AML](FedML/python/examples/federate/prebuilt_jobs/fedgraphnn/AML)
### Federated learning
```
cd AML
python AML_FL.py --cf fedml_config.yaml
```
### Centralized and local training
```
python main.py
```
