import dgl
import torch
import numpy as np
import pandas as pd
import networkx as nx
from dgl.data import DGLDataset
import matplotlib.pyplot as plt
from collections import defaultdict

selected_features = ["province", "confirmed", "latitude", "longitude", "avg_temp", "max_wind_speed", "n_crisis",
                     'label', "end_lat", "end_lon"]
train_features = selected_features[1:6]
smoothing = 2


class CovidDataset(DGLDataset):
    def __init__(self, file_path):
        self.covid_data = pd.read_csv(file_path)
        # self.date = _date
        self.class_names = ['very safe', 'safe', 'moderately safe', 'moderate', 'moderately unsafe', 'dangerous']
        self.num_classes = len(self.class_names)
        super().__init__(name='covid_regions')

    def process(self):
        edges = defaultdict(int)
        node_features = torch.zeros((len(self.covid_data['province'].unique()), len(train_features)))
        node_labels = torch.zeros((len(self.covid_data['province'].unique()),))

        for day, date_ in enumerate(self.covid_data['date'].unique()[::-1], 1):
            multiplier = smoothing / day
            selected_data = self.covid_data.loc[self.covid_data["date"] == str(date_), selected_features]
            selected_data['id'] = selected_data['province'].astype('category').cat.codes.to_numpy()
            # selected_data['label'] = pd.cut(selected_data['confirmed'], [-1, 100, 1000, np.inf],
            #                                 labels=self.class_names)

            node_features = torch.from_numpy(
                selected_data[train_features].to_numpy() * multiplier) + node_features * (1 - multiplier)
            node_labels = torch.tensor(list(selected_data['label'].astype('int'))) * multiplier \
                          + node_labels * (1 - multiplier)

            for item in selected_data.itertuples():
                matches = selected_data[(round(selected_data['latitude'], 0) == round(item.end_lat, 0)) & (
                        round(selected_data['longitude'], 0) == round(item.end_lon, 0)) & (
                                                selected_data['id'] != item.id)]
                if not matches.empty:
                    for row in matches.itertuples():
                        edges[(item.id, row.id)] = row.n_crisis * multiplier + edges[(item.id, row.id)] * \
                                                   (1 - multiplier)

        edges_src, edges_dst = [torch.tensor(x) for x in zip(*edges)]
        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=node_features.shape[0])
        self.graph = dgl.add_self_loop(self.graph)
        self.graph.ndata['feat'] = node_features.double()
        self.graph.ndata['label'] = torch.round((node_labels * (len(self.class_names) - 1) / node_labels.max())).long()
        # self.graph.edata['weight'] = torch.tensor(edges.values())

        n_nodes = node_features.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

# dataset = CovidDataset('abc1_small.csv')
# graph = dataset[0]
#
# # Since the actual graph is undirected, we convert it for visualization
# # purpose.
# nx_G = graph.to_networkx().to_undirected()
# # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
# pos = nx.kamada_kawai_layout(nx_G)
# nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
# plt.show()
