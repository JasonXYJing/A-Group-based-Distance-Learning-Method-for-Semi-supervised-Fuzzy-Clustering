import random
import math
import numpy as np
from numpy import *
import operator
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from metric_learn.constraints import Constraints
import torch
from torch import Tensor
import networkx as nx
import time
import fcm
from sklearn import metrics
import collections


class NLGDL():
    def __init__(self, data_set):
        self.data_set = data_set
        self.num_hidden_neurons = 200
        self.graph_neighbors = 5
        self.num_neighbors = 3
        self.learning_rate = 0.2
        self.max_iter_num = 300

    def find_K_Neighbors(self, data_point):
        neigh = NearestNeighbors()
        neigh.fit(self.data_set)
        K_Neighbors = neigh.kneighbors(data_point, self.num_neighbors+1, return_distance=False)
        return K_Neighbors

    def find_group(self, data_point, index_k_neighbors, cannot_link):
        my_neighbors = []
        for i in index_k_neighbors[0]:
            if (data_point, self.data_set[i]) not in cannot_link:
                my_neighbors.append(self.data_set[i])
            else:
                continue
        total_distance = 0
        for nei in my_neighbors:
            total_distance += pow(np.linalg.norm(list(map(operator.sub, data_point[0], nei))), 2)
        group_radius = total_distance / (len(index_k_neighbors[0]) - 1)
        group = []
        for neii in my_neighbors:
            if pow(np.linalg.norm(list(map(operator.sub, data_point[0], neii))), 2) <= group_radius:
                group.append(neii)
        return group

    def calculate_group_hd(self, link_pair, cannot_link):
        # data_point must be inputted as [[·]]
        group_index_a = self.find_K_Neighbors([link_pair[0]])
        group_index_b = self.find_K_Neighbors([link_pair[1]])
        a_group = self.find_group([link_pair[0]], group_index_a, cannot_link)
        b_group = self.find_group([link_pair[1]], group_index_b, cannot_link)
        return a_group, b_group

    def generate_graph(self, must_index, cannot_index):
        neigh = NearestNeighbors(n_neighbors=self.graph_neighbors)
        neigh.fit(self.data_set)
        A = neigh.kneighbors_graph(self.data_set, mode='connectivity')
        a_matrix = A.toarray()
        alist = []
        for item in a_matrix:
            alist.append(np.where(item == 1)[0] + 1)
        G = nx.make_small_graph(["adjacencylist", "C_4", len(self.data_set), alist])
        for meach in must_index:
            G.add_edge(meach[0], meach[1], weight=0.00001)
        for ceach in cannot_index:
            G.add_edge(ceach[0], ceach[1], weight=0.00001)
        b = nx.betweenness_centrality(G)
        return b

    def calculate_new_HD(self, set_a, set_b, model):
        a = []
        a_index = []
        for item_a in set_a:
            a2b = []
            x = torch.tensor([item_a]).float()
            for item_b in set_b:
                y = torch.tensor([item_b]).float()
                output1, output2 = model(x), model(y)
                dis = (output1 - output2).norm().pow(2)
                a2b.append(dis)
            a.append(min(a2b))
            a_index.append([item_a, set_b[a2b.index(min(a2b))]])
        a2b_max = max(a)
        a2b_index = a_index[a.index(a2b_max)]
        b = []
        b_index = []
        for item_b in set_b:
            b2a = []
            x = torch.tensor([item_b]).float()
            for item_a in set_a:
                y = torch.tensor([item_a]).float()
                output1, output2 = model(x), model(y)
                dis = (output1 - output2).norm().pow(2)
                b2a.append(dis)
            b.append(min(b2a))
            b_index.append([item_b, set_a[b2a.index(min(b2a))]])
        b2a_max = max(b)
        b2a_index = b_index[b.index(b2a_max)]
        if a2b_max > b2a_max:
            return a2b_max, a2b_index
        else:
            return b2a_max, b2a_index

    def train(self, must_link_set, cannot_link_set, must_index, cannot_index, data_dim, betweenness):
        last_loss = 0
        must_group = []
        cannot_group = []
        D_in = D_out = data_dim
        for must_pair in must_link_set:
            group_ma, group_mb = self.calculate_group_hd(must_pair, cannot_link_set)
            must_group.append([group_ma, group_mb])
        for cannot_pair in cannot_link_set:
            group_ca, group_cb = self.calculate_group_hd(cannot_pair, cannot_link_set)
            cannot_group.append([group_ca, group_cb])

        model = torch.nn.Sequential(
            torch.nn.Linear(D_in, self.num_hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(self.num_hidden_neurons, D_out),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        iter_num = 0
        while iter_num <= self.max_iter_num:
            m_total = 0
            c_total = 0
            m_index = 0
            c_index = 0
            for must_pair in must_group:
                new_hd, m_data_pair = self.calculate_new_HD(must_pair[0], must_pair[1], model)
                mweights = betweenness[must_index[m_index][0]] + betweenness[must_index[m_index][1]]
                mx, my = torch.tensor([m_data_pair[0]]).float(), torch.tensor([m_data_pair[1]]).float()
                moutput1, moutput2 = model(mx), model(my)
                dis = (moutput1 - moutput2).norm().pow(2) * mweights
                m_total += dis
                m_index += 1
            for cannot_pair in cannot_group:
                new_chd, c_data_pair = self.calculate_new_HD(cannot_pair[0], cannot_pair[1], model)
                cweights = betweenness[cannot_index[c_index][0]] + betweenness[cannot_index[c_index][1]]
                cx, cy = torch.tensor([c_data_pair[0]]).float(), torch.tensor([c_data_pair[1]]).float()
                coutput1, coutput2 = model(cx), model(cy)
                diss = (coutput1 - coutput2).norm().pow(2)
                disss = max(6 - diss, torch.tensor(0).float()) * cweights
                c_total += disss
                c_index += 1
            m_size = 1 / len(must_link_set)
            c_size = 1 / len(cannot_link_set)
            current_loss = m_size * m_total + c_size * c_total
            if abs(current_loss - last_loss) <= 0.00001:
                break
            else:
                optimizer.zero_grad()
                current_loss.backward()
                optimizer.step()
            last_loss = current_loss
            iter_num += 1
        return model


