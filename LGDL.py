from sklearn.neighbors import NearestNeighbors
import networkx as nx
from numpy import *
import numpy as np
import operator
from collections import Counter
from pylab import *
from numpy.linalg import inv, cholesky
from numpy.linalg import eig
from metric_learn.constraints import Constraints
from sklearn import metrics
import collections


class LGDL():
    def __init__(self, data_set):
        self.data_set = data_set
        self.num_constraint = 10
        self.graph_neighbors = 5
        self.num_neighbors = 3
        self.learning_rate = 0.2
        self.max_iter_num = 300

    def find_K_Neighbors(self, data_point):
        neigh = NearestNeighbors()
        neigh.fit(self.data_set)
        K_Neighbors = neigh.kneighbors(data_point, self.num_neighbors + 1, return_distance=False)
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

    def Check_mahalanobis_matrix(self, mah_matrix):
        vals, vecs = eig(mah_matrix)
        for i in range(len(vals)):
            vals[i] = max(vals[i], 0)
        Lambda = np.diag(vals)
        positive_m = np.dot(np.dot(vecs, Lambda), vecs.T)
        return positive_m

    def calculate_new_HD(self,set_a, set_b, M_matrix):
        a = []
        a_index = []
        for item_a in set_a:
            a2b = []
            for item_b in set_b:
                dab = np.array(item_a) - np.array(item_b)
                new_distance_ab = np.dot(np.dot(dab, M_matrix), dab)
                a2b.append(new_distance_ab)
            a.append(min(a2b))
            a_index.append([item_a, set_b[a2b.index(min(a2b))]])
        a2b_max = max(a)
        a2b_index = a_index[a.index(a2b_max)]
        b = []
        b_index = []
        for item_b in set_b:
            b2a = []
            for item_a in set_a:
                dba = np.array(item_b) - np.array(item_a)
                new_distance_ba = np.dot(np.dot(dba, M_matrix), dba)
                b2a.append(new_distance_ba)
            b.append(min(b2a))
            b_index.append([item_b, set_a[b2a.index(min(b2a))]])
        b2a_max = max(b)
        b2a_index = b_index[b.index(b2a_max)]
        if a2b_max > b2a_max:
            return a2b_max, a2b_index
        else:
            return b2a_max, b2a_index

    def train(self,  must_link, cannot_link, must_index, cannot_index, feature_num,  betweenness):
        iter_num = 0
        M = np.eye(feature_num)
        last_cost = 0
        must_group = []
        cannot_group = []

        for must_pair in must_link:
            group_ma, group_mb = self.calculate_group_hd(must_pair, cannot_link)
            must_group.append([group_ma, group_mb])

        for cannot_pair in cannot_link:
            group_ca, group_cb = self.calculate_group_hd(cannot_pair, cannot_link)
            cannot_group.append([group_ca, group_cb])

        loss_cost = []
        while iter_num <= self.max_iter_num:
            must_pd_cost = 0
            cannot_pd_cost = 0
            must_cost = 0
            cannot_cost = 0
            m_index = 0
            c_index = 0
            for must_pair in must_group:
                new_hd, m_data_pair = self.calculate_new_HD(must_pair[0], must_pair[1], M)
                mweights = betweenness[must_index[m_index][0]] + betweenness[must_index[m_index][1]]
                must_cost += new_hd * mweights
                m_dif = np.array(m_data_pair[0]) - np.array(m_data_pair[1])
                mpd_result = m_dif.reshape(-1, 1) * m_dif.reshape(-1, 1).T* mweights
                must_pd_cost += mpd_result
                m_index += 1

            for cannot_pair in cannot_group:
                new_chd, c_data_pair = self.calculate_new_HD(cannot_pair[0], cannot_pair[1], M)
                cweights = betweenness[cannot_index[c_index][0]] + betweenness[cannot_index[c_index][1]]
                hinge_loss_c = (max(6 - new_chd, 0)).real
                cannot_cost += hinge_loss_c * cweights
                c_dif = np.array(c_data_pair[0]) - np.array(c_data_pair[1])
                cpd_result = (c_dif.reshape(-1, 1) * c_dif.reshape(-1, 1).T * cweights).real
                cannot_pd_cost += hinge_loss_c * cpd_result
                c_index += 1

            pd_cost = (must_pd_cost / len(must_link) - cannot_pd_cost / len(cannot_link)).real
            current_cost = (must_cost / len(must_link) + cannot_cost / len(cannot_link)).real
            loss_cost.append(current_cost)

            if abs(current_cost - last_cost) < 0.00001:
                break
            elif current_cost - last_cost < 0:
                learning_rate = 1.1 * self.learning_rate
                M = self.Check_mahalanobis_matrix(M - learning_rate * pd_cost)
            else:
                learning_rate = 0.5 * self.learning_rate
                M = self.Check_mahalanobis_matrix(M - learning_rate * pd_cost)
            last_cost = current_cost
            iter_num += 1
        M = self.Check_mahalanobis_matrix(M)
        return M
