import numpy as np
import numpy.linalg as alg
from collections import defaultdict


class Graph:
    def __init__(self, pathToFile):
        raw_data = np.genfromtxt(pathToFile, '\t', skip_header=4)

        self._graph = defaultdict(set)
        self.fill(raw_data)

    def add_entry(self, nodeFrom, nodeTo):
        if len(self._graph[nodeFrom]) == 0:
            self._graph[nodeFrom] = [[], []]

        if len(self._graph[nodeTo]) == 0:
            self._graph[nodeTo] = [[], []]

        self._graph[nodeFrom][0].append(nodeTo)
        self._graph[nodeTo][1].append(nodeFrom)

    def fill(self, rawData):
        for node1, node2 in rawData:
            self.add_entry(node1, node2)

    def get_node(self, nodeId):
        return self._graph[nodeId]

    def get_all_nodes(self):
        return self._graph.keys()

    def get_size(self):
        return len(self._graph)

    def dead_end_removal(self):

        # we will work as proposed by https://hunglvosu.github.io/res/deadend.pdf
        # first we need to determine the degree of the nodes
        degrees = defaultdict(int)
        to_remove_queue = []

        for node, lists in self._graph.items():
            degrees[node] = len(lists[0])
            if len(lists[0]) == 0:
                to_remove_queue.append(node)

        progress_list = []
        while len(to_remove_queue) > 0:
            currently_processed_node = to_remove_queue.pop()
            if currently_processed_node not in progress_list:
                progress_list.append(currently_processed_node)
                for node_lowered_deg in self._graph[currently_processed_node][1]:
                    degrees[node_lowered_deg] = degrees[node_lowered_deg] - 1
                    if degrees[node_lowered_deg] == 0:
                        to_remove_queue.append(node_lowered_deg)

        for node in progress_list:
            del self._graph[node]

        return progress_list


def graph_to_matrix(graph):
    all_nodes = graph.get_all_nodes()
    graphID_to_matrixID = defaultdict(int)
    j=0

    for nodeID in all_nodes:
        graphID_to_matrixID[nodeID]=j
        j += 1

    matrix = np.zeros([len(all_nodes),len(all_nodes)])

    for nodeID in all_nodes:
        node = graph.get_node(nodeID)
        node_matrixID = graphID_to_matrixID[nodeID]
        outgoing_list = node[0]
        outgoing_size = len(outgoing_list)

        if outgoing_size == 0:
            continue

        factor = 1 / outgoing_size

        for out_node in outgoing_list:
            out_node_matixID = graphID_to_matrixID[out_node]
            matrix[node_matrixID][out_node_matixID] = factor

    return graphID_to_matrixID, matrix


def google_page_rank(dense_input_matrix, beta, iter):
    primary_size = dense_input_matrix.shape[0]
    # this creates a start vector for the power iteration with 1/N as values
    last_rank_vector = np.reshape((np.ones(primary_size) * (1 / primary_size)), (primary_size, -1))

    while iter > 0:
        iter -= 1
        new_rank_vetor = dense_input_matrix @ last_rank_vector
        last_threshhold = alg.norm((last_rank_vector - new_rank_vetor), 1)
        print("current distance: " + str(last_threshhold))
        last_rank_vector = new_rank_vetor

    # now we need to accommodate for the random teleports
    random_tp_prob_vector = np.reshape((np.ones(primary_size) * (1 - beta) * (1 / primary_size)), (primary_size, -1))
    rank_vector_google = (last_rank_vector * beta) + random_tp_prob_vector
    return rank_vector_google


if __name__ == '__main__':
    graph = Graph("web-Stanford_small.txt")

    graph.dead_end_removal()

    graphId_to_matrixId, matrix = graph_to_matrix(graph)
    beta = 0.8
    iter = 5

    rank_vector = google_page_rank(matrix,beta,iter)
    print(rank_vector)

