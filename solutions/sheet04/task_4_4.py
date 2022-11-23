import numpy as np
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

if __name__ == '__main__':
    graph = Graph("web-Stanford_small.txt")
    print("\ntask a) is done - the chosen datastructures are")
    print("a dictionary with all unique node ids and two lists in each of the dicts entries")
    print("When looking up the node 1 the following is returned:")
    print(graph.get_node(1))
    print("1 has to outgoing connections and none incoming")

    print("\n-------------------------------------------------\n")
    print("Performing the recursive dead end removal.")
    print("Before the dead end removal there are "+str(graph.get_size())+" nodes in the graph")
    removed_nodes = graph.dead_end_removal()
    print("After the dead end removal there are "+str(graph.get_size())+" nodes in the graph")
    print("The following is the list of the removed nodes in order:")
    print(removed_nodes)

