# What needs to be made?

# 1. You need a proper TSP approx
# 2. You need dijkstras to calculate way between the nodes
# 3. You need a function which takes a graph and a set of cluster (centers) + locations
# The function should find the shortest distances between all those clusters
# Then the function should return a new graph containing only those clusters and the shortest paths
# between them. Then you can give this to TSP to calculate the best route between the clusters.

import tspy
from tspy.solvers import TwoOpt_solver
from input_generator import GraphCreator
from dijkstra import Graph


def tsp_approx(ad_mat):
    """
    Provided with an adjacency matrix, it will return tsp path for that input.
    :param ad_mat_file: The adjacency matrix
    :return: Tsp path in indexes
    """
    tsp = tspy.TSP()
    tsp.read_mat(ad_mat)

    two = TwoOpt_solver('NN')
    return two.solve(tsp)


def calculate_cluster_distances(ad_mat, indexes):
    distances = {}
    graph = Graph()
    # Each node maintains a list with nodes distances to itself
    for i in indexes:
        for j in indexes:
            distances[i] = {j:0}
    for i in indexes:
        for j in indexes:
            try:
                distances[i][j] = graph.dijkstra(ad_mat, i)[0][j]
            except:
                distances[i][j] = [i]

    return distances


def create_cluster_graph(cluster_matrix):
    matrix = [['' for j in range(len(cluster_matrix))] for i in range(len(cluster_matrix))]
    row_numbers = {}
    row = 0
    for i in cluster_matrix.keys():
        row_numbers[i] = row
        row += 1
    for key in cluster_matrix.keys():
        for keykey in cluster_matrix[key].keys():
            matrix[row_numbers[key]][row_numbers[keykey]] = cluster_matrix[key][keykey]

    return matrix


def get_distance_dict(adj_mat):
    """
    Returns a dictionary of the shortest distances from all node to all others in the parameter adjacency matrix.
    :param adj_mat: A adjacency matrix (mxn array) for the graph to calculate distances from
    :return: A dictionary with shortest distances from all nodes to all others
    """
    distances = {}
    graph = Graph()
    for i in range(len(adj_mat)):
        distances[i] = {}
    for i in range(len(adj_mat)):
        for j in range(len(adj_mat)):
            distances[i][j] = graph.dijkstra(adj_mat, i)[0][j]

    return distances


def get_distance_dict_fast(adj_mat):
    """
    Returns a dictionary of the shortest distances from all node to all others in the parameter adjacency matrix.
    :param adj_mat: A adjacency matrix (mxn array) for the graph to calculate distances from
    :return: A dictionary with shortest distances from all nodes to all others
    """
    distances = {}
    dijkstras = {}
    graph = Graph()
    for i in range(len(adj_mat)):
        distances[i] = {}
        dijkstras[i] = graph.dijkstra(adj_mat, i)[0]
    for i in range(len(adj_mat)):
        for j in range(len(adj_mat)):
            distances[i][j] = dijkstras[i][j]

    return distances


def get_path_dict(adj_mat):
    """
    Returns a dictionary where the key values are node-indexes and the values are a new dict.
    The keys in this new dict are all nodes in graph. The values for these keys are shortest paths from original
    node to key node.
    :param adj_mat:
    :return:
    """
    paths = {}
    graph = Graph()
    for i in range(len(adj_mat)):
        paths[i] = {}
    for i in range(len(adj_mat)):
        for j in range(len(adj_mat)):
            if i == j:
                paths[i][j] = [i]
            else:
                paths[i][j] = graph.dijkstra(adj_mat, i)[j+1]

    return paths


def find_shortest_traversal(matrix, node_list):
    """
    Given a matrix and a list of nodes, this function will output the shortest traversal visiting the nodes in that order.
    :param matrix: The matrix
    :param node_list: The list of nodes
    :return: List with path
    """
    path = [node_list[0]]
    graph = Graph()
    for i in range(len(node_list[:-1])):
        path.extend(graph.dijkstra(graph=matrix, src=node_list[i])[node_list[i+1]+1][1:])
    return path


def algorithm(input_file):
    mat = GraphCreator.get_matrix_from_file(input_file)
    # Do some stuff to get the clusters
    # Cluster-centers should come from aprils funciton
    cluster_centers = [1, 5, 7, 14, 44, 32]  # Should be aprils_output.keys()
    # Calculate the closest distances between the clusters so that we can run TSP
    cluster_distances = calculate_cluster_distances(mat, cluster_centers)
    # Create a graph with only clusters to use for TSP
    cluster_graph = create_cluster_graph(cluster_distances)
    # Do the TSP approx:
    tsp = tsp_approx(cluster_graph)
    # Convert back to indexes in original graph
    nodes_in_original = [cluster_centers[i] for i in tsp]
    # Then find actual path between nodes:
    path_in_original = find_shortest_traversal(mat, nodes_in_original)
    print(path_in_original)

algorithm('tst.in')



