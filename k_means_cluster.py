import random
import copy
from dijkstra import Graph
from read_file import File

# DRAFT:

# debug values
debug = False
debug_startingk = False
debug_k_means = True

f = File("inputs/50.in")
f.readFile()
graph = f.getGraph()
homes = f.getHomes()
k = 5 #Will work on approximation later
first_center = 9 #random.randint(0,50 )#31 #test at 9

class K_Means_Cluster():

    #def __init__(self):


    """
    Returns a list of the shortest distances from all node to all others in the parameter adjacency matrix.
    index corresponds to node, value correponds to distances.
    :param adj_mat: A adjacency matrix (mxn array) for the graph to calculate distances from
    :return: A list with shortest distances from all nodes to all others
    """
    def get_distance_list_fast(self, adj_mat):
        distances = [0 for j in range(len(adj_mat))]
        dijkstras = [0 for j in range(len(adj_mat))]
        graph = Graph()
        for i in range(len(adj_mat)):
            distances[i] = [0 for j in range(len(adj_mat))]
            dijkstras[i] = graph.dijkstra(adj_mat, i)[0]
        for i in range(len(adj_mat)):
            for j in range(len(adj_mat)):
                distances[i][j] = dijkstras[i][j]

        return distances


    '''
    Choose k starting centers by starting with a random location and choosing next center to be the furthest location from the current center.
    Following centers will be chosen to be the furthest location from all previously chosen centers.
    :param distances_copy: adjacency matrix consisting of distances between every pair of locations
            center_arr: Empty list of length k to store generated k centers.
                         First element initalized to be the randomly generated starting center
            i: iteration count, start from 0
            k: approximated k value for creating k clusters
    :return: list of center location indices
    '''
    def k_starting_centers(self, distances_copy, distances, center_arr, i, k):
        if i == (k-1):
            return center_arr
        # elif (i == 1):
        # farthest_loc = distances_copy[center_arr[i]].index(max(distances_copy[center_arr[i]]))
        # else:
        if debug_startingk:
            print("center_arr: ", center_arr)
        cluster_dict = self.clustering(center_arr, distances, homes, k, False)
        farthest_loc = center_arr[0]
        for center in cluster_dict.keys():
            max_dist = 0
            for cluster_p in cluster_dict[center]:
                dist = distances_copy[center][cluster_p]
                if dist > max_dist:
                    max_dist = dist
                    farthest_loc = cluster_p

        if debug_startingk:
            print("farthest_loc: ", farthest_loc)
            print(i, "cluster_dict: ", cluster_dict)
        if farthest_loc not in center_arr:
            center_arr.append(farthest_loc)
            # center_arr[i+1] = farthest_loc
            return self.k_starting_centers(distances_copy, distances, center_arr, i + 1, k)
        else:
            distances_copy[center_arr[i]][
                farthest_loc] = -1  # To guarantee that this point will not be the max another time
            return self.k_starting_centers(distances_copy, distances, center_arr, i, k)


    def clustering(self, k_centers, distances, homes, k, verify):
        if debug:
            print("clustering")
        cluster_points = []
        centers_dict = {key: [] for key in k_centers}

        for i in homes: #we only want to assign homes to centers #(len(graph)):
            if i not in k_centers:
                center_dist = [distances[i][j] for j in k_centers]
                center_index = center_dist.index(min(center_dist))
                center_for_loc = k_centers[center_index]
                if center_for_loc in centers_dict.keys():
                    centers_dict.update(center_for_loc = centers_dict[center_for_loc].append(i))
        centers_dict.pop('center_for_loc', None)
        if verify:
            centers_dict = self.verify_clustering(distances, centers_dict, homes, k)
        return centers_dict

    def verify_clustering(self, distances, centers_dict, homes, k):
        # If a cluster is of size one and the center is not a location, then change center to home
        # If a cluster is of size zero, then replace that one center.
        print(centers_dict)
        for i in (centers_dict):
            if len(centers_dict[i]) == 1:
                if i not in homes:
                    home = centers_dict[i][0]
                    centers_dict.pop(i)
                    centers_dict[home] = home
            if len(centers_dict[i]) == 0:
                print("EMPTY DICT")
                # Call k_starting centers and make sure not to choose faulty center by changing its distance to any node as -1.
                # Then rerun clustering()
                dist_copy = copy.deepcopy(distances)
                dist_copy[i] = [-1 for i in range(len(distances[i]))]
                k_start_centers = self.k_starting_centers(dist_copy, distances, [0 for i in range(k)], 0, k)
                centers_dict = self.clustering(k_start_centers, distances, homes, k, True)

        return centers_dict


    def improved_centers(self, centers_dict, distances, graph, k_centers):
        # Compute average distance between center and each cluster point for each cluster
        cluster_avg = {key: None for key in k_centers}
        for center in centers_dict.keys():
            cluster_sum = 0
            for cluster_p in centers_dict[center]:
                cluster_sum += distances[center][cluster_p]
            cluster_avg[center] = cluster_sum / len(centers_dict[center])

        # Finding new centers
        # Choose ANY location with the minimum difference of its distance to average distance
        new_centers = []
        for center in centers_dict.keys():
            min_diff = 100
            min_loc = center
            # for cluster_p in centers_dict[center]:
            for i in range(len(graph)):
                diff = abs(distances[center][i] - cluster_avg[center])
                if diff < min_diff:
                    min_diff = diff
                    min_loc = i
            new_centers.append(min_loc)
        return new_centers

    def total_dist(self, centers_dict, distances):
        # To compute total sum of sum of distances of center to all of its cluster points
        total_sum = 0
        for center in centers_dict.keys():
            for cluster_p in centers_dict[center]:
                total_sum += distances[center][cluster_p]
        return total_sum


    def k_means_clustering(self, centers_dict, prev_totalsum, curr_totalsum, distances, homes, i, k, epsilon, graph, k_centers):
        if debug_k_means:
            print("prev_totalsum: ", prev_totalsum)
            print("curr_totalsum: ", curr_totalsum)
        if abs(prev_totalsum - curr_totalsum) < epsilon:
            return centers_dict
        if debug_k_means:
            print("centers_dict: ", centers_dict)
        # Generate initial k centers
        center = [0 for i in range(k)]
        center[0] = first_center

        if i == 1:
            k_centers = self.k_starting_centers(copy.deepcopy(distances), distances, center, 0, k)

        # Cluster homes wrt inital centers
        centers_dict = self.clustering(k_centers, distances, homes, k, True)

        # Generate improved centers
        new_centers = self.improved_centers(centers_dict, distances, graph, k_centers)
        # Compute total sum of distances between centers and corresponding cluster points
        new_totalsum = self.total_dist(centers_dict, distances)
        if debug_k_means:
            print(new_totalsum)
        return self.k_means_clustering(centers_dict, curr_totalsum, new_totalsum, distances, homes, i + 1, k, epsilon, graph,
                                  new_centers)





# Temporary:
k_means = K_Means_Cluster()

distances = k_means.get_distance_list_fast(graph)
center = [0]#[0 for i in range(k)]
center[0] = first_center
k_start_centers = k_means.k_starting_centers(copy.deepcopy(distances), distances, center, 0, k)
print("starting k centers: ", k_start_centers)
centers_dict = k_means.clustering(k_start_centers, distances, homes, k, True)
print("first set of clusters: ", centers_dict)
new_centers = k_means.improved_centers(centers_dict, distances, graph, k_start_centers)
print("new center locations: ", new_centers)
totaldistance = k_means.total_dist(centers_dict, distances)
print("Total sum of distances between centers and corresponding cluster points: ",totaldistance)

print("\n\nTesting for convergence: ")
print("first_center: ",first_center)
epsilon = 10
centers_dict = k_means.k_means_clustering(None, 0, 1000, distances, homes, 1, k, epsilon, graph, [])
print("\nfinal clusters once sums converged: ", centers_dict)
