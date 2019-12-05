import random
import copy
import math
from dijkstra import Graph
from read_file import File

# DRAFT:

# debug values
debug = False
debug_startingk = False
debug_k_medians = True

f = File("inputs/50.in")
f.readFile()
graph = f.getGraph()
homes = f.getHomes()
print("homes: ",len(homes))
k = 20 #Will work on approximation later
first_center = random.randint(0,49)#31 #test at 9 # 33

class K_Medians_Cluster():

    def __init__(self):
        # key: total sum distance
        # value: centers for that total sum distance
        self.sumdist_cent_hist = {}
        self.sumdist_hist = []


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
        if debug_startingk:
            print("size of home: ", len(homes))
            print("size: ", len(cluster_dict[center_arr[0]]))
            print("starting cluster_dict: ", cluster_dict)
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
        if debug:
            print("whole dictionary: ", centers_dict)
            print("keys: ", centers_dict.keys())
        replace_key = {}
        remove_key = {}
        for i in (centers_dict):
            if debug:
                print("i (the key error): ", i)
            if len(centers_dict[i]) == 1:
                if i not in homes:
                    home = centers_dict[i][0]
                    replace_key[i] = [home]
            if len(centers_dict[i]) == 0:
                '''
                MADE CHANGES:
                When a center has no points, delete cluster.
                Previously, I tried to rerun k_starting_centers 
                and get a new set of k centers (bc clearly this is not a good set of centers), but that caused so many errors. 
                
                '''
                if i not in homes:
                    print("Here 2")
                    remove_key[i] = []
                else:
                    centers_dict[i] = [i]
                '''
                 print("Here 2")
                if debug:
                    print("i: ", i, "centers_dict[i]: ", centers_dict[i])
                    print("EMPTY DICT")
                # Call k_starting centers and make sure not to choose faulty center by changing its distance to any node as -1.
                # Then rerun clustering()
                dist_copy = copy.deepcopy(distances)
                dist_copy[i] = [-1 for i in range(len(distances[i]))]
                k_start_centers = self.k_starting_centers(dist_copy, distances, [0 for i in range(k)], 0, k)
                centers_dict = self.clustering(k_start_centers, distances, homes, k, True)
                '''
        for i in replace_key:
            centers_dict.pop(i)
            print(replace_key[i][0])
            centers_dict[replace_key[i][0]] = replace_key[i]

        for i in remove_key:
            centers_dict.pop(i)

        return centers_dict

    def improved_centers(self, centers_dict, distances, graph):
        # Compute average distance between center and each cluster point for each cluster
        '''
        CHANGES TO MAKE:
        compute sum of distance to all points in the cluster to every location.
        One with the min sum is the new center.
        :param centers_dict:
        :param distances:
        :param graph:
        :param k_centers:
        :return:
        '''


        # Get all cluster points in each cluster
        # Compute the sum of all distances from each location in matrix to all cluster points to all points
        # Result: list of location's sum of distances
        # The location with the min sum is the new center of that cluster


        new_centers = []
        for center in centers_dict:
            loc_sumdist = [0 for i in range(len(graph))]
            for loc in range(len(graph)):
                sumdist = 0
                for cluster_p in centers_dict[center]:
                    sumdist += distances[loc][cluster_p]
                loc_sumdist[loc] = sumdist
            new_centers.append(loc_sumdist.index(min(loc_sumdist)))

        '''
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
        '''

        return new_centers

    def total_dist(self, centers_dict, distances):
        # To compute total sum of sum of distances of center to all of its cluster points
        total_sum = 0
        for center in centers_dict.keys():
            for cluster_p in centers_dict[center]:
                total_sum += distances[center][cluster_p]
        return total_sum


    def k_medians_clustering(self, centers_dict, prev_totalsum, curr_totalsum, distances, homes, i, k, epsilon, graph, k_centers):
        if debug_k_medians:
            print("prev_totalsum: ", prev_totalsum)
            print("curr_totalsum: ", curr_totalsum)

        if i > 1:
            # Checking archive if current total sum was already generated in the past
            # (to avoid generating the same centers)
            if curr_totalsum in self.sumdist_hist:
                if debug_k_medians:
                    print("Already saw this total sum in the past. Return best centers from archive.")
                # return the centers of the minimum total sum distance generated
                return self.sumdist_cent_hist[self.sumdist_hist[self.sumdist_hist.index(min(self.sumdist_hist))]]
            else:
                if debug_k_medians:
                    print("curr_centers_dict: ", centers_dict)
                    print("storing: ", centers_dict.keys())

                # Store total sum and center points into archive
                self.sumdist_hist.append(curr_totalsum)
                self.sumdist_cent_hist[curr_totalsum] = centers_dict.keys()

        # Setting a limit as the total sums converge
        # Sometimes, totalsum may be greater than previous totalsum but that's okay.
        # That is why we choose to set a limit rather checking if sums are increasing i.e. quality of center points are decreasing.
        if abs(prev_totalsum - curr_totalsum) < epsilon:
            return self.sumdist_cent_hist[self.sumdist_hist[self.sumdist_hist.index(min(self.sumdist_hist))]]

        if debug_k_medians:
            print("centers_dict: ", centers_dict)
        # Generate initial k centers
        center = []
        center.append(first_center)

        if i == 1:
            k_centers = self.k_starting_centers(copy.deepcopy(distances), distances, center, 0, k)

        # Cluster homes wrt inital centers
        centers_dict = self.clustering(k_centers, distances, homes, k, True)

        # Compute total sum of freshly computed distances between centers and corresponding cluster points
        new_totalsum = self.total_dist(centers_dict, distances)

        # Generate improved centers
        new_centers = self.improved_centers(centers_dict, distances, graph)

        if debug_k_medians:
            print(new_totalsum)
        return self.k_medians_clustering(centers_dict, curr_totalsum, new_totalsum, distances, homes, i + 1, k, epsilon, graph,
                                  new_centers)





# Temporary:
k_medians = K_Medians_Cluster()


distances = k_medians.get_distance_list_fast(graph)
'''
center = [0]#[0 for i in range(k)]
center[0] = first_center
k_start_centers = k_medians.k_starting_centers(copy.deepcopy(distances), distances, center, 0, k)
print("starting k centers: ", k_start_centers)
centers_dict = k_medians.clustering(k_start_centers, distances, homes, k, True)
print("first set of clusters: ", centers_dict)
new_centers = k_medians.improved_centers(centers_dict, distances, graph)
print("new center locations: ", new_centers)
totaldistance = k_medians.total_dist(centers_dict, distances)
print("Total sum of distances between centers and corresponding cluster points: ",totaldistance)
'''
print("\n\nTesting for convergence: ")
print("first_center: ",first_center)
epsilon = 0
centers_dict = k_medians.k_medians_clustering(None, 0, math.inf, distances, homes, 1, k, epsilon, graph, [])
print("\nfinal clusters once sums converged: ", centers_dict)


dict = {3:[1,2], 4:{3,4}, 5:[1]}
print(type(dict.keys()))

