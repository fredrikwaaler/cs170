# April Shin

import random
import copy
import math
from dijkstra import Graph
from read_file import File
from input_generator import GraphCreator

# DRAFT:

# debug values
debug = False
debug_startingk = False
debug_k_medians = False
debug_k = False
debug_approx_k = False

k_is_approximated = True

class K_Medians_Cluster():

    def __init__(self, homes, graph, num_trials):
        # key: total sum distance
        # value: centers for that total sum distance
        self.sumdist_cent_hist = {}

        # key: total sum distance
        # value: centers and corresponding cluster points for that toal sum distance
        self.sumdist_cluster_hist = {}

        self.sumdist_hist = [] # list of total sum distances
        self.homes = homes
        self.graph = graph
        self.first_center = 0
        self.distances = self.get_distance_list_fast(self.graph)
        self.k_potentials = []
        self.k_trials(num_trials) # Generate num_trials potential k values to find best value of k to cluster
        self.ith_k = 0
        self.dunn_index = 0



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
    Generate all k values you want apply to k_medians_clustering for k approximation
    The k that generates best cluster_quality is the number 
    '''
    def k_trials(self, num_trials):
        # Divide length of graph into intervals
        interval_len = math.ceil(len(self.homes)/num_trials)

        if debug_k:
            print("num trials: ", num_trials)
            print("interval length: ", interval_len)
            print("home length: ", len(self.homes))

        # Choose a value randomly in each interval to set as our trial k value
        self.k_potentials = []
        pos = 1

        # Generate random ints within appropriate intervals until you have num_trials k values or already traversed through all intervals.
        while (len(self.k_potentials) < num_trials) and (pos < len(self.graph)):
            if pos + interval_len <= len(self.graph):
                upper_limit = pos + interval_len - 1
                if debug_k:
                    print("upper_limit: ", upper_limit)
                    print("lower_limit: ", pos)
            else:
                # If at the last interval of the length of the graph
                upper_limit = len(self.graph) - 1
            self.k_potentials.append(pos + int((upper_limit - pos)/2))
            pos += interval_len

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
    def k_starting_centers(self, distances_copy, center_arr, i, k):
        if debug_startingk:
            print("first_center: ",self.first_center)

        if i == (k-1):
            if debug_startingk:
                print("starting k: ", i + 1, "/" ,k, " centers chosen. center_arr: ", center_arr)
            return center_arr

        if debug_startingk:
            print("center_arr: ", center_arr)

        cluster_dict = self.clustering(center_arr, k, False)

        if debug_startingk:
            print("size of home: ", len(self.homes))
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
        if farthest_loc not in center_arr:
            if debug_startingk:
                print("farhtest_loc here")
            center_arr.append(farthest_loc)
            if debug_startingk:
                print("starting k: ", i + 1, "/", k, " centers chosen. cluster_dict: ", cluster_dict)
            # center_arr[i+1] = farthest_loc
            return self.k_starting_centers(distances_copy, center_arr, i + 1, k)
        else:
            if debug_startingk:
                print("else here")
            distances_copy[center_arr[i]][farthest_loc] = -1  # To guarantee that this point will not be the max another time
            return self.k_starting_centers(distances_copy, center_arr, i, k)


    def clustering(self, k_centers, k, homesBool):
        if debug:
            print("clustering")
        centers_dict = {key: [] for key in k_centers}
        if homesBool:
            # We only want to assign homes to centers
            for i in self.homes:
                if i not in k_centers:
                    center_dist = [self.distances[i][j] for j in k_centers]
                    center_index = center_dist.index(min(center_dist))
                    center_for_loc = k_centers[center_index]
                    centers_dict.update(center_for_loc=centers_dict[center_for_loc].append(i))
            centers_dict.pop('center_for_loc', None)
            centers_dict = self.verify_clustering(centers_dict, k)

        else:
            # Clustering method is also used for generating starting k centers wrt all locations
            for i in range(len(self.graph)):
                if i not in k_centers:
                    center_dist = [self.distances[i][j] for j in k_centers]
                    center_index = center_dist.index(min(center_dist))
                    center_for_loc = k_centers[center_index]
                    centers_dict.update(center_for_loc = centers_dict[center_for_loc].append(i))
            centers_dict.pop('center_for_loc', None)

        return centers_dict

    def verify_clustering(self, centers_dict, k):
        # If a cluster is of size one and the center is not a location, then change center to home
        # If a cluster is of size zero, and if center is not a home, then remove cluster. Otherwise, add home center as a cluster point as well.
        if debug:
            print("whole dictionary: ", centers_dict)
            print("keys: ", centers_dict.keys())
        replace_key = {}
        remove_key = {}
        for i in (centers_dict):
            if debug:
                print("i (the key error): ", i)
            if len(centers_dict[i]) == 1:
                if i not in self.homes:
                    home = centers_dict[i][0]
                    replace_key[i] = [home]
            if len(centers_dict[i]) == 0:
                # When a center has no points, delete cluster.
                if i not in self.homes:
                    if debug:
                        print("Here 2")
                    remove_key[i] = []
                else:
                    centers_dict[i] = [i]

        for i in replace_key:
            centers_dict.pop(i)
            if debug:
                print("replace key: ", replace_key[i][0])
            centers_dict[replace_key[i][0]] = replace_key[i]

        for i in remove_key:
            centers_dict.pop(i)

        return centers_dict


    def improved_centers(self, centers_dict):
        # Get all cluster points in each cluster
        # Compute the sum of all distances from each location in matrix to all cluster points to all points
        # Result: list of location's sum of distances
        # The location with the min sum is the new center of that cluster

        new_centers = []
        for center in centers_dict:
            loc_sumdist = [0 for i in range(len(self.graph))]
            for loc in range(len(self.graph)):
                sumdist = 0
                for cluster_p in centers_dict[center]:
                    sumdist += self.distances[loc][cluster_p]
                loc_sumdist[loc] = sumdist
            new_centers.append(loc_sumdist.index(min(loc_sumdist)))

        return new_centers

    def clustering_quality(self, centers_dict):
        # To compute total sum of sum of distances of center to all of its cluster points
        # and compute average of the sum of average sum of each cluster
        cluster_avg = [0 for i in range(len(centers_dict))]
        count = 0

        cluster_sums = []
        for center in centers_dict.keys():
            cluster_sum = 0
            for cluster_p in centers_dict[center]:
                #total_sum += self.distances[center][cluster_p]
                cluster_sum += self.distances[center][cluster_p]
            cluster_sums.append(cluster_sum)
            cluster_avg[count] = cluster_sum/len(centers_dict[center])
            count += 1

        sum_avg = 0
        for avg in cluster_avg:
            sum_avg += avg
        avgavgsum_cluster = sum_avg/len(cluster_avg)

        # Also computing Dunn Index
        # a = find the min of (center1 - center2) squared
        # b = find the max of (sum of distances from center to cluster points)
        # dunn_index = max(a/b)
        # We want the biggest ratio of smallest distance between centers over biggest diameter cluster
        # in other words we want the smallest distance between centers to be as big as possible and want our diameter to be as small as possible

        min_center_diff = math.inf
        for center1 in centers_dict:
            for center2 in centers_dict:
                if center1 != center2:
                    dist = self.distances[center1][center2]
                    if dist < min_center_diff:
                        min_center_diff = float(dist)
        if debug_k_medians:
            print("min_center_diff: ", min_center_diff)
        max_cluster_diameter = float(max(cluster_sums))
        if debug_k_medians:
            print("max_cluster_diameter: ", max_cluster_diameter)
        self.dunn_index = min_center_diff/max_cluster_diameter
        if debug_k_medians:
            print("freshly computed self.dunn_index: ", self.dunn_index)

        return avgavgsum_cluster


    def get_min_sum(self):
        return min(self.sumdist_hist)


    #def k_medians_clustering(self, centers_dict=None, prev_totalsum=0, curr_totalsum=math.inf, i=1, k=10, epsilon=0.5, k_centers=[]):
    def k_medians_clustering(self, centers_dict=None, prev_totalsum=0, curr_totalsum=math.inf, i=1, epsilon=0.5, k_centers=[]):

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

                minsum = self.sumdist_hist[self.sumdist_hist.index(min(self.sumdist_hist))]
                centers = self.sumdist_cent_hist[minsum].copy()
                cl_dict = self.sumdist_cluster_hist[minsum].copy()
                self.sumdist_cluster_hist = {}
                self.sumdist_hist = []
                self.sumdist_cent_hist = {}
                return centers, cl_dict, self.dunn_index
            else:
                if debug_k_medians:
                    print("curr_centers_dict: ", centers_dict)
                    print("storing: ", centers_dict.keys())
                centers = []
                for center in centers_dict:
                    centers.append(center)

                # Store total sum and center points into archive
                self.sumdist_hist.append(curr_totalsum)
                self.sumdist_cent_hist[curr_totalsum] = centers
                self.sumdist_cluster_hist[curr_totalsum] = centers_dict

                if debug:
                    print("self.sumdist_hist: ",self.sumdist_hist)
                    print("self.sumdist_cent_hist: ", self.sumdist_cent_hist)

        # Setting a limit as the total sums converge
        # Sometimes, totalsum may be greater than previous totalsum but that's okay.
        # That is why we choose to set a limit rather checking if sums are increasing i.e. quality of center points are decreasing.
        if abs(prev_totalsum - curr_totalsum) < epsilon:
            minsum = self.sumdist_hist[self.sumdist_hist.index(min(self.sumdist_hist))]
            centers = self.sumdist_cent_hist[minsum].copy()
            cl_dict = self.sumdist_cluster_hist[minsum].copy()
            self.sumdist_cluster_hist = {}
            self.sumdist_hist = []
            self.sumdist_cent_hist = {}
            return centers, cl_dict, self.dunn_index

        k = self.k_potentials[self.ith_k]
        if debug_k_medians:
            print("centers_dict: ", centers_dict)
        # Generate initial k centers
        if i == 1:
            center = []
            center.append(self.first_center)

            if k_is_approximated:
                k_centers = self.k_starting_centers(copy.deepcopy(self.distances), center, 0, k)
            else:
                if len(self.graph) <= 50:
                    if len(self.graph) == 1 or len(self.graph) == 2:
                        k = int(len(self.graph)/2)
                    else:
                        k = int(len(self.graph)/3)
                elif len(self.graph) <= 100:
                    k = int(len(self.graph)/5.5)
                elif len(self.graph) <= 200:
                    k = int(len(self.graph)/11)
                k_centers = self.k_starting_centers(copy.deepcopy(self.distances), center, 0, k)

        # Cluster homes wrt inital centers
        centers_dict = self.clustering(k_centers, k, True)

        # Compute total sum of freshly computed distances between centers and corresponding cluster points
        new_totalsum = self.clustering_quality(centers_dict)
        if debug_k_medians:
            print("dunn_index", self.dunn_index)

        # Generate improved centers
        new_centers = self.improved_centers(centers_dict)

        return self.k_medians_clustering(centers_dict, curr_totalsum, new_totalsum, i + 1, epsilon, new_centers)


    def approx_best_clustering(self):
        self.first_center = random.randint(0, len(self.graph) - 1)
        all_clustering = {}
        if debug_approx_k:
            print("self.k_potentials: ", self.k_potentials)
        for i in self.k_potentials:
            if debug_approx_k:
                print("k: ", i)
            centers, centers_dict, dunn_index = self.k_medians_clustering()
            all_clustering[dunn_index] = [centers, centers_dict]
            self.ith_k += 1

        # The bigger the value of cluster_quality, the better
        max_dunn_index = -1
        best_clustering = []
        for dunn_index in all_clustering:
            if debug_approx_k:
                print("dunn_index(bigger better): ", dunn_index)
            if dunn_index > max_dunn_index:
                max_dunn_index = dunn_index
                best_clustering = all_clustering[dunn_index]
        if debug_approx_k:
            print("\nbest clustering quality(bigger better): ",max_dunn_index)
            print("best clustering: ", best_clustering)
        # Return list of center points and dictionary of centers and corresponding cluster points
        return best_clustering[0], best_clustering[1]

"""
# Temporary:


distances = k_medians.get_distance_list_fast(graph)

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



import os

directory = os.fsencode("inputs")

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    #filename = "inputs/" + str(i+1) + "_50.in"
    #f = File("inputs/216_50.in")


#f = File("inputs/" + filename)

filename = "_200.in"
g = GraphCreator()
graph = g.get_matrix_from_file("inputs/" + filename)
homes = g.get_home_indices("inputs/" + filename)

#f = File("inputs/226_50.in")
#f.readFile()
#graph = f.getGraph()
#homes = f.getHomes()
print("homes: ",homes)
#k = 5 #Will work on approximation later
k_medians = K_Medians_Cluster(homes, graph, 6)

print("\n\nTesting for convergence: ")
#print("first_center: ",first_center)
epsilon = 0
results = k_medians.approx_best_clustering()
#centers_dict = k_medians.k_medians_clustering()#None, 0, math.inf, 1, k, epsilon, [])
print("\nfinal clusters once sums converged: ", results)


"""