# April Shin

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

class K_Medians_Cluster():

    def __init__(self, homes, graph):
        # key: total sum distance
        # value: centers for that total sum distance
        self.sumdist_cent_hist = {}

        # list of total sum distances
        self.sumdist_hist = []
        self.homes = homes
        self.graph = graph
        self.first_center = random.randint(0,49)#len(self.graph)-1)
        self.distances = self.get_distance_list_fast(self.graph)


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
        print(distances)
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
    def k_starting_centers(self, distances_copy, center_arr, i, k):

        first_center = random.randint(0, 49)

        if i == (k-1):
            return center_arr
        # elif (i == 1):
        # farthest_loc = distances_copy[center_arr[i]].index(max(distances_copy[center_arr[i]]))
        # else:
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
            print(i, "cluster_dict: ", cluster_dict)
        if farthest_loc not in center_arr:
            center_arr.append(farthest_loc)
            # center_arr[i+1] = farthest_loc
            return self.k_starting_centers(distances_copy, center_arr, i + 1, k)
        else:
            distances_copy[center_arr[i]][
                farthest_loc] = -1  # To guarantee that this point will not be the max another time
            return self.k_starting_centers(distances_copy, center_arr, i, k)


    def clustering(self, k_centers, k, verify):
        if debug:
            print("clustering")
        centers_dict = {key: [] for key in k_centers}

        for i in self.homes: #we only want to assign homes to centers #(len(graph)):
            if i not in k_centers:
                center_dist = [self.distances[i][j] for j in k_centers]
                center_index = center_dist.index(min(center_dist))
                center_for_loc = k_centers[center_index]
                if center_for_loc in centers_dict.keys():
                    centers_dict.update(center_for_loc = centers_dict[center_for_loc].append(i))
        centers_dict.pop('center_for_loc', None)
        if verify:
            centers_dict = self.verify_clustering(centers_dict, k)
        return centers_dict

    def verify_clustering(self, centers_dict, k):
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
                if i not in self.homes:
                    home = centers_dict[i][0]
                    replace_key[i] = [home]
            if len(centers_dict[i]) == 0:
                '''
                MADE CHANGES:
                When a center has no points, delete cluster.
                Previously, I tried to rerun k_starting_centers 
                and get a new set of k centers (bc clearly this is not a good set of centers), but that caused so many errors. 
                
                '''
                if i not in self.homes:
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

    def improved_centers(self, centers_dict):
        # Compute average distance between center and each cluster point for each cluster
        '''
        CHANGES TO MAKE:
        compute sum of distance to all points in the cluster to every location.
        One with the min sum is the new center.
        :param centers_dict:
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
            loc_sumdist = [0 for i in range(len(self.graph))]
            for loc in range(len(self.graph)):
                sumdist = 0
                for cluster_p in centers_dict[center]:
                    sumdist += self.distances[loc][cluster_p]
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

    def total_dist(self, centers_dict):
        # To compute total sum of sum of distances of center to all of its cluster points
        total_sum = 0
        for center in centers_dict.keys():
            for cluster_p in centers_dict[center]:
                total_sum += self.distances[center][cluster_p]
        return total_sum


    # Quantifies the quality of the clusters
    def silhouette_avg(self, centers_dict):

        """
        Compute silhouette coefficient of each cluster.
        Take the average for the whole graph.

        Define a(i) to be the mean distance of point (i) w.r.t to all the other points in the cluster its assigned.
        (mean intra-cluster distance)
        We can interpret a(i) as how well the point is assigned to the cluster. Smaller the value better the assignment.

        Define b(i) to be the mean distance of point(i) w.r.t. to other points in its closet neighboring cluster.
        (mean nearest_cluster distance)

        s(i) = (b(i) - a(i))/max(b(i), a(i))
        """

        # Have a dictionary of dictionary:
        # {Center: {cluster_p: mean_dist, clusterp_p: mean_dist .... }}
        aa = {}
        for center in centers_dict:
            ac  ={}
            cluster_p_sum = 0
            for cluster_p1 in centers_dict[center]:
                for cluster_p2 in centers_dict[center]:
                    if cluster_p1 != cluster_p2:
                        cluster_p_sum += self.distances[cluster_p1][cluster_p2]



        '''
        a = []
        for center in centers_dict:
            cluster_p_sum = 0
            for cluster_p1 in centers_dict[center]:
                for cluster_p2 in centers_dict[center]:
                    if cluster_p1 != cluster_p2:
                        cluster_p_sum += self.distances[cluster_p1][cluster_p2]
            a.append(cluster_p_sum/(len(centers_dict[center])-1))
        '''


        b = []
        # To compute b, compute mean distance between every cluster point to cluster points in other clusters
        # The min mean distance is the closest neighboring cluster

        # To compute b(i)
        # For each cluster point:
        # 1. Compute sum of distance from point to other points of different clusters
        # 2. Compute mean of each sum
        # 3. Choose neighboring cluster that yields min mean sum with cluster point to be closest neighbor
        for center in centers_dict:
            for cluster_p in centers_dict[center]:
                neighbor_sum = 0
                for neighbor_center in centers_dict:
                    if center != neighbor_center:
                        for neighbor_p in centers_dict[neighbor_center]:
                            neighbor_sum += self.distances[cluster_p][neighbor_p]
                neighbor_mean = neighbor_sum/len(centers_dict[neighbor_center])

    # Once you compute the silhouette coefficient, get mean of all coeff of points for each cluster.
    # Compute mean of those means to whole graph


        """
        # Locate closest neighboring cluster by finding closest neighboring center

         closest_neighbors = {}
        for center1 in centers_dict:
            min_dist = math.inf
            closest_center = center1
            for center2 in centers_dict:
                if center1 != center2:
                    cent_dist = distances[center1][center2]
                    if cent_dist < min_dist:
                        min_dist = cent_dist
                        closest_center = center2
            closest_neighbors[center1] = closest_center

        # Compute sum of distances for cluster point to cluster points in neighboring cluster
        for center in centers_dict:
            neighbor_distsum = 0
            for cluster_p in centers_dict[center]:
                for neighbor_p in centers_dict[closest_neighbors[center]]:
                    neighbor_distsum += distances[cluster_p][neighbor_p]
            b.append(neighbor_distsum/)
        """





    def k_medians_clustering(self, centers_dict=None, prev_totalsum=0, curr_totalsum=math.inf, i=1, k=5, epsilon=0, k_centers=[]):
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
                centers = []
                for center in centers_dict:
                    centers.append(center)

                # Store total sum and center points into archive
                self.sumdist_hist.append(curr_totalsum)
                self.sumdist_cent_hist[curr_totalsum] = centers

        # Setting a limit as the total sums converge
        # Sometimes, totalsum may be greater than previous totalsum but that's okay.
        #\That is why we choose to set a limit rather checking if sums are increasing i.e. quality of center points are decreasing.
        if abs(prev_totalsum - curr_totalsum) < epsilon:
            return self.sumdist_cent_hist[self.sumdist_hist[self.sumdist_hist.index(min(self.sumdist_hist))]]

        if debug_k_medians:
            print("centers_dict: ", centers_dict)
        # Generate initial k centers
        center = []
        center.append(self.first_center)

        if i == 1:
            k_centers = self.k_starting_centers(copy.deepcopy(self.distances), center, 0, k)

        # Cluster homes wrt inital centers
        centers_dict = self.clustering(k_centers, k, True)

        # Compute total sum of freshly computed distances between centers and corresponding cluster points
        new_totalsum = self.total_dist(centers_dict)


        # Generate improved centers
        new_centers = self.improved_centers(centers_dict)

        if debug_k_medians:
            print("new_total sum down here: ", new_totalsum)
            print("curr totalsum down here: ", curr_totalsum)
        return self.k_medians_clustering(centers_dict, curr_totalsum, new_totalsum, i + 1, k, epsilon, new_centers)




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


f = File("inputs/10_50.in")
f.readFile()
graph = f.getGraph()
homes = f.getHomes()
print(len(graph))
print(len(homes))



#k = 5 #Will work on approximation later
k_medians = K_Medians_Cluster(homes, graph)

print("\n\nTesting for convergence: ")
#print("first_center: ",first_center)
epsilon = 0
centers_dict = k_medians.k_medians_clustering()#None, 0, math.inf, 1, k, epsilon, [])
print("\nfinal clusters once sums converged: ", centers_dict)


dict = {3:[1,2], 4:{3,4}, 5:[1]}
print(type(dict.keys()))
"""


