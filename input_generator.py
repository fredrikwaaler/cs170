import random
from string import ascii_letters, digits
# Number of locations
num_loc = int(input("Number of locations:"))
# Number of home3
num_homes = random.randint(int(num_loc/2), int((num_loc/3)*2))
# Location names (a-z, A-Z, max 20 chars)
valid_chars = ascii_letters
locations = []
for loc in range(num_loc):
    name = ''
    for i in range(random.randint(5, 10)):
        name += random.choice(valid_chars)
    locations += [name]
# Home names (subset of location)
location_copy = locations.copy()
homes = []
for home in range(num_homes):
    homes.append(location_copy.pop(random.randint(0, len(location_copy)-1)))
# Start/end location
start = locations[random.randint(0, len(locations)-1)]

"""
print(num_loc)
print(num_homes)
print(locations)
print(homes)
print(start)
"""

adjacency_matrix = []
# First generate with all distances as one:
for loc_num in range(len(locations)):
    # Initialize node with no incident edges
    adjacency = ['x' for i in range(num_loc)]
    # Add already existing edges
    for prev_adjacency in range(loc_num-1):
        if adjacency_matrix[prev_adjacency][loc_num-1] != 'x':
            adjacency[prev_adjacency] = adjacency_matrix[prev_adjacency][loc_num-1]

    # Count number of incident edges:
    edge_num = 0
    for node in adjacency:
        if node != 'x':
            edge_num += 1

    # Aiming for an average of 1 to (num_loc / 10) edges incident for every node.
    new_edges = 0
    if edge_num < int(num_loc/10):
        # New edges are only added between this node and upcoming nodes.
        # We find out how many new edges we possibly can add, and generate a random subset of these.
        # Max function make sure we don't go for adding more edges than what is possible ot subsequent nodes
        new_edges = random.randint(1, min(int(num_loc/10)-edge_num, num_loc-loc_num))

    # Ensure that graph is connected
    if edge_num == 0 and new_edges == 0:
        new_edges = 1


    # Nodes we can add new edges to:
    pos_new_neighbours = [i for i in range(loc_num + 1, num_loc)]
    for new_edge in range(new_edges):
        adjacency[pos_new_neighbours.pop(random.randint(1, len(pos_new_neighbours)-1))] = 1

    adjacency_matrix.append(adjacency)


for adjacency in adjacency_matrix:
    print(adjacency)

# One line for each location - entry  j in line i marks the distance of edge from node i to j. 'x' if no edge.
# index i,i should also be 'x', no edge from a node to itself. # Distances max value 2bil, max 5 decimals.
# Graph must be connected - any line needs an entry to represent an edge from every node.
# Enforce triangle inequality



