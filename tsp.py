import pandas as pd
from math import sqrt, inf
import random
import matplotlib.pyplot as plt

# Load data from csv file, return list of coordinates
def load_data(file):
    """
    Function to load data from csv file.
    :param file: csv file with point coordinates stored in attributes 'coord_X' and 'coord_Y'
    :return: list of coordinates of points
    """
    try:
        with open(file) as f:
            data = pd.read_csv(f, sep=";")
            nodes = data[["coord_X", "coord_Y"]].values.tolist()
        return nodes
    except FileNotFoundError:
        print(f"Cannot open file. The file does not exist or the path to the file is incorrect.")
        quit()
    except PermissionError:
        print(f"Program doesn't have permisson to acces file.")
        quit()
    except Exception as e:
        print(f"Unexpected error opening file: {e}")
        quit()

# Calculate distance of two nodes
def distance(node1, node2):
    return sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)

# Nearest Neighbor algorithm
def nearest_neighbor(nodes : list, start_node = -1):
    # Initialize status of all points to new/unprocessed, initialize Hamiltonian cycle and its weight
    status = ['N']*len(nodes)
    cycle = []
    W = 0

    # Check if the index of starting node is in the list index, else select a random index
    if 0 <= start_node <= len(nodes)-1:
        start_index = start_node
    else: start_index = random.randrange(0, len(nodes))

    # Select starting point, set it as current node, add it to cycle and change its status to processed
    u_i = nodes[start_index]
    start = u_i
    cycle.append(u_i)
    status[start_index] = 'P'

    # While there are still unprocessed nodes left
    while 'N' in status:
        # Initialize minimum distance to infinity
        min_dist = inf

        # Iterate over unprocessed nodes
        for i in range(len(nodes)):
            if status[i] == "N":
                u = nodes[i]

                # Calculate distance of current node and other nodes and save the minimum and index of nearest neighbor
                dist = distance(u_i, u)
                if dist < min_dist:
                    min_dist = dist
                    nn = i
        
        # Add nearest neighbor to cycle, calculate W, set nn as current node and change its status
        cycle.append(nodes[nn])
        W += min_dist
        u_i = nodes[nn]
        status[nn] = 'P'
    
    # Calculate distance between last and starting point, add it to W and add starting point to the end of cycle
    end_dist = distance(start, u_i)
    W += end_dist
    cycle.append(start)

    return cycle, W

# Best Insertion algorithm
def best_insertion(nodes : list):
    # Copy all nodes to list of unprocessed nodes, initialize Hamiltonian cycle and its weight
    u_nodes = nodes.copy()
    cycle = []
    W = 0

    # Select three random starting nodes and add them to start cycle
    start = random.sample(nodes, k=3)

    # Remove starting nodes from unprocessed nodes, add them to cycle and add first point to the end of cycle
    for u in start:
        u_nodes.remove(u)
        cycle.append(u)
    start.append(start[0])

    # Calculate distance of starting cycle
    for k in range(3):
        dist_s = distance(start[k], start[k+1])
        W += dist_s
    
    while u_nodes:
        # Initialize minimum distance to infinity
        min_dist = inf

        # Select random point from uprocessed nodes
        index = random.randrange(0, len(u_nodes))
        u = u_nodes[index]

        # Iterate over all nodes in the cycle
        for i in range(len(cycle)):
            # Set index of the other node of the edge
            if i != len(cycle)-1:
                i2 = i + 1 
            else: i2 = 0

            # Calculate difference of the distance, save the minumum and indexes of nodes
            dist = distance(cycle[i], u) + distance(u, cycle[i2]) - distance(cycle[i], cycle[i2])
            if dist < min_dist:
                min_dist = dist
                best_u = index
                u_ind = i + 1
        
        # Add the best node to the cycle before the given node and calculate W
        cycle.insert(u_ind, u_nodes[best_u])
        u_nodes.remove(u_nodes[best_u])
        W += min_dist
    
    # Add starting node to the end of cycle
    cycle.append(start[0])

    return cycle, W

# Plot the Hamiltonian cycle
def plot_cycle(cycle):
    # Save X and Y coordinates of cycle´s nodes to separate lists
    x = []
    y = []
    for u in cycle:
        x.append(u[0])
        y.append(u[1])

    # Plot the cycle
    ax = plt.axes()
    ax.scatter(x, y, c='indigo')
    ax.plot(x, y, c='crimson')
    ax.set_xlabel('X')
    ax.set_ylabel('Y', rotation=0, labelpad=15)
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.show()

# Repetitions of the algorithm and results
def tsp(input_file, algorithm = 'NN', repetitions = 10, start_node = -1, plot = True, print_result = False, export_result = True):
    """
    Function of Traveling Salesman Problem.
    :param input_file: csv file with point coordinates stored in attributes 'coord_X' and 'coord_Y'
    :param algorithm: 'NN' for Nearest Neighbor (default), 'BI' for Best Insertion
    :param repetitions: int, number of repetitions of given algorithm (default 10)
    :param start_node: int, index of starting node for NN (default is set to random)
    :param plot: bool, plot the best cycle (default True)
    :param print_result: bool, print weights of the repetitions to console (default False)
    :param export_result: bool, export weights of the repetitions to csv file (default True)
    :return: best Hamilton cycle (list), weight of best cycle (float), all cycles (list), all weights (list)
    """
    # Load data into list of nodes
    nodes = load_data(input_file)

    # Create lists of repeted cycles and its weights
    cycle_rep = []
    W_rep = []
    if algorithm == 'NN':
        for i in range(repetitions):
            cycle, W = nearest_neighbor(nodes, start_node)
            cycle_rep.append(cycle)
            W_rep.append(W)
    elif algorithm == 'BI':
        for i in range(repetitions):
            cycle, W = best_insertion(nodes)
            cycle_rep.append(cycle)
            W_rep.append(W)
    else: 
        print("Use 'NN' or 'BI' for one of the algorithms.")
        quit()
    
    # Find the miminum of W and corresponding cycle
    min_W = min(W_rep)
    ind = W_rep.index(min_W)
    min_cycle = cycle_rep[ind]

    # Plot the Hamiltonian cycle with minimum weight
    if plot == True:
        plot_cycle(min_cycle)
    
    # Print results of W to the console
    if print_result == True:
        print(f"Algorithm: {algorithm}, {input_file}")
        for i in range(repetitions):
            print(f"Cycle path {i+1}: {W_rep[i]:.3f} m.")
    
    # Export resulst of W to csv file
    if export_result == True:
        with open(f"result_{repetitions}_{algorithm}_{input_file}", 'w', newline = '') as f:
            f.write(str(f"Algorithm: {algorithm}, {input_file}\n"))
            for i in range(repetitions):
                f.write(str(f"{W_rep[i]:.3f}\n"))
    
    return min_cycle, min_W, cycle_rep, W_rep

# Paths to files with input data
input_file = "ČR_obce_10_tis.csv"
#input_file = "LIB_post_office.csv"

# Loading data into list and NN and BI algorithm
# nodes = load_data(input_file)
# cycle, W = nearest_neighbor(nodes)
# cycle, W = best_insertion(nodes)

# Complex function of Traveling Salesman Problem
min_cycle, min_W, cycle_rep, W_rep = tsp(input_file, algorithm='NN', repetitions=10, plot=True, print_result=False, export_result=True)