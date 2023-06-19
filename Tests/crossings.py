import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = '16'
plt.rcParams['figure.dpi'] = '200'
plt.rcParams['backend'] = 'QtAgg'

# def find_coarsest_grid_size(lines):
#     # Find the min and max values of x and y coordinates in the lines
#     x_min = np.min(lines[:, [0, 2]])
#     x_max = np.max(lines[:, [0, 2]])
#     y_min = np.min(lines[:, [1, 3]])
#     y_max = np.max(lines[:, [1, 3]])
#
#     # Compute the distance between x and y min and max values
#     x_dist = x_max - x_min
#     y_dist = y_max - y_min
#
#     # Initialize the minimum grid size to the maximum of x_dist and y_dist
#     min_grid_size = max(x_dist, y_dist)
#
#     # Check each line segment to find the minimum distance between any two points
#     for line in lines:
#         x1, y1, x2, y2 = line
#         dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#         if dist < min_grid_size:
#             min_grid_size = dist
#
#     # Return the minimum grid size rounded up to the nearest integer
#     return int(np.ceil(min_grid_size))
#
#
# def lines_to_binary_matrix_grid(lines, grid_size=1):
#     x_min = np.min(lines[:, [0, 2]])
#     x_max = np.max(lines[:, [0, 2]])
#     y_min = np.min(lines[:, [1, 3]])
#     y_max = np.max(lines[:, [1, 3]])
#     width = x_max - x_min
#     height = y_max - y_min
#     num_cols = int(np.ceil(width / grid_size))
#     num_rows = int(np.ceil(height / grid_size))
#     binary_matrix = np.zeros((num_rows, num_cols), dtype=np.uint8)
#     for line in lines:
#         x1, y1, x2, y2 = line
#         x1_idx = int(np.floor((x1 - x_min) / grid_size))
#         y1_idx = int(np.floor((y1 - y_min) / grid_size))
#         x2_idx = int(np.floor((x2 - x_min) / grid_size))
#         y2_idx = int(np.floor((y2 - y_min) / grid_size))
#         for i, j in zip(np.linspace(x1_idx, x2_idx, max(abs(x2_idx - x1_idx), abs(y2_idx - y1_idx)) + 1, dtype=int),
#                         np.linspace(y1_idx, y2_idx, max(abs(x2_idx - x1_idx), abs(y2_idx - y1_idx)) + 1, dtype=int)):
#             if i < 0 or i >= num_cols or j < 0 or j >= num_rows:
#                 continue
#             binary_matrix[j, i] = 1
#     return binary_matrix


def lines_to_binary_matrix(lines):
    x_min = np.min(lines[:, [0, 2]])
    x_max = np.max(lines[:, [0, 2]])
    y_min = np.min(lines[:, [1, 3]])
    y_max = np.max(lines[:, [1, 3]])
    num_cols = x_max - x_min
    num_rows = y_max - y_min
    binary_matrix = np.zeros((num_rows, num_cols), dtype=np.uint8)
    for line in lines:
        x1, y1, x2, y2 = line
        x1_idx = x1 - x_min
        y1_idx = y1 - y_min
        x2_idx = x2 - x_min
        y2_idx = y2 - y_min
        for i, j in zip(np.linspace(x1_idx, x2_idx, max(abs(x2_idx - x1_idx), abs(y2_idx - y1_idx)) + 1, dtype=int),
                        np.linspace(y1_idx, y2_idx, max(abs(x2_idx - x1_idx), abs(y2_idx - y1_idx)) + 1, dtype=int)):
            if i < 0 or i >= num_cols or j < 0 or j >= num_rows:
                continue
            binary_matrix[j, i] = 1
    return binary_matrix


def count_crossings(lines):
    min_crossings_for_each_line = np.zeros(len(lines))
    binary_matrix = lines_to_binary_matrix(lines)
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line
        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)

        try:
            line_matrix = lines_to_binary_matrix(np.array([line]))
            line_matrix = line_matrix[1:-1, 1:-1]
            sub_matrix = binary_matrix[y_min + 1:y_max - 1, x_min + 1:x_max - 1]

            line_matrix_down = np.zeros_like(line_matrix)
            line_matrix_down[1:, :] = line_matrix[0:-1, :]
            crossings_down = line_matrix_down * ((sub_matrix - line_matrix) + 1) // 2

            line_matrix_right = np.zeros_like(line_matrix)
            line_matrix_right[:, 1:] = line_matrix[:, 0:-1]
            crossings_right = line_matrix_right * ((sub_matrix - line_matrix) + 1) // 2


            min_crossings_for_each_line[i] = min(np.sum(crossings_down),np.sum(crossings_right))

        except:
            continue
        print(i, min_crossings_for_each_line[i])

    return min_crossings_for_each_line


def get_edge_positions(graph):
    # Obtain the Kamada-Kawai layout
    layout = nx.kamada_kawai_layout(graph)

    # Initialize the array to hold the edge positions
    edge_positions = np.zeros((graph.number_of_edges(), 4))

    # Iterate over the edges and add their positions to the array
    for i, edge in enumerate(graph.edges()):
        # Get the positions of the two nodes connected by the edge
        x1, y1 = layout[edge[0]]
        x2, y2 = layout[edge[1]]

        # Add the positions to the edge positions array
        edge_positions[i] = [x1, y1, x2, y2]

    return edge_positions


# %%
N_lines = 150
lines = np.random.uniform(0, 1, (N_lines, 4)) * N_lines * 10
# lines = lines * 100
lines = lines.astype('int')

# lines = np.array([[7,0,7,10],
#                  [5,5,15,5]])
# binary_matrix = lines_to_binary_matrix_grid(lines,grid_size=1)
binary_matrix = lines_to_binary_matrix(lines)

fig, ax = plt.subplots()
ax.imshow(binary_matrix, cmap='Greys')
fig.show()

# %%
line_index = 1
line = lines[line_index, :]
x1, y1, x2, y2 = line
x_min = min(x1, x2)
x_max = max(x1, x2)
y_min = min(y1, y2)
y_max = max(y1, y2)

line_matrix = lines_to_binary_matrix(np.array([line]))
line_matrix = line_matrix[1:-1,1:-1]
sub_matrix = binary_matrix[y_min+1:y_max-1, x_min+1:x_max-1]

# line_matrix = lines_to_binary_matrix(np.array([line]))
# sub_matrix = binary_matrix[y_min:y_max, x_min:x_max]

line_matrix_down = np.zeros_like(line_matrix)
line_matrix_down[1: ,:] = line_matrix[0:-1, :]

line_matrix_right = np.zeros_like(line_matrix)
line_matrix_right[:, 1:] = line_matrix[:, 0:-1]

# sub_matrix = binary_matrix[int(y_min * 1/grid_size):int(y_max * 1/grid_size), int(x_min * 1/grid_size):int(x_max * 1/grid_size) ]

fig, ax = plt.subplots()
ax.imshow(sub_matrix, cmap='Greys')
fig.show()
fig, ax = plt.subplots()
ax.imshow(line_matrix, cmap='Greys')
fig.show()

# %%
crossings_down = line_matrix_down*((sub_matrix -line_matrix)+1)//2
crossings_right = line_matrix_right*((sub_matrix -line_matrix)+1)//2

fig, ax = plt.subplots()
ax.imshow(crossings_down, cmap='Greys')
fig.show()
fig, ax = plt.subplots()
ax.imshow(crossings_right, cmap='Greys')
fig.show()

print(np.sum(crossings_down))
print(np.sum(crossings_right))

#%%
min_crossings_for_each_line = count_crossings(lines)
print(np.sum(min_crossings_for_each_line))
plt.plot(min_crossings_for_each_line)
plt.show()
#%%
import networkx as nx
import matplotlib.pyplot as plt

N = 100
# Create the cycle graph
G = nx.cycle_graph(N)

# Add chord edges
for i in range(0, N, 2):
    for j in range(i+3, i+10-3, 2):
        G.add_edge(i, j % 10)

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)

# Show the plot
plt.show()