import heapq

"""
Takes a 2D position (x, y) and flattens it into a 1D array index
"""
def flatten(position, rows):
    return (position[1] * rows) + position[0]

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    distance = 0
    target_positions = dict()

    # We can get the "position" of a tile by using divmod(), the first value is the x coord, second is y coord.
    # Calculate target positions
    for index, value in enumerate(to_state):
        if value != 0:
            position = divmod(index, 3)[::-1] # Position of (x, y) in a 3x3 grid.
            target_positions[value] = position

    for index, value in enumerate(from_state):
        if value != 0:
            x1, y1 = target_positions[value]
            x2, y2 = divmod(index, 3)[::-1]

            distance += abs(x1 - x2) + abs(y1 - y2)

    return distance


def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    succ_states = list()

    # Loop through all the positions in the grid, calculate the positions moves from that position and enter as a state
    # A tile needs an open adjacent space (0) to generate a new state (no diagonals!).

    for index, tile in enumerate(state):
        if tile != 0:
            x1, y1 = divmod(index, 3)[::-1]
            # Extremely long lambda expression to generate viable positions to move a tile to.
            adj_coords = list(filter(lambda coord:  0 <= coord[0] <= 2 and 0 <= coord[1] <= 2 and state[flatten(coord, 3)] == 0
                                     ,[(x1 - 1, y1), (x1 + 1, y1), (x1, y1 - 1), (x1, y1 + 1)]))

            for new_pos in adj_coords:
                old_index = flatten((x1, y1), 3)
                new_index = flatten(new_pos, 3)
                new_state = state[::] # Deep copy

                new_state[old_index], new_state[new_index] = new_state[new_index], new_state[old_index]

                succ_states.append(new_state)

    return sorted(succ_states)

def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    max_len = -1
    queue = []
    visited = set()
    path = dict()
    found_solution = False # Should be always true for this homework

    # Add information for start node
    cost = get_manhattan_distance(state, goal_state)
    iteration = 0

    # Calculate the cost of the node
    # Format is (g + h, state, (g, h, parent))
    heapq.heappush(queue, (cost, state, (0, cost, -1)))

    # Dict is keyed by iteration and then valued by (state, move(g), h, iteration of parent)
    path[iteration] = (state, 0, cost, -1)
    final_node = tuple()

    while len(queue) != 0:
        max_len = max(max_len, len(queue))
        new_state = heapq.heappop(queue)
        visited.add(tuple(new_state[1]))

        if new_state[1] == goal_state:
            final_node = (new_state[2][0], new_state[2][2])
            found_solution = True
            break

        states = get_succ(new_state[1])
        iteration += 1

        # Take the popped parent and then make sure that it can be referenced for the path
        path[iteration] = (new_state[1], new_state[2][0], new_state[2][1], new_state[2][2])

        # Add the unvisited successors to the queue
        for s in states:
            g = new_state[2][0] + 1

            if tuple(s) not in visited:
                h = get_manhattan_distance(s, goal_state)
                heapq.heappush(queue, (g + h, s, (g, h, iteration)))

    # Backtrack to construct path
    if found_solution:
        current_iter = final_node[1]
        final_path = [f'{goal_state} h=0 moves: {final_node[0]}']

        while current_iter > -1:
            node = path[current_iter]
            final_path.append(f'{node[0]} h={node[2]} moves: {node[1]}')
            current_iter = node[3]

        while len(final_path) > 0:
            print(final_path.pop())
        print(f'Max queue length: {max_len}')

if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    # print_succ([2, 5, 1, 4, 0, 6, 7, 0, 3])
    # print()
    # print_succ([6, 0, 0, 3, 5, 1, 7, 2, 4])
    #
    # print(get_manhattan_distance([2, 5, 1, 4, 0, 6, 7, 0, 3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    # print()
    #
    # solve([2, 5, 1, 4, 0, 6, 7, 0, 3])
    # print()
