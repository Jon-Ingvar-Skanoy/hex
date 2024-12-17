# graphhandler.py

from __future__ import annotations

import numpy as np
from GraphTsetlinMachine.graphs import Graphs
from enum import IntEnum

class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    UP_LEFT = 4
    UP_RIGHT = 5
    DOWN_LEFT = 6
    DOWN_RIGHT = 7

class Connection(IntEnum):
    EDGE_UP = 8
    EDGE_DOWN = 9
    EDGE_LEFT = 10
    EDGE_RIGHT = 11

class GraphHandler:
    def __init__(self, 
                paths: dict = None,
                files: dict = None,
                board_size: int = 3,
                data_array: np.array = None,
                symbols: list = None,
                hypervector_size: int = 64,
                hypervector_bits: int = 2,
                double_hashing: bool = False,
                init_with: GraphHandler = None,
                verbose_level: int = 0
                ):
        
        self.paths = paths
        self.files = files
        self.board_size = board_size
        self.data_array = data_array
        self.n_nodes = board_size**2
        self.symbols = ['RED', 'BLUE','UP', 'DOWN', 'RIGHT','LEFT'] if symbols is None else symbols

        for i in range(board_size):
            self.symbols.append(f'ROW_{i}')
            self.symbols.append(f'COL_{i}')

        self.max_index = self.n_nodes - 1
        self.right_index = self.max_index + 1
        self.left_index = self.max_index + 2
        self.down_index = self.max_index + 3
        self.up_index = self.max_index + 4

        self.hypervector_size = hypervector_size
        self.hypervector_bits = hypervector_bits
        self.double_hashing = double_hashing
        self.init_with = init_with
        self.verbose_level = verbose_level # 0: no output, 1: minimal output (TODO: tqdm), 2: verbose output, 3: debug output

        if init_with is not None:
            self.graphs = Graphs(self.data_array.shape[0], init_with=init_with.graphs)
        else:
            self.graphs = Graphs(self.data_array.shape[0],
                                symbols=self.symbols,
                                hypervector_size=self.hypervector_size,
                                hypervector_bits=self.hypervector_bits,
                                double_hashing=self.double_hashing)
 
 
    def set_n_nodes(self):
        print(f"Setting number of nodes to {self.n_nodes+4}") if self.verbose_level > 1 else None

        a_range = range(self.data_array.shape[0])
        for graph_id in a_range:
            self.graphs.set_number_of_graph_nodes(
                graph_id=graph_id,
                number_of_graph_nodes=self.n_nodes+4
            )

    def get_connections(self, index):
        print(f"Getting connections for index {index}") if self.verbose_level > 2 else None

        x = index % self.board_size
        y = index // self.board_size

        connections = []
        directions = []

        # Upper connections
        if y > 0:
            connections.append(x + (y - 1) * self.board_size)  # Directly above
            directions.append(Direction.UP)
            if y % 2 == 0 and x > 0:  # Even row: diagonal left
                connections.append(x - 1 + (y - 1) * self.board_size)
                directions.append(Direction.UP_LEFT)
            elif y % 2 == 1 and x < self.board_size - 1:  # Odd row: diagonal right
                connections.append(x + 1 + (y - 1) * self.board_size)
                directions.append(Direction.UP_RIGHT)
            else:
                connections.append(self.up_index)
                directions.append(Connection.EDGE_UP)
        else:
            connections.append(self.up_index)
            directions.append(Connection.EDGE_UP)

        # Left and right connections
        if x > 0:
            connections.append(x - 1 + y * self.board_size)
            directions.append(Direction.LEFT)
        else:
            connections.append(self.left_index)
            directions.append(Connection.EDGE_LEFT)

        if x < self.board_size - 1:
            connections.append(x + 1 + y * self.board_size)
            directions.append(Direction.RIGHT)
        else:
            connections.append(self.right_index)
            directions.append(Connection.EDGE_RIGHT)

        # Lower connections
        if y < self.board_size - 1:
            connections.append(x + (y + 1) * self.board_size)
            directions.append(Direction.DOWN)
            if y % 2 == 0 and x > 0:  # Even row: diagonal left
                connections.append(x - 1 + (y + 1) * self.board_size)
                directions.append(Direction.DOWN_LEFT)
            elif y % 2 == 1 and x < self.board_size - 1:  # Odd row: diagonal right
                connections.append(x + 1 + (y + 1) * self.board_size)
                directions.append(Direction.DOWN_RIGHT)
            else:
                connections.append(self.down_index)
                directions.append(Connection.EDGE_DOWN)
        else:
            connections.append(self.down_index)
            directions.append(Connection.EDGE_DOWN)

        return connections, directions

    def add_nodes(self):
        print("Adding nodes") if self.verbose_level > 1 else None

        self.graphs.prepare_node_configuration()
        a_range = range(self.data_array.shape[0])
        b_range = range(self.n_nodes)
        for graph_id in a_range:
            for node_id in b_range:
                nr_neighbours = len(self.get_connections(node_id)[0])
                self.graphs.add_graph_node(graph_id, node_id, nr_neighbours)
            self.graphs.add_graph_node(graph_id, self.right_index, self.board_size)
            self.graphs.add_graph_node(graph_id, self.left_index, self.board_size)
            self.graphs.add_graph_node(graph_id, self.down_index, self.board_size)
            self.graphs.add_graph_node(graph_id, self.up_index, self.board_size)

    def add_edges(self):
        print("Adding edges and properties") if self.verbose_level > 1 else None

        self.graphs.prepare_edge_configuration()
        a_range = range(self.data_array.shape[0])
        b_range = range(self.n_nodes+4)
        for graph_id in a_range:
            for node_id in b_range:
                neighbors, directions = self.get_connections(node_id)
                edge_type = 0
                if node_id < self.n_nodes:
                    for neighbor_id, dir in zip(neighbors, directions):
                        self.graphs.add_graph_node_edge(graph_id, node_id, neighbor_id, dir)
                    
                    node_value = self.data_array[graph_id, node_id]
                    if node_value == 1:
                        self.graphs.add_graph_node_property(graph_id, node_id, 'RED')
                    elif node_value == -1:
                        self.graphs.add_graph_node_property(graph_id, node_id, 'BLUE')
                    
                    row = node_id // self.board_size
                    col = node_id % self.board_size

                    self.graphs.add_graph_node_property(graph_id, node_id, f'ROW_{row}')
                    self.graphs.add_graph_node_property(graph_id, node_id, f'COL_{col}')
                
                if node_id == self.right_index:
                    neighbors = [i for i in range(self.board_size-1, self.board_size**2, self.board_size)]
                    edge_type = Connection.EDGE_RIGHT
                    for neighbor_id in neighbors:
                        self.graphs.add_graph_node_edge(graph_id, node_id, neighbor_id, edge_type)
                    self.graphs.add_graph_node_property(graph_id, node_id, 'RIGHT')
                if node_id == self.left_index:
                    neighbors = [i for i in range(0, self.board_size**2, self.board_size)]
                    edge_type = Connection.EDGE_LEFT
                    for neighbor_id in neighbors:
                        self.graphs.add_graph_node_edge(graph_id, node_id, neighbor_id, edge_type)
                    self.graphs.add_graph_node_property(graph_id, node_id, 'LEFT')
                if node_id == self.down_index:
                    neighbors = [i for i in range(self.board_size**2-self.board_size, self.board_size**2, 1)]
                    edge_type = Connection.EDGE_DOWN
                    for neighbor_id in neighbors:
                        self.graphs.add_graph_node_edge(graph_id, node_id, neighbor_id, edge_type)
                    self.graphs.add_graph_node_property(graph_id, node_id, 'DOWN')
                if node_id == self.up_index:
                    neighbors = [i for i in range(self.board_size)]
                    edge_type = Connection.EDGE_UP
                    for neighbor_id in neighbors:
                        self.graphs.add_graph_node_edge(graph_id, node_id, neighbor_id, edge_type)
                    self.graphs.add_graph_node_property(graph_id, node_id, 'UP')
        
    def encode(self):
        print("Encoding graphs") if self.verbose_level > 1 else None
        self.graphs.encode()
        
    def build_complete_graphs(self):
        self.set_n_nodes()
        self.add_nodes()
        self.add_edges()
        self.encode()
        