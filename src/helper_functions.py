import os
import platform
import io
import sys
from pathlib import Path
import networkx as nx
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def get_machine_info():
    machine_name = platform.node()  
    user = os.getenv("USER") or os.getenv("USERNAME") 
    os_name = platform.system()  # Get os
    
    return machine_name, os_name, user

def get_paths(machine_name, os_name, user, drive_path=None):
    drive_path = "/mnt/f/TsetlinModels" if drive_path is None else drive_path
    if machine_name == "Corsair" and os_name == "Linux" and user == "jon":
        windows_drive = Path(drive_path)
        paths = {
            "data": windows_drive / "data",
            "models": windows_drive / "models",
            "graphs": windows_drive / "graphs",
            "studies": windows_drive / "studies",
        }
    else:
        paths = {
            "data": Path("data"),
            "models": Path("models"),
            "graphs": Path("graphs"),
        }
    return paths

def capture_output(func, *args, **kwargs):
    """
    Capture the printed output of a function.
    
    Parameters:
        func: The function to call whose output is to be captured.
        *args, **kwargs: Arguments to pass to the function.
    
    Returns:
        str: The captured output as a string.
    """
    # Create a StringIO object to capture output
    captured_output = io.StringIO()
    # Redirect stdout to the StringIO object
    sys.stdout = captured_output
    try:
        func(*args, **kwargs)  # Call the function
    finally:
        # Reset stdout to its original state
        sys.stdout = sys.__stdout__
    return captured_output.getvalue()

def hexagonal_layout(board_size):
    """Generate positions for a hexagonal board layout."""
    pos = {}
    for i in range(board_size):
        for j in range(board_size):
            x = j + 0.5 * (i % 2)  # Stagger every other row
            y = -i  # Flip y-axis for visualization
            pos[i * board_size + j] = (x, y)
    return pos

def visualize_parsed_graph(graph_id, nodes_data, edges_data):
    G = nx.DiGraph()  # Directed graph

    # Add nodes with attributes
    for node_id, attributes in nodes_data:
        G.add_node(node_id, **attributes)

    # Add edges with attributes
    for source, dest, edge_type in edges_data:
        G.add_edge(source, dest, edge_type=edge_type)

    # Define positions for visualization
    pos = nx.spring_layout(G)

    # Define node colors based on labels
    node_colors = ['red' if 'RED' in G.nodes[node].get('labels', []) else
                   'blue' if 'BLUE' in G.nodes[node].get('labels', []) else 'gray'
                   for node in G.nodes]

    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500)
    edge_labels = nx.get_edge_attributes(G, 'edge_type')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title(f"Graph #{graph_id}")
    plt.show()

def visualize_hex_graph(graph_id, nodes_data, edges_data, board_size):
    G = nx.DiGraph()  # Directed graph

    # Add nodes with attributes
    for node_id, attributes in nodes_data:
        G.add_node(node_id, **attributes)

    # Add edges with attributes
    for source, dest, edge_type in edges_data:
        G.add_edge(source, dest, edge_type=edge_type)

    # Generate hexagonal layout for visualization
    pos = hexagonal_layout(board_size)

    # Define node colors based on labels
    node_colors = ['red' if 'RED' in G.nodes[node].get('labels', []) else
                   'blue' if 'BLUE' in G.nodes[node].get('labels', []) else 'gray'
                   for node in G.nodes]

    # Draw the graph
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500)
    
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'edge_type')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Set graph title
    plt.title(f"Graph #{graph_id} (Hex Game)")
    plt.show()


## Functions provided by Vojtech Halenka
##draw a simple 2D graph using networkx. draw_simple_graph(obj<Graphs>*, int<graph_id>*, str<savefinename>)
def draw_simple_graph(gt, graph_id, filename='plotgraph.png'):
		colorslist =cm.rainbow(np.linspace(0, 1, len(gt.edge_type_id)))
		G = nx.MultiDiGraph()
		pos = nx.spring_layout(G)
		arc_rad = 0.2

		for node_id in range(0,gt.number_of_graph_nodes[graph_id]):
			for node_edge_num in range(0, gt.graph_node_edge_counter[gt.node_index[graph_id] + node_id]):
				edge_index = gt.edge_index[gt.node_index[graph_id] + node_id] + node_edge_num
				G.add_edge(str(node_id), str(gt.edge[edge_index][0]), weight=gt.edge[edge_index][1])

		pos = nx.spring_layout(G)
		nx.draw_networkx_nodes(G, pos)
		nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

		legend_elements=[]
		
		for k in range(len(gt.edge_type_id)):
			eset = [(u, v) for (u, v, d) in G.edges(data=True) if int(d["weight"]) == k]
			elabls = [d for (u, v, d) in G.edges(data=True) if int(d["weight"]) == k]
			le = 'Unknown'
			for ln in gt.edge_type_id.keys():
				if gt.edge_type_id[ln] == k:
					le = ln
					break
			legend_elements.append(Line2D([0], [0], marker='o', color=colorslist[k], label=le, lw=0))
			
			nx.draw_networkx_edges(G, pos, edgelist=eset, edge_color = colorslist[k], connectionstyle=f'arc3, rad = {arc_rad*k}')
		print(legend_elements)
		plt.title('Graph '+str(graph_id))
		plt.legend(handles=legend_elements, loc='upper left')
		plt.savefig(filename, dpi=300, bbox_inches='tight')

##print the nodes, seperated by (). show_graph_nodes(obj<Graphs>*, int<graph_id>*)
def show_graph_nodes(gt, graph_id):
		graphstr ='Graph#'+str(graph_id)+':\n'
		for node_id in range(gt.number_of_graph_nodes[graph_id]):
			nodestr='Node#'+str(node_id)+'('
			for (symbol_name, symbol_id) in gt.symbol_id.items():
				match = True
				for k in gt.hypervectors[symbol_id,:]:
					chunk = k // 32
					pos = k % 32

					if (gt.X[gt.node_index[graph_id] + node_id][chunk] & (1 << pos)) == 0:
						match = False

				if match:
					nodestr+= symbol_name+' '
				else:
					nodestr+= '*'+' '
			nodestr+= ')'+'\n'
			graphstr+= nodestr
		print(graphstr)

##print the edges, grouped by source node. show_graph_edges(obj<Graphs>*, int<graph_id>*)
def show_graph_edges(gt, graph_id):
	graphstr ='Graph#'+str(graph_id)+':\n'
	for node_id in range(0,gt.number_of_graph_nodes[graph_id]):
		for node_edge_num in range(0, gt.graph_node_edge_counter[gt.node_index[graph_id] + node_id]):
			edge_index = gt.edge_index[gt.node_index[graph_id] + node_id] + node_edge_num
			edgestr='Edge#'+str(edge_index)
		
			edgestr+= ' SrcNode#'+str(node_id)

			edgestr+= ' DestNode#'+str(gt.edge[edge_index][0])

			edgestr+= ' Type#'+str(gt.edge[edge_index][1])+'\n'
		
			graphstr+= edgestr
		graphstr+= '\n'
	print(graphstr)

## Functions extended from Vojtech Halenka's code       
def draw_simple_graph2(gt, graph_id, board_size=3, filename='plotgraph.png'):
    """
    Draws a graph aligned like a grid.
    
    Args:
        gt: Graph object containing graph data.
        graph_id: ID of the graph to plot.
        board_size: Size of the grid layout (e.g., 3x3).
        filename: Name of the file to save the graph plot.
    """
    colorslist = cm.rainbow(np.linspace(0, 1, len(gt.edge_type_id)))
    G = nx.MultiDiGraph()
    arc_rad = 0.2

    # 1. Add edges to the graph
    for node_id in range(0, gt.number_of_graph_nodes[graph_id]):
        for node_edge_num in range(0, gt.graph_node_edge_counter[gt.node_index[graph_id] + node_id]):
            edge_index = gt.edge_index[gt.node_index[graph_id] + node_id] + node_edge_num
            G.add_edge(str(node_id), str(gt.edge[edge_index][0]), weight=gt.edge[edge_index][1])

    # 2. Define a grid-based layout
    pos = {}
    for node_id in range(gt.number_of_graph_nodes[graph_id]):
        row = node_id // board_size  # Determine row
        col = node_id % board_size   # Determine column
        pos[str(node_id)] = (col, -row)  # Negative row to flip the grid correctly in matplotlib

    # 3. Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightblue")
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

    # 4. Draw edges with color coding
    legend_elements = []
    for k in range(len(gt.edge_type_id)):
        eset = [(u, v) for (u, v, d) in G.edges(data=True) if int(d["weight"]) == k]
        le = 'Unknown'
        for ln in gt.edge_type_id.keys():
            if gt.edge_type_id[ln] == k:
                le = ln
                break
        legend_elements.append(Line2D([0], [0], marker='o', color=colorslist[k], label=le, lw=0))
        nx.draw_networkx_edges(G, pos, edgelist=eset, edge_color=colorslist[k], connectionstyle=f'arc3,rad={arc_rad * k}')

    # 5. Add legend and title
    plt.title(f'Graph {graph_id}')
    plt.legend(handles=legend_elements, loc='upper left')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def draw_simple_graph3(gt, graph_id, board_size=5, filename='skewed_grid.png'):
    """
    Draws a graph aligned in a grid layout with skewed rows.

    Args:
        gt: Graph object containing graph data.
        graph_id: ID of the graph to plot.
        board_size: Number of nodes along one dimension of the grid.
        filename: Name of the file to save the graph plot.
    """
    colorslist = cm.rainbow(np.linspace(0, 1, len(gt.edge_type_id)))
    G = nx.MultiDiGraph()

    total_nodes = gt.number_of_graph_nodes[graph_id]
    visible_nodes = total_nodes - 4  # Exclude last 4 nodes

    # Add edges only for visible nodes
    for node_id in range(visible_nodes):
        for node_edge_num in range(gt.graph_node_edge_counter[gt.node_index[graph_id] + node_id]):
            edge_index = gt.edge_index[gt.node_index[graph_id] + node_id] + node_edge_num
            target_node = gt.edge[edge_index][0]
            # Ignore edges leading to or from the last 4 nodes
            if target_node < visible_nodes:
                G.add_edge(str(node_id), str(target_node), weight=gt.edge[edge_index][1])

    # Define positions for visible nodes
    pos = {}
    grid_rows = (visible_nodes + board_size - 1) // board_size
    skew_offset = 0.5  # Offset to skew rows to the right
    scale_factor = 2.0  # Scale the horizontal spacing

    for node_id in range(visible_nodes):
        row = node_id // board_size
        col = node_id % board_size
        pos[str(node_id)] = (col * scale_factor + row * skew_offset, -row)

    # Create the figure and adjust size
    plt.figure(figsize=(15, 10))  # Increase the width and height of the figure

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_size=1200, node_color="lightblue", edgecolors="black", )
    nx.draw_networkx_labels(G, pos, font_size=16, font_color="black", font_weight="bold")

    # Draw edges with curved style and color coding
    legend_elements = []
    for k in range(len(gt.edge_type_id)):
        eset = [
            (u, v) for (u, v, d) in G.edges(data=True) if int(d["weight"]) == k
        ]
        if not eset:  # Skip if no edges for this type
            continue
        le = "Unknown"
        for ln in gt.edge_type_id.keys():
            if gt.edge_type_id[ln] == k:
                le = ln
                break
        legend_elements.append(
            Line2D([0], [0], marker="o", color=colorslist[k], label=le, lw=0)
        )
        nx.draw_networkx_edges(G, pos, edgelist=eset, edge_color=colorslist[k], width=1.5)

    # Add title and legend
    plt.title(f"Graph {graph_id} on {board_size}x{grid_rows} Skewed Grid")
    plt.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=16)
    plt.axis("off")  # Hide axis
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
