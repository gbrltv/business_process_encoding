import networkx as nx
import pm4py

def create_graph(log):
    graph = nx.Graph()

    dfg = pm4py.discover_dfg(log)
    for edge in dfg[0].items():
        graph.add_weighted_edges_from([(edge[0][0], edge[0][1], edge[1])])

    return graph
