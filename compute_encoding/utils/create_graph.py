import networkx as nx
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery


def create_graph(import_path: str) -> nx.Graph:
    """
    Creates a graph using the pm4py library and converts to a networkx DiGraph

    Parameters
    -----------------------
    import_path: str,
        Path and file name to be imported
    Returns
    -----------------------
    graph: nx.DiGraph()
        A graph generated from the event log (includes edge weights based on transition occurrences)
    """
    graph = nx.Graph()

    log = xes_importer.apply(import_path, variant=xes_importer.Variants.LINE_BY_LINE)
    dfg = dfg_discovery.apply(log)
    for edge in dfg:
        graph.add_weighted_edges_from([(edge[0], edge[1], dfg[edge])])

    return graph
