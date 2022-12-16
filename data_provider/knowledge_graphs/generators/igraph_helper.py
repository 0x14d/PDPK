from igraph import Graph,Vertex


def add_vertex_to_graph(graph: Graph, vertex_name: str) -> Vertex:
    """Only adds a vertex to the graph that has not yet been added to it,
    if vertex is already present return already present vertex instead

    Args:
        graph (Graph): graph to add the vertex to
        vertex_name (str): name of the vertex

    Returns:
        Vertex: vertex where vertex['name'] == vertex_name
    """
    try:
        if vertex_name in graph.vs['name']:
            return graph.vs.find(name=vertex_name)
        else:
            return graph.add_vertex(vertex_name)
    except KeyError:
        return graph.add_vertex(vertex_name)