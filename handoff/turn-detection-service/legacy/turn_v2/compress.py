"""ONNX graph ordering for the INT4 export."""

def sort_graph_with_captures(graph, outer_names=()):
    """Sort nodes including the implicit outer-scope inputs of If subgraphs."""
    import onnx
    def subgraphs(node):
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield attribute.g
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                yield from attribute.graphs
    def free_names(value):
        defined = {item.name for item in value.input} | {item.name for item in value.initializer}
        defined.update(name for node in value.node for name in node.output if name)
        referenced = {name for node in value.node for name in node.input if name}
        referenced.update(item.name for item in value.output)
        for node in value.node:
            for child in subgraphs(node):
                referenced.update(free_names(child))
        return referenced - defined
    defined = {item.name for item in graph.input} | {item.name for item in graph.initializer}
    all_names = set(outer_names) | defined | {name for node in graph.node for name in node.output if name}
    dependencies = []
    for node in graph.node:
        required = {name for name in node.input if name}
        for child in subgraphs(node):
            required.update(free_names(child))
            sort_graph_with_captures(child, all_names)
        dependencies.append((node, required))
    available, ordered = set(outer_names) | defined, []
    while dependencies:
        pending = []
        for node, required in dependencies:
            if required <= available:
                ordered.append(node)
                available.update(name for name in node.output if name)
            else:
                pending.append((node, required))
        if len(pending) == len(dependencies):
            raise ValueError("Unresolved graph inputs or cyclic graph dependencies")
        dependencies = pending
    del graph.node[:]
    graph.node.extend(ordered)
