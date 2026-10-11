"""Map array leaves in the public CLRS feedback containers."""
from clrs._src.probing import DataPoint


def map_arrays(fn, tree, *others):
    if isinstance(tree, DataPoint):
        return DataPoint(tree.name, tree.location, tree.type_,
                         fn(tree.data, *(x.data for x in others)))
    if isinstance(tree, dict):
        return {k: map_arrays(fn, v, *(x[k] for x in others)) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        values = [map_arrays(fn, v, *(x[i] for x in others)) for i, v in enumerate(tree)]
        return type(tree)(*values) if hasattr(tree, '_fields') else type(tree)(values)
    if tree is None:
        return None
    return fn(tree, *others)
