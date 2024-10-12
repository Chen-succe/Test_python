import networkx as nx
import matplotlib.pyplot as pl


computing_graph = {
    'k1': ['linear_01'],
    'b1': ['linear_01'],
    'x': ['linear_01'],
    'linear_01':['loss'],
    'y_true': ['loss']
}
nx.draw(nx.DiGraph(computing_graph), with_labels=True)
pl.show()