import random
from chenchenjiang_flow.nn.core import Placeholer
from collections import defaultdict
from functools import reduce


def forward_and_backward(graph):
    # execute all the forward method of sorted_nodes
    for n in graph:
        n.forward()

    for n in graph[::-1]:
        n.backward()


def feedforward(graph):
    for i_ in graph:
        i_.forward()


def backward(graph):
    for i_ in graph[::-1]:
        i_.backward()


def toplogic(graph):
    order = []
    while graph:
        all_nodes_have_output = set(graph.keys())
        all_nodes_have_inputs = set(reduce(lambda x, y: x + y, graph.values()))
        nodes_only_have_output_no_input = all_nodes_have_output - all_nodes_have_inputs
        if nodes_only_have_output_no_input:
            n = random.choice(list(nodes_only_have_output_no_input))
            if len(graph) == 1:
                order.append(n)
                order += graph[n]
            else:
                order.append(n)
            graph.pop(n)

            for _, links in graph.items():
                if n in links:
                    links.remove(n)

        else:
            raise TypeError('This graph cannot get toplogic order, which has a circle ')
    return order


def convert_all_placeholder_to_graph(feed):
    computing_graph = defaultdict(list)
    need_expand = list(feed.keys())
    while need_expand:
        n = need_expand.pop(0)
        if n in computing_graph:
            continue
        if isinstance(n, Placeholer): n.value = feed[n]
        for m in n.outputs:
            computing_graph[n].append(m)
            need_expand.append(m)
    return computing_graph


def optimize(order, leaning_rate=1e-3):
    for node in order:
        if node.is_trainable:
            node.value = node.value + (-1) * node.gradient[node] * leaning_rate


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
