import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from chenchenjiang_flow.nn.core import *
from chenchenjiang_flow.utils.utlities import *


# 创建节点
node_x = Placeholer('x', is_trainable=False)
node_k1 = Placeholer('k1', is_trainable=True)
node_b1 = Placeholer('b1', is_trainable=True)
node_linear01 = Linear('linear01', inputs=[node_x, node_k1, node_b1])
node_y_ture = Placeholer('y_true', is_trainable=False)
node_sigmoid = Sigmoid('sigmoid', inputs=[node_linear01])
node_k2 = Placeholer('k2', is_trainable=True)
node_b2 = Placeholer('b2', is_trainable=True)
node_linear_02 = Linear('linear_02', inputs=[node_k2, node_sigmoid, node_b2])
node_loss = Loss('loss', inputs=[node_linear_02, node_y_ture])
computing_graph = {
    node_k1: [node_linear01],
    node_b1: [node_linear01],
    node_x: [node_linear01],
    node_linear01: [node_sigmoid],
    node_sigmoid: [node_linear_02],
    node_k2: [node_linear_02],
    node_linear_02: [node_loss],
    node_y_ture: [node_loss]
}

graph = nx.DiGraph(computing_graph)
layout = nx.layout.spring_layout(graph)

order = toplogic(computing_graph)

color = ('c', 'red')
before, changed = color


def animate(step):
    # map_colors = [changed if node in order[:step] else before for node in graph]  # 前馈
    map_colors = [changed if node in order[::-1][:step] else before for node in graph]  # 反向传播
    print(map_colors)
    nx.draw(graph, layout, node_color=map_colors, with_labels=True)


ax = plt.gca()
fig = plt.gcf()
ani = FuncAnimation(fig, animate, interval=300)
plt.show()

# 画出排好的序的图的顺序
