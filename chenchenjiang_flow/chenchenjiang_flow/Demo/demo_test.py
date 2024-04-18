from chenchenjiang_flow.nn.core import *
from chenchenjiang_flow.utils.utlities import *
from sklearn.datasets import load_boston
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation

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

# 设置待喂数据
boston_data = load_boston()
x_rm, y_ = boston_data['data'][:, 5], boston_data['target']  # 使用数据中第五列，即房间数
feed_dict = {node_x: x_rm,
             node_k1: random.random(),
             node_b1: random.random(),
             node_k2: random.random(),
             node_b2: random.random(),
             node_y_ture: y_}

# 喂数据给节点
computing_graph_ = convert_all_placeholder_to_graph(feed_dict)


# 获得图的计算顺序
order = toplogic(computing_graph_)
print(order)
epoch = 100
batch_num = len(x_rm)
losses = []
for e in tqdm(range(epoch)):
    for b in range(batch_num):
        loss = 0
        index = np.random.choice(range(len(x_rm)))
        node_x.value = x_rm[index]
        node_y_ture.value = y_[index]
        forward_and_backward(order)
        optimize(order, leaning_rate=1e-3)
        loss += node_loss.value
    losses.append(loss / batch_num)

# plt.plot(losses)
# plt.show()

print(losses[-10:])


# 预测函数
def predict(x):
    node_x.value = x
    feedforward(order)
    return node_linear_02.value


for i in range(5):
    index = np.random.randint(0, 10)
    result = predict(index)
    print(result)

plt.scatter(x_rm, y_)
plt.scatter(x_rm, [predict(x_) for x_ in x_rm])
plt.show()
