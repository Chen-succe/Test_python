# coding:utf-8

class CPU:
    pass


class Disk:
    pass


class Computer:
    def __init__(self, cpu, disk):
        self.cpu = cpu
        self.disk = disk


# (1) 变量赋值
cpu1 = CPU()
cpu2 = cpu1
print(cpu1)
print(cpu2)
# 发现，这两个的地址相同，这就是赋值，虽然是两个变量，但指向的是同一个对象。

# （2） 类的浅拷贝
print('---------------------------')
disk = Disk()  # 创建一个硬盘类的对象
computer = Computer(cpu1, disk)  # 创建一个计算机类的对象

# 浅拷贝
import copy

computer2 = copy.copy(computer)  # 把computer拷贝一份给computer2
print(computer, computer.cpu, computer.disk)
print(computer2, computer2.cpu, computer2.disk)
# 打印发现，computer和computer2是不一样的，但是后面的cpu和disk都指向相同的地址。

# 深拷贝

computer3 = copy.deepcopy(computer)
print(computer, computer.cpu, computer.disk)
print(computer3, computer3.cpu, computer3.disk)
# 打印发现，三个指向的地址都不一样。是完全拷贝了一份新的，包括子对象也重新创建了新的。
