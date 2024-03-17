class A:
    a = 1

    def __init__(self, b):
        self.b = b

    def add(self, a):
        print('计算结果为{}+{}={}'.format(self.a, a, self.a+a))



a1 = A(2)
a2 = A(3)
print(a1.a)  # 1
a1.a = 10
print(A.a)  # 1
print(a1.a)  # 10
print(a2.a)  #
A.a = 11
print(a1.a)  #
print(a2.a)  # 11
a1.add(1)
a2.add(1)