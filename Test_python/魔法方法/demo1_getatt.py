class A:
    def __init__(self):
        self.data = 'abc'
        self.count = 0

    def __getattribute__(self, item):
        if item == 'data':
            self.count += 1
        # return super().__getattribute__(item)
        return object.__getattribute__(self, item)  # 等价于上面这句


o = A()
b = o.data
a = o.data
print(o.count)


class B(A):
    def __init__(self):
        # object.__init__(self)  # 不行
        super(B, self).__init__()


o2 = B()
print(o2)
print(o2.data)


class C:
    _attr = {}

    def __init__(self):
        self.data = 'abc'

    def __getattr__(self, item):
        if item not in self._attr:
            raise AttributeError
        return self._attr[item]

    def __setattr__(self, key, value):
        self._attr[key] = value


oc = C()
ooc = C()
print(oc.data)
# print(oc.test)
oc.data = 'xyz'
print(ooc.data)