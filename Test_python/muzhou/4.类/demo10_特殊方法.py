class Person(object):
    def __new__(cls, *args, **kwargs):
        print('__new__别调用，cls 的id值为{0}'.format(id(cls)))
        obj = super().__new__(cls)  # 这一步是new创建对象，等同于类创建实例
        print('创建对象的id为{0}'.format(id(obj)))
        return obj

    def __init__(self, name, age):
        print('__init__被调用，self的id值为{0}'.format(id(self)))
        self.name = name
        self.age = age


print(id(object))
print(id(Person))
p1 = Person('san', 8)
p2 = Person('a', 3)
print(id(p1))
print(id(p2))
# 一共三个地址，object的地址，类的内存地址，以及创建的对象示例的内存地址。

