class Animal:
    def __init__(self, name):
        self.name = name
        # print('创建')
        print('创建一个animal实例，该实例为{}'.format(self.name))


class RunMixin:
    def run(self):
        print('{}正在跑'.format(self.name))


class SwimMixin:
    def swim(self):
        print("{}正在游泳".format(self.name))


class FlyMixin:
    def fly(self):
        print('{}正在飞'.format(self.name))


class Duck(Animal, RunMixin, SwimMixin, FlyMixin):
    # 它继承了上面所有类
    pass


duck = Duck('鸭子')
# 看一下，duck能否调用其他类的行为方法
duck.fly()
duck.swim()
duck.run()

# 由于duck继承了Animal，所以有Animal的属性self.name
# duck还继承其他几个类，所以，又可以把self.name带入到其他几个类中使用。
# 这是它的一种特有的继承方式，混合继承
'''
当前的继承方式是一种混合继承
    1. mixin 功能是单一的
    2. mixin类不继承其他类型的类（除了object）
mixin因为功能单一，并且没有复杂的继承关系，特别好管理
我们在去使用mixin的时候，尽量避免在子类中使用super。

在django-rest-framework中经常使用到混合继承，
'''


