class Animal:
    def run(self):
        print('动物在跑')


class Dog(Animal):
    def run(self):
        print('狗在跑')


class Cat(Animal):
    def run(self):
        print('猫在跑')


dog = Dog()
dog.run()
cat = Cat()
cat.run()