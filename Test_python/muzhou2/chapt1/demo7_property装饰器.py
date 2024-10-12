class Person:
    def __init__(self, name, salary):
        self.name = name
        self.__salary = salary

    def show(self):
        print('员工{}的年薪为:{}'.format(self.name, self.__salary))

    @property
    def salary(self):
        print('员工{}的年薪为:{}'.format(self.name, self.__salary))
        return self.__salary

    @salary.setter
    def salary(self, num):
        self.__salary = num


p1 = Person('chen', '70w')
# p1.show()
# # print(p1.__salary)  # 报错
# print(dir(p1))
# print(p1._Person__salary)  # 70w
salary = p1.salary
p1.salary = '80w'
salary = p1.salary
