from typing import Iterator, Iterable


class A:
    def __init__(self, stu_list):
        self.stu_list = stu_list
        self.count = 0

    # def __iter__(self):
    #     return self

    def __next__(self):
        if self.count <= len(self.stu_list) - 1:
            tem = self.stu_list[self.count]
            self.count += 1
            return tem
        else:
            # return  修改为下面语句
            raise StopIteration

    # def __getitem__(self, item):
    #     return self.stu_list[item]


a = A(['a', 'b', 'c'])
# a = iter(a)
a = next(a)

print(isinstance(a, Iterator))  # True
print(isinstance(a, Iterable))  # True
# print(next(a))
# print(next(a))
# print(next(a))
# print(next(a))
for i in a:
    print(i)

tp = tuple((1, 2, 3, 4))

print(tp)
tp = iter(tp)
for i in tp:
    print(i)

dt = {'a': '1', 'b': 2, 'c': 3}
dt = iter(dt)
for i in dt:
    print(i)

