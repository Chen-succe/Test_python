from collections import abc

__all__ = ['abc']


# 自定义类实现判断元素是否在容器内      #成员检测in功能实现
class A:
    def __init__(self, stu_list):
        self.lst = stu_list
        self.index = 0

    def __contains__(self, value):
        for v in self:
            if v is value or v == value:
                return True
        return False

    def __iter__(self):
        '''
        __iter__方法作用是返回一个迭代器对象本身。我们便了一使用迭代器对象来遍历集合中的元素。
        :return:
        '''
        return self

    def __getitem__(self, index):
        return self[index]

    def __next__(self):
        if self.index < len(self.lst):
            # 返回下一个元素
            element = self.lst[self.index]
            self.index += 1
            return element
        else:
            raise StopIteration


a = A(['lili', 'xiao'])


# print('lili' in a)
# for i in a:
#     print(i)


def test_raise():
    print('hi')
    if 'q':
        # raise Exception('出错啦')
        raise StopIteration


# test_raise()

list_a = [i for i in range(10)]
print(list_a)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
list_a[:4] = ['a', 'b', 'c']
print(list_a)  # ['a', 'b', 'c', 4, 5, 6, 7, 8, 9]

# 在列表开头插入数据
list_a[:0] = ['a', 'b']
print(list_a)  # ['a', 'b', 'a', 'b', 'c', 4, 5, 6, 7, 8, 9]

list_a[0] = ['a', 'b']
print(list_a)  # [['a', 'b'], 'b', 'a', 'b', 'c', 4, 5, 6, 7, 8, 9]

print('------------------------------')

'''
自定义支持切片的序列类
'''


# from _collections_abc
class PersonGroup:
    def __init__(self, group_name, school_name, staffs: list):
        self.group_name = group_name
        self.school_name = school_name
        self.staffs = staffs

    # 实现序列反转
    def __reversed__(self):
        return self.staffs.reverse()

    # 实现切片的关键方法。
    def __getitem__(self, item):
        cls = type(self)
        if isinstance(item, slice):
            return cls(group_name=self.group_name, school_name=self.school_name, staffs=self.staffs[item])
        return cls(group_name=self.group_name, school_name=self.school_name, staffs=[self.staffs[item]])

    def __len__(self):
        return len(self.staffs)

    def __iter__(self):
        return self

    def __contains__(self, item):
        # 成员检测方法
        if item in self.staffs:
            return True
        else:
            return False


stu_group = PersonGroup('python学习小组', '图灵学院', ['安娜', '王科', '李颖'])

# 改造后想看到具体数据要使用staffs属性。
print(stu_group[:2].staffs)
print(stu_group[::-1].staffs)
print(stu_group[:2][::-1].staffs)
print('安娜' in stu_group.staffs)

import array

my_array = array.array('i')
# my_array.append('love')
my_array.append(1)
my_array.append(2)
print(my_array)
for i in my_array:
    print(i)

my_array2 = array.array('f')
my_array2.append(2)
print(my_array2)

a = dict()
from collections.abc import Mapping, MutableMapping  # 可变的映射类型
# python中的字典是一种映射类型。
print(isinstance(a, MutableMapping))  # True
print(isinstance(a, Mapping))  # True
print(isinstance(a, dict))  # True


import copy
dict_a = {'name': 'lina'}
dict_b = dict_a.copy()
dict_c = copy.copy(dict_a)

# 不可变对象元祖也有深浅拷贝，因为元祖里面可能有列表元素。
tuple_a = (1, 2, [1,2,3])
tuple_b = copy.deepcopy(tuple_a)
tuple_b[2][0] = 0
print(tuple_a)
print(tuple_b)


# 自定义字典，使用python定义的接口UserDict
from collections import UserDict

class MyDict(UserDict):
    def __setitem__(self, key, value):
        super().__setitem__(key, value * 2)


my_dict = MyDict(one=1)
print(my_dict)
p_dict = dict(one=1)
print(p_dict)

print(type(type))  # <class 'type'>
print(type(object))  # <class 'type'>
print(object.__base__)  # None
print(MyDict.__base__)
