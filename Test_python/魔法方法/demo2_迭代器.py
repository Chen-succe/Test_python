class A:
    def __init__(self, n_ls):
        self.name_list = n_ls

    # def __next__(self):
    #     pass

    # def __iter__(self):
    #     return self

    def __getitem__(self, item):
        return self.name_list[item]


a = A(['a', 'n', 'c'])
iter_a = iter(a)
# print(next(iter_a))
# print(next(iter_a))
for i in iter_a:
    print(i)


class B:
    def __init__(self, n_ls):
        self.name_list = n_ls
        self.index = 0

    def __next__(self):
        if self.index >= len(self.name_list):
            raise StopIteration
        else:
            value = self.name_list[self.index]
            self.index += 1
            return value

    def __iter__(self):
        return self

    # def __getitem__(self, item):
    #     return self.name_list[item]


b = B(['a', 'n', 'c'])
iter_b = iter(b)
# print(next(iter_b))
# print(next(iter_b))
for i in iter_b:
    print(i)