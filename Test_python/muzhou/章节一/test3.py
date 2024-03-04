a = set()
a.add(1)
a.add('2')
a.add('2')
print(a)
print(type(a))
b = ()
print(b)
print(type(b))

string_e = '23wf1'
list_e = []
number_e = 0
dict_e = {}
tuple_e = ()
tuple_e2 = tuple()
set_e = set()
print(string_e, list_e, number_e, dict_e, tuple_e, tuple_e2, set_e)

t = ('1', 1, [12, 3])
print(t)
print(min(string_e))
len(string_e)

a = 20,30,50
print(a)
a = ('a',)
print(type(a))
b = a*10
print(b)
list_a = [3,6,6,0,9,1,1,1,2,3,4, 5, 7]
a = set(list_a)
print(a)
a = [1,2,'123', (1,2)]
dict_e = {'a': 1}
b = set(dict_e)
print(b, type(b))

set_add = {1,2, (1,2,3)}
print(set_add)
set_add.add((1,2,3,4))
print(set_add)
set_add.add('1')
print(set_add)
# set_add.add(dict_e)
print(set_add)

aa = {'w', 'b', 'c'}
print(aa)
set_add.update(aa)
print(set_add)

aaa = ['a','c','g','d','4','h','m','b','e']
print(set(aaa))



