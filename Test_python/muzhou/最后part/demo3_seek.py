file = open('c.txt', 'r')
# file.seek(4)
print(file.read())
print(file.tell())
file.close()