with open('追加.txt', 'a') as f:
    f.write('hello')

file = open('追加2.txt', 'a')
file.write('hello')
file.close()