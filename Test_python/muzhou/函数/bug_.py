def password():
    for i in range(3):
        username = input('please input your name:')
        pw = input('please input your password:')
        if i < 3:
            if username == 'a' and pw == 'a':
                print('success')
                break
            else:
                print('wrong password or username')
    else:
        print('sorry, time out')


password()


def else_solo():
    print('hello')
    for i in range(6):
        print('I am {}'.format(i))
        if i < 1:
            print(i)
        else:
            print('I am older than 1')
    else:
        print('hhh')


else_solo()
