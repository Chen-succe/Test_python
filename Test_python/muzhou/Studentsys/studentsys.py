# 学生信息系统
import os

filename = 'student.txt'


def main():
    while True:
        menu()
        choice = int(input('请选择：'))
        if choice in [0, 1, 2, 3, 4, 5, 6, 7]:
            if choice == 0:
                answer = input('您确定要退出系统吗(y/n)')
                if answer == 'y' or answer == 'Y':
                    print('感谢您的使用！！')
                    break
                else:
                    continue
            elif choice == 1:
                insert()
            elif choice == 2:
                search()
            elif choice == 3:
                delete()
            elif choice == 4:
                modify()
            elif choice == 5:
                sort()
            elif choice == 6:
                total()
            elif choice == 7:
                show()


def menu():
    print('********************学生信息管理系统*****************')
    print('------------------------功能菜单-------------------')
    print('\t\t\t\t\t1. 录入学生信息')
    print('\t\t\t\t\t2. 查找学生信息')
    print('\t\t\t\t\t3. 删除学生信息')
    print('\t\t\t\t\t4. 修改学生信息')
    print('\t\t\t\t\t5. 排序')
    print('\t\t\t\t\t6. 统计学生总人数')
    print('\t\t\t\t\t7. 显示所有学生信息')
    print('\t\t\t\t\t0. 退出')
    print('-------------------------------------------------')


def insert():
    student_list = []  # 学生信息列表
    while True:  # 不断询问是否继续录入。
        id = input('请输入ID(例如1001):')
        if not id:  # 如果是空id，则退出
            break
        name = input('请输出姓名:')
        if not name:  # 如果是空name，则退出
            break
        # 如果id和姓名都不为空，那么就继续录入成绩
        # 但是成绩容易录错，所以，需要用try模块
        try:
            English = int(input('请输入英语成绩:'))
            python = int(input('请输入python成绩:'))
            java = int(input('请输入Java成绩:'))

        except:
            # 这里成绩输入可能出错，所有出错情况我们都提示，输入无效。
            print('成绩输入无效， 不是整数类型，请重新输入')
            continue
            #  这里要用continue，继续录入
        student = {'id': id, 'name': name, 'English': English, 'python': python, 'java': java}
        student_list.append(student)
        answer = input('是否要继续添加(y/n)')
        if answer == 'y' or answer == 'Y':
            continue
        else:
            break

    # 退出循环后，要调用save函数，将学生信息存储到文件中。
    # 调用save()函数
    save(student_list)
    print('学生信息录入完毕！')


def save(lst):
    # 保存需要传入一个学生列表，这里可能会出错，打开文件可能会出错，以追加的方式打开，如果里面是空的，会报错
    # 但我实验，追加的情况，空文件，不会报错。
    try:
        stu_txt = open(filename, 'a', encoding='utf-8')
    except:
        stu_txt = open(filename, 'w', encoding='tuf-8')
    for item in lst:
        stu_txt.write(str(item) + '\n')  # 这里别忘记将写入的内容转成str字符串类型的。
    stu_txt.close()  # 最后， 别忘记关闭文件


def delete():
    while True:
        student_id = input('请输入要删除学生的id:')
        if student_id != '':  # 如果id不为空，则去文件中找是否有这个id
            if os.path.exists(filename):  # 如果文件不为空，才能打开
                with open(filename, 'r', encoding='utf-8') as file:  # 打开模式为d读r
                    student_old = file.readlines()
            else:
                student_old = []  # 如果文件不存在，则将这个列表设置为空就行，因为后面需要用到做判断。
            flag = False  # 标记是否删除
            if student_old:
                # 如果列表有内容
                # 则将删除后的学生信息内容重新写入到这个文件中
                # 疑问：这样操作工程量大，而且容易导致之前的信息全部被删除。
                with open(filename, 'w', encoding='utf-8') as f:
                    d = {}
                    for item in student_old:
                        d = dict(eval(item))  # 将字符串转成字典
                        if d['id'] != student_id:  # 如果和要删除的id不相等，则保留，写入新的文件中，否则，将flag设置为True
                            f.write(str(d) + '\n')
                        else:
                            flag = True
                        if flag:
                            print('id为{}的学生信息已被删除'.format(student_id))
                        else:
                            print('没有找到id为{}的学生信息'.format(student_id))
            # 上面是列表中有数据的情况，如果列表中没有数据呢，直接提示信息，并退出循环
            else:
                print('无学生信息')
                break
            show()  # 删除之后重新显示所有学生信息
            answer = input('是否继续删除y/n?')
            if answer == 'y' or answer == 'Y':
                continue
            else:
                break


def modify():
    # 先展示学生信息
    show()
    # 判断学生信息文件是否存在
    if os.path.exists(filename):
        # 文件存在的情况下，去打开文件
        with open(filename, 'r', encoding='utf-8') as file:
            student_old = file.readlines()
    else:
        # 如果文件不存在，就退出
        return
    # 文件存在的情况下，继续操作，修改学生信息
    student_id = input('请输入要修改的学员ID:')
    # 需要判断输入的内容是否为空。
    # if student_id:
    with open(filename, 'w', encoding='utf-8') as wf:
        for item in student_old:
            d = dict(eval(item))
            if d['id'] == student_id:
                print('找到学生信息，可以修改他的相关信息！')
                while True:
                    try:
                        d['name'] = input('请输入姓名')
                        d['English'] = input('请输入英语成绩')
                        d['python'] = input('请输入python成绩')
                        d['java'] = input('请输入java成绩')
                    except:
                        print('输入有误，请重新输如')
                    # 这里如果输入有误，会重新到while这里，重新输入，如果输入没有误，则会往下执行
                    # 这里，如果输入没有误，我们退出while循环，记住，这里很重要，while搭配try使用时，
                    # 一定要注意，什么时候退出while循环，只要满足条件了就可以退出循环了，所以，我们加一个else
                    else:
                        break
                        # 这里使用try……except……else模块，没有异常会执行else模块中的内容。
                # 退出循环后，我们要将内容进行写入
                wf.write(str(d) + '\n')
                print('修改成功！')
            # 这是修改成功的，那那些id不同的呢，即不需要修改的那些学生信息呢？
            # 所以，还要把原来的信息写进去
            else:
                # 这些是id和要修改的id不同的那些学生信息，这些不需要修改，重新写入新文件中就好。
                wf.write(str(d) + '\n')
        # 修改完一个学生后，我们要询问，是否还要继续修改
        answer = input('是否要继续修改其他学生信息呢y/n?')
        if answer == 'y' or answer == 'Y':
            modify()  # 如果是，要继续修改，则再次执行修改函数modify
            # 如果是否，则函数执行到这里就结束了。


def search():
    # while True:  # 第一次写这里写错了，多加了一个while True
    student_query = []
    while True:
        id = ''
        name = ''
        if os.path.exists(filename):
            # 如果文件存在，就要问你，是根据id查找还是名字查找
            mode = input('按ID查找请输入1， 按姓名查找请输入2:')
            if mode == '1':
                id = input('请输入学生ID:')
            elif mode == '2':
                name = input('请输入学生姓名:')
            else:
                # 如果输入的不是1也不是2，提示重新输入
                print('您的输入有误，请重新输入')
                search()  # 怎么重新输入呢，就是重新执行该函数。
            # 到这里，文件也存在，模式也选择了，打开文件，模式为读取
            with open(filename, 'r', encoding='utf-8') as rf:
                student = rf.readlines()
                for item in student:
                    d = dict(eval(item))
                    # 这里使用id和name不为空判断使用哪个模式，很巧妙的设置。
                    # 一开始先设置id和name都为空
                    if id != '':
                        if d['id'] == id:
                            student_query.append(d)
                    elif name != '':
                        if d['name'] == name:
                            student_query.append(d)
            # 这里就查完了，需要展示查询结果
            show_student(student_query)
            # 清空列表
            student_query.clear()  # 为什么要清空呢，因为如果要继续查询，则不会影响下一次查询。否则列表里是有东西的
            answer = input('是否要继续查询？Y/N')
            if answer == 'y' or answer == 'Y':
                continue  # 继续while True中的循环
            else:
                break
        else:
            # 如果文件存在，则可以查询，如果不存在，提示暂未保存学员。
            print('暂未保存学员')
            return


def show_student(lst):
    # 该函数是展示查询结果，是要按照格式化去显示的。
    if len(lst) == 0:  # 如果列表信息为0，说明没有查到信息
        print('没有查询到学生信息，无数据显示')
        return  # 直接return结束函数， 可以看到，一般有return，都是干函数被别的函数调用的，这种函数一般有return
    # 定义标题显示格式
    format_title = '{:^6}\t{:^12}\t{:^8}\t{:^10}\t{:^8}'  # 字符^表示居中对齐
    print(format_title.format('ID', '姓名', '英语成绩', 'python成绩', 'Java成绩', '总成绩'))
    # 定义内容的显示格式
    format_content = '{:^6}\t{:^12}\t{:^8}\t{:^10}\t{:^8}'
    for item in lst:
        print(format_content.format(item.get('id'),  # 因为是字典，可以使用字典的get方法
                                    item.get('name'),
                                    item.get('English'),
                                    item.get('python'),
                                    item.get('java'),
                                    int(item.get('English')) + int(item.get('python')) + int(item.get('java'))))
        # 总成绩这里直接计算，注意从字典取出的时候是字符串类型的，需要转成int类型。


def sort():
    # 首先调用我们的show()方法，显示所有成员信息
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as rf:
            students = rf.readlines()
        student_new = []
        for item in students:
            d = eval(item)
            student_new.append(d)
    else:
        return
    #  到这里，已经拿到了学生信息的列表格式，可以使用列表的sort函数排序了。
    asc_or_desc = input('请选择（0升序，1降序）:')
    if asc_or_desc == '0':
        asc_or_desc_bool = False
    elif asc_or_desc == '1':
        asc_or_desc_bool = True
    else:
        # 如果输入有误，则重新进入sort函数
        print('您的输入有误，请重新输入')
        sort()
    mode = input('请选择排序方式(1.按英语成绩 2.按python成绩 3.按java成绩 0.按总成绩排序):')
    # 现在就差排序操作了
    # 排序操作使用的是列表的sort函数， sort函数中有一个key和reverse
    # reverse上面也已经交代了，asc_or_desc_bool参数，False表示升序，True表示降序
    # key是排序项，这里使用匿名函数lambda，这里自己课外是补充这个知识，很好理解。
    # 获取参数x中的['English']某个键，将其转化成int，再赋给x，key就可以按照第几个位置进行排序。
    # if mode == '1':
    #     student_new.sort(key=lambda x: int(x['English']), reverse=asc_or_desc_bool)
    # elif mode == '2':
    #     student_new.sort(key=lambda x: int(x['python']), reverse=asc_or_desc_bool)
    # elif mode == '3':
    #     student_new.sort(key=lambda x: int(x['java']), reverse=asc_or_desc_bool)
    # elif mode == '0':
    #     student_new.sort(key=lambda x: int(x['English'] + x['python'] + x['java']), reverse=asc_or_desc_bool)
    if mode == '1':
        student_new.sort(key=sort_a, reverse=asc_or_desc_bool)
    elif mode == '2':
        student_new.sort(key=lambda x: int(x['python']), reverse=asc_or_desc_bool)
    elif mode == '3':
        student_new.sort(key=lambda x: int(x['java']), reverse=asc_or_desc_bool)
    elif mode == '0':
        student_new.sort(key=lambda x: int(x['English'] + x['python'] + x['java']), reverse=asc_or_desc_bool)
    else:
        print('您的输入有误，请重新输入：')
        sort()
    # 上面这些都完成，则将排序好的学生信息，进行展示
    show_student(student_new)


def sort_a(x):
    # 例如传入student_new['English']
    return int(x['English'])


def total():
    # 两种情况，文件存在和文件不存在。
    # 文件存在又有两种情况：录入了和还没有录入学生信息。
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as rf:
            students = rf.readlines()
            if students:  # 如果读出的文件有信息
                print('一共有{}名学生'.format(len(students)))
            else:  # 若读出的文件为空
                print('暂未录入学生信息')
    else:
        print('暂未保存数据')


def show():
    student_lst = []
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as rf:
            students = rf.readlines()
            '''
            # 这是我写的，下面是课程写的，很有对比。
            # # 这里缺少一步判断列表是否为空，如果为空，说明无学生信息。
            # if len(students) == 0:
            #     print('暂无学生信息')
            # else:
            #     for item in students:
            #         item_d = dict(eval(item))
            #         student_lst.append(item_d)
            #     show_student(student_lst)
            '''
            for item in students:
                student_lst.append(eval(item))
            if student_lst:  # 这样来判断列表是否为空
                show_student(student_lst)

    else:
        print('暂未录入学生信息')


# 心得： 在循环中，如果有变量或者列表等变量，要注意是否需要再函数前或者后将变量或者列表置为空。

if __name__ == '__main__':
    main()
