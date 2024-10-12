import turtle

p = turtle.Pen()

radius = [10, 20, 30, 40]
p.width(3)
my_color = ('red', 'green', 'yellow', 'black')

for i, j in zip(radius, range(len(radius))):
    p.penup()
    p.goto(0, -i)
    p.pendown()
    p.color(my_color[j])
    # 像下面这样写，在比如8个半径时，可以循环取这4个颜色
    # 半径对颜色数取余。
    # p.color(my_color[i % len(my_color)])
    p.circle(i)

turtle.done()
