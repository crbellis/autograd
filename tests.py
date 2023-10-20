import numpy as np
from main import Value, f, draw_dot

def t1():
    xs = np.arange(-5, 5, 0.25)
    ys = f(xs)
    # plt.plot(xs, ys)
    # plt.show()

    h = 0.0000000001
    x = -3
    slope = (f(x+h) - f(x))/h # how much the function changes
    print(slope)


def t2():
    h = 0.0000000001
    a = 2.0
    b = -3.0
    c = 10.0
    d = a*b + c
    print(d)
    d2 = (a+h)*b + c
    slope = (d2 - d)/h

    print("d1: ", d)
    print("d2: ", d2)
    print(slope)


def t3():
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'

    draw_dot(L).render('./test.gv', view=True)


def t4():
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")

    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    b = Value(6.8813735870195432, label="b")

    x1w1 = x1 * w1; x1w1.label = "x1w1"
    x2w2 = x2 * w2; x2w2.label = "x2w2"

    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1w1x2w2"

    n = x1w1x2w2 + b; n.label = "n"
    o = n.tanh(); o.label = "o"

    o.backward()

    draw_dot(o).render('./test.gv', view=True)

def t5():
    a = Value(3, label="a")
    b = a + a; b.label = "b"
    b.backward()
    draw_dot(b).render('./test.gv', view=True)