import math
import random
from typing import Literal, Tuple, Set, Union, List, Callable, Sequence
from graphviz import Digraph


class Value:
    def __init__(
        self,
        data: Union[int, float],
        _children: Union[Tuple['Value', ...], Tuple[()]] = (),
        _op: str = '',
        label: str = ""
    ) -> None:
        self.data: Union[int, float] = data
        self.grad: float = 0.0
        self._prev: Set['Value'] = set(_children)
        self._op = _op
        self.label = label
        self._backward: Callable[[], None] = lambda: None
    
    def __repr__(self) -> str:
        return f"Value[data={self.data}]"

    def __add__(self, other: Union['Value', int, float]) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out
    
    def __neg__(self) -> 'Value':
        return self * -1
    
    def __sub__(self, other: Union['Value', int, float]) -> 'Value':
        return self + (-other)

    def __radd__(self, other: Union['Value', int, float]) -> 'Value': 
        return self.__add__(other)

    def __mul__(self, other: Union['Value', int, float]) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Union['Value', int, float]) -> 'Value':
        return self.__mul__(other)
    
    def __truediv__(self, other: Union['Value', int, float]) -> 'Value':
        return self * other**-1

    def __pow__(self, other: Union['Value', int, float]) -> 'Value':
        assert isinstance(other, (int, float)), "exponent must be a scalar"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward() -> None:
            self.grad += other * self.data**(other - 1) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> 'Value':
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), "tanh")
        def _backward() -> None:
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def exp(self) -> 'Value':
        x = self.data
        out = Value(math.exp(x), (self,), "exp")
        def _backward() -> None:
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def backward(self) -> None:
        topo = []
        visited = set()
        def build_topo(v: 'Value') -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


class Neuron:
    def __init__(self, n_in: int):
       self.w: List[Value] = [Value(random.uniform(-1, 1)) for _ in range(n_in)] 
       self.b: Value = Value(random.uniform(-1, 1))

    def __call__(self, x: Sequence[Union[Value, float]]) -> Value:
        # w * x + b
        act: Value = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self) -> List[Value]:
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in: int, n_out: int) -> None:
        self.neurons = [Neuron(n_in) for _ in range(n_out)]
    
    def __call__(self, x: Sequence[Union[Value, float]]) -> List[Value]:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self) -> List[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, n_in: int, n_outs: List[int]) -> None:
        sz = [n_in] + n_outs
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(n_outs))]

    def __call__(self, x: Sequence[Union[Value, float]]) -> Union[Sequence[Union[Value, float]], Value]:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self) -> List[Value]:
        return [p for layer in self.layers for p in layer.parameters()]


def trace(root: Value) -> Tuple[Set[Value], Set[Tuple[Value, Value]]]:
    # builds a set of all nodes and edges in a graph
    nodes: Set[Value] = set()
    edges: Set[Tuple[Value, Value]] = set()
    def build(v: Value) -> None:
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def draw_dot(root: Value) -> Digraph:
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
    
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(
            name = uid, 
            label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
            shape='record'
        )
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name = uid + n._op, label = n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


def f(x: Union[float, int]) -> float:
    return 3*x**2 - 4*x + 5 # -> 6x - 4


def MSE(y_true: List[float], y_pred: List[Sequence[Union[Value, float]]]) -> Union[Value, int]:
    return sum([(y_p - y_t)**2 for y_t, y_p in zip(y_true, y_pred)])


def train_ex() -> None:
    n = MLP(3, [4, 4, 1])
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    # view computational graph 
    # draw_dot(n(xs[0])).render('./test.gv', view=True)
    print("Num of params: ", len(n.parameters()))

    lr = 0.05
    # epochs
    for _ in range(100):
        y_pred = [n(x) for x in xs]
        print(y_pred)
        # forward pass
        loss = MSE(ys, y_pred) 

        # reset gradients
        for p in n.parameters():
            p.grad = 0

        # backward pass
        loss.backward()

        # udpate
        for p in n.parameters():
            p.data -= lr * p.grad

        print("MSE: ", loss.data)
    print(y_pred) 

    # display backprop
    draw_dot(loss).render('./test.gv', view=True)


def main() -> None:
    train_ex()


if __name__ == '__main__':
    main()