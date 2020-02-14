from typing import *
from dataclasses import dataclass
import ipdb
import typing
from jax.interpreters.xla import DeviceArray


def func(x: int):
    print(x)


# Differentiable = TypeAlias(Union)
T = TypeVar('T')


# class Differentiable(Generic[T]):
#     pass

def differentiable():
    pass


# Differentiable = Union[differentiable, T]
Differentiable = NewType('Differentiable', Union[DeviceArray, float])


@dataclass
class A:
    w1: int

    w3: Differentiable
    w2: float = func
    w4: func = None


print(A.__annotations__)
print(A.__dataclass_fields__)
# ipdb.set_trace()

a = A(1, 2, 3.)

func(a.w3)

print(A.__annotations__['w3'])
print((A.__annotations__['w4']))
w3t = A.__annotations__['w3']
print(w3t is Differentiable)
print()
