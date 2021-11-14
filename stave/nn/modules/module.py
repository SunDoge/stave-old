from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from jax import numpy as jnp
from jax import random as jrandom
from jax._src.random import PRNGKey
from jax.interpreters.xla import DeviceArray, _DeviceArray
# from ..decorator import BUFFER, CONSTANT, FIELDS, MODULE, NODE_TYPE, NodeType
from dataclasses import dataclass
from stave.utils._functools import cached_property
import jax
from dataclasses import dataclass


def _addindent(s_, num_spaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(num_spaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


@dataclass(repr=False)
class _Module:

    def train(self, mode=True):
        self._train(mode=mode)

        for _key, value in self.__dict__.items():
            if isinstance(value, _Module):
                value.train(mode=mode)

        return self

    def _train(self, mode=True):
        pass

    def eval(self):
        return self.train(mode=False)

    def register_buffer(self, name: str, tensor):
        pass

    def register_parameter(self, name: str, param):
        pass

    def add_module(self, name: str, module: '_Module'):
        pass

    def _reset_parameters(self, rng: DeviceArray):
        pass

    def initialize(
        self,
        seed: int = 42,
        recurse: bool = True,
        rng: Optional[DeviceArray] = None
    ) -> DeviceArray:
        if rng is None:
            rng = jrandom.PRNGKey(seed)

        self._reset_parameters(rng)

        if recurse:
            for _key, value in self.__dict__.items():
                if isinstance(value, _Module):
                    rng, module_rng = jrandom.split(rng)
                    value.initialize(
                        seed=seed, recurse=recurse, rng=module_rng
                    )

        return rng

    def __repr__(self):
        """
        TODO
        :return:
        """
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self.__dict__.items():
            if isinstance(module, _Module):
                mod_str = repr(module)
                mod_str = _addindent(mod_str, 2)
                child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    def extra_repr(self):
        return ''

    def _get_name(self):
        return self.__class__.__name__

    def parameters(self, recurse: bool = True) -> List[Union[DeviceArray, '_Module']]:
        pass

    def named_parameters(self, prefix: str = '', recurse:  bool = True) -> Tuple[str, Union[DeviceArray, '_Module']]:
        pass


_FLOAT32 = jax.numpy.float32


def _default_init_method(key, shape, dtype):
    data = jax.random.uniform(key, shape, dtype)
    return data


InitMethod = Callable[[Any, Any, Any], DeviceArray]
Tensor = _DeviceArray

@dataclass
class Parameter:

    shape: Sequence[int] = ()
    data: Tensor = None
    init_method: InitMethod = _default_init_method
    requires_grad: bool = True

    def replace_data(self) -> DeviceArray:
        data = self.data
        self.data = None
        return data

    def init_data(self, key: DeviceArray, dtype: Any) -> DeviceArray:
        k1, k2 = jax.random.split(key)
        self.data = self.init_method(k2, self.shape, dtype)
        return k1
        

@dataclass
class Buffer(Parameter):
    requires_grad: bool = False


@dataclass
class Model:

    params: Dict[str, Parameter]
    buffers: Dict[str, Buffer]
    pure_forward: Callable

    def __iter__(self):
        yield from [self.params, self.buffers, self.pure_forward]


class Module:

    def init(self, key: Tensor = PRNGKey(42), dtype: Any = _FLOAT32):
        for name, parameter in self.named_parameters():
            key = parameter.init_data(key, dtype)

        for name, buffer in self.named_buffers():
            key = buffer.init_data(key, dtype)

        def pure_forward(parameters: Dict[str, Tensor], buffers: Dict[str, Tensor], *args, **kwargs):
            self.pack(parameters, buffers)
            output = self.forward(*args, **kwargs)
            _, new_buffers = self.unpack()
            return output, new_buffers

        parameters, buffers = self.unpack()
        return Model(parameters, buffers, pure_forward)

    def forward(self, *args, **kwargs):
        pass

    def pack(self, parameters: Dict[str, Tensor], buffers: Dict[str, Tensor]):
        for key, value in self.named_parameters():
            value.data = parameters[key]

        for key, value in self.named_buffers():
            value.data = buffers[key]

    def unpack(self):
        parameters = {k: v.replace_data() for k, v in self.named_parameters()}
        buffers = {k: v.replace_data() for k, v in self.named_buffers()}
        return parameters, buffers

    def named_parameters(self, prefix: str = '', requires_grad: bool = True):
        for key, value in self.__dict__.items():
            if isinstance(value, Module):
                for sub_key, sub_value in value.named_parameters(prefix=key + '.'):
                    yield prefix + sub_key, sub_value
            elif isinstance(value, Parameter) and value.requires_grad == requires_grad:
                yield prefix + key, value

    def named_modules(self, prefix: str = ''):
        pass

    def named_buffers(self, prefix: str = ''):
        return self.named_parameters(prefix=prefix, requires_grad=False)

    def _named_attributes(self, classes):
        for key, value in self.__dict__.items():
            if isinstance(value, Module):
                pass

