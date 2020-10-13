from typing import Any, Dict, List, Optional, Tuple, Union

from jax import numpy as jnp
from jax import random as jrandom
from jax.interpreters.xla import DeviceArray
# from ..decorator import BUFFER, CONSTANT, FIELDS, MODULE, NODE_TYPE, NodeType
from dataclasses import dataclass
from stave.utils._functools import cached_property


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
class Module:

    def train(self, mode=True):
        self._train(mode=mode)

        for _key, value in self.__dict__.items():
            if isinstance(value, Module):
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

    def add_module(self, name: str, module: 'Module'):
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
                if isinstance(value, Module):
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
            if isinstance(module, Module):
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

    def parameters(self, recurse: bool = True) -> List[Union[DeviceArray, 'Module']]:
        pass

    def named_parameters(self, prefix: str = '', recurse:  bool = True) -> Tuple[str, Union[DeviceArray, 'Module']]:
        pass
