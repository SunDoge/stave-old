import functools
from collections import OrderedDict

from jax import random as jrandom
from jax.tree_util import register_pytree_node
from ..parameter import Parameter
from jax.interpreters.xla import DeviceArray
from dataclasses import dataclass, field
from typing import Dict, List
from torch import nn


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


class Module:
    # def __init__(self):
    #     """
    #     _args, _kwargs用于存储创建参数
    #     这些参数可以在differentiable中动态添加
    #     暂时不确定是动态添加好还是静态添加好
    #     """
    #     self._args = None
    #     self._kwargs = None
    #
    #     self._parameters = OrderedDict()
    #     self._buffers = OrderedDict()
    #     self._modules = OrderedDict()
    #
    #     self.training = True

    def train(self, mode=True):
        if hasattr(self, 'training'):
            setattr(self, 'training', True)

        for k, v in self.__annotations__.items():
            if issubclass(v, Module):
                getattr(self, k).train(mode=mode)

        return self

    def eval(self):
        return self.train(mode=False)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def register_buffer(self, name: str, tensor):
        pass

    def register_parameter(self, name: str, param):
        pass

    def add_module(self, name: str, module: 'Module'):
        pass

    # def __getattr__(self, name: str):
    #     if '_parameters' in self.__dict__:
    #         _parameters = self.__dict__['_parameters']
    #         if name in _parameters:
    #             return _parameters[name]
    #     if '_buffers' in self.__dict__:
    #         _buffers = self.__dict__['_buffers']
    #         if name in _buffers:
    #             return _buffers[name]
    #     if '_modules' in self.__dict__:
    #         modules = self.__dict__['_modules']
    #         if name in modules:
    #             return modules[name]
    #     raise AttributeError("'{}' object has no attribute '{}'".format(
    #         type(self).__name__, name))
    #
    # def __setattr__(self, name, value):
    #     def remove_from(*dicts):
    #         for d in dicts:
    #             if name in d:
    #                 del d[name]
    #
    #     params = self.__dict__.get('_parameters')
    #     if isinstance(value, Parameter):
    #         if params is None:
    #             raise AttributeError(
    #                 "cannot assign parameters before Module.__init__() call")
    #         remove_from(self.__dict__, self._buffers, self._modules)
    #         self.register_parameter(name, value)
    #     elif params is not None and name in params:
    #         if value is not None:
    #             raise TypeError("cannot assign '{}' as parameter '{}' "
    #                             "(torch.nn.Parameter or None expected)"
    #                             .format(type(value), name))
    #         self.register_parameter(name, value)
    #     else:
    #         modules = self.__dict__.get('_modules')
    #         if isinstance(value, Module):
    #             if modules is None:
    #                 raise AttributeError(
    #                     "cannot assign module before Module.__init__() call")
    #             remove_from(self.__dict__, self._parameters, self._buffers)
    #             modules[name] = value
    #         elif modules is not None and name in modules:
    #             if value is not None:
    #                 raise TypeError("cannot assign '{}' as child module '{}' "
    #                                 "(torch.nn.Module or None expected)"
    #                                 .format(type(value), name))
    #             modules[name] = value
    #         else:
    #             buffers = self.__dict__.get('_buffers')
    #             if buffers is not None and name in buffers:
    #                 if value is not None and not isinstance(value, DeviceArray):
    #                     raise TypeError("cannot assign '{}' as buffer '{}' "
    #                                     "(torch.Tensor or None expected)"
    #                                     .format(type(value), name))
    #                 buffers[name] = value
    #             else:
    #                 object.__setattr__(self, name, value)

    def reset_parameters(self, rng):
        pass

    def initialize(self, seed=42, recurse=True, rng=None):
        if rng is None:
            rng = jrandom.PRNGKey(seed)
        self.reset_parameters(rng)

        if recurse:
            # for module in self._modules.values():
            for k, v in self.__annotations__.items():
                if issubclass(v, Module):
                    rng, layer_rng = jrandom.split(rng)
                    getattr(self, k).initialize(seed=seed, recurse=recurse, rng=rng)

    def deconstruct(self):
        children = {}
        aux_data = {}
        for k, v in self.__annotations__.items():
            if v is differentiable or issubclass(v, Module) or v is Parameter:
                children[k] = getattr(self, k)
            else:
                aux_data[k] = getattr(self, k)
        # print(children)
        # print(aux_data)
        return children, aux_data

    @classmethod
    def from_args(cls, *args, **kwargs):
        return cls()

    def __repr__(self):
        """
        TODO
        :return:
        """
        # return super().__repr__()
        return 'fuck'


def differentiable(cls: Module, use_dataclass=True):
    # @functools.wraps(cls)
    # def wrapper(*args, **kwargs):
    #     """
    #     Rewrite __init__ of class
    #     store args and kwargs
    #     """
    #     obj = cls(*args, **kwargs)
    #     obj._args = args
    #     obj._kwargs = kwargs
    #     return obj
    if use_dataclass:
        cls = dataclass(cls, repr=False)

    def flatten(obj: Module) -> (List[Dict], Dict):
        """
        get args, kwargs of obj as aux_data
        get params of obj as children
        return aux_data, children
        """

        children, aux_data = obj.deconstruct()

        # print('=' * 100)
        # print(obj, children)
        # print('=' * 100)
        return [children], aux_data

    def unflatten(aux_data: Dict, children: List[Dict]):
        """
        obj = cls(aux_data)
        obj.load(children)
        return obj
        """

        aux_data.update(children[0])
        # print(cls, aux_data)
        obj = cls(
            **aux_data
        )

        return obj

    register_pytree_node(
        cls,
        flatten,
        unflatten,
    )

    return cls
