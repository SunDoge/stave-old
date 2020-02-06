import functools
from jax.tree_util import register_pytree_node
from collections import OrderedDict


class Module:

    def __init__(self):
        """
        _args, _kwargs用于存储创建参数
        这些参数可以在differentiable中动态添加
        暂时不确定是动态添加好还是静态添加好
        """
        self._args = None
        self._kwargs = None

        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._modules = OrderedDict()

        self.training = True

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

    def __getattr__(self, name: str):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))


def differentiable(cls: Module):
    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        """
        Rewrite __init__ of class
        store args and kwargs
        """
        obj = cls(*args, **kwargs)
        obj._args = args
        obj._kwargs = kwargs
        return obj

    def flatten(obj: Module):
        """
        get args, kwargs of obj as aux_data
        get params of obj as children
        return aux_data, children
        """
        # aux_data = obj
        # aux_data = None
        # children = obj.parameters()
        # children = [list(obj._parameters.values())]
        aux_data = (obj._args, obj._kwargs)
        children = [obj._parameters, obj._modules]

        # print('=' * 100)
        # print(obj, children)
        # print('=' * 100)
        return children, aux_data

    def unflatten(aux_data, children):
        """
        obj = cls(aux_data)
        obj.load(children)
        return obj
        """
        args, kwargs = aux_data
        # print(cls, aux_data)
        obj = wrapper(*args, **kwargs)
        obj._parameters = children[0]
        obj._modules = children[1]
        return obj

    register_pytree_node(
        cls,
        flatten,
        unflatten,
    )

    return wrapper
