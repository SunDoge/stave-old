from dataclasses import dataclass
from typing import List, Dict
from .modules import Module
from jax.tree_util import register_pytree_node


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
