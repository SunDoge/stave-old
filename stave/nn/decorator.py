from dataclasses import Field, dataclass, field
from typing import Any, List, Dict, Type, TypeVar, Tuple
from .modules import Module
from jax.tree_util import register_pytree_node
import logging


_logger = logging.getLogger(__name__)


_FIELDS = '__dataclass_fields__'
_NODE_TYPE = 'node_type'
_CHILDREN = 'children'
_AUXILIARY_DATA = 'auxiliary_data'


class NodeType:
    PARAMETER = 'Parameter'
    BUFFER = 'Buffer'
    CONSTANT = 'Constant'


PARAMETER = {
    _NODE_TYPE: NodeType.PARAMETER
}

BUFFER = {
    _NODE_TYPE: NodeType.BUFFER
}

CONSTANT = {
    _NODE_TYPE: NodeType.CONSTANT
}

T = TypeVar('T')
Keys = Dict[str, List[str]]


def _get_keys(node: Any) -> Keys:
    """
    记录哪些变量是children，哪些是aux data
    """
    keys: Keys = {
        _CHILDREN: [],
        _AUXILIARY_DATA: [],
    }

    fields: Dict[str, Field] = getattr(node, _FIELDS)

    for key, value in fields.items():
        node_type = value.metadata.get(_NODE_TYPE, NodeType.CONSTANT)
        if node_type == NodeType.PARAMETER or node_type == NodeType.BUFFER:
            keys[_CHILDREN].append(key)
        else:
            keys[_AUXILIARY_DATA].append(key)

    return keys


def differentiable(cls: Type[T]) -> Type[T]:

    keys = _get_keys(cls)

    def _tree_flatten(node: Module) -> Tuple[Tuple[Dict[str, Any]], Dict[str, Any]]:
        children = {}
        aux_data = {}
        for key in keys[_CHILDREN]:
            children[key] = getattr(node, key)

        for key in keys[_AUXILIARY_DATA]:
            aux_data[key] = getattr(node, key)

        _logger.debug('=' * 50)
        _logger.debug('flatten: %s', cls)
        _logger.debug('aux_data: %s', aux_data)
        _logger.debug('children: %s', children)
        return (children,), aux_data

    def _tree_unflatten(aux_data: Tuple[Dict[str, Any]], children: Dict[str, Any]) -> Module:
        _logger.debug('=' * 50)
        _logger.debug('unflatten: %s', cls)
        _logger.debug('aux_data: %s', aux_data)
        _logger.debug('children: %s', children)
        return cls(**aux_data, **children[0])  # type: ignore

    register_pytree_node(cls, _tree_flatten, _tree_unflatten)

    return cls
