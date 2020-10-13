from dataclasses import Field

from typing import Any, List, Dict, Type, TypeVar, Tuple
from jax.tree_util import register_pytree_node
import logging


_logger = logging.getLogger(__name__)

FIELDS = '__dataclass_fields__'

NODE_KEY = 'pytree_node'

PYTREE_NODE = {
    NODE_KEY: True
}

T = TypeVar('T')
# Keys = Dict[str, List[str]]
Keys = Tuple[List[str], List[str]]
Fields = Dict[str, Field]
NodeData = Tuple[Any, ...]


def _get_keys(node: Any) -> Keys:
    """
    记录哪些变量是children，哪些是aux data
    """
    child_keys = []
    auxiliary_keys = []

    fields: Dict[str, Field] = getattr(node, FIELDS)

    for key, value in fields.items():
        if value.metadata.get(NODE_KEY, False):
            child_keys.append(key)
        else:
            auxiliary_keys.append(key)

    return child_keys, auxiliary_keys


def differentiable(cls: Type[T]) -> Type[T]:

    child_keys, auxiliary_keys = _get_keys(cls)

    def _tree_flatten(node: Any) -> Tuple[NodeData, NodeData]:
        children = tuple(getattr(node, key) for key in child_keys)
        aux_data = tuple(getattr(node, key) for key in auxiliary_keys)

        _logger.debug('=' * 50)
        _logger.debug('flatten: %s', cls)
        _logger.debug('aux_data: %s', aux_data)
        _logger.debug('children: %s', children)
        return children, aux_data

    def _tree_unflatten(aux_data: NodeData, children: NodeData) -> Any:
        _logger.debug('=' * 50)
        _logger.debug('unflatten: %s', cls)
        _logger.debug('aux_data: %s', aux_data)
        _logger.debug('children: %s', children)
        child_args = tuple(zip(child_keys, children))
        aux_args = tuple(zip(auxiliary_keys, aux_data))
        kwargs = dict(child_args + aux_args)
        return cls(**kwargs)  # type: ignore

    register_pytree_node(cls, _tree_flatten, _tree_unflatten)

    return cls
