from typing import List, Tuple, Union
from .module import Module


class Seq(Module):

    def __init__(self, items: List[Union[Module, Tuple[str, Module]]]) -> None:
        super().__init__()
        if not isinstance(items[0], tuple):
            items = [(str(i), item) for i, item in enumerate(items)]

        for key, value in items:
            setattr(self, key, value)

        self.items = items

    def __getitem__(self, key: str) -> Module:
        return getattr(self, key)

    def forward(self, x):
        for _key, value in self.items:
            x = value(x)
        return x
