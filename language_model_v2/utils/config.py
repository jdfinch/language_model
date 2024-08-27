
from __future__ import annotations

import dataclasses as dc
from dataclasses import dataclass
import inspect as ins
import functools as ft

import typing as T


def config(cls=None, **kwargs):
    if cls is None:
        return ft.partial(config, **kwargs)
    cls = dc.dataclass(cls)
    init = getattr(cls, '__init__', lambda self: None)
    init_sig = ins.signature(init)
    def __init__(self, *args, **kwargs):
        self.__default__ = {
            **{p.name: p.default for p in init_sig.parameters.values() if p.default is not p.empty},
            **{f.name: f.default for f in dc.fields(cls) if f.default is not dc.MISSING},
            **{f.name: f.default_factory() for f in dc.fields(cls) if f.default_factory is not dc.MISSING},
        }
        bound = init_sig.bind(self, *args, **kwargs).arguments
        self.__config__ = {k: v for i, (k, v) in enumerate(bound.items()) if i}
        init(self, *args, **kwargs) # noqa
        del self.__config__
        del self.__default__
    cls.__init__ = __init__
    return cls

class Config:
    __config__: dict[str, T.Any]
    __default__: dict[str, T.Any]


if __name__ == '__main__':

    vars().update(dataclass=config)

    @dataclass
    class A(Config):
        x: int
        y: int = 0
        z: list[str] = dc.field(default_factory=list)

        def __post_init__(self):
            print(f'{self.__config__ = }')
            print(f'{self.__default__ = }')


    @dataclass
    class B(A):
        z: list[str] = dc.field(default_factory=lambda: [1, 2, 3])

    a = A(1, y=2)
    print(f'{a = }')
    print(f'{vars(a) = }')
    b = B(8)
    print(f'{b = }')
