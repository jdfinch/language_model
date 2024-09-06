
import typing as T


R: T.TypeVar = T.TypeVar('R')


def setter(f: T.Callable[[T.Any, T.Any], R]) -> R:
    return Setter(f)  # noqa

class Setter:
    def __init__(self, f):
        self.f = f
        self.name = f.__name__
        self.__doc__ = f.__doc__

    def __set__(self, obj, value):
        obj.__dict__[self.name] = self.f(obj, value)


if __name__ == '__main__':

    import dataclasses as dc


    @dc.dataclass
    class Foo:
        x: float
        y: list | T.Iterable

        @setter
        def y(self, value):
            return list(value)


    foo = Foo(2, 'abc')
    print(foo)

    myvar = foo.y

    print(foo)
    print(type(foo.y))
