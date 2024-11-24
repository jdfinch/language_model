

def get_name_of_subclass(self, base_cls):
    for supercls in self.__class__.__mro__:
        if issubclass(supercls, base_cls):
            return supercls.__name__
    return None