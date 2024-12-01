

def get_name_of_subclass(self, base_cls):
    for supercls in self.__class__.__mro__:
        if issubclass(supercls, base_cls):
            name = supercls.__name__
            if name.endswith('Config'):
                name = name[:-len('Config')]
            return name
    return None