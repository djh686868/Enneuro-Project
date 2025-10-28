class tensor:
    ##... other methods ...
    def as_tensor(x):
        if isinstance(x, tensor):
            return x
        return tensor(x)