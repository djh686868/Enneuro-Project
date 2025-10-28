class Tensor:
    ##... other methods ...
    def as_tensor(x):
        if isinstance(x, Tensor):
            return x
        return Tensor(x)