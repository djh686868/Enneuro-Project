class Tensor:
    ##... other methods ...
    def as_Tensor(x):
        if isinstance(x, Tensor):
            return x
        return Tensor(x)