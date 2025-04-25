import torch

class ComplexModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def double(self):
        r"""Casts all floating point parameters and buffers to ``double`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        def fun(t):
            if t.is_floating_point():
                return t.double()
            elif t.is_complex():
                return t.cdouble()
            else:
                return t
        return self._apply(fun)

    def float(self):
        r"""Casts all floating point parameters and buffers to ``float`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        def fun(t):
            if t.is_floating_point():
                return t.float()
            elif t.is_complex():
                return t.cfloat()
            else:
                return t
        return self._apply(fun)

    def half(self):
        r"""Casts all floating point parameters and buffers to ``half`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        def fun(t):
            if t.is_floating_point():
                return t.half()
            elif t.is_complex():
                raise NotImplementedError
            else:
                return t
        return self._apply(fun)