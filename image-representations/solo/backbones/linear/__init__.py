

from .linear import Linear

def linear(method, *args, **kwargs):
    return Linear(kwargs["in_features"], kwargs["out_features"], kwargs.get("positive_weights", False))

__all__ = ["linear"]
