

from probssl.decoder.base import BaseDecoder
from probssl.decoder.linear import LinearDecoder
from probssl.decoder.mlp import MLPDecoder


DECODERS = {
    # base classes
    "base": BaseDecoder,
    # backbones
    "linear": LinearDecoder,
    "mlp": MLPDecoder,
}
__all__ = [
    "base",
    "linear",
    "mlp",
]
