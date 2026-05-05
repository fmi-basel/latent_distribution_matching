

from probssl.methods.base import BaseMethod
from probssl.methods.kalmanSSL import KalmanSSL
from probssl.methods.qalmanSSL import QalmanSSL

METHODS = {
    # base classes
    "base": BaseMethod,
    # methods
    "kalmanSSL": KalmanSSL,
    "qalmanSSL": QalmanSSL,
}
__all__ = [
    "kalmanSSL",
    "qalmanSSL",
]
