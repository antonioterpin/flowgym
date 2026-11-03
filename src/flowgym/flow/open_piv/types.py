# pyright: strict
"""Method selector types for the OpenPIV port."""

from typing import Literal, TypeAlias

SubpixelMethod: TypeAlias = Literal["gaussian", "parabolic", "centroid"]
CorrelationMethod: TypeAlias = Literal["circular", "linear"]
SignalToNoiseMethod: TypeAlias = Literal["peak2peak", "peak2mean"]
