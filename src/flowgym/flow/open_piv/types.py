# pyright: strict
"""Selector types and configuration parsing for the OpenPIV port.

Typed callers use the literal aliases. Parsers retain a runtime backstop for
untyped YAML/CLI input and direct Python calls, including during JIT tracing.
"""

from typing import Literal, TypeAlias, get_args

SubpixelMethod: TypeAlias = Literal["gaussian", "parabolic", "centroid"]
CorrelationMethod: TypeAlias = Literal["circular", "linear"]
SignalToNoiseMethod: TypeAlias = Literal["peak2peak", "peak2mean"]

_SUBPIXEL_METHODS: tuple[SubpixelMethod, ...] = get_args(SubpixelMethod)
_CORRELATION_METHODS: tuple[CorrelationMethod, ...] = get_args(
    CorrelationMethod
)
_SIG2NOISE_METHODS: tuple[SignalToNoiseMethod, ...] = get_args(
    SignalToNoiseMethod
)


def parse_subpixel_method(value: str) -> SubpixelMethod:
    """Parse a subpixel selector from configuration.

    Args:
        value: Method name supplied by the caller.

    Returns:
        The validated method selector.

    Raises:
        ValueError: If the value is not a supported subpixel method.
    """
    if value not in _SUBPIXEL_METHODS:
        raise ValueError(
            f"Unknown subpixel_method {value!r}; "
            f"expected one of {_SUBPIXEL_METHODS}."
        )
    return value


def parse_correlation_method(value: str) -> CorrelationMethod:
    """Parse a correlation selector from configuration.

    Args:
        value: Method name supplied by the caller.

    Returns:
        The validated method selector.

    Raises:
        ValueError: If the value is not a supported correlation method.
    """
    if value not in _CORRELATION_METHODS:
        raise ValueError(
            f"Unknown correlation_method {value!r}; "
            f"expected one of {_CORRELATION_METHODS}."
        )
    return value


def parse_sig2noise_method(value: str) -> SignalToNoiseMethod:
    """Parse a signal-to-noise selector from configuration.

    Args:
        value: Method name supplied by the caller.

    Returns:
        The validated method selector.

    Raises:
        ValueError: If the value is not a supported signal-to-noise method.
    """
    if value not in _SIG2NOISE_METHODS:
        raise ValueError(
            f"sig2noise_method not supported: {value!r}; "
            f"expected one of {_SIG2NOISE_METHODS}."
        )
    return value
