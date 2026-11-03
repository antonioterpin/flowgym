# pyright: strict
"""Selector types and configuration parsing for the OpenPIV port.

Typed callers use the literal aliases. Parsers retain a runtime backstop for
untyped YAML/CLI input and direct Python calls, including during JIT tracing.
"""

from typing import Literal, TypeAlias, cast, get_args

SubpixelMethod: TypeAlias = Literal["gaussian", "parabolic", "centroid"]
CorrelationMethod: TypeAlias = Literal["circular", "linear"]
SignalToNoiseMethod: TypeAlias = Literal["peak2peak", "peak2mean"]


def parse_subpixel_method(value: str) -> SubpixelMethod:
    """Parse a subpixel selector from configuration.

    Args:
        value: Method name supplied by the caller.

    Returns:
        The validated method selector.

    Raises:
        ValueError: If the value is not a supported subpixel method.
    """
    options = get_args(SubpixelMethod)
    if value not in options:
        raise ValueError(
            f"Unknown subpixel_method {value!r}; expected one of {options}."
        )
    return cast(SubpixelMethod, value)


def parse_correlation_method(value: str) -> CorrelationMethod:
    """Parse a correlation selector from configuration.

    Args:
        value: Method name supplied by the caller.

    Returns:
        The validated method selector.

    Raises:
        ValueError: If the value is not a supported correlation method.
    """
    options = get_args(CorrelationMethod)
    if value not in options:
        raise ValueError(
            f"Unknown correlation_method {value!r}; expected one of {options}."
        )
    return cast(CorrelationMethod, value)


def parse_sig2noise_method(value: str) -> SignalToNoiseMethod:
    """Parse a signal-to-noise selector from configuration.

    Args:
        value: Method name supplied by the caller.

    Returns:
        The validated method selector.

    Raises:
        ValueError: If the value is not a supported signal-to-noise method.
    """
    options = get_args(SignalToNoiseMethod)
    if value not in options:
        raise ValueError(
            f"sig2noise_method not supported: {value!r}; "
            f"expected one of {options}."
        )
    return cast(SignalToNoiseMethod, value)
