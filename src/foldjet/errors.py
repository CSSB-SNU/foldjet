"""Public exceptions that callers can catch without importing inference backends."""


class FoldJetError(Exception):
    """Base class for FoldJet lifecycle and input errors."""


class InputError(FoldJetError, ValueError):
    """An input or per-call sampling option is invalid."""


class ModelClosedError(FoldJetError, RuntimeError):
    """A closed predictor cannot execute inference."""


class ConcurrentCallError(FoldJetError, RuntimeError):
    """A predictor or trunk state is already serving another operation."""


class TrunkStateError(FoldJetError, ValueError):
    """A trunk state is closed or belongs to another predictor."""
