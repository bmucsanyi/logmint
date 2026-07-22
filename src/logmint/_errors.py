"""Exception types raised by logmint."""


class LogmintError(Exception):
    """Base class for every error raised by logmint."""


class ConfigError(LogmintError):
    """The config cannot be canonicalised into a run identity."""


class CollisionError(LogmintError):
    """A different config already occupies this run directory."""


class AlreadyFinishedError(LogmintError):
    """The run already recorded a terminal ``finished`` status."""


class RecordError(LogmintError):
    """A record cannot be written as specified."""


class BlobError(LogmintError):
    """A blob reference is malformed or does not resolve."""


class CorpusError(LogmintError):
    """The corpus on disk violates an invariant the reader depends on."""
