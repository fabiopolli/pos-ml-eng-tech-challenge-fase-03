import logging

import structlog

_logging_configured = False


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structlog once per process. Idempotent across ``create_app``.

    The renderer chain intentionally includes ``format_exc_info`` so
    ``logger.exception`` actually surfaces the traceback in JSON output. The
    generic exception handler in ``triage_ml.api.app`` relies on this to
    produce traceback-bearing log records for unexpected 500s.
    """

    global _logging_configured
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    if not _logging_configured:
        logging.basicConfig(format="%(message)s", level=getattr(logging, log_level))
        _logging_configured = True
