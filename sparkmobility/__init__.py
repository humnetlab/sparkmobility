import logging
import os
import urllib.request
from importlib.metadata import PackageNotFoundError, version

from .logging_setup import configure_logging  # noqa: F401

logger = logging.getLogger(__name__)
# stdlib convention: ship a NullHandler so library use without
# configure_logging() stays silent rather than emitting a warning.
logger.addHandler(logging.NullHandler())

# JAR name tracks the Python package version so the two sides cannot drift.
# Fall back to the sparkmobility-scala default version if package metadata
# isn't available (running straight from source without install).
try:
    __version__ = version("sparkmobility")
except PackageNotFoundError:
    __version__ = "1.0.0"

JAR_NAME = f"sparkmobility-{__version__}.jar"
# Published by sparkmobility-scala's .github/workflows/release.yml on tag push.
JAR_URL = (
    f"https://github.com/humnetlab/sparkmobility-scala/releases/download/"
    f"v{__version__}/{JAR_NAME}"
)
JAR_PATH = os.path.join(os.path.dirname(__file__), "lib", JAR_NAME)


def ensure_jar():
    if not os.path.exists(JAR_PATH):
        logger.info("JAR not found at %s; downloading from %s", JAR_PATH, JAR_URL)
        os.makedirs(os.path.dirname(JAR_PATH), exist_ok=True)
        urllib.request.urlretrieve(JAR_URL, JAR_PATH)
        logger.info("JAR download complete")


ensure_jar()

from .settings import config, configure_env, download_and_extract  # noqa: E402, F401

download_and_extract()
configure_env()
