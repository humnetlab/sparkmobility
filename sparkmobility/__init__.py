import os
import urllib.request
from importlib.metadata import PackageNotFoundError, version

# JAR name tracks the Python package version so the two sides cannot drift.
# Fall back to the sparkmobility-scala default version if package metadata
# isn't available (running straight from source without install).
try:
    __version__ = version("sparkmobility")
except PackageNotFoundError:
    __version__ = "0.1.0"

JAR_NAME = f"sparkmobility-{__version__}.jar"
JAR_URL = f"https://storage.googleapis.com/sparkmobility/{JAR_NAME}"
JAR_PATH = os.path.join(os.path.dirname(__file__), "lib", JAR_NAME)


def ensure_jar():
    if not os.path.exists(JAR_PATH):
        print(f"JAR file not found at {JAR_PATH}. Downloading from GCS...")
        os.makedirs(os.path.dirname(JAR_PATH), exist_ok=True)
        urllib.request.urlretrieve(JAR_URL, JAR_PATH)
        print("Download complete.")


ensure_jar()

from .settings import config, configure_env, download_and_extract  # noqa: E402, F401

download_and_extract()
configure_env()
