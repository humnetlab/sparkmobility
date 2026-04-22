import os
import tarfile
import urllib.request

from .config.load_config import load_config


class ConfigDict:
    def __init__(self):
        self._values = load_config()

    def __getitem__(self, key):
        return self._values[key]

    def __setitem__(self, key, value):
        self._values[key] = value

    def __repr__(self):
        return f"Config({self._values})"

    def as_dict(self):
        return dict(self._values)

    def reset(self):
        self._values = load_config()


config = ConfigDict()

SPARK_VERSION = "3.5.5"
SPARK_TGZ_URL = f"https://dlcdn.apache.org/spark/spark-{SPARK_VERSION}/spark-{SPARK_VERSION}-bin-hadoop3-scala2.13.tgz"
INSTALL_DIR = os.path.expanduser("~/.spark")
SPARK_DIR = os.path.join(INSTALL_DIR, f"spark-{SPARK_VERSION}-bin-hadoop3-scala2.13")


def download_and_extract():
    if not os.path.exists(SPARK_DIR):
        os.makedirs(INSTALL_DIR, exist_ok=True)
        tgz_path = os.path.join(INSTALL_DIR, f"spark-{SPARK_VERSION}.tgz")

        print("Downloading Spark...")
        urllib.request.urlretrieve(SPARK_TGZ_URL, tgz_path)

        print("Extracting Spark...")
        with tarfile.open(tgz_path) as tar:
            tar.extractall(INSTALL_DIR)

        os.remove(tgz_path)
        print("Spark installed at:", SPARK_DIR)
    else:
        print("Spark already installed.")


def configure_env():
    os.environ["SPARK_HOME"] = SPARK_DIR
    os.environ["PATH"] = f"{os.path.join(SPARK_DIR, 'bin')}:{os.environ['PATH']}"
    os.environ["PYSPARK_PYTHON"] = "python"
    os.environ["PYSPARK_DRIVER_PYTHON"] = "python"

    print("Environment variables set for current session.")
    print(
        "To make this persistent, add the following to your shell config (e.g., .bashrc):"
    )
    print(f'export SPARK_HOME="{SPARK_DIR}"')
    print('export PATH="$SPARK_HOME/bin:$PATH"')
