import atexit
import functools
import os

from pyspark.sql import SparkSession

from sparkmobility import config

# One SparkSession per Python process. Building a session spins up a JVM
# and py4j gateway, which is expensive — callers like
# plot_mobility_distributions() and StayDetection's multi-step flows
# previously paid that cost on every decorated call. We cache here and
# let atexit tear it down when the interpreter exits.
_active_session: SparkSession | None = None


def _build_spark_session() -> SparkSession:
    # In PySpark local mode the driver JVM is launched by py4j before
    # spark.driver.extraJavaOptions can take effect, so -Djava.io.tmpdir
    # has to be injected via the env var the JVM reads at startup.
    # Otherwise hsperfdata lands in /tmp and warns when /tmp is full.
    tmpdir = str(config["TEMP_DIR"])
    java_opts = f"-Djava.io.tmpdir={tmpdir}"
    existing = os.environ.get("_JAVA_OPTIONS", "")
    if java_opts not in existing:
        os.environ["_JAVA_OPTIONS"] = f"{java_opts} {existing}".strip()
    os.environ.setdefault("TMPDIR", tmpdir)

    spark = (
        SparkSession.builder.master(f"local[{config['CORES']}]")
        .appName("SparkMobility")
        .config("spark.jars", config["SPARKMOBILITY_JAR"])
        .config("spark.executor.memory", f"{config['MEMORY']}g")
        .config("spark.driver.memory", f"{config['MEMORY']}g")
        .config("spark.sql.files.ignoreCorruptFiles", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.local.dir", f"{config['TEMP_DIR']}")
        .config(
            "spark.driver.extraJavaOptions", f"-Djava.io.tmpdir={config['TEMP_DIR']}"
        )
        .config(
            "spark.executor.extraJavaOptions", f"-Djava.io.tmpdir={config['TEMP_DIR']}"
        )
        .config("spark.sql.shuffle.partitions", "1000")
        .config("spark.default.parallelism", "96")
        .config("spark.shuffle.io.maxRetries", "10")
        .config("spark.shuffle.io.retryWait", "5s")
        .config("spark.reducer.maxReqsInFlight", "1")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel(config["LOG_LEVEL"])

    log4jLogger = spark._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)
    LOGGER.info("pyspark script logger initialized")
    return spark


def _shutdown_session() -> None:
    global _active_session
    if _active_session is not None:
        try:
            _active_session.stop()
        finally:
            _active_session = None


def create_spark_session() -> SparkSession:
    global _active_session
    if _active_session is None:
        _active_session = _build_spark_session()
        atexit.register(_shutdown_session)
    return _active_session


def spark_session(func):
    """Decorator that injects the cached SparkSession as the first argument.

    The session is shared across all decorated calls in a process and torn
    down at interpreter exit, so chained calls like get_stays() →
    get_home_work_od_matrix() → summarize() reuse a single JVM.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        spark = create_spark_session()
        return func(spark, *args, **kwargs)

    return wrapper
