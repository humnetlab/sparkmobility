![wheels](https://github.com/humnetlab/sparkmobility/actions/workflows/wheels.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/sparkmobility?cacheSeconds=3600)
![release](https://img.shields.io/github/v/release/humnetlab/sparkmobility?include_prereleases&cacheSeconds=3600)
![GitHub contributors](https://img.shields.io/github/contributors/humnetlab/sparkmobility?cacheSeconds=3600)


<div style="display: flex; align-items: center; justify-content: flex-start;">
  <img src="resources/sparkmobility_icon.png" width="150" style="margin-right: 20px;">
  <h1 style="margin: 0; text-align: left;">sparkmobility - A Spark-based Python Library for Processing, Modeling, and Analyzing Large Mobility Datasets</h1>
</div>


`sparkmobility` is a library for processing large mobility dataset, including Location-Based Service (LBS) using the [Apache Spark](https://spark.apache.org) framework. This Python repository serves as the main interface between `sparkmobility` and users. The Scala repository holds various data processing pipelines which can be found at [sparkmobility-scala](https://github.com/humnetlab/sparkmobility-scala).

Key features of `sparkmobility` include:

- Apache Spark-based implementation of the stay detection algorithm published by [Zheng et al. (2010)](https://dl.acm.org/doi/10.1145/1772690.1772795).

- Inference of home and work locations.

- Extract and visualize mobility metrics and patterns from the stay points output by the stay detection algorithm.

- Generate synthetic trajectories and OD flow patterns using the detected stay points and human mobility models including gravity model, rank-based EPR model, and TimeGeo.


## Table of contents
1. [Installation](#installation)
2. [Examples](#examples)
	- [Import and configure sparkmobility](#Import)
	- [MobilityDataset](#MobilityDataset)
	- [StayDetection](#StayDetection)
	- [UserSelection](#UserSelection)


<a id='installation'></a>
## Installation

`sparkmobility` requires Python 3.11+ and a Java runtime (for Spark). Install from PyPI:

```
pip install sparkmobility
```

On first import, `sparkmobility` downloads Apache Spark into `~/.spark` and fetches the matching Scala JAR from the [sparkmobility-scala GitHub Releases](https://github.com/humnetlab/sparkmobility-scala/releases). The JAR version is pinned to the installed package version (e.g. `sparkmobility-1.0.0.jar` for v1.0.0), so the Python and Scala sides cannot drift.


<a id='examples'></a>
## Examples

<a id='Import'></a>
### Import and configure `sparkmobility`

To import `sparkmobility`, simply call the following:

```python
>>> import sparkmobility as sm
>>> sm.config["CORES"] = 8
>>> sm.config["MEMORY"] = 32
>>> sm.config["LOG_LEVEL"] = "ERROR"
>>> sm.config["TEMP_DIR"] = "/my_path/to_tmp_folder"
```

On first import, you will see Spark installation + environment-variable messages printed to stdout, e.g.:

    Spark installed at: /home/<user>/.spark/spark-3.5.5-bin-hadoop3-scala2.13
    Environment variables set for current session.
    To make this persistent, add the following to your shell config (e.g., .bashrc):
    export SPARK_HOME="/home/<user>/.spark/spark-3.5.5-bin-hadoop3-scala2.13"
    export PATH="$SPARK_HOME/bin:$PATH"

Spark sessions can be configured through the `sparkmobility` configuration file. They include:

- `sm.config['CORES']` sets the number of CPU cores for the parallelism in spark ;
- `sm.config['MEMORY']` sets the amount of memory allocated for both the executor and driver in spark ;
- `sm.config['LOG_LEVEL']` sets the level of messages during compute ;
- `sm.config['TEMP_DIR']` sets the path to the directory that holds the temporary files when running the pipelines. It is important to set it to a directory that has sufficient storage in disk to prevent out of storage error.


<a id='MobilityDataset'></a>
### Initialize a `MobilityDataset`

In sparkmobility, the class `MobilityDataset` describes the mobility dataset. It does NOT hold the raw data or the detected stay points in memory but store various attributes of the dataset. The mandatory input fields include:

- `dataset_name` (type: str) ;
- `raw_data_path` (type: str) ;
- `processed_data_path` (type: str) ;
- `column_mappings` (type: dict) ;

Additionally, it is optional to define the time period and region of interests, which help reduce the computation time during the stay detection phase by selecting a subset of records:
- `start_datetime` (type: datetime) ;
- `end_datetime` (type: datetime) ;
- `longitude` (type: list);
- `latitude` (type: list);
- `time_zone` (type: str) specifies the local time zone of the region of interest.


Initialize a `MobilityDataset`:

```python
>>> from sparkmobility.dataset import MobilityDataset
>>> # create a MobilityDataset
>>> myDataset = MobilityDataset(
        dataset_name="example_dataset",
        raw_data_path="example_dataset_raw_lbs",
        processed_data_path="example_dataset_output",
        column_mappings={"caid": "caid",
                         "latitude": "latitude",
                         "longitude": "longitude",
                         "utc_timestamp": "utc_timestamp"},
        start_datetime="2019-01-01 00:00:00",
        end_datetime="2025-01-31 23:59:59",
        longitude=[-118.9448, -117.6463], # LA region
        latitude=[33.7037, 34.3373],
        time_zone="America/Los_Angeles",
    )
```

```python
>>> print(type(myDataset))
```
	<class 'sparkmobility.dataset.MobilityDataset'>


<a id='StayDetection'></a>
### Conduct `StayDetection`

`StayDetection` is a process for detecting the stay points and their respective stay duration from the raw mobility dataset, which comes in the format of (USER_ID, TIME, LNG, LAT). To call `StayDetection`:

```python
>>> from sparkmobility.processing.stay_detection import StayDetection
# Initialize the StayDetection instance
>>> stays = StayDetection(MobilityDataset=myDataset)
# Conduct stay detection
>>> stays.get_stays(hex_resolution=9)
# Compute OD flow matrix for trips between home and work locations
>>> stays.get_home_work_od_matrix(hex_resolution=7)
# Compute mobility distributions
>>> stays.summarize()
```

Argument `hex_resolution` specifies the resolution of the hexagonal grids in the output data. The output of the `StayDetection` module is automatically saved to the directory `processed_data_path` when `MobilityDataset` is first initialized. The structures are:

```
📦 processed_data_path
 ┣ 📂 HomeWorkODMatrix
 ┃ ┗ 📂 Resolution7
 ┣ 📂 Metrics
 ┃ ┣ 📂 DailyVisitedLocations
 ┃ ┣ 📂 DepartureTimeDistribution
 ┃ ┗ 📂 StayDurationDistribution
 ┣ 📂 StayPoints
 ┣ 📂 StayPointsWithHomeWork
 ┗ 📜 config.json
```

To visualize the output of `StayDetection`, we can call the following visualization functions:


#### Plot mobiilty distributions:
```python
>>> from sparkmobility.visualization.population import plot_mobility_distributions
# plot distribution of the number of daily visited locations N, the depature time t, and the stay duration delta_t
>>> fig, ax = plot_mobility_distributions(myDataset)
```

![Plot distributions](examples/mobility_distributions.png)

#### Plot home work OD flow:
```python
>>> from sparkmobility.visualization.population import plot_flow
# Load the OD flow matrix for trips between home and work locations
>>> flow_df = myDataset.load_home_work_flow(hex_resolution=7).toPandas()
# Plot the flow
>>> plot_flow(flow_df[flow_df["flow"] > 20])
```

![Plot flow](examples/home_work_flow.png)

#### Plot user trajectories:

```python
>>> import pyspark.sql.functions as F
>>> from sparkmobility.visualization.individual import plot_trajectories
# Load detected stay points
>>> stay_points = myDataset.load_stays()
# Filter for a user's trajectory
>>> stay_points = stay_points.filter(
        F.col("caid") == "0006e4cac5385960141fee505fbb73922c27309b34c45a8c5bb0bf03ace****"
    ).toPandas()
>>> plot_trajectories(stay_points)
```

![Plot trajectory](examples/sample_trajectory.png)

#### Plot distribution of home and work locations

```python
myDataset.plot_home_locations(hex_resolution=9)
```

![Plot home](examples/distribution_home_locations.png)

```python
myDataset.plot_work_locations(hex_resolution=9)
```
![Plot home](examples/distribution_work_locations.png)

<a id='UserSelection'></a>
### Select active users using `UserSelection` based on stay points

The `UserSelection` modules filters the users in the stay points dataset based on the active timespan of each user and the number of stay points detected:

- `num_stay_points_range` type(list) ;
- `time_span_days_range` type(list) ;

The method `UserSelection.filter_users` returns a visualization of the number of users by the criteria and saves the stay points of the selected users to `processed_data_path/FilteredUserStayPoints`

```python
>>> from sparkmobility.processing.user_selection import UserSelection
# Create an instance variable for the UserSelection module
>>> user_selection = UserSelection(myDataset)
# Filter users based on the number of stay points and the active timespan
>>> fig, ax = user_selection.filter_users(
        num_stay_points_range=[100, 800],
        time_span_days_range=[15, 30]
    )
```

![Plot user selection](examples/user_selection.png)


# Related packages
[*scikit-mobility*](https://github.com/scikit-mobility/scikit-mobility) is a similar package that deals with mobility datasets. *scikit-mobility* uses two data structures, trajectories (`TrajDataFrame`) and mobility flows (`FlowDataFrame`), to manage, analyze, and model human mobility data. Instead of using pandas, `sparkmobility` levearges Apache Spark. The input mobility data are structured as Resilient Distributed Dataset (RDD) during processing. The gravity and ranked-based EPR models implemented in `sparkmobility` are adopted from *scikit-mobility* implementations.
