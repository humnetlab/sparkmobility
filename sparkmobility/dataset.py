import geopandas as gpd
import h3
import h3spark as h3F
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from shapely.geometry import Polygon

from sparkmobility.utils.session import create_spark_session
from sparkmobility.visualization.population import plot_count


class MobilityDataset:
    def __init__(
        self,
        dataset_name,
        raw_data_path,
        column_mappings,
        processed_data_path=None,
        start_datetime="2020-01-01 00:00:00",
        end_datetime="2030-12-31 23:59:59",
        longitude=None,
        latitude=None,
        time_zone="America/Mexico_City",
        time_format="UNIX",
    ):
        self.dataset_name = dataset_name
        self.input_path = raw_data_path
        if processed_data_path is None:
            processed_data_path = f"{dataset_name}_processed"
        self.output_path = processed_data_path
        self.column_names = column_mappings
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.longitude = longitude
        self.latitude = latitude
        self.time_zone = time_zone
        self.time_format = time_format
        self.df = None

    def load_stays(self):
        try:
            df = create_spark_session().read.parquet(
                self.output_path + "/StayPointsWithHomeWork"
            )
        except Exception as e:
            print(f"Error loading stays: {e}")
            return None
        return df

    def load_stay_count(self, hex_resolution):
        df = self.load_stays()
        df = self.h3_to_parent(df, hex_resolution, h3_index_col_name="h3_index")
        df = df.groupby(["h3_index_res" + str(hex_resolution)]).count().toPandas()
        df["h3_index_res" + str(hex_resolution)] = df[
            "h3_index_res" + str(hex_resolution)
        ].apply(lambda x: h3.api.basic_int.int_to_str(x))

        df["geometry"] = df["h3_index_res" + str(hex_resolution)].apply(
            lambda ix: Polygon([(lon, lat) for (lat, lon) in h3.cell_to_boundary(ix)])
        )
        stay_count = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
        return stay_count

    def load_home_work_flow(self, hex_resolution):
        try:
            df = create_spark_session().read.parquet(
                self.output_path + "/HomeWorkODMatrix/Resolution" + str(hex_resolution)
            )
        except Exception as e:
            print(f"Error loading home-work flow data: {e}")
            return None
        return df

    def load_od_flow(self, hex_resolution):
        df = self.load_stays()
        df = self.h3_to_parent(df, hex_resolution, h3_index_col_name="h3_index")
        w = Window.partitionBy(F.col(self.column_names["caid"])).orderBy(
            F.col("local_time")
        )
        stays_lagged = df.withColumn(
            "h3_index_prev", F.lag(F.col("h3_index_res" + str(hex_resolution))).over(w)
        )
        od_flow = (
            stays_lagged.na.drop(
                subset=["h3_index_prev", "h3_index_res" + str(hex_resolution)]
            )
            .groupBy("h3_index_prev", "h3_index_res" + str(hex_resolution))
            .count()
            .withColumnRenamed("count", "flow")
            .withColumnRenamed("h3_index_prev", "origin")
            .withColumnRenamed("h3_index_res" + str(hex_resolution), "destination")
        )

        od_flow = (
            od_flow.withColumn("lat_lng_origin", h3F.cell_to_latlng(F.col("origin")))
            .withColumn("lat_lng_destination", h3F.cell_to_latlng(F.col("destination")))
            .withColumn(
                "distance",
                self.haversine(
                    F.col("lat_lng_origin.lat"),
                    F.col("lat_lng_origin.lon"),
                    F.col("lat_lng_destination.lat"),
                    F.col("lat_lng_destination.lon"),
                ),
            )
            .drop("lat_lng_origin", "lat_lng_destination")
            .toPandas()
        )

        od_flow["origin"] = od_flow["origin"].apply(
            lambda x: h3.api.basic_int.int_to_str(x)
        )
        od_flow["destination"] = od_flow["destination"].apply(
            lambda x: h3.api.basic_int.int_to_str(x)
        )

        return od_flow

    def plot_home_locations(self, hex_resolution):
        df = self.load_stays()
        home_loc = (
            df.groupBy("caid")
            .agg({"home_h3_index": "first"})
            .withColumnRenamed("first(home_h3_index)", "home_h3_index")
        )
        home_pd = home_loc.groupBy("home_h3_index").count().toPandas()

        home_pd = home_pd.dropna()
        home_pd = home_pd[home_pd["home_h3_index"] != "None"].reset_index(drop=True)

        return plot_count(home_pd, "home_h3_index", hex_resolution)

    def plot_work_locations(self, hex_resolution):
        df = self.load_stays()
        work_loc = (
            df.groupBy("caid")
            .agg({"work_h3_index": "first"})
            .withColumnRenamed("first(work_h3_index)", "work_h3_index")
        )
        work_pd = work_loc.groupBy("work_h3_index").count().toPandas()

        work_pd = work_pd.dropna()
        work_pd = work_pd[work_pd["work_h3_index"] != "None"].reset_index(drop=True)

        return plot_count(work_pd, "work_h3_index", hex_resolution)

    @staticmethod
    def h3_to_parent(df, hex_resolution, h3_index_col_name="h3_index"):
        return df.withColumn(
            "h3_index_res" + str(hex_resolution),
            h3F.cell_to_parent(F.col(h3_index_col_name), F.lit(hex_resolution)),
        )

    @staticmethod
    def haversine(colLat1, colLon1, colLat2, colLon2):
        r = F.lit(6371000.0)  # meters
        theta1, theta2 = F.radians(colLat1), F.radians(colLat2)
        d_theta = F.radians(colLat2 - colLat1)
        d_lambda = F.radians(colLon2 - colLon1)
        a = F.pow(F.sin(d_theta / 2), 2) + F.cos(theta1) * F.cos(theta2) * F.pow(
            F.sin(d_lambda / 2), 2
        )
        c = F.lit(2.0) * F.atan2(
            F.sqrt(F.least(F.lit(1.0), a)),
            F.sqrt(F.greatest(F.lit(0.0), F.lit(1.0) - a)),
        )
        return r * c / 1000.0  # return in kilometers
