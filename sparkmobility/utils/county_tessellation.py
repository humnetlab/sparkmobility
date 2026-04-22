import os

import censusdis.data as ced
import geopandas as gpd
import h3
from censusdis.maps import ShapeReader
from shapely.geometry import Polygon
from shapely.geometry.multipolygon import MultiPolygon

TOTAL_POP = "B01003_001E"


def tesselate_county(
    state_fips_code,  # TODO: Change to allow multiple states and counties
    county_fips_code,
    hex_resolution,
    year=2020,
    census_dataset="acs/acs5",
    projection_crs="EPSG:3310",
    census_variables=["NAME", TOTAL_POP],
    census_geo_level="tract",  # TODO: Implement this argument
):
    """
    Fills a U.S. county with H3 hexagons and estimates their population
    based on Census population data.

    Parameters
    ----------
    state_fips_code
        U.S. Census state FIPS code for the state containing the county
        to be tiled. See https://www.census.gov/library/reference/code-lists/ansi.html
        for more info. This is a two-digit string. For California it is `'06'`.
    county_fips_code
        U.S. Census county FIPS code for the county to be tiled.
        See https://www.census.gov/library/reference/code-lists/ansi.html
        for more info. This is a three-digit string.
        For Alameda County, CA it is `'001'`.
    year
        The year to download data for. This is
        an integer year, for example, `2020`.
    hex_resolution
        The resolution of H3 hexagons with which to tile the county.
        This is an integer between 0 and 15.
    census_dataset
        The dataset to download from. For example `"acs/acs5"`,
        `"dec/pl"`, or `"timeseries/poverty/saipe/schdist"`.
    projection_crs
        The coordinate reference system (CRS) to project the data into when taking
        area measurements. This is a string, and `'EPSG:3310'` is used for California.
        **However**, The appropriate CRS for the area being examined should be used.
    census_variables
        The census variables to download, for example `["NAME", "B01001_001E"]`.
    census_geo_level
        The Census geography to query at the lowest level. This is a string like `'tract'`
        for Census tracts or `'bg'` for Census block groups. Make sure this geography
        is available for the specified dataset and year.
    """

    # Download total population data for each tract in the county
    # as well as the shapefiles for each tract.
    gdf_tract = ced.download(
        census_dataset,
        year,
        census_variables,
        state=state_fips_code,
        county=county_fips_code,
        tract="*",
        with_geometry=True,
    ).rename(columns={TOTAL_POP: "Total Population"})

    # Specify and create the directory where the shapefile for
    # all U.S. counties should be stored.
    SHAPEFILE_ROOT = os.path.join(os.environ["HOME"], "data", "shapefiles")
    os.makedirs(SHAPEFILE_ROOT, exist_ok=True)

    # Initialize the shapefile reader to save to SHAPEFILE_ROOT
    # and look at data from year.

    reader = ShapeReader(SHAPEFILE_ROOT, year)

    # Load the county shapefile into a Geopandas DataFrame and save it to SHAPEFILE_ROOT
    counties_gdf = reader.read_shapefile(
        "us", "county"
    )  # TODO: This should read from SHAPEFILE_ROOT if the file exists.
    county_gdf = counties_gdf[
        counties_gdf["GEOID"] == state_fips_code + county_fips_code
    ]

    # Get H3 hexagon indices that tile the polygon
    def hex_idx_from_polygon(polygon, hex_resolution):
        if isinstance(polygon, Polygon):
            parsed = [(lat, lng) for lng, lat in polygon.exterior.coords]
        elif isinstance(polygon, MultiPolygon):
            parsed = []
            for poly in polygon.geoms:
                parsed.extend([(lat, lng) for lng, lat in poly.exterior.coords])
        else:
            raise TypeError("Expected Polygon or MultiPolygon")

        h3_poly = h3.LatLngPoly(parsed)
        hex_ids = h3.h3shape_to_cells(h3_poly, res=hex_resolution)

        return hex_ids

    # Get the geometries of the hexagons in hex_ids
    def hex_geometries_from_idx(hex_ids):
        return [
            Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(idx)])
            for idx in hex_ids
        ]

    hex_ids = hex_idx_from_polygon(county_gdf["geometry"].iloc[0], hex_resolution)
    hex_geometries = hex_geometries_from_idx(hex_ids)

    # Combine the hexagon indices and geometries into one GDF for the whole county.
    hex_gdf = gpd.GeoDataFrame(index=hex_ids, geometry=hex_geometries).set_crs(
        county_gdf.crs
    )

    # Join each hexagon with all overlapping tracts
    intersection_gdf = hex_gdf.sjoin(
        gdf_tract.rename(columns={"Total Population": "Tract Population"}),
        how="left",
        predicate="intersects",
    ).dropna(subset="index_right", axis=0)

    # Select the hexagon and tract geometry columns
    # and convert to a projected CRS for more accurate
    # area measurements.
    left_geom = intersection_gdf.geometry.to_crs(projection_crs)
    right_geom = gdf_tract.loc[intersection_gdf["index_right"]].geometry.to_crs(
        projection_crs
    )

    # Find the areas of all overlaps between hexagons and tracts
    areas = left_geom.intersection(right_geom, align=False).area / 1e6

    # Find the ratio between the overlap areas and the total tract areas
    area_ratios = (
        areas.to_numpy() / (right_geom.area / 1e6).to_numpy()
    )  # !! Had to do this in numpy because pandas wasn't dividing properly

    # Scale populations accordingly
    intersection_gdf["Scaled Population"] = (
        area_ratios * intersection_gdf["Tract Population"]
    )

    # Fill sum up overlapping, scaled populations for each hexagon
    hex_populations = (
        intersection_gdf.groupby(intersection_gdf.index)
        .agg({"geometry": "first", "Scaled Population": "sum"})
        .rename(columns={"Scaled Population": "Hexagon Population"})
        .set_geometry(
            "geometry"
        )  # !! For some reason the geometry col gets ignored along the way and here we respecify it
        .set_crs(gdf_tract.crs)  # !! Also the CRS gets lost
    )

    return hex_populations.reset_index()
