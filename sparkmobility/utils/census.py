import geopandas as gpd
import pandas as pd

from sparkmobility.utils.county_tessellation import tesselate_county


def obtain_population_data(
    state_fips_codes,
    county_fips_codes,
    hex_resolution,
    year,
    census_dataset,
    projection_crs,
    census_variables,
):
    """Iterate states/counties and tessellate to H3 grid with census attributes."""
    pop_df = gpd.GeoDataFrame()
    for state_fips in state_fips_codes:
        for county_fips in county_fips_codes.get(state_fips, []):
            pop_df_i = tesselate_county(
                state_fips_code=state_fips,
                county_fips_code=county_fips,
                hex_resolution=hex_resolution,
                year=year,
                census_dataset=census_dataset,
                projection_crs=projection_crs,
                census_variables=census_variables,
            )
            pop_df = pd.concat([pop_df, pop_df_i], ignore_index=True)
    return pop_df
