import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def get_data_from_ibge(endpoint):
    response = requests.get(endpoint)
    # explode all keys to columns from dicts inside localidade and serie columns
    res_data = pd.DataFrame(response.json()[0]["resultados"][0]["series"])
    res_data = res_data.join(pd.json_normalize(res_data["localidade"])).drop(
        columns=["localidade"]
    )
    res_data = res_data.join(pd.json_normalize(res_data["serie"])).drop(
        columns=["serie"]
    )
    res_data = res_data.rename(columns={"2024": "populacao"})
    return res_data


def load_cities(
    path,
    endpoint="https://servicodados.ibge.gov.br/api/v3/agregados/6579/periodos/2024/variaveis/9324?localidades=N6[all]",
):
    """Load cities from a CSV file and return a GeoDataFrame."""
    data = pd.read_csv(path)
    cities = get_data_from_ibge(endpoint=endpoint)

    # Convert tipos to match for the merge
    cities['id'] = cities['id'].astype(str)
    data['codigo_ibge'] = data['codigo_ibge'].astype(str)

    # Merge CSV data with IBGE population data
    merged_data = data.merge(
        cities,
        how="left",
        left_on="codigo_ibge",
        right_on="id",
        suffixes=("", "_ibge"),
    )

    # Filter out rows without population data and sort by population
    merged_data = merged_data.dropna(subset=['populacao']).sort_values(by="populacao", ascending=False)[
        [
            "codigo_ibge",
            "nome",
            "populacao",
            "latitude",
            "longitude",
        ]
    ]

    # Clean and prepare the data
    merged_data['populacao'] = merged_data['populacao'].astype(int)

    # Select top cities by population
    df = merged_data.sort_values(by=['populacao'], ascending=False).head(50)[['nome', 'latitude', 'longitude', 'populacao']]

    geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, crs="EPSG:4326", geometry=geometry)
    return gdf
