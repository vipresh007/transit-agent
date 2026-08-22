"""Tools that talk to the outside world: geocoding, weather, points of interest.

All three services are free and need no API key. They are also volunteer-run
(Nominatim, Overpass), so be a good citizen: identify yourself in the
User-Agent and don't hammer them.
"""

import json

import requests

# Nominatim's usage policy requires a real User-Agent identifying your app.
HEADERS = {"User-Agent": "transit-agent-learning-project/0.1"}
TIMEOUT = 20

def geocode(place: str) -> str:
    """Turn a place name into coordinates. 'CN Tower, Toronto' -> lat/lon."""
    r = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": place, "format": "json", "limit": 1},
        headers=HEADERS,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    results = r.json()
    if not results:
        return f"No location found for {place!r}. Try adding a city or country."

    hit = results[0]
    return json.dumps(
        {
            "name": hit.get("display_name"),
            "lat": float(hit["lat"]),
            "lon": float(hit["lon"]),
        }
    )


def get_weather(lat: float, lon: float, date: str = "") -> str:
    """Forecast for a coordinate. date is YYYY-MM-DD; blank means today."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "timezone": "auto",
        "forecast_days": 7,
    }
    if date:
        params["start_date"] = date
        params["end_date"] = date
        params.pop("forecast_days")

    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params=params,
        headers=HEADERS,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    daily = r.json()["daily"]

    days = [
        {
            "date": d,
            "high_c": hi,
            "low_c": lo,
            "rain_chance_pct": rain,
        }
        for d, hi, lo, rain in zip(
            daily["time"],
            daily["temperature_2m_max"],
            daily["temperature_2m_min"],
            daily["precipitation_probability_max"],
        )
    ]
    return json.dumps(days)


# Overpass speaks its own query language. Each category maps to a LIST of tag
# selectors, because OSM tagging is inconsistent in practice: TTC subway
# stations appear as station=subway, as railway=station + subway=yes, and as
# public_transport=station. Querying only one spelling returns nothing, and
# an agent that gets nothing back tends to retry with different radii forever
# rather than concluding the tag is wrong.
POI_TAGS = {
    "museum": ['[tourism=museum]'],
    "gallery": ['[tourism=gallery]'],
    "attraction": ['[tourism=attraction]'],
    "viewpoint": ['[tourism=viewpoint]'],
    "park": ['[leisure=park]'],
    "cafe": ['[amenity=cafe]'],
    "restaurant": ['[amenity=restaurant]'],
    "bar": ['[amenity=bar]'],
    "subway_station": [
        '[station=subway]',
        '[railway=station][subway=yes]',
        '[railway=station][transport=subway]',
    ],
}


def find_pois(lat: float, lon: float, category: str, radius_m: int = 1500) -> str:
    """Find points of interest near a coordinate. See POI_TAGS for categories."""
    selectors = POI_TAGS.get(category)
    if selectors is None:
        return f"Unknown category {category!r}. Valid options: {', '.join(POI_TAGS)}"

    # Overpass QL: union of every tagging variant, nodes and ways, in radius.
    clauses = "\n      ".join(
        f"node{tag}(around:{radius_m},{lat},{lon});\n"
        f"      way{tag}(around:{radius_m},{lat},{lon});"
        for tag in selectors
    )
    query = f"""
    [out:json][timeout:25];
    (
      {clauses}
    );
    out center 60;
    """

    r = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={"data": query},
        headers=HEADERS,
        timeout=60,
    )
    r.raise_for_status()

    places = {}
    for el in r.json().get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name or name in places:
            continue  # unnamed POIs are noise; union queries return dupes
        places[name] = {
            "name": name,
            "lat": el.get("lat") or el.get("center", {}).get("lat"),
            "lon": el.get("lon") or el.get("center", {}).get("lon"),
            "opening_hours": tags.get("opening_hours"),
        }

    if not places:
        # Tell the agent what to do next. A bare "no results" invites it to
        # retry the same call at a slightly different radius, forever.
        return (
            f"No {category} found within {radius_m}m. Do NOT retry with a "
            f"different radius — the answer will be the same. For transit "
            f"stops, query the stops table via query_transit instead: it has "
            f"stop_lat/stop_lon and is authoritative for the TTC."
        )
    return json.dumps(list(places.values())[:20])
