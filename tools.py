"""
Tools the agent can call.

Every tool here is a plain Python function. Two rules:
  1. It takes JSON-friendly arguments (str, int, float, bool).
  2. It returns a string, or something json.dumps() can handle.

That's it. There is no magic. The "agent" part is just the model
choosing which of these to call and with what arguments.

All three services below are free and need no API key.
"""

import json
import requests

# Nominatim's usage policy requires a real User-Agent identifying your app.
# Be a good citizen: they run this for free.
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


# Overpass speaks its own query language. These are the tags we know how to ask for.
POI_TAGS = {
    "museum": '[tourism=museum]',
    "gallery": '[tourism=gallery]',
    "attraction": '[tourism=attraction]',
    "viewpoint": '[tourism=viewpoint]',
    "park": '[leisure=park]',
    "cafe": '[amenity=cafe]',
    "restaurant": '[amenity=restaurant]',
    "bar": '[amenity=bar]',
    "subway_station": '[railway=station][station=subway]',
}


def find_pois(lat: float, lon: float, category: str, radius_m: int = 1500) -> str:
    """Find points of interest near a coordinate. See POI_TAGS for categories."""
    tag = POI_TAGS.get(category)
    if tag is None:
        return f"Unknown category {category!r}. Valid options: {', '.join(POI_TAGS)}"

    # Overpass QL: find nodes+ways with this tag inside a radius, return centers.
    query = f"""
    [out:json][timeout:25];
    (
      node{tag}(around:{radius_m},{lat},{lon});
      way{tag}(around:{radius_m},{lat},{lon});
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

    places = []
    for el in r.json().get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue  # unnamed POIs are noise for our purposes
        places.append(
            {
                "name": name,
                "lat": el.get("lat") or el.get("center", {}).get("lat"),
                "lon": el.get("lon") or el.get("center", {}).get("lon"),
                "opening_hours": tags.get("opening_hours"),
            }
        )

    if not places:
        return f"No {category} found within {radius_m}m."
    return json.dumps(places[:20])


# ---------------------------------------------------------------------------
# Schemas: how we describe the functions above to the model.
#
# This is the part people find surprising. The model never sees your code --
# it only sees these descriptions. If a tool gets called wrongly, the fix is
# almost always here, in the wording, not in the function.
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "geocode",
            "description": (
                "Convert a place name or address into latitude/longitude coordinates. "
                "Call this first whenever you need coordinates for another tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "place": {
                        "type": "string",
                        "description": "Place name, ideally with city, e.g. 'Kensington Market, Toronto'",
                    }
                },
                "required": ["place"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get the daily weather forecast for a coordinate, up to 7 days ahead. "
                "Use this to decide between indoor and outdoor activities."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "date": {
                        "type": "string",
                        "description": "Optional single date as YYYY-MM-DD. Omit for a 7-day forecast.",
                    },
                },
                "required": ["lat", "lon"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_pois",
            "description": (
                "Find named points of interest near a coordinate, with opening hours "
                "when OpenStreetMap has them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "category": {
                        "type": "string",
                        "enum": list(POI_TAGS.keys()),
                    },
                    "radius_m": {
                        "type": "integer",
                        "description": "Search radius in metres. Default 1500.",
                    },
                },
                "required": ["lat", "lon", "category"],
            },
        },
    },
]

# Name -> function, so the loop can dispatch on whatever the model asks for.
TOOL_FUNCTIONS = {
    "geocode": geocode,
    "get_weather": get_weather,
    "find_pois": find_pois,
}
