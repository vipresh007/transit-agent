"""The tool registry: what the model is told exists, and what actually runs.

Three exports, and they must stay in sync:
  TOOL_SCHEMAS    JSON schemas sent to the model. It never sees your code --
                  only these. A tool called wrongly is usually a wording bug
                  here, not a bug in the function.
  TOOL_FUNCTIONS  name -> callable, for dispatch.
  SCHEDULE_TOOLS  which tools can produce a VERIFIED schedule time. Kept here
                  rather than in agent.py because it went stale twice when it
                  lived far from the tools it names.
"""

from .geo import POI_TAGS, find_pois, geocode, get_weather
from .guides import guides_status, search_guides
from .journey import plan_journey
from memory import (
    TOOL_FUNCTIONS as MEMORY_FUNCTIONS,
    TOOL_SCHEMAS as MEMORY_SCHEMAS,
)
from .transit import (
    describe_transit_schema,
    find_direct_trips,
    find_nearby_stops,
    query_transit,
)

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
    {
        "type": "function",
        "function": {
            "name": "find_nearby_stops",
            "description": (
                "Find TTC stops near a coordinate, with distance and the routes "
                "serving each. ALWAYS use this to go from a place to its stops. "
                "Do NOT search the stops table by place name — stop names are "
                "intersections like 'College St at Augusta Ave', so "
                "neighbourhood and landmark names will never match."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "radius_m": {
                        "type": "integer",
                        "description": "Search radius in metres. Default 800.",
                    },
                },
                "required": ["lat", "lon"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_guides",
            "description": (
                "Search Wikivoyage travel guides for Toronto: what a "
                "neighbourhood is like, what's worth seeing, where to eat, "
                "local quirks and advice. Use for subjective or descriptive "
                "questions ('what's Kensington Market like?', 'somewhere to "
                "eat near the Distillery'). Do NOT use for schedules, "
                "departure times or routes — the guides contain prose, not "
                "timetables, and plan_journey is authoritative for those."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What you want to know, in natural "
                        "language. Full questions work better than keywords.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "How many passages to return. Default 4.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_journey",
            "description": (
                "Plan a real journey between two coordinates. Searches nearby "
                "stops at both ends for a direct ride, and failing that "
                "computes single-transfer options with a real interchange and "
                "verified departure/arrival times for each leg. "
                "USE THIS FIRST for any A-to-B question — it replaces manually "
                "picking stops, guessing an interchange, and writing journey "
                "SQL, all of which are error-prone."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin_lat": {"type": "number"},
                    "origin_lon": {"type": "number"},
                    "dest_lat": {"type": "number"},
                    "dest_lon": {"type": "number"},
                    "after_time": {
                        "type": "string",
                        "description": "Earliest departure, HH:MM:SS. Default 08:00:00.",
                    },
                    "service_id": {
                        "type": "string",
                        "description": "'1' weekday, '2' Saturday, '3' Sunday.",
                    },
                    "origin_name": {
                        "type": "string",
                        "description": "What the user called the origin, e.g. "
                        "'Kensington Market'. Used to label the walking legs.",
                    },
                    "dest_name": {
                        "type": "string",
                        "description": "What the user called the destination.",
                    },
                },
                "required": ["origin_lat", "origin_lon", "dest_lat", "dest_lon"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_direct_trips",
            "description": (
                "Find scheduled trips that serve BOTH stops on the same vehicle, "
                "origin before destination, with real departure and arrival "
                "times. Use this INSTEAD of hand-writing journey SQL. If it "
                "returns no trips, a transfer is required — call it again for "
                "each leg via an interchange stop."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin_stop_id": {
                        "type": "string",
                        "description": "stop_id from find_nearby_stops. Platforms "
                        "are direction-specific; pick the one whose 'serves' "
                        "headsigns point the way you're going.",
                    },
                    "dest_stop_id": {"type": "string"},
                    "after_time": {
                        "type": "string",
                        "description": "Earliest departure, HH:MM:SS. Default 08:00:00.",
                    },
                    "service_id": {
                        "type": "string",
                        "description": "'1' weekday, '2' Saturday, '3' Sunday.",
                    },
                },
                "required": ["origin_stop_id", "dest_stop_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_transit_schema",
            "description": (
                "Get the schema of the TTC transit database, including column "
                "names and important quirks. ALWAYS call this before writing "
                "any SQL with query_transit."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_transit",
            "description": (
                "Run a read-only SQL SELECT against the TTC schedule database "
                "(routes, trips, stop_times, stops, calendar, calendar_dates). "
                "Use for real departure times, route lookups, and service days. "
                "Call describe_transit_schema first. If the query errors, read "
                "the message and try a corrected query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A single SELECT statement, no trailing semicolon.",
                    }
                },
                "required": ["sql"],
            },
        },
    },
] + MEMORY_SCHEMAS

# Name -> function, so the loop can dispatch on whatever the model asks for.
TOOL_FUNCTIONS = {
    "geocode": geocode,
    "get_weather": get_weather,
    "find_pois": find_pois,
    "describe_transit_schema": describe_transit_schema,
    "query_transit": query_transit,
    "find_nearby_stops": find_nearby_stops,
    "find_direct_trips": find_direct_trips,
    "plan_journey": plan_journey,
    "search_guides": search_guides,
    **MEMORY_FUNCTIONS,
}

# Tools whose output can constitute a VERIFIED schedule time.
#
# Declared here, next to the tools themselves, and imported by agent.py. It
# lived in agent.py as a hardcoded set and went stale twice — once when
# find_direct_trips was added, again when plan_journey was. A registry that
# sits far away from the thing it registers will always drift; keeping it
# adjacent at least makes the omission visible when you add a tool.
SCHEDULE_TOOLS = {"query_transit", "find_direct_trips", "plan_journey"}
