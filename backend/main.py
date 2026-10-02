"""
ChronoSky API
=============

A thin FastAPI wrapper around the astronomy engine and star catalogue.

Run it from the PROJECT ROOT (the folder containing de440s.bsp and
hip_main.dat) with:

    uvicorn backend.main:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger docs, or
open frontend/index.html in your browser for the Time Travel UI.
"""

import io
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.astronomy.engine import (
    get_planet_positions,
    get_orbit_positions,
    get_iss_passes,
    get_satellite_passes,
    get_lagna,
    get_eclipses,
    get_panchang,
)
from backend.astronomy.catalogue.star_catalog import (
    get_visible_stars,
    get_named_star_positions,
)
from backend.astronomy.catalogue.sky_map import render_sky_map_png


app = FastAPI(
    title="ChronoSky",
    description="Historical astronomy engine — see the sky at any date, time, and place.",
    version="0.2.0",
)

# Allow the frontend (opened as a local file, or served from any port)
# to call this API from the browser. Fine for local development / a
# beta demo - if this ever goes on the public internet, narrow this
# down to the frontend's real origin instead of "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default observer location (Dammam, Saudi Arabia) - matches the
# defaults already used in engine.py.
DEFAULT_LAT = 26.4207
DEFAULT_LON = 50.0888

# NASA Exoplanet Archive's public TAP service. No API key required.
EXOPLANET_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# NASA JPL's live Deep Space Network status feed. No API key required.
DSN_NOW_URL = "https://eyes.nasa.gov/dsn/data/dsn.xml"

# NOAA's public Space Weather Prediction Center feeds. No API key required.
KP_INDEX_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
XRAY_FLARE_URL = "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-latest.json"
NOAA_SCALES_URL = "https://services.swpc.noaa.gov/products/noaa-scales.json"

# A few well-known spacecraft codes the DSN feed uses, mapped to
# friendlier display names. Anything not in here is shown as-is -
# the feed covers many missions and this list isn't exhaustive.
SPACECRAFT_NAMES = {
    "VGR1": "Voyager 1", "VGR2": "Voyager 2", "NHPC": "New Horizons",
    "JNO": "Juno", "MRO": "Mars Reconnaissance Orbiter",
    "M01O": "Mars Odyssey", "JWST": "James Webb Space Telescope",
    "SOHO": "SOHO", "PLC": "Parker Solar Probe", "PSP": "Parker Solar Probe",
    "EMM": "Hope Mars Mission", "MVN": "MAVEN", "TGO": "ExoMars TGO",
    "IMAP": "IMAP", "STA": "STEREO-A", "STB": "STEREO-B", "RST": "Roman Space Telescope",
    "CHDR": "Chandra X-ray Observatory", "ESCG": "Euclid", "GAIA": "Gaia",
    "LUCY": "Lucy", "OSAM": "OSIRIS-APEX", "PERS": "Perseverance (Mars)",
    "CURI": "Curiosity (Mars)", "MSL": "Curiosity (Mars)",
}


@app.get("/api")
def api_info():
    return {
        "name": "ChronoSky API",
        "endpoints": {
            "/planets": "Positions of Sun, Moon, and planets for a given date/time/location",
            "/stars/visible": "All bright stars currently above the horizon",
            "/stars/named": "Positions of ChronoSky's curated named stars",
            "/skymap": "PNG polar sky map for a given date/time/location",
            "/orbits": "Heliocentric x/y/z positions of the planets, for the orbit hologram",
            "/exoplanets": "Recently confirmed exoplanets from NASA's Exoplanet Archive",
            "/iss-passes": "Upcoming naked-eye-visible ISS passes for a location",
            "/deep-space-network": "Live NASA Deep Space Network status - which probes, what data rate",
            "/lagna": "Vedic Ascendant (Lagna/Rashi) and Moon's Nakshatra, Lahiri ayanamsha",
            "/satellite-passes": "Upcoming visible passes for iss, hubble, or tiangong",
            "/eclipses": "Upcoming lunar eclipses and approximate solar eclipse visibility",
            "/panchang": "The five limbs of the Hindu daily almanac for a date/time/location",
            "/space-weather": "Live Kp index, latest solar flare, and NOAA activity scales",
        },
    }


@app.get("/planets")
def planets(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    latitude: float = DEFAULT_LAT,
    longitude: float = DEFAULT_LON,
):
    """Positions of the Sun, Moon, and planets at a given date/time/location."""
    return get_planet_positions(year, month, day, hour, minute, latitude, longitude)


@app.get("/stars/visible")
def stars_visible(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    latitude: float = DEFAULT_LAT,
    longitude: float = DEFAULT_LON,
):
    """Every bright catalogue star above the horizon at the given date/time/location."""
    return get_visible_stars(year, month, day, hour, minute, latitude, longitude)


@app.get("/stars/named")
def stars_named(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    latitude: float = DEFAULT_LAT,
    longitude: float = DEFAULT_LON,
):
    """Positions of ChronoSky's curated named stars (Polaris, Sirius, Vega, ...)."""
    return get_named_star_positions(year, month, day, hour, minute, latitude, longitude)


@app.get("/skymap")
def skymap(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    latitude: float = DEFAULT_LAT,
    longitude: float = DEFAULT_LON,
):
    """A PNG polar sky map of visible stars at the given date/time/location."""
    png_bytes = render_sky_map_png(year, month, day, hour, minute, latitude, longitude)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@app.get("/orbits")
def orbits(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
):
    """Heliocentric x/y/z AU positions of the planets - for the orbit hologram."""
    return get_orbit_positions(year, month, day, hour, minute)


@app.get("/exoplanets")
def exoplanets(limit: int = 24):
    """
    Recently confirmed exoplanets, fetched live from NASA's public
    Exoplanet Archive (no API key needed). Proxied through this
    backend so the browser doesn't have to deal with the upstream
    service's own CORS rules.
    """
    query = (
        f"SELECT TOP {limit} pl_name, hostname, disc_year, discoverymethod, "
        f"pl_orbper, pl_rade, pl_bmasse, pl_eqt, sy_dist "
        f"FROM pscomppars "
        f"WHERE disc_year IS NOT NULL "
        f"ORDER BY disc_year DESC"
    )

    try:
        response = requests.get(
            EXOPLANET_TAP_URL,
            params={"query": query, "format": "json"},
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach NASA's Exoplanet Archive: {error}",
        )

    return response.json()


@app.get("/exoplanets/earth-twin")
def earth_twin():
    """
    The confirmed exoplanet currently closest to Earth in both size and
    temperature - a real, data-driven "best known Earth analog", not a
    fixed pick. Queried live from NASA's Exoplanet Archive and ranked
    here by how close each candidate's radius and equilibrium
    temperature are to Earth's own.
    """
    query = (
        "SELECT pl_name, hostname, disc_year, discoverymethod, "
        "pl_rade, pl_bmasse, pl_eqt, sy_dist "
        "FROM pscomppars "
        "WHERE pl_rade IS NOT NULL AND pl_eqt IS NOT NULL "
        "AND pl_rade BETWEEN 0.5 AND 1.8 "
        "AND pl_eqt BETWEEN 200 AND 320"
    )

    try:
        response = requests.get(
            EXOPLANET_TAP_URL,
            params={"query": query, "format": "json"},
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach NASA's Exoplanet Archive: {error}",
        )

    rows = response.json()
    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No close Earth analogs found in the current archive data.",
        )

    def similarity_score(row):
        radius_diff = (row["pl_rade"] - 1.0)
        temp_diff = (row["pl_eqt"] - 288) / 100
        return radius_diff ** 2 + temp_diff ** 2

    best = min(rows, key=similarity_score)
    return best


@app.get("/lagna")
def lagna(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    latitude: float = DEFAULT_LAT,
    longitude: float = DEFAULT_LON,
):
    """The Vedic Ascendant (Lagna/Rashi) and Moon's Nakshatra for a given date, time, and location - Lahiri ayanamsha."""
    return get_lagna(year, month, day, hour, minute, latitude, longitude)


@app.get("/satellite-passes")
def satellite_passes(satellite: str = "iss", latitude: float = DEFAULT_LAT, longitude: float = DEFAULT_LON, days: int = 5):
    """
    Upcoming naked-eye-visible passes of a chosen satellite (iss,
    hubble, or tiangong) for a location - same real TLE-based
    prediction as the ISS tracker, generalized to other bright
    satellites.
    """
    try:
        return get_satellite_passes(satellite, latitude, longitude, days)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except requests.RequestException as error:
        raise HTTPException(status_code=502, detail=f"Could not fetch satellite tracking data: {error}")


@app.get("/eclipses")
def eclipses(latitude: float = DEFAULT_LAT, longitude: float = DEFAULT_LON, days: int = 180):
    """
    Upcoming lunar eclipses (real Skyfield geometry) and an
    approximate solar eclipse visibility check, for a location.
    """
    return get_eclipses(latitude, longitude, days)


@app.get("/panchang")
def panchang(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    latitude: float = DEFAULT_LAT,
    longitude: float = DEFAULT_LON,
    local_year: int = None,
    local_month: int = None,
    local_day: int = None,
):
    """
    The five limbs of the Hindu daily almanac (Tithi, Vaara, Nakshatra,
    Yoga, Karana) for a date/time/location. year/month/day/hour/minute
    must be the correct UTC instant; pass local_year/local_month/
    local_day (the calendar date as the person actually experiences
    it) so Vaara (the weekday) is correct even when the UTC date
    differs from the local one - e.g. an early-morning local time that
    falls on the previous day in UTC.
    """
    return get_panchang(
        year, month, day, hour, minute, latitude, longitude,
        local_year, local_month, local_day,
    )


@app.get("/space-weather")
def space_weather():
    """
    Live space weather: the current planetary Kp geomagnetic index
    (an aurora indicator), the most recent solar flare, and NOAA's
    current R/S/G activity scales. Each sub-fetch is independent, so
    one feed being unavailable doesn't break the others.
    """
    result = {}

    try:
        kp_resp = requests.get(KP_INDEX_URL, timeout=10)
        kp_resp.raise_for_status()
        kp_data = kp_resp.json()
        if isinstance(kp_data, list) and len(kp_data) > 1:
            header, latest = kp_data[0], kp_data[-1]
            result["kp_index"] = dict(zip(header, latest))
    except Exception as error:
        result["kp_index_error"] = str(error)

    try:
        flare_resp = requests.get(XRAY_FLARE_URL, timeout=10)
        flare_resp.raise_for_status()
        flare_data = flare_resp.json()
        result["latest_flare"] = (
            flare_data[0] if isinstance(flare_data, list) and flare_data else flare_data
        )
    except Exception as error:
        result["flare_error"] = str(error)

    try:
        scales_resp = requests.get(NOAA_SCALES_URL, timeout=10)
        scales_resp.raise_for_status()
        result["noaa_scales"] = scales_resp.json()
    except Exception as error:
        result["scales_error"] = str(error)

    return result


@app.get("/iss-passes")
def iss_passes(latitude: float, longitude: float, days: int = 5):
    """
    Upcoming times the ISS should be visible to the naked eye from a
    given location, using a live TLE (no API key) and real sunlight/
    twilight geometry - not just "when it's overhead."
    """
    try:
        return get_iss_passes(latitude, longitude, days)
    except requests.RequestException as error:
        raise HTTPException(
            status_code=502,
            detail=f"Could not fetch ISS tracking data: {error}",
        )


@app.get("/deep-space-network")
def deep_space_network():
    """
    Which real NASA spacecraft the Deep Space Network is talking to
    right now, and at what data rate - live from NASA JPL's own public
    DSN Now feed (no API key required).
    """
    try:
        response = requests.get(DSN_NOW_URL, timeout=15)
        response.raise_for_status()
    except requests.RequestException as error:
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach NASA's Deep Space Network feed: {error}",
        )

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Could not parse the Deep Space Network feed: {error}",
        )

    current_station = None
    signals = []

    for element in root:
        if element.tag == "station":
            current_station = element.get("friendlyName")
        elif element.tag == "dish":
            target = element.find("target")
            target_name = target.get("name") if target is not None else None
            if not target_name or target_name in ("DSN", "DSS"):
                continue

            down_signals = [
                s for s in element.findall("downSignal")
                if s.get("active") == "true"
            ]
            if not down_signals:
                continue

            best = max(down_signals, key=lambda s: float(s.get("dataRate") or 0))
            downleg_km_raw = target.get("downlegRange")
            distance_km = None
            if downleg_km_raw not in (None, "", "-1"):
                distance_km = float(downleg_km_raw)

            signals.append({
                "spacecraft_code": target_name,
                "spacecraft_name": SPACECRAFT_NAMES.get(target_name, target_name),
                "station": current_station,
                "dish": element.get("name"),
                "activity": element.get("activity"),
                "data_rate_bps": float(best.get("dataRate") or 0),
                "band": best.get("band"),
                "distance_km": distance_km,
            })

    timestamp_el = root.find("timestamp")
    return {
        "generated_at_epoch_ms": timestamp_el.text if timestamp_el is not None else None,
        "signals": signals,
    }


# ============================================================
# SERVE THE FRONTEND
# ============================================================
# Mounted last, at the root path, so every API route above still
# takes priority - only requests that don't match an API route above
# fall through to serving the frontend's static files (index.html,
# etc). This lets one deployed service host both the API and the
# webpage at the same URL, with no separate frontend host and no
# cross-origin requests needed once deployed.

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
