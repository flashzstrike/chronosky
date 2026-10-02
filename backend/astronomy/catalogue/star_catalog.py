from pathlib import Path

from skyfield.api import load, Star, wgs84
from skyfield.data import hipparcos
import numpy as np

# ============================================================
# IMPORTANT STARS
# ============================================================

IMPORTANT_STARS = {
    "Polaris": 11767,
    "Sirius": 32349,
    "Betelgeuse": 27989,
    "Rigel": 24436,
    "Vega": 91262,
    "Altair": 97649,
    "Deneb": 102098,
    "Aldebaran": 21421,
    "Antares": 80763,
    "Spica": 65474,
    "Arcturus": 69673,
    "Procyon": 37279,
    "Capella": 24608,
    "Regulus": 49669,
}

# Keep only reasonably bright stars (lower magnitude = brighter)
MAGNITUDE_LIMIT = 4.0


# ============================================================
# CHRONOSKY
# Historical Star Catalogue
# ============================================================

# Project root = the folder that holds de440s.bsp / hip_main.dat.
# backend/astronomy/catalogue/star_catalog.py -> catalogue -> astronomy -> backend -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DE440S_PATH = PROJECT_ROOT / "de440s.bsp"
HIP_MAIN_PATH = PROJECT_ROOT / "hip_main.dat"

print("Loading astronomical data for star catalogue...")

# Loaded ONCE at import time and reused by every function below -
# the old version reloaded the ephemeris and re-parsed the whole
# 53MB catalogue file on every single function call.
ts = load.timescale()
planets = load(str(DE440S_PATH))
earth = planets["earth"]

print("Loading Hipparcos star catalogue...")

# IMPORTANT: hip_main.dat (as downloaded/extracted here) is plain,
# uncompressed text - it is NOT the .gz file Skyfield's loader
# fetches by default. hipparcos.load_dataframe() defaults to
# compression="gzip", so it must be told compression=None here or
# it will fail trying to gunzip a file that isn't gzipped.
with open(HIP_MAIN_PATH, "rb") as f:
    hip_df = hipparcos.load_dataframe(f)

# Columns provided by Skyfield: magnitude, ra_degrees, dec_degrees,
# parallax_mas, ra_mas_per_year, dec_mas_per_year, ra_hours, epoch_year
# (indexed by HIP number).
bright_df = hip_df[hip_df["magnitude"] <= MAGNITUDE_LIMIT].copy()

# Some catalogue entries have missing proper motion / parallax -
# default those to 0 instead of dropping the star entirely.
bright_df[["parallax_mas", "ra_mas_per_year", "dec_mas_per_year"]] = bright_df[
    ["parallax_mas", "ra_mas_per_year", "dec_mas_per_year"]
].fillna(0.0)

# RA/Dec are mandatory fields - drop anything missing those.
bright_df = bright_df.dropna(subset=["ra_degrees", "dec_degrees"])

# One vectorized Star object representing every bright star at once.
# Skyfield can observe all of them in a single call, which is far
# faster than the original per-star Python loop.
bright_stars = Star.from_dataframe(bright_df)
bright_star_hip_ids = bright_df.index.to_numpy()
bright_star_magnitudes = bright_df["magnitude"].to_numpy()


# ============================================================
# CATALOGUE SUMMARY
# ============================================================

print()
print("======================================")
print("       CHRONOSKY STAR CATALOGUE")
print("======================================")
print()

print(
    f"Bright stars loaded: {len(bright_df)}"
)

print()
print("Star objects created successfully.")


# ============================================================
# HISTORICAL STAR POSITIONS
# ============================================================

def get_star_positions(
    year,
    month,
    day,
    hour=0,
    minute=0
):

    # Historical time
    t = ts.utc(
        year,
        month,
        day,
        hour,
        minute
    )

    # Astrometric / apparent position of every bright star at once
    astrometric = earth.at(t).observe(bright_stars)
    apparent = astrometric.apparent()

    # Right Ascension / Declination / Distance
    ra, dec, distance = apparent.radec()

    return (
        ra.hours.tolist(),
        dec.degrees.tolist(),
        distance.au.tolist(),
    )


# ============================================================
# VISIBLE STARS
# ============================================================

def get_visible_stars(
    year,
    month,
    day,
    hour,
    minute,
    latitude,
    longitude
):

    # Historical time
    t = ts.utc(
        year,
        month,
        day,
        hour,
        minute
    )

    # Create observer location
    location = wgs84.latlon(
        latitude,
        longitude
    )

    # Combine Earth + observer location
    observer = earth + location

    # Observe every bright star at once
    astrometric = observer.at(t).observe(bright_stars)
    apparent = astrometric.apparent()

    # Calculate altitude and azimuth
    alt, az, distance = apparent.altaz()

    altitude = alt.degrees
    azimuth = az.degrees
    distance_au = distance.au

    visible_stars = []

    # Only keep stars above the horizon
    for i in range(len(altitude)):

        if altitude[i] > 0:

            visible_stars.append({
                "index": int(i),
                "hip_id": int(bright_star_hip_ids[i]),
                "altitude": float(altitude[i]),
                "azimuth": float(azimuth[i]),
                "distance_au": float(distance_au[i]),
                "magnitude": float(bright_star_magnitudes[i])
            })

    return visible_stars


# ============================================================
# NAMED STAR POSITIONS
# ============================================================

def get_named_star_positions(
    year,
    month,
    day,
    hour,
    minute,
    latitude,
    longitude
):
    """
    Calculate the historical apparent positions
    of ChronoSky's important named stars.
    """

    # Create the historical time
    t = ts.utc(
        year,
        month,
        day,
        hour,
        minute
    )

    # Observer location
    observer = earth + wgs84.latlon(
        latitude,
        longitude
    )

    named_positions = {}

    for star_name, hip_id in IMPORTANT_STARS.items():

        try:
            # Get the star from the full (unfiltered) Hipparcos dataframe
            star_data = hip_df.loc[[hip_id]]

            star = Star.from_dataframe(star_data)

            # Observe the star from our historical location
            astrometric = observer.at(t).observe(star)

            # Convert to apparent position
            apparent = astrometric.apparent()

            # Altitude / Azimuth
            altitude, azimuth, distance = apparent.altaz()

            named_positions[star_name] = {
                "hip_id": hip_id,
                "altitude": float(np.asarray(altitude.degrees).squeeze()),
                "azimuth": float(np.asarray(azimuth.degrees).squeeze()),
                "distance_ly": float(np.asarray(distance.au / 63241.077).squeeze())
            }

        except Exception as error:

            print(
                f"Could not calculate {star_name}: {error}"
            )

    return named_positions


# Test Program
if __name__ == "__main__":

    print()
    print("Testing historical star positions...")
    print()

    # Apollo 11 landing-era test
    ra, dec, distance = get_star_positions(
        1969,
        7,
        20,
        20,
        17
    )

    print(
        f"Stars calculated: {len(ra)}"
    )

    print()
    print("Historical date:")
    print("20 July 1969")
    print("20:17 UTC")

    print()
    print("First 10 stars:")
    print()

    for i in range(
        min(10, len(ra))
    ):

        print(
            f"Star {i + 1}: "
            f"RA={ra[i]:.4f}h "
            f"Dec={dec[i]:.4f}° "
            f"Distance={distance[i]:.2f} AU"
        )
        print()
    print("Testing visible stars...")
    print()

    visible = get_visible_stars(
        1969,
        7,
        20,
        20,
        17,
        26.4207,
        50.0888
    )

    print(
        f"Visible stars: {len(visible)}"
    )

    print()
    print("First 10 visible stars:")
    print()

    for star in visible[:10]:

        print(
            f"Star index: {star['index']} | "
            f"Altitude: {star['altitude']:.2f}° | "
            f"Azimuth: {star['azimuth']:.2f}° | "
            f"Distance: {star['distance_au']:.2f} AU"
        )

    print()
    print("Testing named star positions...")
    print() 

    named_stars = get_named_star_positions(
        1969,
        7,
        20,
        20,
        17,
        26.4207,
        50.0888
    )

    print(f"Named stars calculated: {len(named_stars)}")

    print()

    for name, data in named_stars.items():
        print(
            f"{name}: "
            f"Altitude={data['altitude']:.2f}° | "
            f"Azimuth={data['azimuth']:.2f}° | "
            f"Distance={data['distance_ly']:.2f} ly"
        )
