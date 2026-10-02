from pathlib import Path
from datetime import date
import math

import requests
from skyfield.api import (
    load,
    wgs84,
    load_constellation_map,
    load_constellation_names,
    EarthSatellite,
)
from skyfield import almanac, eclipselib


# ============================================================
# CHRONOSKY
# Historical Astronomy Engine
# ============================================================

# Project root = the folder that holds de440s.bsp / hip_main.dat.
# Computed from this file's own location so it works no matter
# what your terminal's current working directory is (VS Code's
# "Run" button, a debugger, or uvicorn from anywhere).
#   backend/astronomy/engine.py -> backend/astronomy -> backend -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DE440S_PATH = PROJECT_ROOT / "de440s.bsp"

print("Loading astronomical data...")

# Load JPL DE440s planetary ephemeris
planets = load(str(DE440S_PATH))

# Skyfield time scale
ts = load.timescale()
#Load constellation boundaries
constellation_at = load_constellation_map()
#Convert constellation abbreviations to full names
constellation_names = dict(load_constellation_names())


# ============================================================
# CELESTIAL BODIES
# ============================================================

sun = planets["sun"]
moon = planets["moon"]

mercury = planets["mercury"]
venus = planets["venus"]
earth = planets["earth"]
mars = planets["mars barycenter"]
jupiter = planets["jupiter barycenter"]
saturn = planets["saturn barycenter"]
uranus = planets["uranus barycenter"]
neptune = planets["neptune barycenter"]

# Planets only (no Sun/Moon/Earth) - used by get_orbit_positions for
# the solar-system hologram view.
ORBIT_BODIES = {
    "Mercury": mercury,
    "Venus": venus,
    "Earth": earth,
    "Mars": mars,
    "Jupiter": jupiter,
    "Saturn": saturn,
    "Uranus": uranus,
    "Neptune": neptune,
}


# ============================================================
# MAIN ASTRONOMY FUNCTION
# ============================================================

def get_planet_positions(
    year,
    month,
    day,
    hour=0,
    minute=0,
    latitude=26.4207,
    longitude=50.0888
):

    # Create exact time
    t = ts.utc(year, month, day, hour, minute)

    # Create observer on Earth
    observer = earth + wgs84.latlon(latitude, longitude)

    # Celestial bodies
    bodies = {
        "Sun": sun,
        "Moon": moon,
        "Mercury": mercury,
        "Venus": venus,
        "Earth": earth,
        "Mars": mars,
        "Jupiter": jupiter,
        "Saturn": saturn,
        "Uranus": uranus,
        "Neptune": neptune,
    }

    positions = {}

    # Calculate every body
    for name, body in bodies.items():

        # Position relative to Earth
        position = body.at(t) - earth.at(t)

        x, y, z = position.position.au

        data = {
            "x_au": x,
            "y_au": y,
            "z_au": z
        }

        # Earth itself cannot be observed from the observer
        if name != "Earth":

            apparent = observer.at(t).observe(body).apparent()

            altitude, azimuth, distance = apparent.altaz()

            ra, dec, distance_radec = apparent.radec()

            data["altitude_deg"] = altitude.degrees
            data["azimuth_deg"] = azimuth.degrees
            data["distance_au"] = distance.au
            data["ra_hours"] = ra.hours
            data["dec_deg"] = dec.degrees

            #Identify the constellation containing the object
            constellation_abbr = constellation_at(apparent)
            constellation_name = constellation_names.get(
                constellation_abbr,
                constellation_abbr
            )
            data["constellation"] = constellation_name

            # Moon phase (for the phase icon) - 0=new, 90=first quarter,
            # 180=full, 270=last quarter.
            if name == "Moon":
                phase_angle = almanac.moon_phase(planets, t).degrees
                data["phase_deg"] = phase_angle
                data["illumination_fraction"] = (
                    1 - math.cos(math.radians(phase_angle))
                ) / 2

        positions[name] = data

    return positions


# ============================================================
# ORBIT (HOLOGRAM) POSITIONS
# ============================================================

def get_orbit_positions(year, month, day, hour=0, minute=0):
    """
    Heliocentric (Sun-centered) x/y/z positions in AU for every planet,
    for the solar-system hologram view. Unlike get_planet_positions
    (which is geocentric - relative to Earth, for what an observer
    sees in their sky), this is relative to the Sun, which is what you
    want to draw an actual top-down map of the solar system.
    """

    t = ts.utc(year, month, day, hour, minute)

    positions = {}

    for name, body in ORBIT_BODIES.items():
        position = body.at(t) - sun.at(t)
        x, y, z = position.position.au

        positions[name] = {
            "x_au": x,
            "y_au": y,
            "z_au": z,
        }

    return positions


# ============================================================
# SATELLITE VISIBILITY PREDICTOR (ISS and others)
# ============================================================

SATELLITE_CATALOG = {
    "iss": {"name": "International Space Station", "catnr": 25544},
    "hubble": {"name": "Hubble Space Telescope", "catnr": 20580},
    "tiangong": {"name": "Tiangong Space Station", "catnr": 48274},
}


def get_satellite_passes(satellite_key, latitude, longitude, days=5):
    """
    Upcoming times a given satellite will be visible to the naked eye
    from a given location: high enough above the horizon, lit by the
    Sun, and the sky dark enough (observer past civil twilight) to see
    it. Uses a live TLE fetched from Celestrak - no API key needed.
    """
    info = SATELLITE_CATALOG.get(satellite_key)
    if info is None:
        raise ValueError(f"Unknown satellite key: {satellite_key}")

    url = (
        "https://celestrak.org/NORAD/elements/gp.php"
        f"?CATNR={info['catnr']}&FORMAT=TLE"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    lines = [line for line in response.text.strip().splitlines() if line.strip()]
    name, line1, line2 = lines[0], lines[1], lines[2]
    satellite = EarthSatellite(line1, line2, name, ts)

    observer_topos = wgs84.latlon(latitude, longitude)
    observer = earth + observer_topos

    t0 = ts.now()
    t1 = ts.tt_jd(t0.tt + days)

    times, events = satellite.find_events(
        observer_topos, t0, t1, altitude_degrees=10.0
    )

    passes = []
    current = {}

    for t, event in zip(times, events):
        if event == 0:  # rise
            current = {"rise": t}
        elif event == 1:  # culminate
            current["culminate"] = t
            difference = satellite - observer_topos
            alt, az, _ = difference.at(t).altaz()
            current["max_altitude_deg"] = alt.degrees
        elif event == 2:  # set
            current["set"] = t

            culm_t = current.get("culminate", t)
            sunlit = satellite.at(culm_t).is_sunlit(planets)
            sun_alt, _, _ = observer.at(culm_t).observe(sun).apparent().altaz()
            observer_is_dark = sun_alt.degrees < -6

            if sunlit and observer_is_dark and "rise" in current:
                passes.append({
                    "rise_utc": current["rise"].utc_iso(),
                    "culminate_utc": culm_t.utc_iso(),
                    "set_utc": current["set"].utc_iso(),
                    "max_altitude_deg": current.get("max_altitude_deg"),
                })

            current = {}

    return passes


def get_iss_passes(latitude, longitude, days=5):
    """Backwards-compatible wrapper - ISS passes specifically."""
    return get_satellite_passes("iss", latitude, longitude, days)


# ============================================================
# LAGNA (VEDIC ASCENDANT) & NAKSHATRA
# ============================================================

RASHIS = [
    {"name": "Mesha", "english": "Aries", "lord": "Mars"},
    {"name": "Vrishabha", "english": "Taurus", "lord": "Venus"},
    {"name": "Mithuna", "english": "Gemini", "lord": "Mercury"},
    {"name": "Karka", "english": "Cancer", "lord": "Moon"},
    {"name": "Simha", "english": "Leo", "lord": "Sun"},
    {"name": "Kanya", "english": "Virgo", "lord": "Mercury"},
    {"name": "Tula", "english": "Libra", "lord": "Venus"},
    {"name": "Vrishchika", "english": "Scorpio", "lord": "Mars"},
    {"name": "Dhanu", "english": "Sagittarius", "lord": "Jupiter"},
    {"name": "Makara", "english": "Capricorn", "lord": "Saturn"},
    {"name": "Kumbha", "english": "Aquarius", "lord": "Saturn"},
    {"name": "Meena", "english": "Pisces", "lord": "Jupiter"},
]

NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni",
    "Uttara Phalguni", "Hasta", "Chitra", "Swati", "Vishakha",
    "Anuradha", "Jyeshtha", "Mula", "Purva Ashadha", "Uttara Ashadha",
    "Shravana", "Dhanishta", "Shatabhisha", "Purva Bhadrapada",
    "Uttara Bhadrapada", "Revati",
]


def _lahiri_ayanamsha(t_centuries):
    """
    Lahiri (Chitrapaksha) ayanamsha - the official Indian government
    standard - as a linear approximation in degrees. Accurate to a
    small fraction of a degree within a few centuries of J2000, which
    is more than enough precision for Rashi/Nakshatra determination.
    """
    return (23.85 + 1.397 * t_centuries) % 360


def get_lagna(year, month, day, hour, minute, latitude, longitude):
    """
    The Vedic Ascendant (Lagna) and Moon's Nakshatra for a given date,
    time, and location - real spherical astronomy (local sidereal
    time, obliquity of the ecliptic, the standard ascendant formula),
    converted from the tropical to the sidereal zodiac via the Lahiri
    ayanamsha.
    """
    t = ts.utc(year, month, day, hour, minute)

    jd_ut1 = t.ut1
    t_centuries = (jd_ut1 - 2451545.0) / 36525.0

    # Greenwich Mean Sidereal Time (standard IAU/Meeus polynomial),
    # then Local Sidereal Time by adding longitude (east-positive).
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd_ut1 - 2451545.0)
        + 0.000387933 * t_centuries ** 2
        - (t_centuries ** 3) / 38710000.0
    ) % 360
    lst_deg = (gmst_deg + longitude) % 360

    # Mean obliquity of the ecliptic (standard IAU polynomial).
    obliquity_deg = (
        23.439291
        - 0.0130042 * t_centuries
        - 0.00000016 * t_centuries ** 2
        + 0.000000504 * t_centuries ** 3
    )

    lst_rad = math.radians(lst_deg)
    eps_rad = math.radians(obliquity_deg)
    lat_rad = math.radians(latitude)

    # Standard ascendant formula (Meeus). Its raw output lands on
    # whichever of the ascendant/descendant axis points the atan2
    # branch happens to pick - by definition the Ascendant is the
    # *rising* (easterly) one, exactly 180 deg from the other, so an
    # explicit flip to that branch is required, not optional.
    numerator = -math.cos(lst_rad)
    denominator = (
        math.sin(eps_rad) * math.tan(lat_rad)
        + math.cos(eps_rad) * math.sin(lst_rad)
    )
    raw_ascendant_deg = math.degrees(math.atan2(numerator, denominator))
    tropical_ascendant_deg = (raw_ascendant_deg + 180) % 360

    ayanamsha_deg = _lahiri_ayanamsha(t_centuries)
    sidereal_ascendant_deg = (tropical_ascendant_deg - ayanamsha_deg) % 360

    rashi_index = int(sidereal_ascendant_deg // 30)
    degree_in_rashi = sidereal_ascendant_deg % 30

    # Moon's tropical ecliptic longitude -> sidereal -> Nakshatra (27
    # lunar mansions of 13°20' each) and pada (quarter, 3°20' each).
    _, moon_lon, _ = earth.at(t).observe(moon).apparent().ecliptic_latlon()
    moon_sidereal_lon = (moon_lon.degrees - ayanamsha_deg) % 360

    nakshatra_span = 360.0 / 27.0
    nakshatra_index = int(moon_sidereal_lon // nakshatra_span)
    position_in_nakshatra = moon_sidereal_lon % nakshatra_span
    pada = int(position_in_nakshatra // (nakshatra_span / 4)) + 1

    return {
        "ayanamsha_deg": ayanamsha_deg,
        "ayanamsha_system": "Lahiri (Chitrapaksha)",
        "tropical_ascendant_deg": tropical_ascendant_deg,
        "sidereal_ascendant_deg": sidereal_ascendant_deg,
        "rashi": RASHIS[rashi_index],
        "degree_in_rashi": degree_in_rashi,
        "nakshatra": NAKSHATRAS[nakshatra_index],
        "nakshatra_pada": pada,
        "moon_sidereal_longitude_deg": moon_sidereal_lon,
    }


def get_eclipses(latitude, longitude, days=180):
    """
    Upcoming lunar eclipses (real geometry, via Skyfield's own
    eclipselib) and an approximate check for solar eclipses. Skyfield
    does not include full solar eclipse path/shadow geometry, so the
    solar side here is an honest approximation: at each New Moon,
    check how close the Sun and Moon appear from this location - if
    within about the width of their disks, at least a partial solar
    eclipse is likely visible here. It will not distinguish total from
    annular from partial the way real path-of-totality maps do.
    """
    t0 = ts.now()
    t1 = ts.tt_jd(t0.tt + days)

    observer_topos = wgs84.latlon(latitude, longitude)
    observer = earth + observer_topos

    # Lunar eclipses - real Skyfield geometry.
    lunar_type_names = ["Penumbral", "Partial", "Total"]
    times, types, _details = eclipselib.lunar_eclipses(t0, t1, planets)

    lunar_events = []
    for t, typ in zip(times, types):
        alt, _az, _d = observer.at(t).observe(moon).apparent().altaz()
        lunar_events.append({
            "type": lunar_type_names[int(typ)],
            "utc": t.utc_iso(),
            "moon_above_horizon": alt.degrees > 0,
        })

    # Solar eclipses - approximate, as explained above.
    new_moon_times, phases = almanac.find_discrete(t0, t1, almanac.moon_phases(planets))

    solar_events = []
    for t, phase in zip(new_moon_times, phases):
        if int(phase) != 0:  # 0 = New Moon
            continue
        sun_pos = observer.at(t).observe(sun).apparent()
        moon_pos = observer.at(t).observe(moon).apparent()
        separation_deg = sun_pos.separation_from(moon_pos).degrees
        if separation_deg < 1.6:
            sun_alt, _az, _d = sun_pos.altaz()
            solar_events.append({
                "utc": t.utc_iso(),
                "separation_deg": separation_deg,
                "sun_above_horizon": sun_alt.degrees > 0,
            })

    return {
        "lunar_eclipses": lunar_events,
        "solar_eclipses_approx": solar_events,
    }


# ============================================================
# PANCHANG (Hindu daily almanac)
# ============================================================

TITHI_NAMES = [
    "Pratipada", "Dwitiya", "Tritiya", "Chaturthi", "Panchami", "Shashthi",
    "Saptami", "Ashtami", "Navami", "Dashami", "Ekadashi", "Dwadashi",
    "Trayodashi", "Chaturdashi", "Purnima",
    "Pratipada", "Dwitiya", "Tritiya", "Chaturthi", "Panchami", "Shashthi",
    "Saptami", "Ashtami", "Navami", "Dashami", "Ekadashi", "Dwadashi",
    "Trayodashi", "Chaturdashi", "Amavasya",
]

YOGA_NAMES = [
    "Vishkambha", "Priti", "Ayushman", "Saubhagya", "Shobhana", "Atiganda",
    "Sukarma", "Dhriti", "Shula", "Ganda", "Vriddhi", "Dhruva", "Vyaghata",
    "Harshana", "Vajra", "Siddhi", "Vyatipata", "Variyana", "Parigha",
    "Shiva", "Siddha", "Sadhya", "Shubha", "Shukla", "Brahma", "Indra",
    "Vaidhriti",
]

CHARA_KARANAS = ["Bava", "Balava", "Kaulava", "Taitila", "Gara", "Vanija", "Vishti"]

WEEKDAYS = ["Somvar", "Mangalvar", "Budhvar", "Guruvar", "Shukravar", "Shanivar", "Ravivar"]


def get_panchang(
    year, month, day, hour, minute, latitude, longitude,
    local_year=None, local_month=None, local_day=None,
):
    """
    The five limbs (Panchanga) of the Hindu almanac for a given date,
    time, and location: Tithi, Vaara, Nakshatra, Yoga, and Karana -
    all derived from real Sun/Moon ecliptic positions and the Lahiri
    ayanamsha, the same foundation as the Lagna calculation.

    year/month/day/hour/minute must together specify the correct UTC
    instant (Tithi, Nakshatra, Yoga, and Karana only depend on that
    instant, not on which calendar date label is attached to it).
    Vaara (the weekday) is different - it's inherently the LOCAL civil
    calendar date, so pass local_year/local_month/local_day (the date
    as entered/experienced by the person) separately whenever the UTC
    date could differ from it (e.g. an early-morning local time that
    falls on the previous UTC calendar day). If omitted, year/month/day
    is used for Vaara too.
    """
    t = ts.utc(year, month, day, hour, minute)
    t_centuries = (t.ut1 - 2451545.0) / 36525.0
    ayanamsha_deg = _lahiri_ayanamsha(t_centuries)

    _, sun_lon, _ = earth.at(t).observe(sun).apparent().ecliptic_latlon()
    _, moon_lon, _ = earth.at(t).observe(moon).apparent().ecliptic_latlon()
    sun_trop = sun_lon.degrees
    moon_trop = moon_lon.degrees

    # Tithi and Karana depend only on the Moon-Sun angular difference,
    # so the ayanamsha cancels out - tropical or sidereal gives the
    # same result here.
    elongation = (moon_trop - sun_trop) % 360

    tithi_index = int(elongation // 12)
    paksha = "Shukla" if tithi_index < 15 else "Krishna"
    tithi_name = TITHI_NAMES[tithi_index]

    karana_position = int(elongation // 6) + 1  # 1-60
    if karana_position == 1:
        karana_name = "Kimstughna"
    elif karana_position == 58:
        karana_name = "Shakuni"
    elif karana_position == 59:
        karana_name = "Chatushpada"
    elif karana_position == 60:
        karana_name = "Naga"
    else:
        karana_name = CHARA_KARANAS[(karana_position - 2) % 7]

    # Yoga depends on the Sun+Moon SUM, which does not cancel the
    # ayanamsha - both must be properly converted to sidereal first.
    sun_sid = (sun_trop - ayanamsha_deg) % 360
    moon_sid = (moon_trop - ayanamsha_deg) % 360
    yoga_span = 360.0 / 27.0
    yoga_index = int(((sun_sid + moon_sid) % 360) // yoga_span)

    nakshatra_span = 360.0 / 27.0
    nakshatra_index = int(moon_sid // nakshatra_span)
    position_in_nakshatra = moon_sid % nakshatra_span
    pada = int(position_in_nakshatra // (nakshatra_span / 4)) + 1

    vaara_year = local_year if local_year is not None else year
    vaara_month = local_month if local_month is not None else month
    vaara_day = local_day if local_day is not None else day
    vaara = WEEKDAYS[date(vaara_year, vaara_month, vaara_day).weekday()]

    return {
        "tithi": tithi_name,
        "paksha": paksha,
        "vaara": vaara,
        "nakshatra": NAKSHATRAS[nakshatra_index],
        "nakshatra_pada": pada,
        "yoga": YOGA_NAMES[yoga_index],
        "karana": karana_name,
        "ayanamsha_deg": ayanamsha_deg,
    }


# ============================================================
# TEST PROGRAM
# ============================================================

if __name__ == "__main__":

    print()
    print("======================================")
    print("          CHRONOSKY")
    print("     Historical Astronomy Engine")
    print("======================================")
    print()

    # Date
    year = int(input("Enter year: "))
    month = int(input("Enter month: "))
    day = int(input("Enter day: "))

    # Time
    hour = int(input("Enter hour (UTC): "))
    minute = int(input("Enter minute: "))

    # Observer Location
    latitude = float(input("Enter latitude:"))
    longitude = float(input("Enter longitude:"))

    # Calculate positions
    positions = get_planet_positions(
        year,
        month,
        day,
        hour,
        minute,
        latitude,
        longitude,
    )

    

    print()
    print("======================================")
    print("         ASTRONOMICAL DATA")
    print("======================================")
    print()

    print(
        f"Date: {day}/{month}/{year} "
        f"{hour:02d}:{minute:02d} UTC"
    )

    print("Observer location :")
    print(f"Latitude:{latitude}")
    print(f"Longitude:{longitude}")

    print()

    # Display results
    for planet, position in positions.items():

        print(f"☄️ {planet}")

        print(f"   X: {position['x_au']:.6f} AU")
        print(f"   Y: {position['y_au']:.6f} AU")
        print(f"   Z: {position['z_au']:.6f} AU")

        if planet != "Earth":

            print(
                f"   Altitude: "
                f"{position['altitude_deg']:.2f}°"
            )

            print(
                f"   Azimuth:  "
                f"{position['azimuth_deg']:.2f}°"
            )

            print(
                f"   Distance: "
                f"{position['distance_au']:.6f} AU"
            )
            print(
                f" Right Ascension: "
                f" {position['ra_hours']:.4f} hours"
            )
            print(
                f"  Declination: "
                f"{position['dec_deg']:.4f}"
            )
            print(
                 f"  Constellation: "
                 f"{position['constellation']}"
            )

        print()
