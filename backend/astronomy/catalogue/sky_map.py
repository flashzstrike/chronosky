import io
import math

import matplotlib
matplotlib.use("Agg")  # non-interactive backend - safe for a server, no display needed
import matplotlib.pyplot as plt

try:
    # Works when imported as part of the backend package (e.g. by main.py / uvicorn)
    from backend.astronomy.catalogue.star_catalog import get_visible_stars, IMPORTANT_STARS
except ImportError:
    # Works when this file is run directly as a standalone script
    from star_catalog import get_visible_stars, IMPORTANT_STARS


# ============================================================
# CHRONOSKY — SKY MAP
# ============================================================

# Color palette - matches the frontend's navy/brass/starlight theme.
NAVY = "#0E1526"
GRID_COLOR = "#2C3A57"
BRASS = "#C9974B"
BRASS_BRIGHT = "#E0B36C"
STARLIGHT = "#F2F0E8"
DIM_STARLIGHT = "#9AA3B8"

# Reverse lookup so we can label famous stars right on the chart.
HIP_TO_NAME = {hip_id: name for name, hip_id in IMPORTANT_STARS.items()}


def _magnitude_to_marker_size(magnitude):
    """Brighter stars (lower/negative magnitude) draw bigger, not smaller."""
    return max(6, min(160, 90 - 18 * magnitude))


def _build_sky_map_figure(
    year,
    month,
    day,
    hour,
    minute,
    latitude,
    longitude
):
    """
    Build (but don't display or save) the polar sky map figure.

    Azimuth:
        0°   = North
        90°  = East
        180° = South
        270° = West

    Altitude:
        0°  = Horizon
        90° = Zenith
    """

    stars = get_visible_stars(
        year,
        month,
        day,
        hour,
        minute,
        latitude,
        longitude
    )

    theta = [star["azimuth"] * math.pi / 180 for star in stars]
    altitude = [star["altitude"] for star in stars]
    sizes = [_magnitude_to_marker_size(star["magnitude"]) for star in stars]

    fig = plt.figure(figsize=(10, 10), dpi=150, facecolor=NAVY)
    ax = fig.add_subplot(111, polar=True, facecolor=NAVY)

    # Put North at the top, angles increasing clockwise like a compass.
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # Horizon at the outer edge, zenith (straight up) at the center.
    ax.set_rlim(90, 0)

    # Soft halo behind each star, then a bright core on top - gives a
    # glow instead of a flat dot, and reflects real relative brightness.
    if theta:
        ax.scatter(
            theta, altitude, s=[s * 5 for s in sizes],
            color=STARLIGHT, alpha=0.10, linewidths=0, zorder=2,
        )
        ax.scatter(
            theta, altitude, s=[s * 2 for s in sizes],
            color=STARLIGHT, alpha=0.18, linewidths=0, zorder=2,
        )
        ax.scatter(
            theta, altitude, s=sizes,
            color=STARLIGHT, alpha=0.95, linewidths=0, zorder=3,
        )

    # Highlight and label ChronoSky's named stars, if they're up.
    for star, t, alt in zip(stars, theta, altitude):
        name = HIP_TO_NAME.get(star["hip_id"])
        if name is None:
            continue
        ax.scatter(
            [t], [alt], s=[_magnitude_to_marker_size(star["magnitude"]) * 3],
            color=BRASS_BRIGHT, alpha=0.9, linewidths=0.6,
            edgecolors=BRASS, zorder=4,
        )
        ax.annotate(
            name, (t, alt),
            textcoords="offset points", xytext=(7, 6),
            color=BRASS_BRIGHT, fontsize=9, fontweight="bold",
            fontfamily="serif", zorder=5,
        )

    # Grid + cardinal directions, styled to match the app rather than
    # matplotlib's default black-on-white look.
    ax.grid(color=GRID_COLOR, alpha=0.6, linewidth=0.7)
    ax.spines["polar"].set_color(BRASS)
    ax.spines["polar"].set_linewidth(1.3)

    ax.set_xticks([0, math.pi / 2, math.pi, 3 * math.pi / 2])
    ax.set_xticklabels(["N", "E", "S", "W"])
    for label in ax.get_xticklabels():
        label.set_color(BRASS_BRIGHT)
        label.set_fontsize(14)
        label.set_fontweight("bold")
        label.set_fontfamily("serif")

    ax.set_yticks([0, 30, 60, 90])
    ax.set_yticklabels(["90°", "60°", "30°", "0°"])
    for label in ax.get_yticklabels():
        label.set_color(DIM_STARLIGHT)
        label.set_fontsize(9)

    ax.set_title(
        f"ChronoSky — {day}/{month}/{year} "
        f"{hour:02d}:{minute:02d} UTC\n"
        f"Observer: {latitude}°, {longitude}°",
        pad=24,
        color=STARLIGHT,
        fontsize=15,
        fontfamily="serif",
    )

    if not stars:
        ax.text(
            0, 0, "No visible stars found",
            ha="center", va="center",
            color=DIM_STARLIGHT, fontsize=12,
        )

    return fig


def create_sky_map(
    year,
    month,
    day,
    hour,
    minute,
    latitude,
    longitude
):
    """Build and display the sky map interactively (desktop/local use only)."""
    fig = _build_sky_map_figure(
        year, month, day, hour, minute, latitude, longitude
    )
    plt.show()
    return fig


def render_sky_map_png(
    year,
    month,
    day,
    hour,
    minute,
    latitude,
    longitude
):
    """Build the sky map and return it as raw PNG bytes (for API responses)."""
    fig = _build_sky_map_figure(
        year, month, day, hour, minute, latitude, longitude
    )

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    # This file now defaults to the Agg backend (needed so the
    # module also works headless inside the API server), so
    # plt.show() won't pop up a window here - we save a PNG
    # to disk instead for a quick visual check.
    png_bytes = render_sky_map_png(
        1969,
        7,
        20,
        20,
        17,
        26.4207,
        50.0888
    )

    with open("chronosky_test_output.png", "wb") as f:
        f.write(png_bytes)

    print("Saved test sky map to chronosky_test_output.png")
