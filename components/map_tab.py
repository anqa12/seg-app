import io
import time
from functools import lru_cache
import cv2

import numpy as np
import requests
import streamlit as st
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

import folium
from folium.plugins import Draw, Geocoder
from pyproj import Transformer
from streamlit_folium import st_folium

import rasterio
from rasterio.transform import from_origin

from utils.session import reset_image_state, set_native_resolution

# config params
WMS_URL = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/StandardResolution"  # url for ortophomap
RES_M = 0.25  # standard res 25 cm/px
MAX_PX = 4096  # max pxl to avoid wms errors
HTTP_TIMEOUT = (5, 60)  # (connect, read) in seconds
# Geoportal sometimes answers 404/5xx for a request that works a moment later.
# Such answers are retried after these pauses (seconds): 5 attempts in total, ~13 s of waiting.
RETRY_PAUSES = (1.0, 2.0, 4.0, 6.0)
TRANSIENT_STATUSES = (404, 429, 500, 502, 503, 504)
POLAND_BOUNDS = {"min_lat": 48.9, "max_lat": 55.0, "min_lon": 14.0, "max_lon": 24.2}  # Poland bounds for displaying


def _get_http_session() -> requests.Session:
    """
    Create and configure a requests.Session with automated retry logic.
    Only connection problems are retried here. Temporary HTTP errors (404, 429, 5xx) are
    retried in _fetch_wms instead, so that every new attempt can use a new connection.
    """
    session = requests.Session()
    retries = Retry(
        connect=1,
        read=False,
        backoff_factor=1.0,
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


@lru_cache(maxsize=1)
def _transformer() -> Transformer:
    """
    Retrieve or cache the coordinate transformation engine from WGS84 (EPSG:4326)
    to the Polish official projected coordinate system Poland 1992 (EPSG:2180).
    `always_xy=True` enforces consistent (longitude, latitude) / (x, y) ordering.
    """
    return Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)


def _bbox_to_2180(lon_min, lat_min, lon_max, lat_max):
    """
    Convert a bounding box from geographic coordinates (EPSG:4326)
    to projected metric coordinates (EPSG:2180).
    """
    t = _transformer()
    x_min, y_min = t.transform(lon_min, lat_min)
    x_max, y_max = t.transform(lon_max, lat_max)
    return x_min, y_min, x_max, y_max


def _calc_image_size(w_m: float, h_m: float) -> tuple[int, int]:
    """
    Calculate the raster dimensions in pixels based on the target ground resolution (RES_M).
    Downscales proportionally if either dimension exceeds MAX_PX to prevent WMS out-of-memory errors.
    """
    width = max(1, int(round(w_m / RES_M)))
    height = max(1, int(round(h_m / RES_M)))
    if max(width, height) > MAX_PX:
        scale = MAX_PX / max(width, height)
        width = max(64, int(width  * scale))
        height = max(64, int(height * scale))
    return width, height


def _fetch_wms(x_min, y_min, x_max, y_max, width, height,
               on_progress=None, on_status=None) -> tuple[bytes | None, str | None]:
    """
    Query the Geoportal WMS GetMap service to retrieve aerial orthoimagery for a given metric BBOX.
    Iterates through image formats (JPEG, PNG, TIFF) as fallback strategies and reports progress.
    Returns (raw_bytes, None) on success or (None, error_message) on failure.

    Temporary errors (404/429/5xx, connection problems) are retried after growing pauses
    (RETRY_PAUSES), and on_status(text) tells the UI what is going on during the wait.
    """
    bbox_str = f"{x_min:.2f},{y_min:.2f},{x_max:.2f},{y_max:.2f}"

    # Custom headers are required; bare requests User-Agents are blocked by the government firewall (WAF)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "image/jpeg,image/png,image/tiff,image/*,*/*",
    }

    for fmt in ("image/jpeg", "image/png", "image/tiff"):
        params = {
            "SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap",
            "LAYERS": "Raster", "STYLES": "", "SRS": "EPSG:2180",
            "BBOX": bbox_str,
            "WIDTH": str(int(width)), "HEIGHT": str(int(height)),
            "FORMAT": fmt, "TRANSPARENT": "false",
        }
        last_problem = None  # last temporary problem, used for the final message

        for attempt, pause in enumerate((0.0,) + RETRY_PAUSES):
            if pause:
                if on_status:
                    on_status(f"Geoportal chwilowo nie odpowiada ({last_problem}). "
                              f"Ponawiam, próba {attempt + 1} z {len(RETRY_PAUSES) + 1}…")
                time.sleep(pause)

            # A NEW session (= a new connection) for every attempt. A kept-alive connection
            # may stay tied to the same, currently failing, server behind Geoportal's gateway,
            # while a new connection can be served by another one.
            session = _get_http_session()
            try:
                with session.get(WMS_URL, params=params, headers=headers,
                                 timeout=HTTP_TIMEOUT, stream=True) as resp:
                    ct = resp.headers.get("Content-Type", "").lower()

                    if resp.status_code in TRANSIENT_STATUSES:
                        last_problem = f"HTTP {resp.status_code}"
                        # visible in the terminal, to see what Geoportal really answers
                        print(f"[WMS] {fmt}, attempt {attempt + 1}: HTTP {resp.status_code}, "
                              f"body: {resp.content[:150]!r}")
                        continue

                    # Geoportal often returns HTTP 200 with an XML ServiceException payload on internal failure
                    if "xml" in ct or "html" in ct:
                        break

                    if resp.status_code != 200:
                        return None, f"Geoportal odpowiedział błędem HTTP {resp.status_code}."

                    # stream progress in UI
                    total = int(resp.headers.get("Content-Length") or 0)
                    chunks, received, last_update = [], 0, 0.0
                    for chunk in resp.iter_content(chunk_size=256 * 1024):
                        chunks.append(chunk)
                        received += len(chunk)
                        if on_progress and time.monotonic() - last_update > 0.3:
                            on_progress(received, total)
                            last_update = time.monotonic()
                    data = b"".join(chunks)

                # validate data integrity
                if len(data) < 1000 or data.startswith(b"<?xml"):
                    break

                return data, None

            except requests.exceptions.Timeout:
                return None, (
                    f"Geoportal nie odpowiedział w ciągu {HTTP_TIMEOUT[1]} s. "
                    "Zaznacz mniejszy obszar lub spróbuj za chwilę."
                )
            except requests.exceptions.ConnectionError:
                last_problem = "błąd połączenia"
                continue
            except Exception as e:
                return None, f"Nieoczekiwany błąd: {e}"
            finally:
                session.close()

        else:
            # all attempts were used up on temporary errors (the format is not the cause,
            # so there is no point in trying the next one)
            if last_problem == "HTTP 404":
                return None, (
                    "Geoportal wielokrotnie zwrócił błąd 404 (brak danych albo chwilowa awaria po "
                    "jego stronie). Spróbuj za chwilę albo lekko przesuń zaznaczony prostokąt."
                )
            if last_problem == "błąd połączenia":
                return None, (
                    "Serwer Geoportalu jest chwilowo niedostępny. "
                    "Spróbuj ponownie za chwilę."
                )
            return None, (
                f"Geoportal jest przeciążony ({last_problem}). "
                "Spróbuj ponownie za chwilę."
            )

    return None, "Geoportal nie zwrócił obrazu (brak ortofotomapy dla tego obszaru)."


def _is_blank(img_np, threshold=0.98) -> bool:
    """
    Check if an image consists of a single uniform color.
    Downsamples by a stride of 4 for speed optimization.
    """
    if img_np is None or img_np.size == 0:
        return True
    sample = img_np[::4, ::4]
    for ch in range(sample.shape[2]):
        flat = sample[:, :, ch].flatten()
        if np.bincount(flat).max() / len(flat) > threshold:
            return True
    return False


def _to_geotiff(img_np, x_min, y_max, pixel_size) -> bytes | None:
    """
    Encode an RGB NumPy array into GeoTIFF binary bytes in memory, attaching EPSG:2180 spatial georeference.
    """
    try:
        transform = from_origin(x_min, y_max, pixel_size, pixel_size)
        h, w = img_np.shape[:2]
        with rasterio.io.MemoryFile() as mf:
            with mf.open(driver="GTiff", height=h, width=w, count=3,
                         dtype=img_np.dtype, crs="EPSG:2180",
                         transform=transform) as ds:
                for band in range(3):
                    ds.write(img_np[:, :, band], band + 1)
            return mf.read()
    except Exception as e:
        print(f"[GeoTIFF] Błąd zapisu: {e}")
        return None


def _plain_tiff(img_np) -> bytes:
    """Fallback standard TIFF exporter without spatial metadata if GeoTIFF encoding fails."""
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format="TIFF")
    return buf.getvalue()


# MAIN CUI COMPONENT
def render_map_tab() -> None:
    """
    Main function to render the map tab.
    Embeds a Folium map, tracks drawn bounding boxes, validates constraints, and initiates WMS fetching.
    """
    st.caption(
        "Narysuj prostokąt na mapie i kliknij **Pobierz**. "
        "Ortofotomapa standardowa (25 cm/px) pokrywa całą Polskę. "
        "Maksymalny obszar: ok. **1024 m × 1024 m** (4096 × 4096 px)."
    )

    if "map_key_id" not in st.session_state:
        st.session_state["map_key_id"] = 0
    if "current_bbox" not in st.session_state:
        st.session_state["current_bbox"] = None

    # Construct Folium map centered over Poland with spatial boundary limits
    m = folium.Map(
        location=[52.0, 19.5], zoom_start=6, min_zoom=5,
        tiles=None, max_bounds=True,
        min_lat=POLAND_BOUNDS["min_lat"], max_lat=POLAND_BOUNDS["max_lat"],
        min_lon=POLAND_BOUNDS["min_lon"], max_lon=POLAND_BOUNDS["max_lon"],
    )
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri, Maxar, Earthstar Geographics",
        name="Satelita", overlay=False, control=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="OpenStreetMap", name="Mapa (OSM)",
        overlay=False, control=True,
    ).add_to(m)
    folium.LayerControl(position="topright").add_to(m)

    # Only draw rectangles
    Draw(
        draw_options={
            "rectangle": True, "polygon": False, "polyline": False,
            "circle": False, "marker": False, "circlemarker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    # Search Bar for address
    Geocoder(
        collapsed=True,
        position="topleft",
        placeholder="Szukaj adresu lub miejscowości...",
        add_marker=True,
    ).add_to(m)

    col_map, col_info = st.columns([3, 1], gap="medium")

    with col_map:
        # Static key ensures the iframe is preserved without reloading
        map_data = st_folium(
            m,
            height=750,
            width=True,
            key="geoportal_map",
            returned_objects=["all_drawings"],
        )

    # Evaluate all geometry shapes currently present on the canvas
    all_drawings = (map_data.get("all_drawings") or []) if map_data else []
    num_drawings = len(all_drawings)

    can_download = False
    bbox_4326 = None
    x_min = y_min = x_max = y_max = w_m = h_m = 0

    with col_info:
        st.markdown("#### Wybrany obszar")
        max_side_m = MAX_PX * RES_M  # Maximum allowed extent in meters

        # 0 rectangles
        if num_drawings == 0:
            st.info("Narysuj prostokąt na mapie, aby móc pobrać ortofotomapę.")

        # > 1 rectangles
        elif num_drawings > 1:
            st.error(
                f"Wykryto **{num_drawings}** zaznaczone obszary! "
                "Na mapie może znajdować się tylko **jeden prostokąt**. "
                "Użyj ikony kosza po lewej stronie mapy, aby usunąć nadmiar."
            )

        else:
            # one rectangle
            geom = all_drawings[0].get("geometry", {})
            if geom.get("type") == "Polygon":
                coords = geom["coordinates"][0]
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                bbox_4326 = (min(lons), min(lats), max(lons), max(lats))

                lon_min, lat_min, lon_max, lat_max = bbox_4326
                x_min, y_min, x_max, y_max = _bbox_to_2180(
                    lon_min, lat_min, lon_max, lat_max
                )
                w_m = x_max - x_min
                h_m = y_max - y_min

                outside = (
                        lon_max < POLAND_BOUNDS["min_lon"]
                        or lon_min > POLAND_BOUNDS["max_lon"]
                        or lat_max < POLAND_BOUNDS["min_lat"]
                        or lat_min > POLAND_BOUNDS["max_lat"]
                )

                st.write(f"**Wymiary:** {w_m:.0f} × {h_m:.0f} m")
                st.caption(f"Powierzchnia: **{(w_m * h_m) / 10_000:.1f} ha**")

                if outside:
                    st.error("Obszar poza Polską.")
                elif w_m < 50 or h_m < 50:
                    st.warning("⚠Obszar zbyt mały (< 50 m).")
                elif w_m > max_side_m or h_m > max_side_m:
                    st.error(
                        f"Obszar zbyt duży! Limit to **{max_side_m:.0f} m** na bok "
                        f"({MAX_PX} px przy 25 cm/px). Zmniejsz zaznaczenie."
                    )
                else:
                    w_px, h_px = _calc_image_size(w_m, h_m)
                    st.caption(f"Rozdzielczość: **{RES_M:.2f} m/px** (25 cm/px)")
                    st.caption(f"Obraz: **{w_px} × {h_px} px**")
                    can_download = True

        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
        download_clicked = st.button(
            "Pobierz",
            type="primary",
            disabled=not can_download,
        )

    # WMS downloading
    if download_clicked and can_download:
        w_px, h_px = _calc_image_size(w_m, h_m)

        progress = st.progress(0, text="Łączenie z Geoportalem…")

        def _on_progress(received: int, total: int) -> None:
            if total:
                progress.progress(
                    min(80, 20 + int(60 * received / total)),
                    text=f"Pobieranie {received/1e6:.1f} / {total/1e6:.1f} MB",
                )
            else:
                progress.progress(30, text=f"Pobieranie {received/1e6:.1f} MB")

        def _on_status(message: str) -> None:
            progress.progress(15, text=message)

        progress.progress(15, text="Pobieranie ortofotomapy standardowej (25 cm/px)")
        raw_bytes, error = _fetch_wms(x_min, y_min, x_max, y_max, w_px, h_px, _on_progress, _on_status)

        if raw_bytes is None:
            progress.empty()
            st.error(f"{error}")
        else:
            progress.progress(85, text="Wczytywanie i zapis pliku")
            try:
                img_np = np.array(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))
            except Exception as e:
                progress.empty()
                st.error(f"Nie można odczytać obrazu: {e}")
                return

            if _is_blank(img_np):
                progress.empty()
                st.error(
                    "Geoportal zwrócił pusty obraz. "
                    "Sprawdź czy zaznaczony obszar leży w Polsce i spróbuj ponownie."
                )
                with st.expander("Podgląd (diagnostyka)"):
                    st.image(img_np, width=600)
            else:
                # Align array dimensions if the server adjusted width/height to match aspect ratio
                if img_np.shape[:2] != (h_px, w_px):
                    img_np = np.array(
                        Image.fromarray(img_np).resize((w_px, h_px), Image.BILINEAR)
                    )

                geo_bytes = _to_geotiff(img_np, x_min, y_max, RES_M)
                save_bytes = geo_bytes if geo_bytes is not None else _plain_tiff(img_np)

                # Keep the downloaded image to global session state for inference
                reset_image_state()
                st.session_state.update({
                    "img_np":              img_np,
                    "img_source":          "map",
                    "source_res":          RES_M,
                    "_map_native_label":   set_native_resolution(RES_M),
                    "_map_img_np":         img_np,
                    "_map_raw_bytes":      save_bytes,
                    "_map_actual_res":     RES_M,
                    "_map_caption":        "Ortofotomapa standardowa Geoportalu (25 cm/px)",
                    "_map_filename":       (
                        f"ortofoto_{lat_min:.2f}_{lon_min:.2f}_25cm.tiff"
                    ),
                })
                progress.empty()
                st.rerun()

    # Preview of the downloaded image
    if st.session_state.get("_map_img_np") is not None:
        st.markdown("---")
        img = st.session_state["_map_img_np"]
        caption = f"Ortofotomapa z Geoportalu · {img.shape[1]} × {img.shape[0]} px · 25 cm/px",

        # Główny podział: lewa połowa na zdjęcie i przyciski, prawa pusta
        col_img, _ = st.columns([1, 1])

        with col_img:
            # Sztuczne powiększenie dla lupy (pełnego ekranu)
            disp_img = img
            h, w = img.shape[:2]
            if max(h, w) < 800:
                scale = 800 / max(h, w)
                disp_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)

            # Zdjęcie wypełnia dokładnie szerokość lewej kolumny
            st.image(disp_img, caption=st.session_state["_map_caption"], width="stretch")

            # Sub-kolumny na przyciski - dzielą szerokość pod zdjęciem idealnie 50/50
            col_zapisz, col_usun = st.columns(2)

            with col_zapisz:
                st.download_button(
                    "Zapisz",
                    data=st.session_state["_map_raw_bytes"],
                    file_name=st.session_state["_map_filename"],
                    mime="image/tiff",
                    width="stretch",
                    type="primary"
                )
            with col_usun:
                if st.button("Usuń", width="stretch", type="tertiary"):
                    for k in ("_map_img_np", "_map_raw_bytes", "_map_actual_res",
                              "_map_caption", "_map_filename", "_map_native_label"):
                        st.session_state.pop(k, None)
                    reset_image_state()
                    st.rerun()