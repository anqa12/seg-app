import io
import requests
import numpy as np
import streamlit as st
from PIL import Image

import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
from pyproj import Transformer

import rasterio
from rasterio.transform import from_origin

from utils.session import reset_image_state, set_native_resolution

WMS_HIGH_URL = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/HighResolution"
WMS_STANDARD_URL = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/StandardResolution"
MAX_PX = 4096

POLAND_BOUNDS = {"min_lat": 48.9, "max_lat": 55.0, "min_lon": 14.0, "max_lon": 24.2}

convert_to_2180_cache = None


# caches value for conversion from EPSG:4326 to EPSG:2180
def _transformed():

    global convert_to_2180_cache
    if convert_to_2180_cache is None:
        convert_to_2180_cache = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
    return convert_to_2180_cache


# transforms selected region using cached value
def _bbox_to_2180(lon_min, lat_min, lon_max, lat_max):

    t = _transformed()
    x_min, y_min = t.transform(lon_min, lat_min)
    x_max, y_max = t.transform(lon_max, lat_max)
    return x_min, y_min, x_max, y_max


# calculates pixel value for given resolution - takes into account MAX_PX
def _calc_px(w_m, h_m, res_m: float):

    width = max(1, int(round(w_m / res_m)))
    height = max(1, int(round(h_m / res_m)))
    if max(width, height) > MAX_PX:
        scale = MAX_PX / max(width, height)
        width = max(64, int(width * scale))
        height = max(64, int(height * scale))
    actual_res = ((w_m / width) + (h_m / height)) / 2.0
    return width, height, actual_res


# downloads raster from WMS 1.1.1 from Geoportal
def _fetch_wms_tiff(x_min, y_min, x_max, y_max, width, height, service_type="high"):

    url = WMS_HIGH_URL if service_type == "high" else WMS_STANDARD_URL

    # round bbox to 2 decimal points
    bbox_str = f"{x_min:.2f},{y_min:.2f},{x_max:.2f},{y_max:.2f}"

    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": "Raster",
        "STYLES": "",
        "SRS": "EPSG:2180",
        "BBOX": bbox_str,
        "WIDTH": str(int(width)),
        "HEIGHT": str(int(height)),
        "FORMAT": "image/tiff",
        "TRANSPARENT": "false",
    }

    # required website User-Agent, in case of rejecting request (error 401/402/403)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "image/tiff,image/*,*/*",
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=60)

        content_type = resp.headers.get("Content-Type", "")
        if "xml" in content_type.lower() or "html" in content_type.lower():
            return None

        resp.raise_for_status()
        return resp.content
    except requests.exceptions.Timeout:
        st.error("Przekroczono czas oczekiwania (60s). Zmniejsz obszar na mapie.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"Błąd serwera Geoportalu ({service_type}): {e}")
        return None
    except Exception as e:
        st.error(f"Błąd połączenia ({service_type}): {e}")
        return None


def _is_blank_image(img_np, threshold=0.98):
    if img_np is None:
        return True
    for channel in range(img_np.shape[2]):
        ch = img_np[:, :, channel].flatten()
        most_common_frac = np.bincount(ch).max() / len(ch)
        if most_common_frac > threshold:
            return True
    return False


# adds metadata to the .tiff file
def _to_georeferenced_tiff(img_np, x_min, y_max, pixel_w, pixel_h):
    try:
        transform = from_origin(x_min, y_max, pixel_w, pixel_h)
        h, w = img_np.shape[:2]
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(driver="GTiff", height=h, width=w, count=3,
                              dtype=img_np.dtype, crs="EPSG:2180",
                              transform=transform) as ds:
                for band in range(3):
                    ds.write(img_np[:, :, band], band + 1)
            return memfile.read()
    except Exception as e:
        st.caption(f"(Nie udało się dopisać georeferencji: {e})")
        return None


def render_map_tab():
    """Main function for rendering map tab"""

    st.caption(
        "Narysuj prostokąt na mapie, wybierz tryb pobierania i kliknij **Pobierz**. "
        "Ortofotomapa Geoportalu dostępna jest wyłącznie dla Polski."
    )

    # select option for downloading the data
    mode_options = {
        "Auto: Najpierw High-Res (10 cm), w razie braku Standard (25 cm)": "auto",
        "Tylko High-Res (10 cm/px — miasta/aglomeracje)": "force_high",
        "Tylko Standard (25 cm/px — szybciej, cała Polska)": "force_standard",
    }
    selected_mode_label = st.radio(
        "Jakość ortofotomapy:",
        list(mode_options.keys()),
        index=0,
        horizontal=False
    )
    mode = mode_options[selected_mode_label]

    # map
    m = folium.Map(
        location=[52.0, 19.5], zoom_start=6, min_zoom=5,
        tiles=None, max_bounds=True,
        min_lat=POLAND_BOUNDS["min_lat"], max_lat=POLAND_BOUNDS["max_lat"],
        min_lon=POLAND_BOUNDS["min_lon"], max_lon=POLAND_BOUNDS["max_lon"],
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics",
        name="Satelita", overlay=False, control=True,
    ).add_to(m)
    folium.TileLayer(tiles="OpenStreetMap", name="Mapa (OSM)",
                     overlay=False, control=True).add_to(m)
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    Draw(
        draw_options={
            "rectangle": True, "polygon": False, "polyline": False,
            "circle": False, "marker": False, "circlemarker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    map_data = st_folium(m, height=650, width="100%", key="geoportal_map")

    bbox_4326 = None
    if map_data and map_data.get("last_active_drawing"):
        geom = map_data["last_active_drawing"].get("geometry", {})
        if geom.get("type") == "Polygon":
            coords = geom["coordinates"][0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            bbox_4326 = (min(lons), min(lats), max(lons), max(lats))

    if bbox_4326:
        lon_min, lat_min, lon_max, lat_max = bbox_4326
        x_min, y_min, x_max, y_max = _bbox_to_2180(lon_min, lat_min, lon_max, lat_max)
        w_m = x_max - x_min
        h_m = y_max - y_min

        st.markdown(
            f"**Obszar:** {w_m / 1000:.2f} km × {h_m / 1000:.2f} km &nbsp;|&nbsp; "
            f"**Współrzędne EPSG:2180:** X: `{x_min:.0f}` - `{x_max:.0f}`, Y: `{y_min:.0f}` - `{y_max:.0f}`"
        )

        outside_poland = (
                lon_max < POLAND_BOUNDS["min_lon"] or lon_min > POLAND_BOUNDS["max_lon"] or
                lat_max < POLAND_BOUNDS["min_lat"] or lat_min > POLAND_BOUNDS["max_lat"]
        )
        if outside_poland:
            st.error("Zaznaczony obszar leży poza Polską.")

        if st.button("⬇️ Pobierz ortofotomapę (.tiff)", type="primary",
                     width='stretch', disabled=outside_poland):
            progress = st.progress(0, text="Łączenie z usługą Geoportalu…")
            img_np = None
            used_layer_name = ""
            actual_res = None

            # high resolution
            if mode in ("auto", "force_high"):
                progress.progress(20, text="Sprawdzanie HighResolution (10 cm/px)…")
                w_high, h_high, res_high = _calc_px(w_m, h_m, res_m=0.10)
                raw_bytes = _fetch_wms_tiff(x_min, y_min, x_max, y_max, w_high, h_high, service_type="high")

                if raw_bytes:
                    try:
                        img_pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                        temp_np = np.array(img_pil)
                        if not _is_blank_image(temp_np):
                            img_np = temp_np
                            used_layer_name = "HighResolution (~10 cm/px)"
                            actual_res = res_high
                    except Exception:
                        pass

            # standard resolution (if high resolution fails or if selected option was so
            if img_np is None:
                if mode == "force_high":
                    progress.empty()
                    st.error("Brak ortofotomapy HighResolution dla tego obszaru. Zmień tryb na Auto lub Standard.")
                    st.stop()
                else:
                    if mode == "auto":
                        st.info("Brak pokrycia 10 cm/px. Automatycznie przełączenie na warstwę standardową (25 cm/px)...")

                    progress.progress(55, text="Pobieranie StandardResolution (25 cm/px)…")
                    w_std, h_std, res_std = _calc_px(w_m, h_m, res_m=0.25)
                    raw_bytes = _fetch_wms_tiff(x_min, y_min, x_max, y_max, w_std, h_std, service_type="standard")

                    if raw_bytes:
                        try:
                            img_pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                            temp_np = np.array(img_pil)
                            if not _is_blank_image(temp_np):
                                img_np = temp_np
                                used_layer_name = "StandardResolution (~25-50 cm/px)"
                                actual_res = res_std
                        except Exception as e:
                            st.error(f"Błąd odczytu obrazu: {e}")

            # save and summarize
            if img_np is None:
                progress.empty()
                st.error("Nie udało się pobrać zdjęcia (obszar poza zasięgiem lub błąd serwera).")
            else:
                progress.progress(85, text="Przygotowywanie GeoTIFF…")
                real_h, real_w = img_np.shape[:2]
                pixel_w = w_m / real_w
                pixel_h = h_m / real_h
                actual_res = (pixel_w + pixel_h) / 2.0

                geo_bytes = _to_georeferenced_tiff(img_np, x_min, y_max, pixel_w, pixel_h)
                save_bytes = geo_bytes if geo_bytes is not None else raw_bytes

                reset_image_state()
                st.session_state["img_np"] = img_np
                st.session_state["img_source"] = "map"
                st.session_state["source_res"] = round(actual_res, 4)
                set_native_resolution(actual_res)

                st.session_state["_map_img_np"] = img_np
                st.session_state["_map_raw_bytes"] = save_bytes
                st.session_state["_map_actual_res"] = actual_res
                st.session_state["_map_caption"] = (
                    f"Ortofotomapa Geoportalu — {used_layer_name} ({actual_res:.2f} m/px)"
                )
                st.session_state["_map_filename"] = (
                    f"ortofoto_{lat_min:.2f}_{lon_min:.2f}_{actual_res:.2f}mpx.tiff"
                )
                progress.progress(100, text="Pobrano!")
                progress.empty()
                st.rerun()
    else:
        if st.session_state.get("_map_img_np") is None:
            st.caption("Narysuj prostokąt na mapie, aby odblokować przycisk pobierania.")

    # preview and save on disk button
    if st.session_state.get("_map_img_np") is not None:
        st.success(
            f"Pobrano ortofotomapę: "
            f"{st.session_state['_map_img_np'].shape[1]} × {st.session_state['_map_img_np'].shape[0]} px, "
            f"rozdzielczość terenowa: **{st.session_state['_map_actual_res']:.2f} m/px**"
        )
        st.image(
            st.session_state["_map_img_np"],
            caption=st.session_state["_map_caption"],
            width=500,
        )
        st.download_button(
            label="💾 Zapisz plik .tiff",
            data=st.session_state["_map_raw_bytes"],
            file_name=st.session_state["_map_filename"],
            mime="image/tiff",
            width='stretch',
        )
        if st.button("🗑️ Usuń pobraną ortofotomapę", width='stretch'):
            for k in ("_map_img_np", "_map_raw_bytes", "_map_actual_res",
                      "_map_caption", "_map_filename"):
                st.session_state.pop(k, None)
            reset_image_state()
            st.rerun()