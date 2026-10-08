import io
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import numpy as np
import streamlit as st
from PIL import Image

import folium
from branca.element import MacroElement
from jinja2 import Template
from streamlit_folium import st_folium
from folium.plugins import Draw
from pyproj import Transformer

import rasterio
from rasterio.transform import from_origin

from utils.session import reset_image_state, set_native_resolution

# ==========================================
# KONFIGURACJA
# ==========================================
WMS_HIGH_URL = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/HighResolution"
WMS_STANDARD_URL = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/StandardResolution"
MAX_PX = 2500  # Bezpieczny limit dla Geoportalu
POLAND_BOUNDS = {"min_lat": 48.9, "max_lat": 55.0, "min_lon": 14.0, "max_lon": 24.2}


@st.cache_resource(show_spinner=False)
def _get_http_session():
    """Zoptymalizowana sesja HTTP z mechanizmem ponawiania."""
    session = requests.Session()
    retries = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


HTTP_SESSION = _get_http_session()
convert_to_2180_cache = None


# ==========================================
# FUNKCJE POMOCNICZE WMS & GEO
# ==========================================
def _transformed():
    global convert_to_2180_cache
    if convert_to_2180_cache is None:
        convert_to_2180_cache = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
    return convert_to_2180_cache


def _bbox_to_2180(lon_min, lat_min, lon_max, lat_max):
    t = _transformed()
    x_min, y_min = t.transform(lon_min, lat_min)
    x_max, y_max = t.transform(lon_max, lat_max)
    return x_min, y_min, x_max, y_max


def _align_bbox_and_size(x_min, y_min, x_max, y_max, res_m: float):
    """Wymusza idealną rozdzielczość siatki."""
    w_m = x_max - x_min
    h_m = y_max - y_min
    width = max(1, int(round(w_m / res_m)))
    height = max(1, int(round(h_m / res_m)))
    aligned_x_max = x_min + width * res_m
    aligned_y_max = y_min + height * res_m
    return aligned_x_max, aligned_y_max, width, height


def _fetch_wms_tiff(x_min, y_min, x_max, y_max, width, height, service_type="high"):
    """Pobiera raster z 45-sekundowym timeoutem i fallbackiem do PNG."""
    url = WMS_HIGH_URL if service_type == "high" else WMS_STANDARD_URL
    bbox_str = f"{x_min:.2f},{y_min:.2f},{x_max:.2f},{y_max:.2f}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "image/tiff,image/png,image/*,*/*",
    }

    # Pętla: najpierw próbujemy TIFF, jeśli serwer rzuci błędem - ratujemy się PNG
    for fmt in ("image/tiff", "image/png"):
        params = {
            "SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap",
            "LAYERS": "Raster", "STYLES": "", "SRS": "EPSG:2180",
            "BBOX": bbox_str, "WIDTH": str(int(width)), "HEIGHT": str(int(height)),
            "FORMAT": fmt, "TRANSPARENT": "false",
        }

        try:
            resp = HTTP_SESSION.get(url, params=params, headers=headers, timeout=45)
            content_type = resp.headers.get("Content-Type", "").lower()

            if "xml" in content_type or "html" in content_type or resp.content.startswith(b"<?xml"):
                continue  # Błąd Geoportalu w formacie XML -> spróbuj formatu PNG

            if resp.status_code == 200 and len(resp.content) > 1000:
                return resp.content
        except Exception as e:
            print(f"[WMS {service_type} {fmt} Error]: {e}")
            continue

    return None


def _is_blank_image(img_np, threshold=0.98):
    if img_np is None or img_np.size == 0:
        return True
    for channel in range(img_np.shape[2]):
        ch = img_np[:, :, channel].flatten()
        most_common_frac = np.bincount(ch).max() / len(ch)
        if most_common_frac > threshold:
            return True
    return False


def _to_georeferenced_tiff(img_np, x_min, y_max, pixel_w, pixel_h):
    try:
        transform = from_origin(x_min, y_max, pixel_w, pixel_h)
        h, w = img_np.shape[:2]
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(driver="GTiff", height=h, width=w, count=3,
                              dtype=img_np.dtype, crs="EPSG:2180", transform=transform) as ds:
                for band in range(3):
                    ds.write(img_np[:, :, band], band + 1)
            return memfile.read()
    except Exception as e:
        # ZWYKŁY PRINT - zapobiega błędowi wątków (missing ScriptRunContext)
        print(f"[GeoTIFF Warning] Błąd zapisu: {e}")
        return None


# ==========================================
# WTYCZKA MAPY (AUTOMATYCZNE USUWANIE)
# ==========================================
class KeepOnlyLastDrawing(MacroElement):
    """Błyskawicznie czyści mapę z poprzedniego kształtu, gdy tylko zaczniesz rysować nowy."""
    _template = Template("""
        {% macro script(this, kwargs) %}
            var map = {{ this._parent.get_name() }};
            // Usunięcie przy rozpoczęciu rysowania
            map.on('draw:drawstart', function (e) {
                map.eachLayer(function (layer) {
                    if (layer instanceof L.Path || layer instanceof L.Polygon) {
                        map.removeLayer(layer);
                    }
                });
            });
            // Dodatkowe zabezpieczenie po narysowaniu
            map.on('draw:created', function (e) {
                map.eachLayer(function (layer) {
                    if ((layer instanceof L.Path || layer instanceof L.Polygon) && layer !== e.layer) {
                        map.removeLayer(layer);
                    }
                });
            });
        {% endmacro %}
    """)

    def __init__(self):
        super().__init__()
        self._name = "KeepOnlyLastDrawing"


# ==========================================
# GŁÓWNY WIDOK STREAMLIT
# ==========================================
def render_map_tab():
    st.caption(
        "Narysuj prostokąt na mapie, wybierz tryb pobierania i kliknij **Pobierz**. "
        "Ortofotomapa Geoportalu dostępna jest wyłącznie dla Polski."
    )

    mode_options = {
        "Auto: Najpierw High-Res (10 cm), w razie braku Standard (25 cm)": "auto",
        "Tylko High-Res (10 cm/px — miasta/aglomeracje)": "force_high",
        "Tylko Standard (25 cm/px — szybciej, cała Polska)": "force_standard",
    }
    mode = mode_options[st.radio("Jakość ortofotomapy:", list(mode_options.keys()))]

    # Inicjalizacja stanu
    if "map_key_id" not in st.session_state:
        st.session_state["map_key_id"] = 0
    if "current_bbox" not in st.session_state:
        st.session_state["current_bbox"] = None

    # Tworzenie mapy Folium
    m = folium.Map(
        location=[52.0, 19.5], zoom_start=6, min_zoom=5,
        tiles=None, max_bounds=True,
        min_lat=POLAND_BOUNDS["min_lat"], max_lat=POLAND_BOUNDS["max_lat"],
        min_lon=POLAND_BOUNDS["min_lon"], max_lon=POLAND_BOUNDS["max_lon"],
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satelita", overlay=False, control=True).add_to(m)
    folium.TileLayer(tiles="OpenStreetMap", name="Mapa (OSM)", overlay=False, control=True).add_to(m)
    folium.LayerControl(position="topright").add_to(m)

    Draw(
        draw_options={"rectangle": True, "polygon": False, "polyline": False, "circle": False, "marker": False,
                      "circlemarker": False},
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    KeepOnlyLastDrawing().add_to(m)  # Podłączenie automatycznego czyszczenia

    col_map, col_info = st.columns([3, 1])

    with col_map:
        # Zwracamy tylko aktywne rysowanie (żeby mapa nie lagowała)
        map_data = st_folium(
            m, height=750, width="stretch",
            key=f"geoportal_map_{st.session_state['map_key_id']}",
            returned_objects=["last_active_drawing"]
        )

    # Detekcja narysowanego obszaru
    if map_data and map_data.get("last_active_drawing"):
        geom = map_data["last_active_drawing"].get("geometry", {})
        if geom.get("type") == "Polygon":
            coords = geom["coordinates"][0]
            new_bbox = (min(c[0] for c in coords), min(c[1] for c in coords),
                        max(c[0] for c in coords), max(c[1] for c in coords))

            if st.session_state["current_bbox"] != new_bbox:
                st.session_state["current_bbox"] = new_bbox
                st.rerun()

    bbox_4326 = st.session_state.get("current_bbox")

    # Panel boczny z informacjami i walidacją
    with col_info:
        st.markdown("#### Wybrany obszar")
        can_download = False

        if bbox_4326 is None:
            st.info("Narysuj prostokąt na mapie, aby pobrać.")
        else:
            lon_min, lat_min, lon_max, lat_max = bbox_4326
            x_min, y_min, x_max, y_max = _bbox_to_2180(lon_min, lat_min, lon_max, lat_max)
            w_m, h_m = x_max - x_min, y_max - y_min
            area_ha = (w_m * h_m) / 10000.0

            st.write(f"**Wymiary:** {w_m / 1000:.2f} × {h_m / 1000:.2f} km")
            st.caption(f"Powierzchnia: **{area_ha:.1f} ha**")

            # Dynamiczne limity
            max_side_m = 5000.0 if mode == "force_standard" else 2000.0
            max_area_ha = 2500.0 if mode == "force_standard" else 400.0

            outside_poland = (lon_max < POLAND_BOUNDS["min_lon"] or lon_min > POLAND_BOUNDS["max_lon"] or
                              lat_max < POLAND_BOUNDS["min_lat"] or lat_min > POLAND_BOUNDS["max_lat"])

            # Ścisła walidacja
            if outside_poland:
                st.error("Zaznaczony obszar leży poza Polską.")
            elif w_m < 50 or h_m < 50:
                st.warning("Obszar jest zbyt mały (< 50 m).")
            elif w_m > max_side_m or h_m > max_side_m:
                st.error(f"Zbyt duży bok! Limit to **{max_side_m / 1000:.1f} km**.")
            elif area_ha > max_area_ha:
                st.error(f"Zbyt duża powierzchnia! Limit to **{max_area_ha:.0f} ha**.")
            else:
                can_download = True

                # Podgląd końcowego obrazu
                target_res_preview = 0.25 if mode == "force_standard" else 0.10
                _, _, est_w, est_h = _align_bbox_and_size(x_min, y_min, x_max, y_max, target_res_preview)
                st.caption(f"Rozdzielczość: **{target_res_preview:.2f} m/px** ({int(target_res_preview * 100)} cm/px)")
                st.caption(f"Wymiary obrazu: **{est_w} × {est_h} px**")

            if st.button("❌ Wyczyść zaznaczenie", width="stretch"):
                st.session_state["current_bbox"] = None
                st.session_state["map_key_id"] += 1
                st.rerun()

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        download_clicked = st.button("⬇️ Pobierz ortofotomapę (.tiff)", type="primary", width="stretch",
                                     disabled=not can_download)

    # Akcja pobierania
    if download_clicked and can_download and bbox_4326:
        progress = st.progress(0, text="Łączenie z usługą Geoportalu…")
        img_np, used_layer_name, final_res, final_y_max = None, "", None, y_max

        # KROK 1: HighResolution
        if mode in ("auto", "force_high"):
            progress.progress(20, text="Pobieranie High-Res (10 cm/px). To może zająć do 45 sekund…")
            cur_x_max, cur_y_max, w_px, h_px = _align_bbox_and_size(x_min, y_min, x_max, y_max, 0.10)
            raw_bytes = _fetch_wms_tiff(x_min, y_min, cur_x_max, cur_y_max, w_px, h_px, "high")

            if raw_bytes:
                try:
                    temp_np = np.array(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))
                    if not _is_blank_image(temp_np):
                        img_np, used_layer_name, final_res, final_y_max = temp_np, "HighResolution", 0.10, cur_y_max
                except Exception:
                    pass

        # KROK 2: StandardResolution (Fallback lub wymuszony)
        if img_np is None:
            if mode == "force_high":
                progress.empty()
                st.error("Brak ortofotomapy HighResolution dla tego obszaru.")
                st.stop()

            if mode == "auto":
                st.info("Brak pokrycia 10 cm/px. Pobieram warstwę 25 cm/px...")

            progress.progress(40, text="Pobieranie Standard (25 cm/px)…")
            cur_x_max, cur_y_max, w_px, h_px = _align_bbox_and_size(x_min, y_min, x_max, y_max, 0.25)
            raw_bytes = _fetch_wms_tiff(x_min, y_min, cur_x_max, cur_y_max, w_px, h_px, "standard")

            if raw_bytes:
                try:
                    temp_np = np.array(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))
                    if not _is_blank_image(temp_np):
                        img_np, used_layer_name, final_res, final_y_max = temp_np, "StandardResolution", 0.25, cur_y_max
                except Exception:
                    pass

        # KROK 3: Finalizacja
        if img_np is None:
            progress.empty()
            st.error("Nie udało się pobrać zdjęcia z Geoportalu (brak danych lub przeciążenie serwera).")
        else:
            progress.progress(85, text="Zapisywanie GeoTIFF…")
            geo_bytes = _to_georeferenced_tiff(img_np, x_min, final_y_max, final_res, final_res)
            save_bytes = geo_bytes if geo_bytes is not None else raw_bytes

            reset_image_state()
            st.session_state.update({
                "img_np": img_np,
                "img_source": "map",
                "source_res": final_res,
                "_map_native_label": set_native_resolution(final_res),
                "_map_img_np": img_np,
                "_map_raw_bytes": save_bytes,
                "_map_actual_res": final_res,
                "_map_caption": f"Ortofotomapa — {used_layer_name} ({final_res:.2f} m/px)",
                "_map_filename": f"ortofoto_{lat_min:.2f}_{lon_min:.2f}_{final_res:.2f}mpx.tiff"
            })
            progress.empty()
            st.rerun()

    # Podgląd po pobraniu
    if st.session_state.get("_map_img_np") is not None:
        st.markdown("---")
        st.success(
            f"Pobrano: {st.session_state['_map_img_np'].shape[1]} × {st.session_state['_map_img_np'].shape[0]} px, rozdzielczość: **{st.session_state['_map_actual_res']:.2f} m/px**")
        c_prev, c_act = st.columns([1, 1])
        with c_prev:
            st.image(st.session_state["_map_img_np"], caption=st.session_state["_map_caption"],
                     width="stretch")
        with c_act:
            st.download_button("💾 Zapisz plik .tiff", data=st.session_state["_map_raw_bytes"],
                               file_name=st.session_state["_map_filename"], mime="image/tiff", width="stretch")
            if st.button("🗑️ Usuń pobraną ortofotomapę", width="stretch"):
                for k in ("_map_img_np", "_map_raw_bytes", "_map_actual_res", "_map_caption", "_map_filename"):
                    st.session_state.pop(k, None)
                reset_image_state()
                st.rerun()