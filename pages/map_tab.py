"""
map_tab.py
----------
Fragment do wklejenia do segmentation.py jako trzecia zakładka.

Użycie w segmentation.py:
    from map_tab import render_map_tab

    tab_sample, tab_map, tab_upload = st.tabs([
        "📂 Zdjęcia testowe", "🗺️ Pobierz z mapy", "⬆️ Wgraj własne"
    ])

    with tab_map:
        render_map_tab()
"""

import io
import requests
import numpy as np
import streamlit as st
from PIL import Image

try:
    import folium
    from streamlit_folium import st_folium
    from folium.plugins import Draw
    from pyproj import Transformer
    FOLIUM_OK = True
except ImportError:
    FOLIUM_OK = False

# ── Stałe WMS ────────────────────────────────────────────────────────────────
WMS_URL     = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/HighResolution"
WMS_LAYER   = "Raster"
WMS_VERSION = "1.3.0"
WMS_CRS     = "EPSG:2180"
MAX_PX      = 4096   # limit pikseli WMS Geoportalu

_t_to_2180 = None

def _transformer():
    global _t_to_2180
    if _t_to_2180 is None:
        _t_to_2180 = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
    return _t_to_2180


def _bbox_to_2180(lon_min, lat_min, lon_max, lat_max):
    t = _transformer()
    x_min, y_min = t.transform(lon_min, lat_min)
    x_max, y_max = t.transform(lon_max, lat_max)
    return x_min, y_min, x_max, y_max


def _image_size(x_min, y_min, x_max, y_max, target_res: float):
    """Oblicza WIDTH/HEIGHT w px dla zadanej rozdzielczości, przycina do MAX_PX."""
    w_m = x_max - x_min
    h_m = y_max - y_min
    width  = int(round(w_m / target_res))
    height = int(round(h_m / target_res))

    if max(width, height) > MAX_PX:
        scale  = MAX_PX / max(width, height)
        width  = max(64, int(width  * scale))
        height = max(64, int(height * scale))

    actual_res = ((w_m / width) + (h_m / height)) / 2.0
    return width, height, actual_res


def _fetch_wms_tiff(x_min, y_min, x_max, y_max, width, height) -> bytes | None:
    """
    Pobiera fragment ortofotomapy jako GeoTIFF z WMS Geoportalu.
    Bbox w EPSG:2180, kolejność WMS 1.3.0: minNorthing,minEasting,maxNorthing,maxEasting.
    Zwraca surowe bajty pliku TIFF lub None przy błędzie.
    """
    bbox_str = f"{y_min},{x_min},{y_max},{x_max}"
    params = {
        "SERVICE":     "WMS",
        "VERSION":     WMS_VERSION,
        "REQUEST":     "GetMap",
        "LAYERS":      WMS_LAYER,
        "STYLES":      "",
        "CRS":         WMS_CRS,
        "BBOX":        bbox_str,
        "WIDTH":       str(width),
        "HEIGHT":      str(height),
        "FORMAT":      "image/tiff",
        "TRANSPARENT": "false",
    }
    try:
        resp = requests.get(WMS_URL, params=params, timeout=60)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type and "tiff" not in content_type:
            # Geoportal czasem zwraca JPEG nawet przy FORMAT=tiff — obsłuż oba
            st.warning(f"WMS zwrócił: {content_type} — spróbuję wczytać jako obraz.")

        return resp.content

    except requests.exceptions.Timeout:
        st.error("Przekroczono czas oczekiwania (60s). Spróbuj mniejszy obszar lub niższą rozdzielczość.")
        return None
    except Exception as e:
        st.error(f"Błąd pobierania WMS: {e}")
        return None


def render_map_tab():
    """
    Renderuje zakładkę z interaktywną mapą Folium + pobieraniem WMS Geoportalu.
    """
    if not FOLIUM_OK:
        st.error("Brakuje pakietów: `pip install streamlit-folium folium pyproj`")
        return

    st.caption(
        "Narysuj prostokąt na mapie narzędziem ▭, wybierz rozdzielczość i kliknij **Pobierz**. "
        "Ortofotomapa dotyczy tylko obszaru Polski."
    )

    # ── Suwak rozdzielczości ──────────────────────────────────────────────────
    res_options = {
        "10 cm/px  (bardzo duże pliki, wolno)": 0.10,
        "25 cm/px  (zalecane — jak LandCover)": 0.25,
        "50 cm/px":                              0.50,
        "1 m/px    (szybko, mały plik)":         1.00,
    }
    res_label  = st.radio(
        "Rozdzielczość pobierania",
        list(res_options.keys()),
        index=1,
        horizontal=True,
        help="Wpływa na rozmiar obrazu i czas pobierania. "
             "Przy 10 cm/px dla dużego obszaru obraz może być przycięty do 4096 px."
    )
    target_res = res_options[res_label]

    # ── Mapa ─────────────────────────────────────────────────────────────────
    m = folium.Map(location=[52.0, 19.5], zoom_start=7, tiles="OpenStreetMap")
    Draw(
        draw_options={
            "rectangle":    True,
            "polygon":      False,
            "polyline":     False,
            "circle":       False,
            "marker":       False,
            "circlemarker": False,
        },
        edit_options={"edit": False},
    ).add_to(m)

    map_data = st_folium(m, height=480, width="100%", key="geoportal_map")

    # ── Odczyt narysowanego prostokąta ────────────────────────────────────────
    bbox_4326 = None
    if map_data and map_data.get("last_active_drawing"):
        geom = map_data["last_active_drawing"].get("geometry", {})
        if geom.get("type") == "Polygon":
            coords = geom["coordinates"][0]
            lons   = [c[0] for c in coords]
            lats   = [c[1] for c in coords]
            bbox_4326 = (min(lons), min(lats), max(lons), max(lats))

    if not bbox_4326:
        st.caption("Narysuj prostokąt na mapie żeby zobaczyć opcje pobierania.")
        return

    # ── Podgląd parametrów ───────────────────────────────────────────────────
    lon_min, lat_min, lon_max, lat_max = bbox_4326
    x_min, y_min, x_max, y_max        = _bbox_to_2180(lon_min, lat_min, lon_max, lat_max)
    w_m   = x_max - x_min
    h_m   = y_max - y_min
    width, height, actual_res = _image_size(x_min, y_min, x_max, y_max, target_res)

    capped = actual_res > target_res * 1.05   # czy przycięto do MAX_PX

    col_info, col_warn = st.columns([3, 2])
    with col_info:
        st.markdown(
            f"**Obszar:** {w_m/1000:.2f} km × {h_m/1000:.2f} km &nbsp;·&nbsp; "
            f"**Obraz:** {width} × {height} px &nbsp;·&nbsp; "
            f"**Rozdzielczość:** `{actual_res:.4f} m/px`"
        )
    with col_warn:
        if capped:
            st.warning(
                f"⚠️ Obszar zbyt duży dla {target_res} m/px — "
                f"obraz przycięty do {MAX_PX} px. "
                f"Faktyczna rozdzielczość: {actual_res:.4f} m/px."
            )
        elif w_m > 10_000 or h_m > 10_000:
            st.info("Duży obszar — pobieranie może chwilę potrwać.")

    # ── Przycisk pobierania ───────────────────────────────────────────────────
    if not st.button("⬇️ Pobierz ortofotomapę (.tiff)", type="primary",
                     use_container_width=True):
        return

    # Pasek postępu — WMS to jedno żądanie, więc symulujemy etapy
    progress = st.progress(0, text="Łączenie z WMS Geoportalu…")

    progress.progress(15, text="Wysyłanie zapytania GetMap…")
    raw_bytes = _fetch_wms_tiff(x_min, y_min, x_max, y_max, width, height)

    if raw_bytes is None:
        progress.empty()
        return

    progress.progress(70, text="Wczytywanie obrazu…")

    try:
        img_pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img_np  = np.array(img_pil)
    except Exception as e:
        progress.empty()
        st.error(f"Nie udało się wczytać pobranego pliku: {e}")
        return

    progress.progress(100, text="Gotowe!")
    progress.empty()

    # ── Zapisz do session state ───────────────────────────────────────────────
    st.session_state["img_np"]      = img_np
    st.session_state["img_source"]  = "upload"   # traktuj jak upload — brak GT
    st.session_state["mask_gt"]     = None
    st.session_state["source_res"]  = round(actual_res, 4)
    st.session_state["chosen"]      = None
    st.session_state["chosen_ds"]   = None
    # Zapamiętaj też rozdzielczość jako "natywną" żeby automatch w ustawieniach działał
    native_label = f"Natywna ({actual_res:.4f} m/px)"
    st.session_state["resolutions"][native_label] = round(actual_res, 4)
    st.session_state["_map_native_label"] = native_label  # przekazuje do sekcji ustawień

    # ── Wyświetl pobrany obraz ────────────────────────────────────────────────
    st.success(
        f"✅ Pobrano ortofotomapę — "
        f"{img_np.shape[1]} × {img_np.shape[0]} px, "
        f"rozdzielczość: **{actual_res:.4f} m/px**"
    )
    st.image(img_np,
             caption=f"Ortofotomapa Geoportalu — {actual_res:.4f} m/px",
             use_container_width=True)

    # Przycisk do pobrania pliku TIFF
    st.download_button(
        label="💾 Zapisz plik .tiff",
        data=raw_bytes,
        file_name=f"ortofoto_{lat_min:.4f}_{lon_min:.4f}_{actual_res:.4f}mpx.tiff",
        mime="image/tiff",
        use_container_width=True,
    )