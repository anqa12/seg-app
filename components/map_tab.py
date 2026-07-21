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

from utils.session import reset_image_state, set_native_resolution

WMS_URL = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/HighResolution"
WMS_LAYER = "Raster"
WMS_VERSION = "1.3.0"
WMS_CRS = "EPSG:2180"
MAX_PX = 4096

POLAND_BOUNDS = {"min_lat": 48.9, "max_lat": 55.0, "min_lon": 14.0, "max_lon": 24.2}

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
    w_m = x_max - x_min
    h_m = y_max - y_min
    width = int(round(w_m / target_res))
    height = int(round(h_m / target_res))
    if max(width, height) > MAX_PX:
        scale = MAX_PX / max(width, height)
        width = max(64, int(width * scale))
        height = max(64, int(height * scale))
    actual_res = ((w_m / width) + (h_m / height)) / 2.0
    return width, height, actual_res


def _to_georeferenced_tiff(img_np, x_min, y_max, pixel_w, pixel_h):
    try:
        import rasterio
        from rasterio.transform import from_origin
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


def _fetch_wms_tiff(x_min, y_min, x_max, y_max, width, height):
    bbox_str = f"{y_min},{x_min},{y_max},{x_max}"
    params = {
        "SERVICE": "WMS", "VERSION": WMS_VERSION, "REQUEST": "GetMap",
        "LAYERS": WMS_LAYER, "STYLES": "", "CRS": WMS_CRS,
        "BBOX": bbox_str, "WIDTH": str(width), "HEIGHT": str(height),
        "FORMAT": "image/tiff", "TRANSPARENT": "false",
    }
    try:
        resp = requests.get(WMS_URL, params=params, timeout=60)
        resp.raise_for_status()
        return resp.content
    except requests.exceptions.Timeout:
        st.error("⏱️ Przekroczono czas oczekiwania (60s). Spróbuj mniejszy obszar lub niższą rozdzielczość.")
        return None
    except Exception as e:
        st.error(f"❌ Błąd pobierania WMS: {e}")
        return None


def _is_blank_image(img_np: np.ndarray, threshold: float = 0.98) -> bool:
    """
    Sprawdza czy obraz jest praktycznie jednokolorowy (biały/czarny/pusty).
    Zwraca True jeśli ponad `threshold` pikseli ma tę samą wartość na każdym kanale.
    """
    if img_np is None:
        return True
    for channel in range(img_np.shape[2]):
        ch = img_np[:, :, channel].flatten()
        most_common_frac = np.bincount(ch).max() / len(ch)
        if most_common_frac > threshold:
            return True
    return False


def render_map_tab():
    if not FOLIUM_OK:
        st.error("Brakuje pakietów: `pip install streamlit-folium folium pyproj`")
        return

    st.caption(
        "Narysuj prostokąt na mapie narzędziem ▭ (lewy górny róg mapy), "
        "wybierz rozdzielczość i kliknij **Pobierz**. "
        "Ortofotomapa dostępna tylko dla Polski."
    )

    # resolution
    res_options = {
        "10 cm/px  (bardzo duże pliki, wolno)": 0.10,
        "25 cm/px  (zalecane — jak LandCover)": 0.25,
        "50 cm/px":                              0.50,
        "1 m/px    (szybko, mały plik)":         1.00,
    }
    res_label  = st.radio(
        "Rozdzielczość pobierania", list(res_options.keys()),
        index=1, horizontal=True,
    )
    target_res = res_options[res_label]

    # map creation
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

    # bigger map
    map_data = st_folium(m, height=750, width="100%", key="geoportal_map")

    # reading bounding box
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
        width, height, actual_res = _image_size(x_min, y_min, x_max, y_max, target_res)
        capped = actual_res > target_res * 1.05

        col_info, col_warn = st.columns([3, 2])
        with col_info:
            st.markdown(
                f"**Obszar:** {w_m/1000:.2f} km × {h_m/1000:.2f} km &nbsp;·&nbsp; "
                f"**Obraz:** {width} × {height} px &nbsp;·&nbsp; "
                f"**Rozdzielczość:** `{actual_res:.2f} m/px`"
            )
        with col_warn:
            if capped:
                st.warning(f"⚠️ Obszar zbyt duży — przycięty do {MAX_PX} px. "
                           f"Faktyczna rozdzielczość: {actual_res:.2f} m/px.")
            elif w_m > 10_000 or h_m > 10_000:
                st.info("Duży obszar — pobieranie może chwilę potrwać.")

        outside_poland = (
            lon_max < POLAND_BOUNDS["min_lon"] or lon_min > POLAND_BOUNDS["max_lon"] or
            lat_max < POLAND_BOUNDS["min_lat"] or lat_min > POLAND_BOUNDS["max_lat"]
        )
        if outside_poland:
            st.error(
                "❌ Zaznaczony obszar leży poza Polską — ortofotomapa Geoportalu "
                "dostępna wyłącznie dla terytorium Polski."
            )

        if st.button("⬇️ Pobierz ortofotomapę (.tiff)", type="primary",
                     use_container_width=True, disabled=outside_poland):
            progress = st.progress(0, text="Łączenie z WMS Geoportalu…")
            progress.progress(15, text="Wysyłanie zapytania GetMap…")
            raw_bytes = _fetch_wms_tiff(x_min, y_min, x_max, y_max, width, height)

            if raw_bytes is None:
                progress.empty()
            else:
                progress.progress(70, text="Wczytywanie obrazu…")
                img_np = None
                try:
                    img_pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    img_np  = np.array(img_pil)
                except Exception as e:
                    progress.empty()
                    st.error(f"❌ Nie udało się wczytać pobranego pliku: {e}")

                if img_np is not None:
                    # validation - downloaded img
                    if _is_blank_image(img_np):
                        progress.empty()
                        st.error(
                            "❌ Pobrany obraz jest pusty (biały/czarny). "
                            "Geoportal nie ma ortofotomapy dla tego obszaru lub "
                            "żądany fragment leży poza zasięgiem usługi. "
                            "Spróbuj inny obszar, przesuń prostokąt lub zmniejsz go."
                        )
                        # img preview
                        with st.expander("Podgląd pobranego obrazu."):
                            st.image(img_np, caption="Obraz uznany za pusty", width=400)
                    else:
                        progress.progress(100, text="Gotowe!")
                        progress.empty()

                        # check resolutions
                        real_h, real_w = img_np.shape[:2]
                        if (real_w, real_h) != (width, height):
                            actual_res = ((w_m / real_w) + (h_m / real_h)) / 2.0
                            st.warning(
                                f"⚠️ Serwer zwrócił {real_w}×{real_h} px zamiast "
                                f"{width}×{height} px — przeliczam rozdzielczość z "
                                f"faktycznego rozmiaru: **{actual_res:.2f} m/px**."
                            )

                        pixel_w = w_m / real_w
                        pixel_h = h_m / real_h
                        geo_bytes = _to_georeferenced_tiff(img_np, x_min, y_max, pixel_w, pixel_h)
                        save_bytes = geo_bytes if geo_bytes is not None else raw_bytes

                        reset_image_state()
                        st.session_state["img_np"] = img_np
                        st.session_state["img_source"] = "map"
                        st.session_state["source_res"] = round(actual_res, 4)
                        set_native_resolution(actual_res)

                        # Trwały podgląd
                        st.session_state["_map_img_np"] = img_np
                        st.session_state["_map_raw_bytes"] = save_bytes
                        st.session_state["_map_actual_res"] = actual_res
                        st.session_state["_map_caption"] = (
                            f"Ortofotomapa Geoportalu — {actual_res:.2f} m/px"
                        )
                        st.session_state["_map_filename"] = (
                            f"ortofoto_{lat_min:.2f}_{lon_min:.2f}"
                            f"_{actual_res:.2f}mpx.tiff"
                        )
    else:
        if st.session_state.get("_map_img_np") is None:
            st.caption("Narysuj prostokąt na mapie żeby zobaczyć opcje pobierania.")

    # ortophotomap preview
    if st.session_state.get("_map_img_np") is not None:
        st.success(
            f"✅ Pobrano ortofotomapę — "
            f"{st.session_state['_map_img_np'].shape[1]} × "
            f"{st.session_state['_map_img_np'].shape[0]} px, "
            f"rozdzielczość: **{st.session_state['_map_actual_res']:.2f} m/px**"
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
            use_container_width=True,
        )
        if st.button("🗑️ Usuń pobraną ortofotomapę", use_container_width=True):
            for k in ("_map_img_np", "_map_raw_bytes", "_map_actual_res",
                      "_map_caption", "_map_filename"):
                st.session_state.pop(k, None)
            reset_image_state()
            st.rerun()