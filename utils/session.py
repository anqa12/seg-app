"""
session.py
----------
Wspólne zarządzanie session_state dla trzech zakładek wyboru zdjęcia
(Zdjęcia testowe / Pobierz z mapy / Wgraj własne).

WAŻNE: Streamlit renderuje ciało WSZYSTKICH zakładek st.tabs() przy każdym
rerunie — zakładki to tylko UI, nie ma natywnego zdarzenia "zmiana zakładki".
Jedyny niezawodny moment na wyczyszczenie stanu po innych źródłach to chwila,
w której user faktycznie coś wybiera w danej zakładce (klik "Wybierz",
wgranie pliku, pobranie z mapy). Dlatego reset_image_state() wołamy ZAWSZE
na samym początku takiej akcji, zanim ustawimy nowe wartości.
"""

import streamlit as st

# Klucze opisujące AKTUALNIE wybrane zdjęcie i jego pochodzenie — czyszczone
# przy każdej zmianie źródła, żeby dane z poprzedniego źródła (rozdzielczość,
# maska GT, podgląd z mapy) nie zostawały przypisane do nowego zdjęcia.
_RESET_KEYS = [
    "img_np",
    "img_source",
    "mask_gt",
    "chosen",
    "chosen_ds",
    "chosen_dataset_type",
    "_map_img_np",
    "_map_raw_bytes",
    "_map_caption",
    "_map_filename",
    "_map_actual_res",
    "_map_native_label",   # zachowane dla wstecznej kompatybilności, jeśli gdzieś jeszcze czytane
    "_upload_img_np",
    "_upload_raw_name",
    "_upload_detected_res",
    # UWAGA: "_upload_processed_file_id" CELOWO nie jest tutaj — musi przetrwać
    # reset, inaczej ten sam plik nadal "przyklejony" do file_uploadera
    # zostałby przetworzony (i aktywowany) ponownie na następnym rerunie.
]


def reset_image_state():
    """Czyści cały stan powiązany z poprzednio wybranym zdjęciem."""
    for key in _RESET_KEYS:
        st.session_state.pop(key, None)
    st.session_state["mask_gt"] = None
    st.session_state["source_res"] = 0.25

    # Podbij licznik generacji — segmentation.py używa go jako części klucza
    # widgetów wyboru modeli/rozdzielczości, żeby te widgety dostawały świeży
    # stan (i świeże domyślne wartości) przy KAŻDEJ zmianie zdjęcia zamiast
    # trzymać stary wybór (np. rozdzielczość "Natywna" z poprzedniego zdjęcia).
    st.session_state["selection_gen"] = st.session_state.get("selection_gen", 0) + 1

    # Usuń poprzedni wpis "Natywna (...)" — inaczej zostaje na liście
    # rozdzielczości i można go przypadkiem wybrać dla zupełnie innego zdjęcia.
    resolutions = st.session_state.get("resolutions", {})
    for label in [l for l in resolutions if l.startswith("Natywna (")]:
        del resolutions[label]


def set_native_resolution(value: float) -> str:
    """
    Zapisuje `value` jako JEDYNĄ etykietę "Natywna (...)" w słowniku
    rozdzielczości (usuwając poprzednią, jeśli istniała) i zwraca jej nazwę.
    Używane przez upload_tab.py (wykryta rozdzielczość z metadanych) i
    map_tab.py (rozdzielczość pobranego fragmentu ortofotomapy).
    """
    label = f"Natywna ({value:.4f} m/px)"
    resolutions = st.session_state.setdefault("resolutions", {})
    for old in [l for l in resolutions if l.startswith("Natywna (")]:
        del resolutions[old]
    resolutions[label] = round(value, 4)
    return label