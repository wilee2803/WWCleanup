"""
WW Duplikat-Erkennung
Streamlit App zur Erkennung und Bereinigung von Duplikaten in Weindaten.
Unterstützt: Rebsorten, Weinbauregionen, Winzer, Weine
"""

import io
import os
import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz
from rapidfuzz import process as rfprocess

# ── Datei-Konfigurationen ──────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(__file__)

FILE_CONFIGS = {
    "grapevariety": {
        "label":           "Rebsorten",
        "file":            os.path.join(BASE_DIR, "Datafiles", "grapevariety.csv"),
        "col_id":          13,
        "col_name":        15,
        "col_group":       1,
        "group_label":     "Weintyp",
        "split_dash":      False,
        "cuvee_detection": True,
        "out_file":        "grapevariety_cleaned.csv",
    },
    "growingregion": {
        "label":                  "Weinbauregionen",
        "file":                   os.path.join(BASE_DIR, "Datafiles", "growingregion.csv"),
        "col_id":                 1,
        "col_name":               3,
        "col_group":              0,
        "group_label":            "Land",
        "split_dash":             True,
        "cuvee_detection":        False,
        "out_file":               "growingregion_cleaned.csv",
        "pre_known_synonyms_file": os.path.join(BASE_DIR, "Datafiles", "Dupletten_Weinbauregion.xlsx"),
    },
    "producer": {
        "label":           "Winzer",
        "file":            os.path.join(BASE_DIR, "Datafiles", "producer.csv"),
        "col_id":          0,      # Id
        "col_name":        1,      # ShortName
        "col_group":       None,   # Kein Gruppenfeld — alle vs. alle
        "group_label":     "Winzer",
        "split_dash":      False,
        "cuvee_detection": False,
        "out_file":        "producer_cleaned.csv",
        # Zusatz-Felder im Pair-Review
        "detail_cols": [
            (2,  "LongName"),
            (5,  "Verifiziert"),
            (8,  "Erstellt am"),
            (12, "Gelöscht"),
        ],
    },
    "product": {
        "label":              "Weine",
        "file":               os.path.join(BASE_DIR, "Datafiles", "product.csv"),
        "col_id":             0,    # Product.Id
        "col_name":           3,    # Title
        "col_group":          None, # wird beim Laden als _group gesetzt
        "group_label":        "Gruppe",
        "group_label_metric": "Wein-Gruppen",
        "split_dash":         False,
        "cuvee_detection":    False,
        "out_file":           "product_cleaned.csv",
        # Produkt-spezifisch: zusammengesetzter Gruppierschlüssel
        "composite_group":    True,
        "col_producer_id":    2,    # ProducerId
        "col_year":           6,    # ProductionYear
        "col_bottle":         15,   # BottleSize
        "col_wine_type_id":   18,   # GrapeVarietyGroupId (Weinsorte)
        "col_grape_id":       19,   # GrapeVarietyId      (Rebsorte)
        # Joined-Spalten (pandas benennt Duplikate um):
        "col_wine_type_name": 44,   # Title.1  → GrapeVarityGroup.Title
        "col_producer_name":  48,   # ShortName → Producer.ShortName
    },
}

# ── Seiteneinstellungen ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="WW Duplikat-Erkennung",
    page_icon="🍷",
    layout="wide",
)

IS_LOCAL = any(os.path.exists(cfg["file"]) for cfg in FILE_CONFIGS.values())


# ── Hilfsfunktionen ────────────────────────────────────────────────────────────

def normalize(text: str, split_dash: bool = False) -> str:
    if pd.isna(text) or str(text).strip().upper() in ("NULL", "NONE", ""):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"\s+", " ", text).lower().strip()
    if split_dash:
        text = re.sub(r"\s+-\s+", ", ", text)
    tokens = sorted(t.strip() for t in text.split(",") if t.strip())
    return ", ".join(tokens)


def is_cuvee(name: str) -> bool:
    """True wenn der Name mehrere Rebsorten enthält (kommagetrennt)."""
    if not name or str(name).strip().upper() in ("NULL", "NONE", ""):
        return False
    return "," in str(name)


def is_cuvee_master_name(name: str) -> bool:
    """True wenn der Name mit 'cuvee' beginnt und kein Blend ist (kein Komma).
    Erfasst z.B. 'Cuvée', 'Cuvée Weiss', 'Cuvee Rouge' — aber nicht 'Cuvée, Merlot'."""
    n = normalize(name)
    return n.startswith("cuvee") and "," not in n


def load_csv(source) -> tuple[pd.DataFrame | None, str | None]:
    for enc in ("latin-1", "cp1252", "utf-8", "utf-8-sig"):
        try:
            if isinstance(source, (str, os.PathLike)):
                df = pd.read_csv(source, sep=";", encoding=enc, dtype=str,
                                 keep_default_na=False)
            else:
                df = pd.read_csv(io.BytesIO(source), sep=";", encoding=enc,
                                 dtype=str, keep_default_na=False)
            return df, enc
        except Exception:
            continue
    return None, None


def load_bottle_mapping() -> dict:
    """Gibt ein Dict {UUID: lesbarer Name} für Flaschengrößen zurück.
    Primär: packingunittype.csv; Fallback: hardcodierte Werte."""
    path = os.path.join(BASE_DIR, "Datafiles", "packingunittype.csv")
    _fallback = {
        "fe76ab14-c231-e95e-b5d4-3a13412f9a95": ".750L",
        "ce66ab14-c231-e95e-b5d4-3a13412f9a95": ".5L",
        "fe76ab14-c231-e95e-b5d4-3a13412f9b95": ".375L",
        "15e816fd-770c-9570-0ef6-3a13412f9841": "1L",
        "546d4174-f673-63ed-58c1-3a13412f9841": "1.5L",
        "2e76ab14-c231-e95e-b5d4-3a13412f9b95": "2L",
        "a5e816fd-770c-9570-0ef6-3a13412f9841": "3L",
        "55e816fd-770c-9570-0ef6-3a13412f9841": "5L",
        "65e816fd-770c-9570-0ef6-3a13412f9841": "6L",
        "be76ab14-c231-e95e-b5d4-3a13412f9b95": "12L",
    }
    if not os.path.exists(path):
        return _fallback
    try:
        df_btl = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False)
        return dict(zip(df_btl["Id"].str.strip(), df_btl["Title"].str.strip()))
    except Exception:
        return _fallback


def load_wine_type_mapping() -> dict:
    """Gibt ein Dict {UUID: lesbarer Name} für Weinsorten zurück.
    Primär: grapevarietygroup.csv; Fallback: hardcodierte Werte."""
    path = os.path.join(BASE_DIR, "Datafiles", "grapevarietygroup.csv")
    _fallback = {
        "1058504d-c1b3-416f-d533-3a133d623252": "Red Wine",
        "1158504d-c1b3-416f-d533-3a133d623252": "Rosé Wine",
        "1258504d-c1b3-416f-d533-3a133d623252": "Sparkling Wine",
        "1358504d-c1b3-416f-d533-3a133d623252": "Sweet & Dessert Wine",
        "1458504d-c1b3-416f-d533-3a133d623252": "White Wine",
        "1558504d-c1b3-416f-d533-3a133d623252": "Others",
    }
    if not os.path.exists(path):
        return _fallback
    try:
        df_wt = pd.read_csv(path, sep=";", encoding="latin-1", dtype=str,
                            keep_default_na=False)
        return dict(zip(df_wt["Id"].str.strip(), df_wt["Title"].str.strip()))
    except Exception:
        return _fallback


def load_known_synonyms(
    xlsx_path: str,
    df: pd.DataFrame,
    col_id: str,
    col_name: str,
    col_group: str | None,
    split_dash: bool = True,
) -> tuple[list[dict], set, list[str], list[tuple[str, str]]]:
    """Lädt vorab bekannte Synonyme aus einer Excel-Datei (Dupletten_Weinbauregion.xlsx).

    Struktur: Spalten 'Land', 'Weinbauregion' (Master), 'Duplikate'.
    Weinbauregion wird vorwärts gefüllt; eine neue Master-Zeile erkennt man daran,
    dass Weinbauregion in der Originaldatei befüllt ist.

    Gibt zurück:
        synonyms          – Liste von Synonym-Dicts (direkt in den Export-Pool)
        duplicate_ids     – Set der Duplikat-IDs (werden aus Fuzzy-Matching ausgeschlossen)
        unmatched_masters – Masternamen ohne Treffer in der CSV
        unmatched_dupes   – (master_name, dup_name)-Paare ohne Treffer in der CSV
    """
    if not os.path.exists(xlsx_path):
        return [], set(), [], []
    try:
        xl = pd.read_excel(xlsx_path)
    except Exception:
        return [], set(), [], []

    # Lookup: normalisierter Name → (id, group_value)
    name_to_row: dict[str, tuple[str, str]] = {}
    for _, row in df.iterrows():
        norm = normalize(str(row[col_name]), split_dash=split_dash)
        if norm:
            grp = str(row[col_group]) if col_group else ""
            name_to_row[norm] = (str(row[col_id]), grp)

    xl = xl.copy()
    xl["_is_master"] = xl["Weinbauregion"].notna()
    xl["Weinbauregion"] = xl["Weinbauregion"].ffill()

    synonyms:          list[dict]         = []
    duplicate_ids:     set                = set()
    unmatched_masters: list[str]          = []
    unmatched_dupes:   list[tuple[str, str]] = []
    seen_pairs:        set                = set()

    current_master:    str | None = None
    current_master_id: str | None = None
    current_group:     str        = ""

    for _, row in xl.iterrows():
        wb  = str(row.get("Weinbauregion", "")).strip()
        dup = row.get("Duplikate", "")
        dup = "" if pd.isna(dup) else str(dup).strip()

        # Neue Master-Zeile
        if row["_is_master"] and wb and wb != "nan":
            current_master = wb
            norm_master    = normalize(wb, split_dash=split_dash)
            if norm_master in name_to_row:
                current_master_id, current_group = name_to_row[norm_master]
            else:
                current_master_id = None
                current_group     = ""
                if wb not in unmatched_masters:
                    unmatched_masters.append(wb)

        # Duplikat verarbeiten
        if dup and dup != "nan":
            norm_dup = normalize(dup, split_dash=split_dash)
            if norm_dup in name_to_row:
                dup_id, dup_grp = name_to_row[norm_dup]
                pair_key = (current_master_id, dup_id)
                if (
                    current_master_id
                    and dup_id != current_master_id
                    and pair_key not in seen_pairs
                ):
                    seen_pairs.add(pair_key)
                    duplicate_ids.add(dup_id)
                    synonyms.append({
                        "type":           "duplicate",
                        "duplicate_id":   dup_id,
                        "original_id":    current_master_id,
                        "duplicate_name": dup,
                        "original_name":  current_master,
                        "land":           dup_grp or current_group,
                        "confidence_%":   100,
                    })
            else:
                unmatched_dupes.append((current_master or "", dup))

    return synonyms, duplicate_ids, unmatched_masters, unmatched_dupes


def find_candidates(df: pd.DataFrame, id_col: str, name_col: str,
                    group_col: str | None, threshold: int,
                    split_dash: bool = False,
                    exclude_ids: set | None = None,
                    group_none_label: str = "Alle") -> list[dict]:
    """Findet Duplikat-Paare per rapidfuzz cdist (effizient auch bei >1000 Einträgen).

    group_col=None → alle Einträge werden als eine Gruppe behandelt.
    exclude_ids    → werden übersprungen (z.B. bestätigte Cuvées).
    """
    exclude_ids = exclude_ids or set()
    candidates  = []

    # Gruppen bestimmen
    if group_col is None:
        groups = [(group_none_label, df)]
    else:
        groups = list(df.groupby(group_col))

    for group_name, gdf in groups:
        valid = gdf[~gdf[name_col].isin(["NULL", "NONE", ""])].copy()
        valid = valid[valid[name_col].notna()]
        valid = valid[~valid[id_col].isin(exclude_ids)]
        valid = valid.reset_index(drop=True)
        if len(valid) < 2:
            continue

        ids   = valid[id_col].tolist()
        names = valid[name_col].tolist()
        norms = [normalize(n, split_dash=split_dash) for n in names]

        # Leere Normen herausfiltern, Index-Mapping behalten
        valid_idx   = [i for i, n in enumerate(norms) if n]
        valid_norms = [norms[i] for i in valid_idx]
        if len(valid_norms) < 2:
            continue

        # Paarweise Ähnlichkeit per cdist (C-Implementierung, deutlich schneller als Python-Loop)
        matrix = rfprocess.cdist(valid_norms, valid_norms,
                                 scorer=fuzz.ratio, score_cutoff=threshold)

        rows, cols = np.where(np.triu(matrix, k=1) > 0)
        for r, c in zip(rows.tolist(), cols.tolist()):
            orig_r = valid_idx[r]
            orig_c = valid_idx[c]
            candidates.append({
                "group":  group_name,
                "id_a":   ids[orig_r],
                "name_a": names[orig_r],
                "id_b":   ids[orig_c],
                "name_b": names[orig_c],
                "score":  round(float(matrix[r, c]), 1),
            })

    return sorted(candidates, key=lambda x: (-x["score"], x["group"], x["name_a"]))


# ── Session State Init ─────────────────────────────────────────────────────────

_defaults: dict = {
    "df":                          None,
    "encoding":                    None,
    "config":                      None,
    "candidates":                  None,
    "decisions":                   {},
    "col_id":                      None,
    "col_name":                    None,
    "col_group":                   None,
    "file_load_count":             0,    # erhöht bei Datei-Load → setzt Editoren zurück
    "analysis_count":              0,    # erhöht bei Analyse-Start
    "confirmed_cuvee_ids":         set(), # IDs bestätigter Blend-Cuvées (Schritt 2)
    "cuvee_master_per_group":      {},   # {group_val: master_id} aus Schritt 1
    "cuvee_master_non_master_ids": set(), # nicht-Master-Cuvée-IDs aus Schritt 1
    # Bekannte Synonyme (aus Excel-Vorab-Kuratierung)
    "known_synonyms":              [],   # Liste von Synonym-Dicts
    "known_synonym_ids":           set(), # Duplikat-IDs (aus Fuzzy-Matching ausschließen)
    "known_unmatched_masters":     [],   # Masternamen ohne CSV-Treffer
    "known_unmatched_dupes":       [],   # (master, dup)-Paare ohne CSV-Treffer
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Einstellungen")

    dtype_options  = {cfg["label"]: key for key, cfg in FILE_CONFIGS.items()}
    selected_label = st.radio("Datentyp", list(dtype_options.keys()))
    selected_key   = dtype_options[selected_label]
    active_config  = FILE_CONFIGS[selected_key]

    st.divider()

    if IS_LOCAL:
        source_mode = st.radio("Quelle", ["Standard-Datei", "CSV hochladen"],
                               label_visibility="collapsed")
    else:
        source_mode = "CSV hochladen"
        st.caption("☁️ Cloud-Modus — bitte CSV hochladen")

    uploaded_file = None
    if source_mode == "CSV hochladen":
        uploaded_file = st.file_uploader("CSV hochladen", type=["csv"])

    if st.button("📂 Datei laden", use_container_width=True):
        raw = None
        if source_mode == "Standard-Datei":
            raw = active_config["file"]
        elif uploaded_file is not None:
            raw = uploaded_file.read()

        if raw is not None:
            df_new, enc = load_csv(raw)
            if df_new is not None:
                st.session_state.encoding  = enc
                st.session_state.config    = active_config
                st.session_state.col_id    = df_new.columns[active_config["col_id"]]
                st.session_state.col_name  = df_new.columns[active_config["col_name"]]

                if active_config.get("composite_group"):
                    # Zusammengesetzten Gruppierschlüssel aus 5 Feldern aufbauen
                    c_prod  = df_new.columns[active_config["col_producer_name"]]
                    c_type  = df_new.columns[active_config["col_wine_type_name"]]
                    c_year  = df_new.columns[active_config["col_year"]]
                    c_grape = df_new.columns[active_config["col_grape_id"]]
                    c_btl   = df_new.columns[active_config["col_bottle"]]
                    btl_map = load_bottle_mapping()
                    df_new["_group"] = (
                        df_new[c_prod].str.strip() + " / " +
                        df_new[c_type].str.strip() + " / " +
                        df_new[c_year].str.strip() + " / " +
                        df_new[c_grape].str[:8] + " / " +
                        df_new[c_btl].map(lambda x: btl_map.get(x, x[:8]))
                    )
                    st.session_state.col_group = "_group"
                else:
                    col_group_idx = active_config["col_group"]
                    st.session_state.col_group = (
                        df_new.columns[col_group_idx] if col_group_idx is not None else None
                    )

                st.session_state.df                          = df_new
                st.session_state.candidates                  = None
                st.session_state.decisions                   = {}
                st.session_state.confirmed_cuvee_ids         = set()
                st.session_state.cuvee_master_per_group      = {}
                st.session_state.cuvee_master_non_master_ids = set()
                st.session_state.known_synonyms              = []
                st.session_state.known_synonym_ids           = set()
                st.session_state.known_unmatched_masters     = []
                st.session_state.known_unmatched_dupes       = []
                st.session_state.file_load_count            += 1

                # Vorab-Synonyme aus Excel laden (falls konfiguriert)
                xlsx_path = active_config.get("pre_known_synonyms_file")
                if xlsx_path:
                    _cid   = df_new.columns[active_config["col_id"]]
                    _cname = df_new.columns[active_config["col_name"]]
                    _cgrp_idx = active_config.get("col_group")
                    _cgrp  = df_new.columns[_cgrp_idx] if _cgrp_idx is not None else None
                    _syns, _dup_ids, _unm_m, _unm_d = load_known_synonyms(
                        xlsx_path, df_new, _cid, _cname, _cgrp,
                        split_dash=active_config.get("split_dash", False),
                    )
                    st.session_state.known_synonyms          = _syns
                    st.session_state.known_synonym_ids       = _dup_ids
                    st.session_state.known_unmatched_masters = _unm_m
                    st.session_state.known_unmatched_dupes   = _unm_d

                msg = f"✓ {len(df_new)} Zeilen geladen  \nKodierung: {enc}"
                if st.session_state.known_synonyms:
                    msg += f"  \n📌 {len(st.session_state.known_synonyms)} bekannte Synonyme geladen"
                st.success(msg)
            else:
                st.error("Datei konnte nicht gelesen werden.")
        else:
            st.warning("Keine Datei ausgewählt.")

    st.divider()

    threshold = st.slider(
        "Ähnlichkeitsschwelle (%)", 50, 100, 80, 1,
        help="Gilt für Schritt 3: Duplikat-Erkennung.",
    )

    analyse_ready = (
        st.session_state.df is not None
        and st.session_state.config is not None
        and st.session_state.config["label"] == active_config["label"]
    )

    if analyse_ready:
        if active_config["cuvee_detection"]:
            n_master  = len(st.session_state.cuvee_master_per_group)
            n_non_master = len(st.session_state.cuvee_master_non_master_ids)
            n_blends  = len(st.session_state.confirmed_cuvee_ids)
            st.caption(
                f"Schritt 1: **{n_master}** Cuvée-Master  \n"
                f"Schritt 2: **{n_blends}** Blend-Cuvées markiert"
            )
            if n_non_master:
                st.caption(f"↳ {n_non_master} doppelte Cuvée-Einträge entfernt")

        if st.button("🔍 Duplikat-Analyse starten", type="primary",
                     use_container_width=True):
            cfg = st.session_state.config
            with st.spinner("Suche Duplikate..."):
                # Aus Analyse ausschließen: bestätigte Blend-Cuvées + nicht-Master-Einträge
                exclude = (
                    st.session_state.confirmed_cuvee_ids |
                    st.session_state.cuvee_master_non_master_ids |
                    st.session_state.known_synonym_ids
                )
                cands = find_candidates(
                    st.session_state.df,
                    st.session_state.col_id,
                    st.session_state.col_name,
                    st.session_state.col_group,
                    threshold,
                    split_dash=cfg["split_dash"],
                    exclude_ids=exclude,
                    group_none_label=cfg["label"],
                )
            st.session_state.candidates  = cands
            st.session_state.decisions   = {}
            st.session_state.analysis_count += 1
            if cands:
                st.success(f"✓ {len(cands)} Duplikat-Kandidaten")
            else:
                st.info("Keine Duplikate gefunden.")

    if st.session_state.candidates:
        total   = len(st.session_state.candidates)
        decided = len(st.session_state.decisions)
        st.divider()
        st.progress(
            decided / total if total else 0,
            text=f"Duplikate: {decided} / {total} entschieden",
        )


# ── Hauptbereich ───────────────────────────────────────────────────────────────

cfg = st.session_state.config or active_config
st.title(f"🍷 WW Duplikat-Erkennung — {cfg['label']}")

if st.session_state.df is None or st.session_state.config is None:
    st.info("👈 Bitte links einen Datentyp wählen und eine Datei laden.")
    st.stop()

if st.session_state.config["label"] != active_config["label"]:
    st.warning(
        f"Aktuell geladen: **{st.session_state.config['label']}**  \n"
        f"Gewählt: **{active_config['label']}**  \n"
        "Bitte Datei neu laden."
    )
    st.stop()

df        = st.session_state.df
col_id    = st.session_state.col_id
col_name  = st.session_state.col_name
col_group = st.session_state.col_group

# ── Übersicht ──────────────────────────────────────────────────────────────────

st.subheader("📊 Übersicht")
mc1, mc2, mc3, mc4 = st.columns(4)
with mc1:
    st.metric("Einträge gesamt", len(df))
with mc2:
    if col_group is not None:
        metric_label = cfg.get("group_label_metric", cfg["group_label"] + "gruppen")
        st.metric(metric_label, df[col_group].nunique())
    else:
        st.metric("Einträge", len(df))
with mc3:
    n_cuvee = int(df[col_name].apply(is_cuvee).sum()) if cfg["cuvee_detection"] else "–"
    st.metric("Cuvée-Kandidaten", n_cuvee)
with mc4:
    n_cand = len(st.session_state.candidates) if st.session_state.candidates is not None else "–"
    st.metric("Duplikat-Kandidaten", n_cand)

if col_group is not None:
    with st.expander(f"Einträge pro {cfg['group_label']} anzeigen"):
        counts = (
            df.groupby(col_group)[col_id]
            .count()
            .reset_index()
            .rename(columns={col_group: cfg["group_label"], col_id: cfg["label"]})
            .sort_values(cfg["label"], ascending=False)
        )
        st.dataframe(counts, use_container_width=True, hide_index=True)


# ── SCHRITT 1: Cuvée-Master je Weintyp ─────────────────────────────────────────

if cfg["cuvee_detection"]:
    st.divider()
    st.subheader("Schritt 1 — 🏆 Cuvée-Master je Weintyp")
    st.caption(
        "Pro Weintyp wird genau ein 'Cuvée'-Eintrag als Master bestimmt. "
        "Gibt es mehrere, wähle den Master — die übrigen werden als Duplikate behandelt."
    )

    master_mask = df[col_name].apply(is_cuvee_master_name)
    master_rows = df[master_mask].copy()

    if master_rows.empty:
        st.warning("⚠️ Keine Einträge mit Name 'Cuvée' gefunden. Bitte prüfe die Daten.")
        master_per_group      = {}
        non_master_ids        = set()
    else:
        master_per_group = {}
        non_master_ids   = set()
        needs_manual     = []

        for group_val, gdf in master_rows.groupby(col_group):
            entries = gdf[[col_id, col_name]].reset_index(drop=True)
            if len(entries) == 1:
                master_per_group[group_val] = entries[col_id].iloc[0]
            else:
                needs_manual.append((group_val, entries))

        if needs_manual:
            st.info(
                f"Bei **{len(needs_manual)}** Weintypen gibt es mehrere 'Cuvée'-Einträge — "
                "bitte jeweils den Master auswählen:"
            )
            for group_val, entries in needs_manual:
                st.write(f"**{group_val}**")
                options = {
                    f"{row[col_name]}  (ID: {row[col_id][:8]}…)": row[col_id]
                    for _, row in entries.iterrows()
                }
                safe_key = re.sub(r"[^a-zA-Z0-9_]", "_", str(group_val))
                selected_label = st.radio(
                    f"Master für {group_val}",
                    options=list(options.keys()),
                    key=f"cuvee_master_{safe_key}_{st.session_state.file_load_count}",
                    label_visibility="collapsed",
                )
                master_per_group[group_val] = options[selected_label]

        # Nicht-Master-IDs bestimmen
        for group_val, gdf in master_rows.groupby(col_group):
            master_id = master_per_group.get(group_val)
            for _, row in gdf.iterrows():
                if row[col_id] != master_id:
                    non_master_ids.add(row[col_id])

        n_auto   = sum(1 for _, gdf in master_rows.groupby(col_group) if len(gdf) == 1)
        n_manual = len(needs_manual)
        total_groups = n_auto + n_manual

        if non_master_ids:
            st.warning(
                f"**{len(non_master_ids)}** doppelte Cuvée-Einträge werden in der "
                "Synonymtabelle als Duplikate geführt."
            )
        else:
            st.success(
                f"✅ {total_groups} Weintypen — je genau ein Cuvée-Master vorhanden."
            )

    # Immer in Session State schreiben (auch wenn leer)
    st.session_state.cuvee_master_per_group      = master_per_group
    st.session_state.cuvee_master_non_master_ids = non_master_ids


# ── SCHRITT 2: Cuvée-Erkennung (Blends) ────────────────────────────────────────

edited_cuvee: pd.DataFrame | None = None

if cfg["cuvee_detection"]:
    st.divider()
    st.subheader("Schritt 2 — 🍾 Cuvée-Erkennung")
    st.caption(
        "Einträge mit mehreren Rebsorten (kommagetrennt) sind als Cuvée vorausgewählt. "
        "Deaktiviere Einträge, die kein Cuvée sind. "
        "Bestätigte Cuvées werden auf den jeweiligen Cuvée-Master (Schritt 1) gemappt "
        "und aus der Duplikat-Analyse ausgeschlossen."
    )

    cuvee_mask = df[col_name].apply(is_cuvee)
    cuvee_rows = df[cuvee_mask].copy()

    if cuvee_rows.empty:
        st.info("Keine Blend-Cuvée-Kandidaten erkannt.")
    else:
        cuvee_table = pd.DataFrame({
            "id":               cuvee_rows[col_id].tolist(),
            cfg["group_label"]: cuvee_rows[col_group].tolist(),
            "Rebsorte":         cuvee_rows[col_name].tolist(),
            "Cuvée":            True,
        })

        # Master-ID pro Gruppe für Vorschau anzeigen
        master_map = st.session_state.cuvee_master_per_group
        cuvee_table["Master-ID"] = cuvee_table[cfg["group_label"]].map(
            lambda g: master_map.get(g, "⚠️ kein Master")
        )

        edited_cuvee = st.data_editor(
            cuvee_table,
            column_config={
                "id": st.column_config.TextColumn(
                    "ID", disabled=True, width="small"),
                cfg["group_label"]: st.column_config.TextColumn(
                    cfg["group_label"], disabled=True, width="small"),
                "Rebsorte": st.column_config.TextColumn(
                    "Rebsorte", disabled=True),
                "Cuvée": st.column_config.CheckboxColumn(
                    "Als Cuvée markieren", default=True, width="small"),
                "Master-ID": st.column_config.TextColumn(
                    "→ Cuvée-Master", disabled=True, width="medium"),
            },
            hide_index=True,
            use_container_width=True,
            key=f"cuvee_editor_{st.session_state.file_load_count}",
        )

        n_confirmed = int(edited_cuvee["Cuvée"].sum())
        n_total     = len(cuvee_table)
        st.caption(f"**{n_confirmed}** von **{n_total}** als Cuvée markiert — "
                   f"danach links auf **Duplikat-Analyse starten** klicken")

        st.session_state.confirmed_cuvee_ids = set(
            edited_cuvee[edited_cuvee["Cuvée"]]["id"].tolist()
        )


# ── Bekannte Synonyme (aus Excel-Vorab-Kuratierung) ────────────────────────────

if cfg.get("pre_known_synonyms_file"):
    st.divider()
    st.subheader("📌 Bekannte Synonyme")
    known_syns      = st.session_state.known_synonyms
    unm_masters     = st.session_state.known_unmatched_masters
    unm_dupes       = st.session_state.known_unmatched_dupes

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Bekannte Synonympaare", len(known_syns))
    with k2:
        st.metric("Master ohne CSV-Treffer", len(unm_masters))
    with k3:
        st.metric("Duplikate ohne CSV-Treffer", len(unm_dupes))

    if known_syns:
        with st.expander(f"{len(known_syns)} bekannte Synonyme anzeigen"):
            st.caption(
                "Diese Paare sind bereits kuratiert und werden direkt in den Export "
                "übernommen. Die Duplikat-Einträge werden aus der Fuzzy-Analyse ausgeschlossen."
            )
            st.dataframe(
                pd.DataFrame(known_syns).drop(columns="type"),
                use_container_width=True, hide_index=True,
            )
    else:
        xlsx_path = cfg["pre_known_synonyms_file"]
        if not os.path.exists(xlsx_path):
            st.caption("📁 Excel-Datei nicht gefunden — bitte lokal ablegen.")
        else:
            st.caption("Keine Treffer in der CSV — Namen prüfen.")

    if unm_masters:
        with st.expander(f"⚠️ {len(unm_masters)} Master-Namen ohne CSV-Treffer"):
            for m in unm_masters:
                st.write(f"- {m}")

    if unm_dupes:
        with st.expander(f"⚠️ {len(unm_dupes)} Duplikat-Namen ohne CSV-Treffer"):
            for master, dup in unm_dupes:
                st.write(f"- **{master}** ← {dup}")


# ── SCHRITT 3: Duplikat-Review ─────────────────────────────────────────────────

st.divider()
st.subheader("Schritt 3 — 📋 Duplikat-Review")

if st.session_state.candidates is None:
    st.info("👈 Klicke auf **Duplikat-Analyse starten** um fortzufahren.")
    st.stop()

candidates = st.session_state.candidates

if not candidates:
    st.success("✅ Keine Duplikate über dem Schwellwert gefunden.")
else:
    st.caption(
        "Für jedes Paar: Wähle, welcher Eintrag das **Original** ist — "
        "oder verwirf, wenn es kein Duplikat ist."
    )

    by_group: dict[str, list] = {}
    for c in candidates:
        by_group.setdefault(c["group"], []).append(c)

    for group_name, pairs in by_group.items():
        n_open = sum(1 for p in pairs
                     if (p["id_a"], p["id_b"]) not in st.session_state.decisions)
        with st.expander(
            f"**{group_name}** — {len(pairs)} Paare  ({n_open} offen)",
            expanded=n_open > 0,
        ):
            # Wein-Kontext: gemeinsame Attribute einmalig pro Gruppe anzeigen
            if cfg.get("composite_group") and pairs:
                ctx = df[df[col_id] == pairs[0]["id_a"]]
                if not ctx.empty:
                    r      = ctx.iloc[0]
                    c_prod  = df.columns[cfg["col_producer_name"]]
                    c_type  = df.columns[cfg["col_wine_type_name"]]
                    c_year  = df.columns[cfg["col_year"]]
                    c_wtype = df.columns[cfg["col_wine_type_id"]]
                    c_btl   = df.columns[cfg["col_bottle"]]
                    btl_map  = load_bottle_mapping()
                    wt_map   = load_wine_type_mapping()
                    btl_label = btl_map.get(str(r[c_btl]),  str(r[c_btl])[:8]  + "…")
                    wt_label  = wt_map.get(str(r[c_wtype]), str(r[c_wtype])[:8] + "…")
                    st.info(
                        f"**Winzer:** {r[c_prod]}  ·  "
                        f"**Jahrgang:** {r[c_year]}  ·  "
                        f"**Rebsorte:** {r[c_type]}  ·  "
                        f"**Weinsorte:** {wt_label}  ·  "
                        f"**Flaschengröße:** {btl_label}"
                    )

            for pair in pairs:
                key      = (pair["id_a"], pair["id_b"])
                decision = st.session_state.decisions.get(key)
                score    = pair["score"]

                badge  = "🔴" if score >= 95 else ("🟠" if score >= 85 else "🟡")
                status = {
                    "original_a": "✅ A ist Original",
                    "original_b": "✅ B ist Original",
                    "reject":     "❌ Kein Duplikat",
                }.get(decision, "⏳ Ausstehend")

                st.write(f"{badge} **{score}%** — {status}")

                left, mid, right = st.columns([4, 4, 3])

                detail_cols = cfg.get("detail_cols", [])

                def _detail_lines(entry_id: str) -> str:
                    if not detail_cols:
                        return ""
                    rows = df[df[col_id] == entry_id]
                    if rows.empty:
                        return ""
                    r = rows.iloc[0]
                    parts = []
                    for col_idx, col_label in detail_cols:
                        val = r.iloc[col_idx] if col_idx < len(r) else ""
                        val = str(val).strip()
                        if val and val.upper() not in ("NULL", "NONE", ""):
                            parts.append(f"**{col_label}:** {val}")
                    return "  \n".join(parts)

                with left:
                    st.markdown("**Eintrag A**")
                    st.markdown(f"```\n{pair['name_a']}\n```")
                    st.caption(f"ID: {pair['id_a']}")
                    details_a = _detail_lines(pair["id_a"])
                    if details_a:
                        st.markdown(details_a)
                with mid:
                    st.markdown("**Eintrag B**")
                    st.markdown(f"```\n{pair['name_b']}\n```")
                    st.caption(f"ID: {pair['id_b']}")
                    details_b = _detail_lines(pair["id_b"])
                    if details_b:
                        st.markdown(details_b)
                with right:
                    st.markdown("**Entscheidung**")
                    b1, b2, b3 = st.columns(3)
                    uid = f"{key[0][:6]}_{key[1][:6]}"
                    with b1:
                        if st.button("A=Orig", key=f"a_{uid}",
                                     type="primary" if decision == "original_a" else "secondary",
                                     help="A ist das Original"):
                            st.session_state.decisions[key] = "original_a"
                            st.rerun()
                    with b2:
                        if st.button("B=Orig", key=f"b_{uid}",
                                     type="primary" if decision == "original_b" else "secondary",
                                     help="B ist das Original"):
                            st.session_state.decisions[key] = "original_b"
                            st.rerun()
                    with b3:
                        if st.button("✗", key=f"r_{uid}",
                                     type="primary" if decision == "reject" else "secondary",
                                     help="Kein Duplikat"):
                            st.session_state.decisions[key] = "reject"
                            st.rerun()
                st.divider()


# ── Export ─────────────────────────────────────────────────────────────────────

confirmed_pairs   = [(k, v) for k, v in st.session_state.decisions.items() if v != "reject"]
has_cuvee         = (edited_cuvee is not None and bool(edited_cuvee["Cuvée"].any()))
has_non_masters   = bool(st.session_state.cuvee_master_non_master_ids)
has_known_syns    = bool(st.session_state.known_synonyms)
all_pairs_decided = (len(st.session_state.decisions) == len(candidates))

# Export erst anzeigen wenn alle Paare entschieden wurden
if not all_pairs_decided:
    remaining = len(candidates) - len(st.session_state.decisions)
    st.info(f"⏳ Noch **{remaining}** Paare offen — bitte alle entscheiden, dann erscheint der Export.")
    st.stop()

if not confirmed_pairs and not has_cuvee and not has_non_masters and not has_known_syns:
    st.stop()

st.divider()
st.subheader("📥 Export")

out_df = df.copy()

# Interne Hilfsspalten nicht exportieren
for _internal in ["_group"]:
    if _internal in out_df.columns:
        out_df = out_df.drop(columns=_internal)

insert_pos = list(out_df.columns).index(col_id) + 1
out_df.insert(insert_pos, "original_id", "")

if cfg["cuvee_detection"]:
    out_df.insert(insert_pos + 1, "is_cuvee", "")

# Synonymtabelle
synonyms: list[dict] = []

# 0) Schritt 1: doppelte Cuvée-Einträge (nicht-Master → Master)
if cfg["cuvee_detection"] and has_non_masters:
    master_map = st.session_state.cuvee_master_per_group
    for non_master_id in st.session_state.cuvee_master_non_master_ids:
        nm_rows = df[df[col_id] == non_master_id]
        if nm_rows.empty:
            continue
        nm_row    = nm_rows.iloc[0]
        group_val = nm_row[col_group]
        master_id = master_map.get(group_val, "")
        master_rows_df = df[df[col_id] == master_id]
        master_name = master_rows_df[col_name].iloc[0] if not master_rows_df.empty else ""

        out_df.loc[out_df[col_id] == non_master_id, "original_id"] = master_id
        out_df.loc[out_df[col_id] == non_master_id, "is_cuvee"]    = "1"

        synonyms.append({
            "type":           "duplicate",
            "duplicate_id":   non_master_id,
            "original_id":    master_id,
            "duplicate_name": nm_row[col_name],
            "original_name":  master_name,
            cfg["group_label"].lower(): group_val,
            "confidence_%":   100,
        })

# 1) Schritt 3: bestätigte Duplikate
for (id_a, id_b), decision in confirmed_pairs:
    original_id  = id_a if decision == "original_a" else id_b
    duplicate_id = id_b if decision == "original_a" else id_a

    orig_rows = df[df[col_id] == original_id]
    dup_rows  = df[df[col_id] == duplicate_id]

    orig_name = orig_rows[col_name].iloc[0]  if not orig_rows.empty else ""
    dup_name  = dup_rows[col_name].iloc[0]   if not dup_rows.empty  else ""
    group_val = (orig_rows[col_group].iloc[0] if (col_group and not orig_rows.empty) else "")

    pair  = next((c for c in candidates if {c["id_a"], c["id_b"]} == {id_a, id_b}), None)
    score = pair["score"] if pair else 0

    out_df.loc[out_df[col_id] == duplicate_id, "original_id"] = original_id

    synonyms.append({
        "type":           "duplicate",
        "duplicate_id":   duplicate_id,
        "original_id":    original_id,
        "duplicate_name": dup_name,
        "original_name":  orig_name,
        cfg["group_label"].lower(): group_val,
        "confidence_%":   score,
    })

# 2) Schritt 2: bestätigte Blend-Cuvées → auf Master mappen
if cfg["cuvee_detection"] and edited_cuvee is not None:
    master_map = st.session_state.cuvee_master_per_group
    for _, row in edited_cuvee.iterrows():
        is_confirmed = bool(row["Cuvée"])
        blend_id     = row["id"]
        group_val    = row[cfg["group_label"]]
        master_id    = master_map.get(group_val, "")

        out_df.loc[out_df[col_id] == blend_id, "is_cuvee"] = "1" if is_confirmed else "0"

        if is_confirmed:
            if master_id:
                out_df.loc[out_df[col_id] == blend_id, "original_id"] = master_id
            master_rows_df = df[df[col_id] == master_id] if master_id else pd.DataFrame()
            master_name = master_rows_df[col_name].iloc[0] if not master_rows_df.empty else ""

            synonyms.append({
                "type":           "cuvee",
                "duplicate_id":   blend_id,
                "original_id":    master_id,
                "duplicate_name": row["Rebsorte"],
                "original_name":  master_name,
                cfg["group_label"].lower(): group_val,
                "confidence_%":   100,
            })

# Master-Cuvée-Einträge selbst als is_cuvee=1 markieren
if cfg["cuvee_detection"]:
    for master_id in st.session_state.cuvee_master_per_group.values():
        out_df.loc[out_df[col_id] == master_id, "is_cuvee"] = "1"

# Bekannte Synonyme aus Excel voranstellen
if st.session_state.known_synonyms:
    # "land"-Schlüssel auf den entity-spezifischen group_label-Key umbenennen
    group_key = cfg["group_label"].lower()
    for s in st.session_state.known_synonyms:
        if "land" in s and group_key != "land":
            s[group_key] = s.pop("land")
    synonyms = st.session_state.known_synonyms + synonyms

syn_df = pd.DataFrame(synonyms)

# Vorschau
if has_known_syns:
    st.write(f"**{len(st.session_state.known_synonyms)} bekannte Synonyme** (aus Excel-Kuratierung):")
    st.dataframe(
        pd.DataFrame(st.session_state.known_synonyms).drop(columns="type"),
        use_container_width=True, hide_index=True,
    )

if confirmed_pairs:
    n_dup = len(confirmed_pairs)
    st.write(f"**{n_dup} Duplikate** bestätigt (Schritt 3):")
    st.dataframe(
        syn_df[syn_df["type"] == "duplicate"].drop(columns="type"),
        use_container_width=True, hide_index=True,
    )

if has_cuvee or has_non_masters:
    n_cuv      = int(edited_cuvee["Cuvée"].sum()) if edited_cuvee is not None else 0
    n_nm       = len(st.session_state.cuvee_master_non_master_ids)
    parts      = []
    if n_nm:
        parts.append(f"**{n_nm}** doppelte Cuvée-Einträge (Schritt 1)")
    if n_cuv:
        parts.append(f"**{n_cuv}** Blend-Cuvées (Schritt 2)")
    st.write(f"{' + '.join(parts)}:")
    st.dataframe(
        syn_df[syn_df["type"] == "cuvee"][
            ["duplicate_id", "original_id", "duplicate_name", "original_name",
             cfg["group_label"].lower()]
        ],
        use_container_width=True, hide_index=True,
    )

# Download
dcol1, dcol2 = st.columns(2)

with dcol1:
    csv_out = out_df.to_csv(sep=";", index=False).encode("utf-8-sig")
    extras  = ["`original_id`"]
    if cfg["cuvee_detection"]:
        extras.append("`is_cuvee`")
    st.download_button(
        "📄 Ergebnis-CSV herunterladen",
        data=csv_out,
        file_name=cfg["out_file"],
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(f"Gleiche Struktur + {' + '.join(extras)}")

with dcol2:
    syn_name = cfg["out_file"].replace("_cleaned.csv", "_synonyms.csv")
    syn_out  = syn_df.to_csv(sep=";", index=False).encode("utf-8-sig")
    st.download_button(
        "📋 Synonymtabelle herunterladen",
        data=syn_out,
        file_name=syn_name,
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(
        "type=duplicate: Duplikate (Schritt 1+3)  |  "
        "type=cuvee: Blend-Cuvées → Master (Schritt 2)"
    )
