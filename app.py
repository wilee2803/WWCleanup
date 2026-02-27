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
        "label":           "Weinbauregionen",
        "file":            os.path.join(BASE_DIR, "Datafiles", "growingregion.csv"),
        "col_id":          1,
        "col_name":        3,
        "col_group":       0,
        "group_label":     "Land",
        "split_dash":      True,
        "cuvee_detection": False,
        "out_file":        "growingregion_cleaned.csv",
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
    "df":                  None,
    "encoding":            None,
    "config":              None,
    "candidates":          None,
    "decisions":           {},
    "col_id":              None,
    "col_name":            None,
    "col_group":           None,
    "file_load_count":     0,    # erhöht bei Datei-Load → setzt Cuvée-Editor zurück
    "analysis_count":      0,    # erhöht bei Analyse-Start
    "confirmed_cuvee_ids": set(), # IDs bestätigter Cuvées (aus letztem Render)
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
                    df_new["_group"] = (
                        df_new[c_prod].str.strip() + " / " +
                        df_new[c_type].str.strip() + " / " +
                        df_new[c_year].str.strip() + " / " +
                        df_new[c_grape].str[:8] + " / " +
                        df_new[c_btl].str[:8]
                    )
                    st.session_state.col_group = "_group"
                else:
                    col_group_idx = active_config["col_group"]
                    st.session_state.col_group = (
                        df_new.columns[col_group_idx] if col_group_idx is not None else None
                    )

                st.session_state.df                  = df_new
                st.session_state.candidates          = None
                st.session_state.decisions           = {}
                st.session_state.confirmed_cuvee_ids = set()
                st.session_state.file_load_count    += 1
                st.success(f"✓ {len(df_new)} Zeilen geladen  \nKodierung: {enc}")
            else:
                st.error("Datei konnte nicht gelesen werden.")
        else:
            st.warning("Keine Datei ausgewählt.")

    st.divider()

    threshold = st.slider(
        "Ähnlichkeitsschwelle (%)", 50, 100, 80, 1,
        help="Gilt für Schritt 2: Duplikat-Erkennung.",
    )

    analyse_ready = (
        st.session_state.df is not None
        and st.session_state.config is not None
        and st.session_state.config["label"] == active_config["label"]
    )

    if analyse_ready:
        # Hinweis bei Rebsorten, dass Cuvée-Review zuerst kommen sollte
        if active_config["cuvee_detection"]:
            n_cuvee_confirmed = len(st.session_state.confirmed_cuvee_ids)
            st.caption(f"Schritt 1 abgeschlossen: **{n_cuvee_confirmed}** Cuvées markiert")

        if st.button("🔍 Duplikat-Analyse starten", type="primary",
                     use_container_width=True):
            cfg = st.session_state.config
            with st.spinner("Suche Duplikate..."):
                cands = find_candidates(
                    st.session_state.df,
                    st.session_state.col_id,
                    st.session_state.col_name,
                    st.session_state.col_group,
                    threshold,
                    split_dash=cfg["split_dash"],
                    exclude_ids=st.session_state.confirmed_cuvee_ids,
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


# ── SCHRITT 1: Cuvée-Erkennung ─────────────────────────────────────────────────

edited_cuvee: pd.DataFrame | None = None

if cfg["cuvee_detection"]:
    st.divider()
    st.subheader("Schritt 1 — 🍾 Cuvée-Erkennung")
    st.caption(
        "Einträge mit mehreren Rebsorten (kommagetrennt) sind als Cuvée vorausgewählt. "
        "Deaktiviere Einträge, die kein Cuvée sind. "
        "Bestätigte Cuvées werden aus der Duplikat-Analyse ausgeschlossen."
    )

    cuvee_mask = df[col_name].apply(is_cuvee)
    cuvee_rows = df[cuvee_mask].copy()

    if cuvee_rows.empty:
        st.info("Keine Cuvée-Kandidaten erkannt.")
    else:
        cuvee_table = pd.DataFrame({
            "id":               cuvee_rows[col_id].tolist(),
            cfg["group_label"]: cuvee_rows[col_group].tolist(),
            "Rebsorte":         cuvee_rows[col_name].tolist(),
            "Cuvée":            True,
        })

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
            },
            hide_index=True,
            use_container_width=True,
            key=f"cuvee_editor_{st.session_state.file_load_count}",
        )

        n_confirmed = int(edited_cuvee["Cuvée"].sum())
        n_total     = len(cuvee_table)
        st.caption(f"**{n_confirmed}** von **{n_total}** als Cuvée markiert — "
                   f"danach links auf **Duplikat-Analyse starten** klicken")

        # Bestätigte IDs in Session State speichern (für Analyse-Button im nächsten Render)
        st.session_state.confirmed_cuvee_ids = set(
            edited_cuvee[edited_cuvee["Cuvée"]]["id"].tolist()
        )


# ── SCHRITT 2: Duplikat-Review ─────────────────────────────────────────────────

st.divider()
st.subheader("Schritt 2 — 📋 Duplikat-Review")

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
                    st.info(
                        f"**Winzer:** {r[c_prod]}  ·  "
                        f"**Jahrgang:** {r[c_year]}  ·  "
                        f"**Rebsorte:** {r[c_type]}  ·  "
                        f"**Weinsorte-ID:** `{str(r[c_wtype])[:8]}…`  ·  "
                        f"**Flaschen-ID:** `{str(r[c_btl])[:8]}…`"
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
                with left:
                    st.markdown("**Eintrag A**")
                    st.markdown(f"```\n{pair['name_a']}\n```")
                    st.caption(f"ID: {pair['id_a']}")
                with mid:
                    st.markdown("**Eintrag B**")
                    st.markdown(f"```\n{pair['name_b']}\n```")
                    st.caption(f"ID: {pair['id_b']}")
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
all_pairs_decided = (len(st.session_state.decisions) == len(candidates))

# Export erst anzeigen wenn alle Paare entschieden wurden
if not all_pairs_decided:
    remaining = len(candidates) - len(st.session_state.decisions)
    st.info(f"⏳ Noch **{remaining}** Paare offen — bitte alle entscheiden, dann erscheint der Export.")
    st.stop()

if not confirmed_pairs and not has_cuvee:
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

# 1) Bestätigte Duplikate
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

# 2) Bestätigte Cuvées
if cfg["cuvee_detection"] and edited_cuvee is not None:
    for _, row in edited_cuvee.iterrows():
        is_confirmed = bool(row["Cuvée"])
        out_df.loc[out_df[col_id] == row["id"], "is_cuvee"] = "1" if is_confirmed else "0"

        if is_confirmed:
            src_row   = df[df[col_id] == row["id"]]
            group_val = src_row[col_group].iloc[0] if not src_row.empty else ""
            synonyms.append({
                "type":           "cuvee",
                "duplicate_id":   "",
                "original_id":    row["id"],
                "duplicate_name": "",
                "original_name":  row["Rebsorte"],
                cfg["group_label"].lower(): group_val,
                "confidence_%":   100,
            })

syn_df = pd.DataFrame(synonyms)

# Vorschau
if confirmed_pairs:
    n_dup = len(confirmed_pairs)
    st.write(f"**{n_dup} Duplikate** bestätigt:")
    st.dataframe(
        syn_df[syn_df["type"] == "duplicate"].drop(columns="type"),
        use_container_width=True, hide_index=True,
    )

if has_cuvee:
    n_cuv = int(edited_cuvee["Cuvée"].sum())
    st.write(f"**{n_cuv} Cuvées** markiert:")
    st.dataframe(
        syn_df[syn_df["type"] == "cuvee"][
            ["original_id", "original_name", cfg["group_label"].lower()]
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
    st.caption("Enthält Duplikate (type=duplicate) und Cuvées (type=cuvee)")
