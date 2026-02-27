"""
WW Duplikat-Erkennung
Streamlit App zur Erkennung und Bereinigung von Duplikaten in Weindaten.
Unterstützt: Rebsorten (grapevariety.csv), Weinbauregionen (growingregion.csv)
"""

import io
import os
import re
import unicodedata

import pandas as pd
import streamlit as st
from rapidfuzz import fuzz

# ── Datei-Konfigurationen ──────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(__file__)

# Jede Konfiguration beschreibt, wie eine CSV-Datei zu interpretieren ist.
FILE_CONFIGS = {
    "grapevariety": {
        "label":        "Rebsorten",
        "file":         os.path.join(BASE_DIR, "Datafiles", "grapevariety.csv"),
        "col_id":       13,     # GrapeVarity.Id
        "col_name":     15,     # GrapeVarity.Title  (was wird verglichen)
        "col_group":    1,      # GrapeVarityGroup.Title  (Gruppierung: Weintyp)
        "group_label":  "Weintyp",
        "split_dash":   False,  # Bindestrich-Trenner NICHT normalisieren
        "out_file":     "grapevariety_cleaned.csv",
    },
    "growingregion": {
        "label":        "Weinbauregionen",
        "file":         os.path.join(BASE_DIR, "Datafiles", "growingregion.csv"),
        "col_id":       1,      # Id
        "col_name":     3,      # Name  (was wird verglichen)
        "col_group":    0,      # IsoAlp2  (Gruppierung: Land)
        "group_label":  "Land",
        "split_dash":   True,   # "Bordeaux - Pauillac" == "Pauillac, Bordeaux"
        "out_file":     "growingregion_cleaned.csv",
    },
}

# ── Seiteneinstellungen ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="WW Duplikat-Erkennung",
    page_icon="🍷",
    layout="wide",
)


# ── Hilfsfunktionen ────────────────────────────────────────────────────────────

def normalize(text: str, split_dash: bool = False) -> str:
    """Normalisiert einen Namen für den Fuzzy-Vergleich.

    - Entfernt Akzente/Diakritika
    - Kleinschreibung + Whitespace normalisieren
    - split_dash=True: ' - ' wird zu ', ' (z.B. 'Bordeaux - Pauillac' → sortierbar)
    - Tokens werden alphabetisch sortiert (Blend-/Kombinations-Reihenfolge egal)
    """
    if pd.isna(text) or str(text).strip().upper() in ("NULL", "NONE", ""):
        return ""
    text = str(text)
    # Akzente entfernen (NFD → Combining-Zeichen raus)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    # Kleinschreibung + Whitespace
    text = re.sub(r"\s+", " ", text).lower().strip()
    # Bindestrich-Trenner normalisieren (nur für Regionen)
    # Nur " - " mit Leerzeichen — NICHT Bindestriche in Ortsnamen wie "Savigny-les-Beaune"
    if split_dash:
        text = re.sub(r"\s+-\s+", ", ", text)
    # Tokens alphabetisch sortieren → Reihenfolge spielt keine Rolle
    tokens = sorted(t.strip() for t in text.split(",") if t.strip())
    return ", ".join(tokens)


def load_csv(source) -> tuple[pd.DataFrame | None, str | None]:
    """Lädt CSV aus Dateipfad oder Bytes. Probiert mehrere Encodings."""
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
                    group_col: str, threshold: int,
                    split_dash: bool = False) -> list[dict]:
    """Findet alle Paare mit Fuzzy-Ähnlichkeit >= threshold innerhalb derselben Gruppe."""
    candidates = []
    for group_name, gdf in df.groupby(group_col):
        valid = gdf[~gdf[name_col].isin(["NULL", "NONE", ""])].copy()
        valid = valid[valid[name_col].notna()].reset_index(drop=True)
        if len(valid) < 2:
            continue

        ids   = valid[id_col].tolist()
        names = valid[name_col].tolist()
        norms = [normalize(n, split_dash=split_dash) for n in names]

        n = len(ids)
        for i in range(n):
            if not norms[i]:
                continue
            for j in range(i + 1, n):
                if not norms[j]:
                    continue
                score = fuzz.ratio(norms[i], norms[j])
                if score >= threshold:
                    candidates.append({
                        "group":  group_name,
                        "id_a":   ids[i],
                        "name_a": names[i],
                        "id_b":   ids[j],
                        "name_b": names[j],
                        "score":  round(score, 1),
                    })

    return sorted(candidates, key=lambda x: (-x["score"], x["group"], x["name_a"]))


# ── Lokal vs. Cloud erkennen ───────────────────────────────────────────────────
# Auf Streamlit Community Cloud existieren die lokalen Dateien nicht.
# In diesem Fall wird der "Standard-Datei"-Modus ausgeblendet.

IS_LOCAL = any(os.path.exists(cfg["file"]) for cfg in FILE_CONFIGS.values())

# ── Session State Init ─────────────────────────────────────────────────────────

_defaults: dict = {
    "df":         None,
    "encoding":   None,
    "config":     None,   # aktive FILE_CONFIGS-Einträge
    "candidates": None,   # None = noch nicht analysiert, [] = keine Treffer
    "decisions":  {},     # {(id_a, id_b): 'original_a' | 'original_b' | 'reject'}
    "col_id":     None,
    "col_name":   None,
    "col_group":  None,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Einstellungen")

    # Datentyp-Auswahl
    dtype_options = {cfg["label"]: key for key, cfg in FILE_CONFIGS.items()}
    selected_label = st.radio("Datentyp", list(dtype_options.keys()))
    selected_key   = dtype_options[selected_label]
    active_config  = FILE_CONFIGS[selected_key]

    st.divider()

    # Quelle: Standard-Datei (nur lokal) oder Upload
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
            df, enc = load_csv(raw)
            if df is not None:
                st.session_state.df         = df
                st.session_state.encoding   = enc
                st.session_state.config     = active_config
                st.session_state.col_id     = df.columns[active_config["col_id"]]
                st.session_state.col_name   = df.columns[active_config["col_name"]]
                st.session_state.col_group  = df.columns[active_config["col_group"]]
                st.session_state.candidates = None
                st.session_state.decisions  = {}
                st.success(f"✓ {len(df)} Zeilen geladen  \nKodierung: {enc}")
            else:
                st.error("Datei konnte nicht gelesen werden.\nBitte Encoding/Format prüfen.")
        else:
            st.warning("Keine Datei ausgewählt.")

    st.divider()

    threshold = st.slider(
        "Ähnlichkeitsschwelle (%)", 50, 100, 80, 1,
        help="Paare über diesem Wert werden zur Review vorgeschlagen.\n"
             "80 % ist ein guter Startwert.",
    )

    analyse_ready = (
        st.session_state.df is not None
        and st.session_state.config is not None
        and st.session_state.config["label"] == active_config["label"]
    )

    if analyse_ready:
        if st.button("🔍 Analyse starten", type="primary", use_container_width=True):
            cfg = st.session_state.config
            with st.spinner("Suche Duplikate..."):
                cands = find_candidates(
                    st.session_state.df,
                    st.session_state.col_id,
                    st.session_state.col_name,
                    st.session_state.col_group,
                    threshold,
                    split_dash=cfg["split_dash"],
                )
            st.session_state.candidates = cands
            st.session_state.decisions  = {}
            if cands:
                st.success(f"✓ {len(cands)} Kandidatenpaare")
            else:
                st.info("Keine Paare gefunden.")

    # Fortschrittsbalken
    if st.session_state.candidates:
        total   = len(st.session_state.candidates)
        decided = len(st.session_state.decisions)
        st.divider()
        st.progress(
            decided / total if total else 0,
            text=f"Review: {decided} / {total} Paare entschieden",
        )


# ── Hauptbereich ───────────────────────────────────────────────────────────────

cfg = st.session_state.config or active_config
st.title(f"🍷 WW Duplikat-Erkennung — {cfg['label']}")

if st.session_state.df is None or st.session_state.config is None:
    st.info("👈 Bitte links einen Datentyp wählen, Datei laden und die Analyse starten.")
    st.stop()

# Warnung wenn geladene Datei nicht zum gewählten Typ passt
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
    st.metric(cfg["group_label"] + "gruppen", df[col_group].nunique())
with mc3:
    n_cand = len(st.session_state.candidates) if st.session_state.candidates is not None else "–"
    st.metric("Kandidatenpaare", n_cand)
with mc4:
    confirmed = sum(1 for v in st.session_state.decisions.values() if v != "reject")
    st.metric("Bestätigte Duplikate", confirmed)

with st.expander(f"Einträge pro {cfg['group_label']} anzeigen"):
    counts = (
        df.groupby(col_group)[col_id]
        .count()
        .reset_index()
        .rename(columns={col_group: cfg["group_label"], col_id: cfg["label"]})
        .sort_values(cfg["label"], ascending=False)
    )
    st.dataframe(counts, use_container_width=True, hide_index=True)

if st.session_state.candidates is None:
    st.info("👈 Klicke auf **Analyse starten** um Duplikate zu suchen.")
    st.stop()

candidates = st.session_state.candidates

if not candidates:
    st.success("✅ Keine Duplikate über dem Schwellwert gefunden.")
    st.stop()

# ── Review ─────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("📋 Duplikat-Review")
st.caption(
    "Für jedes Paar: Wähle, welcher Eintrag das **Original** ist — "
    "oder verwirf, wenn es kein Duplikat ist."
)

by_group: dict[str, list] = {}
for c in candidates:
    by_group.setdefault(c["group"], []).append(c)

for group_name, pairs in by_group.items():
    n_open = sum(1 for p in pairs if (p["id_a"], p["id_b"]) not in st.session_state.decisions)
    with st.expander(
        f"**{group_name}** — {len(pairs)} Paare  ({n_open} offen)",
        expanded=n_open > 0,
    ):
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
                    if st.button(
                        "A=Orig", key=f"a_{uid}",
                        type="primary" if decision == "original_a" else "secondary",
                        help="A ist das Original — B wird als Duplikat markiert",
                    ):
                        st.session_state.decisions[key] = "original_a"
                        st.rerun()

                with b2:
                    if st.button(
                        "B=Orig", key=f"b_{uid}",
                        type="primary" if decision == "original_b" else "secondary",
                        help="B ist das Original — A wird als Duplikat markiert",
                    ):
                        st.session_state.decisions[key] = "original_b"
                        st.rerun()

                with b3:
                    if st.button(
                        "✗", key=f"r_{uid}",
                        type="primary" if decision == "reject" else "secondary",
                        help="Kein Duplikat — Paar verwerfen",
                    ):
                        st.session_state.decisions[key] = "reject"
                        st.rerun()

            st.divider()


# ── Export ─────────────────────────────────────────────────────────────────────

confirmed_pairs = [(k, v) for k, v in st.session_state.decisions.items() if v != "reject"]

if not confirmed_pairs:
    st.stop()

st.divider()
st.subheader("📥 Export")

out_df = df.copy()

# Spalte 'original_id' direkt nach der ID-Spalte einfügen
insert_pos = list(out_df.columns).index(col_id) + 1
out_df.insert(insert_pos, "original_id", "")

synonyms: list[dict] = []

for (id_a, id_b), decision in confirmed_pairs:
    original_id  = id_a if decision == "original_a" else id_b
    duplicate_id = id_b if decision == "original_a" else id_a

    orig_rows = df[df[col_id] == original_id]
    dup_rows  = df[df[col_id] == duplicate_id]

    orig_name  = orig_rows[col_name].iloc[0]  if not orig_rows.empty else ""
    dup_name   = dup_rows[col_name].iloc[0]   if not dup_rows.empty  else ""
    group_val  = orig_rows[col_group].iloc[0] if not orig_rows.empty else ""

    pair  = next(
        (c for c in candidates if {c["id_a"], c["id_b"]} == {id_a, id_b}), None
    )
    score = pair["score"] if pair else 0

    out_df.loc[out_df[col_id] == duplicate_id, "original_id"] = original_id

    synonyms.append({
        "duplicate_id":   duplicate_id,
        "original_id":    original_id,
        "duplicate_name": dup_name,
        "original_name":  orig_name,
        cfg["group_label"].lower(): group_val,
        "confidence_%":   score,
    })

syn_df = pd.DataFrame(synonyms)

st.write(f"**{len(synonyms)} Duplikate** werden exportiert:")
st.dataframe(syn_df, use_container_width=True, hide_index=True)

dcol1, dcol2 = st.columns(2)

with dcol1:
    csv_out = out_df.to_csv(sep=";", index=False).encode("utf-8-sig")
    st.download_button(
        "📄 Ergebnis-CSV herunterladen",
        data=csv_out,
        file_name=cfg["out_file"],
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(f"Gleiche Struktur wie `{os.path.basename(cfg['file'])}` + Spalte `original_id`")

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
    st.caption("duplicate_id → original_id Mapping für das Originalsystem")
