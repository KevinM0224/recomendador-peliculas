"""Recomendador híbrido de películas (MovieLens) - app Streamlit para desplegar.

Usa solo numpy/scipy/scikit-learn (sin PyTorch): los pesos del NCF se
exportaron a numpy y aquí se recalcula la red a mano.
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st

st.set_page_config(page_title="Recomendador híbrido de películas", page_icon="🎬", layout="wide")

ART = Path(os.environ.get("ARTEFACTOS", Path(__file__).parent / "artefactos"))

CLAVES = {"Matrix Factorization": "mf", "Filtrado colaborativo": "cf_items",
          "Contenido": "contenido", "NCF": "ncf", "Popularidad": "popularidad"}
MODELOS_UI = ["Híbrido", "Matrix Factorization", "Filtrado colaborativo", "Contenido", "NCF", "Popularidad"]
COLS = ["#", "Película", "Géneros", "Puntaje", "IMDb"]


# ---------------------------------------------------------------------------
# 1. Carga de artefactos (una sola vez; Streamlit la reutiliza entre clics)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelos...")
def cargar():
    movies = pd.read_parquet(ART / "movies_app.parquet").sort_values("movie_idx").reset_index(drop=True)
    links = pd.read_parquet(ART / "links.parquet")
    C = sp.load_npz(ART / "contenido_C.npz").tocsr()        # películas x términos (géneros + tags)
    S_kT = sp.load_npz(ART / "S_k.npz").T.tocsr()           # similitud (50 vecinos), traspuesta para "x @ S_k"
    V = np.load(ART / "V50.npy").astype("float32")          # factores del SVD (películas x 50)
    R = sp.load_npz(ART / "R.npz").tocsr()                  # ratings centrados (usuarios x películas)
    W = {k: v.astype("float32") for k, v in np.load(ART / "ncf_pesos.npz").items()}
    with open(ART / "favs.json", encoding="utf-8") as f:
        favs = json.load(f)
    with open(ART / "config.json") as f:
        media_global = json.load(f)["media_global"]
    with open(ART / "meta.pkl", "rb") as f:
        meta = pickle.load(f)

    n_users, n_movies = R.shape
    assert len(movies) == n_movies, "movies_app.parquet y R.npz no coinciden"

    # Popularidad = cuántos usuarios calificaron cada película. Se cuenta con la estructura
    # de R (no con R != 0), así un rating igual a la media del usuario (0.0) sigue contando.
    popularidad = np.bincount(R.indices, minlength=n_movies).astype("float32")

    # Las 3000 películas más populares, para el buscador del usuario nuevo
    titulo2idx = {}
    for i in np.argsort(-popularidad)[:3000]:
        titulo2idx.setdefault(movies.loc[i, "title"], int(i))

    return dict(movies=movies, imdb=links.set_index("movieId")["imdbId"], C=C, S_kT=S_kT, V=V, R=R, W=W,
                favs=favs, media_global=media_global, clf=meta["clf"], clf_nuevo=meta["clf_nuevo"],
                nombres=meta["nombres"], n_users=n_users, n_movies=n_movies,
                log_pop=np.log1p(popularidad), titulo2idx=titulo2idx)


D = cargar()
movies, R, W, C, V, S_kT = D["movies"], D["R"], D["W"], D["C"], D["V"], D["S_kT"]
n_users, n_movies, nombres = D["n_users"], D["n_movies"], D["nombres"]


# ---------------------------------------------------------------------------
# 2. Motor de recomendación
# ---------------------------------------------------------------------------
def url_imdb(mid):
    v = D["imdb"].get(mid)
    return f"https://www.imdb.com/title/tt{int(v):07d}/" if pd.notna(v) else "https://www.imdb.com"


def scores_ncf(u):
    """Red NeuMF (rama GMF + rama MLP) en numpy: puntaje del usuario u para TODAS las películas."""
    gmf = W["gmf_u.weight"][u] * W["gmf_m.weight"]
    mlp = np.concatenate([np.tile(W["mlp_u.weight"][u], (n_movies, 1)), W["mlp_m.weight"]], axis=1)
    for i in (0, 2, 4):                                     # Linear + ReLU, tres veces
        mlp = np.maximum(mlp @ W[f"mlp.{i}.weight"].T + W[f"mlp.{i}.bias"], 0)
    out = np.concatenate([gmf, mlp], axis=1) @ W["out.weight"].T + W["out.bias"]
    return out.ravel()


def z_unseen(s, vistas):
    """Estandariza los puntajes entre las películas NO vistas; las vistas quedan en -1e9."""
    s = s.astype("float32").copy()
    s[vistas] = np.nan
    mu = np.nanmean(s)
    sd = max(float(np.sqrt(np.nanmean((s - mu) ** 2))), 1e-6)
    z = (s - mu) / sd
    z[vistas] = -1e9
    return z


def recomendar(rec, vistas, u_idx, nombre_modelo, n):
    s = {
        "popularidad": D["log_pop"],
        "cf_items": S_kT @ rec,
        "mf": (rec @ V) @ V.T,
        "contenido": ((sp.csr_matrix(rec) @ C) @ C.T).toarray().ravel(),
    }
    if u_idx is not None:                                   # el NCF solo existe para usuarios del dataset
        s["ncf"] = scores_ncf(u_idx)
    z = {k: z_unseen(v, vistas) for k, v in s.items()}

    if nombre_modelo == "Híbrido":
        usados = [k for k in nombres if k in z]
        cand = np.unique(np.concatenate([np.argpartition(-z[k], 49)[:50] for k in usados]))
        X = np.stack([z[k][cand] for k in usados], axis=1)
        prob = (D["clf"] if u_idx is not None else D["clf_nuevo"]).predict_proba(X)[:, 1]
        orden = np.argsort(-prob)[:n]
        ids = cand[orden]
        punt = [f"{p:.1%}" for p in prob[orden]]
    else:
        zz = z[CLAVES[nombre_modelo]]
        ids = np.argsort(-zz)[:n]
        punt = [f"{v:.2f}" for v in zz[ids]]

    return pd.DataFrame({
        "#": range(1, len(ids) + 1),
        "Película": movies.loc[ids, "title"].values,
        "Géneros": movies.loc[ids, "genres"].str.replace("|", ", ", regex=False).values,
        "Puntaje": punt,
        "IMDb": [url_imdb(movies.loc[i, "movieId"]) for i in ids],
    })


# ---------------------------------------------------------------------------
# 3. Lógica de cada pestaña
# ---------------------------------------------------------------------------
def ui_existente(uid, nombre_modelo, n):
    u = int(uid)
    fila = R[u]
    vistas = np.zeros(n_movies, dtype=bool)
    vistas[fila.indices] = True
    gustaron = ", ".join(D["favs"].get(str(u), [])) or "(sin favoritas con 4.5 o más)"
    info = f"**Usuario {u}**: {int(vistas.sum())} películas calificadas. Algunas que le gustaron mucho: {gustaron}"
    return info, recomendar(fila.toarray().ravel(), vistas, u, nombre_modelo, int(n))


def ui_nuevo(gustaron, no_gustaron, nombre_modelo, n):
    if len(gustaron) < 3:
        return "Elige al menos 3 películas que te hayan gustado.", pd.DataFrame(columns=COLS)
    rec = np.zeros(n_movies, dtype="float32")
    vistas = np.zeros(n_movies, dtype=bool)
    for t in gustaron:
        rec[D["titulo2idx"][t]] = 5.0 - D["media_global"]
        vistas[D["titulo2idx"][t]] = True
    for t in no_gustaron:
        rec[D["titulo2idx"][t]] = 1.0 - D["media_global"]
        vistas[D["titulo2idx"][t]] = True
    info = f"Perfil armado con {len(gustaron)} películas que te gustaron y {len(no_gustaron)} que no."
    return info, recomendar(rec, vistas, None, nombre_modelo, int(n))


def mostrar(clave):
    """Dibuja el último resultado guardado de una pestaña (así no se borra al mover otros controles)."""
    if clave in st.session_state:
        info, df = st.session_state[clave]
        st.markdown(info)
        if len(df):
            st.dataframe(df, hide_index=True,
                         column_config={"IMDb": st.column_config.LinkColumn("IMDb", display_text="ver")})


# ---------------------------------------------------------------------------
# 4. El panel (el mismo del notebook 2)
# ---------------------------------------------------------------------------
st.title("🎬 Recomendador híbrido de películas")
st.caption("MovieLens · filtrado colaborativo + Matrix Factorization + contenido + NCF")

tab1, tab2 = st.tabs(["Usuario existente", "Usuario nuevo"])

with tab1:
    uid = st.number_input(f"ID de usuario (0 a {n_users - 1})", min_value=0, max_value=n_users - 1,
                          value=0, step=1)
    m1 = st.radio("Modelo", MODELOS_UI, horizontal=True, key="m1")
    n1 = st.slider("Cuántas recomendaciones", 5, 20, 10, key="n1")
    if st.button("Recomendar", type="primary", key="b1"):
        st.session_state["res1"] = ui_existente(uid, m1, n1)
    mostrar("res1")

with tab2:
    titulos = list(D["titulo2idx"])
    gus = st.multiselect("Películas que te gustaron (mínimo 3)", titulos)
    nog = st.multiselect("Películas que NO te gustaron (opcional)", titulos)
    m2 = st.radio("Modelo", [m for m in MODELOS_UI if m != "NCF"], horizontal=True, key="m2")
    n2 = st.slider("Cuántas recomendaciones", 5, 20, 10, key="n2")
    if st.button("Recomendar", type="primary", key="b2"):
        st.session_state["res2"] = ui_nuevo(gus, nog, m2, n2)
    mostrar("res2")
