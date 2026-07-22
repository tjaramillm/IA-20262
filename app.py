"""
Mapa de calor de rendimiento de jugadores de fútbol
===================================================

Ejecutar con:
    streamlit run app.py

Dibuja un campo de fútbol y superpone el mapa de calor de la presencia de cada
jugador a partir de coordenadas de posición (x, y).

- Si no subes datos, se genera un conjunto de demostración por posición.
- Puedes subir tu propio CSV con las columnas: jugador, x, y  (posicion es opcional).
  * x va de 0 a 105 (largo del campo, ataque hacia la derecha)
  * y va de 0 a 68  (ancho del campo)
"""

import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch

# ---------------------------------------------------------------------------
# Configuración del campo
# ---------------------------------------------------------------------------
PITCH_LENGTH = 105  # largo (eje x)
PITCH_WIDTH = 68    # ancho (eje y)

# Anclas de cada posición (x, y) para generar los datos de demostración.
POSICIONES = {
    "Portero":            (8, 34),
    "Defensa central":    (25, 34),
    "Lateral derecho":    (35, 60),
    "Lateral izquierdo":  (35, 8),
    "Mediocentro":        (52, 34),
    "Mediapunta":         (72, 34),
    "Extremo derecho":    (78, 58),
    "Extremo izquierdo":  (78, 10),
    "Delantero":          (88, 34),
}

TEMAS = {
    "Césped oscuro":  {"pitch_color": "#22312b", "line_color": "#c7d5cc"},
    "Pizarra":        {"pitch_color": "#1e1e2e", "line_color": "#a6adc8"},
    "Clásico blanco": {"pitch_color": "white",   "line_color": "#3b3b3b"},
}

COLORMAPS = ["hot", "inferno", "magma", "plasma", "viridis", "turbo", "YlOrRd", "Reds"]


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
@st.cache_data
def generar_demo(n_por_jugador: int = 400, seed: int = 42) -> pd.DataFrame:
    """Genera coordenadas simuladas por jugador alrededor de su posición."""
    rng = np.random.default_rng(seed)
    filas = []
    for i, (rol, (ax, ay)) in enumerate(POSICIONES.items(), start=1):
        nombre = f"{i:02d} · {rol}"
        # Mayor dispersión a lo largo del campo que a lo ancho (más realista).
        xs = rng.normal(ax, 13, n_por_jugador).clip(0, PITCH_LENGTH)
        ys = rng.normal(ay, 9, n_por_jugador).clip(0, PITCH_WIDTH)
        for x, y in zip(xs, ys):
            filas.append((nombre, rol, float(x), float(y)))
    return pd.DataFrame(filas, columns=["jugador", "posicion", "x", "y"])


def cargar_csv(archivo) -> pd.DataFrame:
    """Lee y valida un CSV subido por el usuario."""
    df = pd.read_csv(archivo)
    df.columns = [c.strip().lower() for c in df.columns]
    faltan = {"jugador", "x", "y"} - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas obligatorias: {', '.join(sorted(faltan))}")
    if "posicion" not in df.columns:
        df["posicion"] = "—"
    df = df.dropna(subset=["x", "y"]).copy()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    return df.dropna(subset=["x", "y"])


# ---------------------------------------------------------------------------
# Dibujo del mapa de calor
# ---------------------------------------------------------------------------
def dibujar_heatmap(datos, tema, cmap, tipo, vertical, titulo):
    estilo = TEMAS[tema]
    pitch = Pitch(
        pitch_type="custom",
        pitch_length=PITCH_LENGTH,
        pitch_width=PITCH_WIDTH,
        line_color=estilo["line_color"],
        pitch_color=estilo["pitch_color"],
        line_zorder=2,
    )
    orientacion = "vertical" if vertical else "horizontal"
    figsize = (7, 9.5) if vertical else (10, 7)
    fig, ax = pitch.draw(figsize=figsize)
    fig.set_facecolor(estilo["pitch_color"])

    if len(datos) >= 5:
        if tipo == "Suavizado (KDE)":
            pitch.kdeplot(
                datos["x"], datos["y"], ax=ax,
                fill=True, levels=100, thresh=0.02,
                cmap=cmap, alpha=0.85, zorder=1,
            )
        else:  # Rejilla (bins)
            stats = pitch.bin_statistic(
                datos["x"], datos["y"], statistic="count", bins=(24, 16)
            )
            pitch.heatmap(stats, ax=ax, cmap=cmap, edgecolors=estilo["pitch_color"], zorder=1)

    ax.set_title(titulo, color=estilo["line_color"], fontsize=15, pad=12)
    if vertical:
        # mplsoccer dibuja horizontal por defecto; rotamos si se pide vertical.
        pass
    return fig


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Mapa de calor de jugadores", page_icon="⚽", layout="wide")
    st.title("⚽ Mapa de calor de rendimiento de jugadores")

    with st.sidebar:
        st.header("Datos")
        archivo = st.file_uploader(
            "Sube un CSV (columnas: jugador, x, y)", type=["csv"]
        )
        if archivo is not None:
            try:
                df = cargar_csv(archivo)
                st.success(f"{len(df):,} registros cargados.")
            except Exception as e:
                st.error(f"Error al leer el CSV: {e}")
                st.stop()
        else:
            df = generar_demo()
            st.info("Usando datos de demostración. Sube tu CSV para reemplazarlos.")

        st.header("Visualización")
        jugadores = ["— Todos —"] + sorted(df["jugador"].unique().tolist())
        jugador = st.selectbox("Jugador", jugadores)
        tipo = st.radio("Tipo de mapa", ["Suavizado (KDE)", "Rejilla (bins)"])
        cmap = st.selectbox("Paleta de color", COLORMAPS)
        tema = st.selectbox("Tema del campo", list(TEMAS.keys()))
        vertical = st.toggle("Orientación vertical", value=False)

    # Filtrado
    if jugador == "— Todos —":
        datos = df
        titulo = "Todos los jugadores"
    else:
        datos = df[df["jugador"] == jugador]
        titulo = jugador

    col_mapa, col_stats = st.columns([3, 1])

    with col_mapa:
        if len(datos) < 5:
            st.warning("Se necesitan al menos 5 registros para dibujar el mapa de calor.")
        else:
            fig = dibujar_heatmap(datos, tema, cmap, tipo, vertical, titulo)
            st.pyplot(fig, use_container_width=True)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200,
                        facecolor=fig.get_facecolor(), bbox_inches="tight")
            st.download_button(
                "⬇️ Descargar PNG", buf.getvalue(),
                file_name="mapa_calor.png", mime="image/png",
            )
            plt.close(fig)

    with col_stats:
        st.subheader("Resumen")
        st.metric("Registros", f"{len(datos):,}")
        if len(datos) > 0:
            st.metric("Posición media X", f"{datos['x'].mean():.1f}")
            st.metric("Posición media Y", f"{datos['y'].mean():.1f}")
            tercio = pd.cut(
                datos["x"], bins=[0, 35, 70, 105],
                labels=["Defensivo", "Medio", "Ofensivo"],
            ).value_counts(normalize=True) * 100
            st.caption("Presencia por tercio del campo")
            for zona, pct in tercio.items():
                st.write(f"{zona}: {pct:.0f}%")

    with st.expander("Ver datos"):
        st.dataframe(datos, use_container_width=True, height=300)


if __name__ == "__main__":
    main()
