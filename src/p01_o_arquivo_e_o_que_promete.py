"""Perfil de primeira passada do dataset — roda antes de qualquer modelagem."""
import pandas as pd
from cabecalho import cabecalho

RAW = "data/raw/spotify_tracks.csv"
AUDIO = ["danceability", "energy", "loudness", "speechiness", "acousticness",
         "instrumentalness", "liveness", "valence", "tempo", "duration_ms"]


def load() -> pd.DataFrame:
    df = pd.read_csv(RAW, index_col=0)
    return df


def main() -> None:
    cabecalho(
        pergunta="Este arquivo e' o que ele promete ser? Quantas linhas, quantas colunas, o que falta e o que se repete.",
        como="Contagens diretas com pandas: shape, nulos, duplicatas, distribuicao de popularity e correlacao com as audio features.",
        ler_assim="Numero de linhas diferente do numero de track_id unicos significa repeticao — e repeticao precisa de explicacao antes de qualquer conta.",
        piso="Um arquivo saudavel teria linhas == ids unicos e nenhum valor sentinela.",
        metricas=['pearson'],
    )
    df = load()
    print(f"shape: {df.shape[0]:,} linhas x {df.shape[1]} colunas\n")

    print("=== nulos ===")
    n = df.isna().sum()
    print(n[n > 0] if n.any() else "nenhum")

    print("\n=== duplicatas ===")
    print(f"linhas totais          : {len(df):,}")
    print(f"track_id unicos        : {df.track_id.nunique():,}")
    print(f"track_id repetidos     : {len(df) - df.track_id.nunique():,}")
    dup = df.duplicated(subset=["track_name", "artists"]).sum()
    print(f"dup por (nome, artista): {dup:,}")
    print(f"linhas 100% identicas  : {df.duplicated().sum():,}")

    print("\n=== generos ===")
    g = df.track_genre.value_counts()
    print(f"{df.track_genre.nunique()} generos | min {g.min()} | max {g.max()} por genero")

    print("\n=== popularity ===")
    print(df.popularity.describe().round(2).to_string())
    print(f"faixas com popularity == 0: {(df.popularity == 0).sum():,} "
          f"({(df.popularity == 0).mean():.1%})")

    print("\n=== correlacao com popularity (Pearson) ===")
    corr = df[AUDIO + ["popularity"]].corr().popularity.drop("popularity")
    print(corr.reindex(corr.abs().sort_values(ascending=False).index).round(3).to_string())

    print("\n=== variancia de popularity explicada por genero ===")
    grand = df.popularity.mean()
    between = df.groupby("track_genre").popularity.agg(["mean", "count"])
    ss_between = (between["count"] * (between["mean"] - grand) ** 2).sum()
    ss_total = ((df.popularity - grand) ** 2).sum()
    print(f"eta^2 (genero) = {ss_between / ss_total:.3f}")


if __name__ == "__main__":
    main()
