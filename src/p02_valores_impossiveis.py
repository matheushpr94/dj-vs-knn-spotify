"""Auditoria de validade de dominio: valores impossiveis ou suspeitos por coluna."""
import pandas as pd
from p01_o_arquivo_e_o_que_promete import load
from cabecalho import cabecalho

UNIT = ["danceability", "energy", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence"]


def main() -> None:
    cabecalho(
        pergunta="Existe valor impossivel nas colunas? Zero que nao e' zero, BPM que nao existe?",
        como="Checa cada coluna contra o dominio documentado no cartao oficial da Spotify, e procura marcador de ausencia disfarcado de valor.",
        ler_assim="Valor fora do dominio e' defeito. Mas pico em exatamente zero costuma ser censura ou falha de extracao, nao medida baixa.",
        piso="Um dado limpo teria 0% fora do dominio.",
    )
    df = load()
    n = len(df)
    print(f"{n:,} linhas\n")

    print("=== linhas com campo textual nulo ===")
    bad = df[df[["artists", "album_name", "track_name"]].isna().any(axis=1)]
    print(bad[["track_id", "artists", "track_name", "popularity", "track_genre"]].to_string())

    print("\n=== features [0,1]: fora do intervalo ===")
    for c in UNIT:
        out = ((df[c] < 0) | (df[c] > 1)).sum()
        print(f"  {c:<17} min {df[c].min():>7.3f}  max {df[c].max():>7.3f}  fora: {out}")

    print("\n=== valores impossiveis / sentinelas ===")
    checks = {
        "tempo == 0 (BPM impossivel)": (df.tempo == 0).sum(),
        "duration_ms == 0": (df.duration_ms == 0).sum(),
        "duration < 30s": (df.duration_ms < 30_000).sum(),
        "duration > 10min": (df.duration_ms > 600_000).sum(),
        "time_signature == 0 (invalido)": (df.time_signature == 0).sum(),
        "loudness > 0 dB (clipping)": (df.loudness > 0).sum(),
        "popularity == 0": (df.popularity == 0).sum(),
        "instrumentalness == 0 exato": (df.instrumentalness == 0).sum(),
    }
    for k, v in checks.items():
        print(f"  {k:<32} {v:>7,}  ({v / n:.2%})")

    print("\n=== dominio das categoricas ===")
    for c in ["key", "mode", "time_signature", "explicit"]:
        print(f"  {c:<15} {sorted(df[c].unique())}")

    print("\n=== coerencia: tempo == 0 tem audio feature valida? ===")
    z = df[df.tempo == 0]
    if len(z):
        print(f"  {len(z)} faixas. energy media {z.energy.mean():.3f} | "
              f"danceability media {z.danceability.mean():.3f} | "
              f"popularity media {z.popularity.mean():.1f}")
        print(f"  quantas tem TODAS as audio features zeradas: "
              f"{(z[UNIT].sum(axis=1) == 0).sum()}")

    print("\n=== campo 'artists' multi-artista ===")
    multi = df.artists.dropna().str.contains(";").sum()
    print(f"  linhas com ';' (mais de um artista): {multi:,} ({multi / n:.1%})")
    print(f"  artists unicos como string crua      : {df.artists.nunique():,}")
    exp = df.artists.dropna().str.split(";").explode().str.strip()
    print(f"  artistas individuais apos split      : {exp.nunique():,}")

    print("\n=== mesma faixa, popularity divergente entre generos? ===")
    v = df.groupby("track_id").popularity.agg(["nunique", "min", "max", "count"])
    div = v[v["nunique"] > 1]
    print(f"  track_id com popularity divergente: {len(div):,}")
    if len(div):
        print(f"  amplitude media da divergencia   : {(div['max'] - div['min']).mean():.1f} pontos")
        print(f"  amplitude maxima                 : {(div['max'] - div['min']).max()} pontos")


if __name__ == "__main__":
    main()
