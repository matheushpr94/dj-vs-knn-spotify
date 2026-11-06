"""Testa se faixas zeradas sao duplicatas delistadas de musicas vivas no dataset."""
import pandas as pd
from prepare import carregar
from cabecalho import cabecalho


def main() -> None:
    cabecalho(
        pergunta="As faixas com popularidade zero sao musicas impopulares, ou copias delistadas de hits?",
        como="Para cada faixa zerada, procura no proprio arquivo outra com mesmo nome e artista e popularidade maior que zero.",
        ler_assim="Se a gemea viva existe e e' popular, o zero nao era impopularidade — era indisponibilidade daquele track_id.",
        piso="Se fossem musicas realmente impopulares, quase nenhuma teria gemea viva.",
    )
    df = carregar()
    ng = df.groupby("track_id").track_genre.nunique().rename("n_gen")
    d = (df.sort_values("popularity", ascending=False)
           .drop_duplicates("track_id").join(ng, on="track_id"))

    best = d.groupby(["track_name", "artists"]).popularity.max().rename("melhor_gemea")
    d = d.join(best, on=["track_name", "artists"])
    d = d.assign(tem_gemea=d.melhor_gemea > d.popularity)

    z47 = d[(d.n_gen.between(4, 7)) & (d.popularity == 0)]
    print(f"zeradas no grupo 4-7 generos : {len(z47):,}")
    print(f"  com gemea viva             : {z47.tem_gemea.sum():,} ({z47.tem_gemea.mean():.1%})")
    print(f"  popularity mediana da gemea: {z47[z47.tem_gemea].melhor_gemea.median():.0f}")

    print("\nteste decisivo — dedup por (nome, artista) em vez de track_id:")
    canon = d.sort_values("popularity", ascending=False).drop_duplicates(["track_name", "artists"])
    for base, lab in [(d, "por track_id"), (canon, "por musica")]:
        g = base.assign(grp=pd.cut(base.n_gen, [0, 1, 3, 7, 9],
                                   labels=["1", "2-3", "4-7", "8-9"]))
        med = g.groupby("grp", observed=True).popularity.median()
        print(f"  {lab:<14} " + " | ".join(f"{k}: {v:.0f}" for k, v in med.items()))


if __name__ == "__main__":
    main()
