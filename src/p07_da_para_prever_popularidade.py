"""Quais correlacoes deste dataset sao armadilha, e por que."""
import numpy as np, pandas as pd
from prepare import carregar, por_faixa, por_musica
from cabecalho import cabecalho

CONT = ["danceability","energy","loudness","speechiness","acousticness",
        "instrumentalness","liveness","valence","tempo","duration_ms"]


def main() -> None:
    cabecalho(
        pergunta="Quais correlacoes deste dataset sao armadilha?",
        como="Matriz entre as continuas, mais o teste de quais pares sao definicionais (uma coluna e' insumo da outra).",
        ler_assim="Correlacao alta entre colunas que se definem uma pela outra nao e' achado sobre musica — e' o algoritmo se correlacionando consigo mesmo.",
        piso="O par mais forte NAO definicional e' o unico que diz algo sobre o mundo.",
        metricas=['pearson', 'spearman', 'r2', 'mae'],
    )
    d = por_faixa(carregar(remover_impossiveis=True))
    print(f"{len(d):,} faixas unicas\n")

    print("=== 1. matriz entre as continuas — os pares mais fortes ===")
    c = d[CONT].corr()
    iu = np.triu_indices_from(c.values, k=1)
    pares = sorted(zip(np.abs(c.values[iu]), c.values[iu],
                       c.index[iu[0]], c.columns[iu[1]]), reverse=True)[:8]
    for _, v, a, b in pares:
        print(f"  {v:+.3f}   {a} × {b}")

    print("\n=== 2. o par energy × loudness e' medida ou definicao? ===")
    r = d.energy.corr(d.loudness)
    print(f"  Pearson: {r:+.3f}")
    print("  O cartao define energy como medida PERCEPTUAL que combina faixa dinamica,")
    print("  volume percebido, timbre, taxa de ataque e entropia. Ou seja: loudness e'")
    print("  INSUMO de energy. Correlacionar os dois mede o algoritmo, nao a musica.")

    print("\n=== 3. instrumentalness: por que Pearson mente aqui ===")
    p = d.instrumentalness.corr(d.valence)
    s = d.instrumentalness.corr(d.valence, method="spearman")
    nz = d[d.instrumentalness > 0]
    pnz = nz.instrumentalness.corr(nz.valence)
    print(f"  com os zeros    : Pearson {p:+.3f} | Spearman {s:+.3f}")
    print(f"  sem os zeros    : Pearson {pnz:+.3f}")
    print(f"  massa em zero   : {(d.instrumentalness == 0).mean():.1%}")
    print("  34% num unico valor faz a nuvem virar duas nuvens. Pearson ajusta uma reta")
    print("  entre elas e reporta 'relacao' que e' so' a distancia entre os dois grupos.")

    print("\n=== 4. loudness mede musica ou masterizacao? ===")
    alb = d.album_name.fillna("")
    rem = d[alb.str.contains("remaster", case=False, regex=False)]
    resto = d[~alb.str.contains("remaster", case=False, regex=False)]
    print(f"  faixas 'remaster' : {len(rem):>6,}  loudness medio {rem.loudness.mean():+.2f} dB")
    print(f"  demais            : {len(resto):>6,}  loudness medio {resto.loudness.mean():+.2f} dB")
    print(f"  positivos (clipping): {(d.loudness > 0).sum()}")

    print("\n=== 5. A ARMADILHA POPULARITY ===")
    m = por_musica(carregar(remover_impossiveis=True))
    print(f"  base correta para popularidade (uma linha por musica viva): {len(m):,}")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    X, y = m[CONT], m.popularity
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=0)
    rf = RandomForestRegressor(n_estimators=180, min_samples_leaf=5,
                               n_jobs=-1, random_state=0).fit(Xtr, ytr)
    pred = rf.predict(Xte)
    print(f"  R2 das audio features           : {r2_score(yte, pred):.3f}")
    print(f"  MAE do modelo                   : {mean_absolute_error(yte, pred):.1f} pontos")
    print(f"  MAE de chutar sempre a mediana  : "
          f"{np.abs(yte - ytr.median()).mean():.1f} pontos")
    print(f"  desvio-padrao de popularity     : {y.std():.1f}")


if __name__ == "__main__":
    main()
