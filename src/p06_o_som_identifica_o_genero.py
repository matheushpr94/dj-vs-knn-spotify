"""As audio features realmente identificam genero? E faixa de popularidade?"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from floresta import prever, probabilidades_em_lotes
from prepare import por_faixa
from cabecalho import cabecalho

F = ["danceability", "energy", "loudness", "speechiness", "acousticness",
     "instrumentalness", "liveness", "valence", "tempo", "duration_ms"]


def main() -> None:
    cabecalho(
        pergunta="O som identifica o genero? E identifica a faixa de popularidade?",
        como="Random Forest com treino e teste separados 70/30 estratificado, mais arvore rasa para ler a regra em portugues.",
        ler_assim="Acerto alto significa assinatura acustica. Onde o modelo erra e' mais informativo que onde acerta. "
                  "ATENCAO na parte 2: este script entra por `por_faixa`, que e' a visao certa para audio feature e a ERRADA "
                  "para popularidade (ver prepare.py). O numero canonico de popularidade e' o de 16-recomputo.txt, secao 3.",
        piso="Chute entre 114 generos = 0,88%. Entre 4 faixas de popularidade = 25%.",
        metricas=['acaso', 'topk', 'confianca'],
    )
    d = por_faixa()

    # ---------- 1. features -> genero ----------
    print("=== 1. audio features conseguem dizer o GENERO? ===")
    X, y = d[F], d.track_genre
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=0, stratify=y)
    # Floresta treinada em lotes: 114 classes x 220 arvores nao cabem em 8 GB de uma
    # vez. Mesmo resultado da floresta inteira (ver floresta.py e selftest_floresta.py).
    pr, classes = probabilidades_em_lotes(Xtr, ytr, Xte, n_arvores=220, lote=20,
                                          min_samples_leaf=2, n_jobs=-1, random_state=0)
    previsto, empates = prever(pr, classes)
    top1 = accuracy_score(yte, previsto)
    top5 = top_k_accuracy_score(yte, pr, k=5, labels=classes)
    chance = 1 / d.track_genre.nunique()
    print(f"  top-1 : {top1:.1%}   (acaso {chance:.2%}  ->  {top1/chance:.0f}x melhor)")
    print(f"  top-5 : {top5:.1%}   (acaso {5*chance:.2%})")
    print(f"  empate exato no topo em {empates:,} de {len(yte):,} faixas — decidido por ordem "
          f"alfabetica, ver floresta.prever")

    res = pd.DataFrame({"y": yte.values, "p": previsto})
    rec = res.assign(hit=res.y == res.p).groupby("y").hit.mean()
    print("\n  generos MAIS identificaveis pelo som:")
    for g, v in rec.sort_values(ascending=False).head(8).items():
        print(f"    {g:<18} {v:.0%}")
    print("\n  generos MENOS identificaveis (o som nao distingue):")
    for g, v in rec.sort_values().head(8).items():
        print(f"    {g:<18} {v:.0%}")

    # ---------- 2. features -> faixa de popularidade ----------
    print("\n=== 2. as mesmas features conseguem dizer a POPULARIDADE? ===")
    nz = d[d.popularity > 0].copy()
    nz["faixa"] = pd.qcut(nz.popularity, 4, labels=["baixa", "media-baixa", "media-alta", "alta"])
    Xa, ya = nz[F], nz.faixa
    Xtr2, Xte2, ytr2, yte2 = train_test_split(Xa, ya, test_size=.25, random_state=0, stratify=ya)
    rf2 = RandomForestClassifier(n_estimators=220, min_samples_leaf=5,
                                 n_jobs=-1, random_state=0).fit(Xtr2, ytr2)
    a2 = accuracy_score(yte2, rf2.predict(Xte2))
    print(f"  acerto: {a2:.1%}   (acaso 25,0%  ->  {a2/.25:.2f}x)")

    # com o genero junto, quanto melhora?
    Xg = pd.get_dummies(nz.track_genre, prefix="g")
    Xb = pd.concat([nz[F].reset_index(drop=True), Xg.reset_index(drop=True)], axis=1)
    Xtr3, Xte3, ytr3, yte3 = train_test_split(Xb, ya, test_size=.25, random_state=0, stratify=ya)
    rf3 = RandomForestClassifier(n_estimators=220, min_samples_leaf=5,
                                 n_jobs=-1, random_state=0).fit(Xtr3, ytr3)
    a3 = accuracy_score(yte3, rf3.predict(Xte3))
    print(f"  somando o genero como feature: {a3:.1%}  (ganho de {a3-a2:+.1%})")

    # ---------- 3. regras legiveis ----------
    print("\n=== 3. regras concretas (arvore rasa) para 6 generos bem distintos ===")
    sub = d[d.track_genre.isin(["classical", "death-metal", "reggaeton",
                                "sleep", "comedy", "hip-hop"])]
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=60,
                                random_state=0).fit(sub[F], sub.track_genre)
    print(f"  acerto dentro desses 6: {dt.score(sub[F], sub.track_genre):.1%} "
          f"(acaso 16,7%)")
    print(export_text(dt, feature_names=F, max_depth=3))


if __name__ == "__main__":
    main()
