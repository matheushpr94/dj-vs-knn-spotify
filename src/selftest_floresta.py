"""Prova que `floresta.probabilidades_em_lotes` da' o mesmo que a floresta inteira.

Caso pequeno e rapido (20 generos, 24 arvores, lotes de 5, que nao divide 24 de
proposito). Compara probabilidade por probabilidade. Saida 0 se igual, 1 se nao.

    uv run python src/selftest_floresta.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from floresta import prever, probabilidades_em_lotes
from prepare import por_faixa

F = ["danceability", "energy", "loudness", "speechiness", "acousticness",
     "instrumentalness", "liveness", "valence", "tempo", "duration_ms"]


def main() -> int:
    d = por_faixa()
    d = d[d.track_genre.isin(sorted(d.track_genre.unique())[:20])]
    Xtr, Xte, ytr, yte = train_test_split(d[F], d.track_genre, test_size=.25,
                                          random_state=0, stratify=d.track_genre)
    params = dict(min_samples_leaf=2, n_jobs=-1, random_state=0)
    inteira = RandomForestClassifier(n_estimators=24, **params).fit(Xtr, ytr)
    p_inteira = inteira.predict_proba(Xte)
    p_lotes, classes = probabilidades_em_lotes(Xtr, ytr, Xte, n_arvores=24, lote=5, **params)

    dif = float(np.abs(p_inteira - p_lotes).max())
    prev_lotes, empates = prever(p_lotes, classes)
    prev_inteira, _ = prever(p_inteira, inteira.classes_)
    mesma_previsao = bool((prev_lotes == prev_inteira).all())
    print(f"maior diferenca de probabilidade : {dif:.2e}  (so' arredondamento de soma: < 1e-12)")
    print(f"empates exatos no topo           : {empates:,} de {len(Xte):,} faixas (decididos por ordem alfabetica)")
    print(f"mesma classe prevista em todas   : {mesma_previsao}")
    return 0 if dif < 1e-12 and mesma_previsao and list(classes) == list(inteira.classes_) else 1


if __name__ == "__main__":
    sys.exit(main())
