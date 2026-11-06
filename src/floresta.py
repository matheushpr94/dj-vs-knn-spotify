"""Floresta aleatoria com memoria limitada, e resultado igual ao da floresta inteira.

POR QUE ISTO EXISTE. A RandomForest de genero tem 114 classes e 220 arvores. Cada
arvore guarda, em cada no, a contagem das 114 classes: medido, ~30 MB por arvore,
~6,6 GB a floresta inteira. Num notebook de 16 GB roda; numa maquina de 8 GB o
sistema mata o processo no meio.

COMO RESOLVE. A previsao da floresta e' a MEDIA das probabilidades das arvores.
Media se calcula somando aos poucos: treina um lote de arvores, soma as
probabilidades que elas dao para o conjunto de teste, joga o lote fora, treina o
proximo. So' um lote fica na memoria por vez.

POR QUE O RESULTADO E' O MESMO. Com `warm_start=True`, o scikit-learn sorteia a
semente de cada arvore nova a partir do mesmo gerador, avancando-o pelas arvores
que ja' existem — e' o que ele faz para que treinar 110 + 110 de igual a treinar
220 de uma vez. As arvores descartadas viram marcadores vazios: ocupam a contagem
que o gerador precisa e nada da memoria. Conferido em
`selftest_floresta.py`: mesmas probabilidades que a floresta inteira.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class _Descartada:
    """Marca o lugar de uma arvore ja' somada. So' existe para ser contada."""


def probabilidades_em_lotes(X_treino, y_treino, X_teste, n_arvores: int, lote: int = 20,
                            **parametros) -> tuple[np.ndarray, np.ndarray]:
    """Devolve (probabilidades em X_teste, classes), iguais as de uma floresta inteira.

    `parametros` vai direto para RandomForestClassifier (random_state, min_samples_leaf...).
    """
    rf = RandomForestClassifier(n_estimators=1, warm_start=True, **parametros)
    soma = None
    feitas = 0
    while feitas < n_arvores:
        feitas = min(feitas + lote, n_arvores)
        rf.set_params(n_estimators=feitas)
        rf.fit(X_treino, y_treino)
        novas = [e for e in rf.estimators_ if not isinstance(e, _Descartada)]
        p = sum(e.predict_proba(np.asarray(X_teste, dtype=np.float32)) for e in novas)
        soma = p if soma is None else soma + p
        rf.estimators_ = [_Descartada() for _ in rf.estimators_]
    return soma / n_arvores, rf.classes_


def prever(prob: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, int]:
    """Classe de maior probabilidade, com EMPATE decidido de forma fixa.

    Com folhas pequenas, a probabilidade de uma floresta e' uma fracao de poucos
    denominadores, e EMPATE EXATO no topo e' comum. O scikit-learn soma as arvores
    em threads, em ordem que varia; a sobra de arredondamento (~1e-16) decidia o
    empate ao acaso. Aqui o arredondamento a 12 casas apaga essa sobra e o empate
    vai para a primeira classe em ordem alfabetica, sempre. Devolve tambem
    quantas linhas tinham empate, para o numero aparecer na saida.
    """
    p = np.round(prob, 12)
    topo = p.max(axis=1, keepdims=True)
    empates = int(((p == topo).sum(axis=1) > 1).sum())
    return classes[p.argmax(axis=1)], empates
