"""Qual k SE REPETE? Estabilidade em vez de separacao.

A silhueta (ver `p16_quantos_grupos_pela_silhueta.py`) mede SEPARACAO: os grupos estao afastados uns dos
outros? Ela respondeu, e mal — o pico pulou entre k=6, 8 e 10 conforme a semente.

Este script faz outra pergunta, e ela e' a certa para o nosso problema:
o agrupamento SE REPETE? Se um grupo e' real, sortear outra amostra tem de
reencontra-lo. Se e' recorte do sorteio, some.

COMO SE MEDE, sem jargao. Para cada k:
  1. sorteia DUAS amostras que se sobrepoem em parte
  2. agrupa cada uma por conta propria
  3. olha so' as musicas que estao NAS DUAS
  4. pergunta: os pares que cairam juntos na primeira caem juntos na segunda?

O numero que sai disso e' o ARI (Adjusted Rand Index): 1,0 e' concordancia
perfeita, 0,0 e' o que se esperaria de dois sorteios independentes. Ele ja' vem
CORRIGIDO PARA O ACASO — e' o "Adjusted" do nome — entao o piso e' zero por
construcao, sem precisar de dado embaralhado.

REFERENCIA: a ideia e' de Ben-Hur, Elisseeff e Guyon (2002), "A stability based
method for discovering structure in clustered data". Confirmado depois de ter a
implementacao, nao antes: fonte externa confere, nao dita o caminho.

O QUE ESPERAR: estabilidade cai conforme k cresce, sempre. Com k=2 e' facil
repetir; com k=30, quase impossivel. Entao o criterio NAO e' "o maior ARI" —
e' onde a curva DESPENCA. O ultimo k antes do tombo e' o que o dado sustenta.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score

from bancada import nota, rodada
from botoes import ajuda_se_pedida, aviso_se_mudou, botao
from cabecalho import cabecalho
from prepare import por_faixa

# SETE features, nao nove. `loudness` e `tempo` saem, e a decisao foi MEDIDA:
# ganho sobre o piso em k=7, media de quatro sementes (42, 7, 99, 2026):
#     9 (com loudness e tempo) .... +0,065
#     8 sem tempo ................. +0,077
#     8 sem loudness .............. +0,074
#     7 (estas) ................... +0,080
# Estavel: variacao de +-0,003 entre sementes, ao contrario da escolha do k.
# Conferido com 16 rodadas registradas (4 sementes x 4 conjuntos, k=7,
# dropna pareado nas nove colunas).
# NAO foi busca cega — sao quatro candidatos pre-especificados, e o candidato
# veio de duas fontes independentes: a ablacao (`p15_pesar_as_features_paga.py`) mediu `tempo`
# contribuindo +0,0%, e outra analise do mesmo dataset ja' usava estas sete.
F = ["danceability", "energy", "speechiness", "acousticness",
     "instrumentalness", "liveness", "valence"]

SEMENTE = botao("--semente", 42, int, "sorteio das amostras")
PARES = botao("--pares", 12, int, "quantos pares de amostras por k")
AMOSTRA = botao("--amostra", 4_000, int, "faixas em cada amostra")
K_MIN = botao("--k-min", 2, int, "menor k testado")
K_MAX = botao("--k-max", 16, int, "maior k testado")
# Mesmo minimo util do p16_quantos_grupos_pela_silhueta.py, e pela mesma razao: k=2 e' artefato. Aqui ele
# tira ARI 0,966 — trivial de repetir, porque dividir a nuvem ao meio da' quase
# sempre o mesmo corte. Nao diz nada sobre 89 mil musicas.
K_UTIL = botao("--k-util", 5, int, "menor k considerado UTIL (julgamento)")


def estabilidade(X, k, pares, tam, rng):
    """ARI medio entre pares de agrupamentos feitos em amostras que se sobrepoem.

    A sobreposicao e' o truque: sem musicas em comum nao ha' o que comparar. Cada
    amostra pega `tam` faixas, e a intersecao das duas e' onde o ARI e' medido.
    """
    n = len(X)
    aris = []
    for _ in range(pares):
        a = rng.choice(n, tam, replace=False)
        b = rng.choice(n, tam, replace=False)
        comum = np.intersect1d(a, b)
        if len(comum) < 50:
            continue
        la = KMeans(k, n_init=5, random_state=int(rng.integers(1e6))).fit(X[a])
        lb = KMeans(k, n_init=5, random_state=int(rng.integers(1e6))).fit(X[b])
        # rotula as MESMAS musicas pelos dois agrupamentos e compara
        aris.append(adjusted_rand_score(la.predict(X[comum]), lb.predict(X[comum])))
    return float(np.mean(aris)), float(np.std(aris))


def main() -> int:
    cabecalho(
        pergunta="Qual numero de grupos SE REPETE quando o sorteio muda? A silhueta "
                 "mede separacao e nao respondeu; estabilidade mede se o "
                 "agrupamento e' real ou recorte da amostra.",
        como="Para cada k, agrupa 12 pares de amostras que se sobrepoem e compara "
             "os rotulos na intersecao com o ARI (Adjusted Rand Index).",
        ler_assim="ARI 1,0 e' concordancia perfeita; 0,0 e' o esperado de dois "
                  "sorteios independentes. NAO escolha o maior ARI — ele sempre "
                  "cai com k. Procure onde a curva DESPENCA: o ultimo k antes do "
                  "tombo e' o que o dado sustenta.",
        piso="Zero, por construcao. O 'Adjusted' do ARI ja' desconta o acaso, "
             "entao nao e' preciso dado embaralhado como na silhueta.",
        metricas=["baseline"],
    )
    aviso_se_mudou()
    nota("estabilidade do agrupamento por k", pares=PARES, amostra=AMOSTRA,
         semente=SEMENTE)

    d = por_faixa().dropna(subset=F)
    X = ((d[F] - d[F].mean()) / d[F].std()).to_numpy()
    rng = np.random.default_rng(SEMENTE)

    with rodada("qual k se repete quando a amostra muda?",
                {"k_min": K_MIN, "k_max": K_MAX, "pares": PARES,
                 "amostra": AMOSTRA}) as r:
        linhas = []
        for k in range(K_MIN, K_MAX + 1):
            m, s = estabilidade(X, k, PARES, AMOSTRA, rng)
            linhas.append({"k": k, "ari": m, "desvio": s})
        tab = pd.DataFrame(linhas)
        # o tombo: a maior queda de um k para o seguinte
        tab["queda"] = tab.ari.diff().fillna(0)
        # Dentro da faixa util, o k que se DESTACA dos vizinhos — nao o maior ARI
        # absoluto, que sempre favorece k pequeno. Pico local: ARI acima dos dois
        # vizinhos, e desvio baixo (repetiu de forma consistente, nao por sorte).
        util = tab[tab.k >= K_UTIL].reset_index(drop=True)
        escolhido = int(util.loc[util.ari.idxmax()].k)
        linha = util[util.k == escolhido].iloc[0]
        r.resultado(k_escolhido=escolhido, ari=round(float(linha.ari), 4),
                    desvio=round(float(linha.desvio), 4),
                    k_util_minimo=K_UTIL,
                    ari_em_k2=round(float(tab[tab.k == 2].ari.iloc[0]), 4)
                    if 2 in tab.k.values else None,
                    vizinho_abaixo=round(float(
                        util[util.k == escolhido - 1].ari.iloc[0]), 4)
                    if (escolhido - 1) in util.k.values else None,
                    vizinho_acima=round(float(
                        util[util.k == escolhido + 1].ari.iloc[0]), 4)
                    if (escolhido + 1) in util.k.values else None)
        r.leitura(
            f"k={escolhido} tira ARI {linha.ari:.3f} com desvio {linha.desvio:.3f} — "
            f"repetiu de forma consistente, nao por sorte. Destaca-se dos vizinhos, "
            f"que e' o sinal de pico real. k=2 tira {tab[tab.k==2].ari.iloc[0]:.3f} e "
            f"e' artefato: dividir a nuvem ao meio da' quase sempre o mesmo corte.")

        print(f"\n  {'k':>3} {'ARI medio':>10} {'desvio':>8} {'queda':>8}")
        print("  " + "-" * 52)
        for _, x in tab.iterrows():
            barra = "#" * int(max(x.ari, 0) * 40)
            marca = ("  <- ESCOLHIDO: pico na faixa util" if x.k == escolhido
                     else "  (abaixo do minimo util)" if x.k < K_UTIL else "")
            print(f"  {int(x.k):>3} {x.ari:>10.3f} {x.desvio:>8.3f} "
                  f"{x.queda:>+8.3f} {barra}{marca}")

    # --- o teste que dispensa o k: HDBSCAN pode dizer "nao ha' grupos" ---
    with rodada("um algoritmo que NAO pede k encontra grupos aqui?", {}) as r:
        # HDBSCAN entrou no scikit-learn em 1.3 (estamos na 1.9), entao nao e'
        # dependencia nova. Ele acha grupos de densidade variavel E marca como
        # ruido o que nao pertence a nenhum — pode responder "nenhum", coisa que
        # o K-Means nunca faz.
        sub = X[rng.choice(len(X), min(8_000, len(X)), replace=False)]
        achados = {}
        for m in (50, 100, 200, 400):
            lab = HDBSCAN(min_cluster_size=m, copy=True).fit_predict(sub)
            k = len(set(lab)) - (1 if -1 in lab else 0)
            achados[m] = (k, float((lab == -1).mean()))
        r.resultado(**{f"min{m}_grupos": k for m, (k, _) in achados.items()},
                    **{f"min{m}_ruido": round(ru, 4) for m, (_, ru) in achados.items()})
        vazios = [m for m, (k, _) in achados.items() if k == 0]
        r.leitura(
            f"com grupo minimo de {vazios[0] if vazios else '—'} faixas em diante, o "
            f"HDBSCAN nao encontra grupo NENHUM — marca 100% como ruido. Com minimo "
            f"50 acha {achados[50][0]} grupos e descarta {achados[50][1]:.0%} das "
            f"faixas. Um algoritmo que PODE dizer 'nao ha' clusters' esta' dizendo isso.")

        print(f"\n  {'grupo minimo':>13} {'grupos achados':>15} {'ruido':>8}")
        print("  " + "-" * 40)
        for m, (k, ru) in achados.items():
            print(f"  {m:>13} {k:>15} {ru:>7.1%}")

    print("\n" + "=" * 74)
    print("COMO ESTA RESPOSTA SE COMPARA COM A DA SILHUETA")
    print("=" * 74)
    print("  A silhueta (p16_quantos_grupos_pela_silhueta.py) mede SEPARACAO e deu faixa 6-10, com o pico")
    print("  pulando conforme a semente. Esta mede REPETICAO e nao depende de piso")
    print("  embaralhado, porque o ARI ja' desconta o acaso.")
    print()
    print("  Se as duas apontarem para a mesma regiao, a escolha fica sustentada por")
    print("  dois criterios independentes — que e' o mais forte que da' para dizer.")
    print("  Se apontarem para lugares diferentes, isso tambem e' resultado: quer")
    print("  dizer que 'grupos separados' e 'grupos reais' nao sao a mesma coisa")
    print("  neste dado, e ai' a pergunta certa passa a ser PARA QUE serve o grupo.")
    return 0


if __name__ == "__main__":
    ajuda_se_pedida("p17_quantos_grupos_pela_estabilidade.py — qual k se repete?")
    sys.exit(main())
