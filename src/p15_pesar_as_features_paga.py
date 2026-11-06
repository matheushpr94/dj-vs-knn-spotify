"""Pesar as 9 colunas melhora o escore de pertencimento, ou 'igual para todas' ja' basta?

Hoje o escore trata as 9 features com o mesmo peso. E' improvavel que `liveness`
importe tanto quanto `energy` para dizer se uma faixa cabe num grupo. Este script
procura pesos melhores — e mede honestamente se valeu.

O CUIDADO QUE MUDA O RESULTADO: procurar os pesos e reportar o acerto no MESMO
dado infla o numero. Qualquer busca acha algo que funciona no dado que ela viu.
Por isso os generos sao partidos em duas metades:

  BUSCA     onde os pesos sao procurados
  RESERVA   onde o resultado e' reportado, e que a busca nunca viu

Se o ganho aparecer na busca e sumir na reserva, os pesos decoraram ruido — e
saber disso e' o proposito da divisao.

TRES CANDIDATOS sao comparados contra o peso igual:
  ablacao       tira uma coluna por vez, para ver quais atrapalham
  importancia   pesos vindos do RandomForest de genero
  sorteio       200 vetores de peso aleatorios, fica o melhor na busca
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from bancada import nota, rodada
from botoes import ajuda_se_pedida, aviso_se_mudou, botao
from cabecalho import cabecalho
from p14_esta_faixa_cabe_na_playlist import F, MIN_FAIXAS, PISO_LARGURA
from prepare import carregar, por_faixa

SEMENTE = botao("--semente", 42, int,
                "sorteio E a divisao busca/reserva (teste de robustez)")
SORTEIOS = botao("--sorteios", 200, int, "quantas tentativas a busca faz")
MAX_POR_GRUPO = botao("--max-por-grupo", 120, int, "teto de faixas por genero")

# EXPERIMENTO QUE ENSINA MAIS: rode com 20, 200 e 2000 sorteios. O ganho na BUSCA
# cresce com as tentativas (achar sorte fica mais facil); o da RESERVA, nao — e a
# diferenca entre os dois E' o problema dos caminhos bifurcados, medido a mao.


def escores_do_grupo(X, catalogo, rng, pesos):
    """Escore de cada faixa do grupo, e de uma faixa sorteada, para os MESMOS pesos.

    Deixa a propria faixa de fora do centro e da largura — vetorizado. Para um
    grupo de n faixas, a media sem a i-esima e' (soma - x_i)/(n-1), e a variancia
    sai da mesma identidade. Fazer isso em laco seria correto e lento demais para
    uma varredura de 200 candidatos.
    """
    n, k = X.shape
    if n <= 3:
        return None, None
    soma, somaq = X.sum(0), (X ** 2).sum(0)
    m = (soma - X) / (n - 1)                              # (n, k) medias sem a i
    v = (somaq - X ** 2) / (n - 1) - m ** 2
    s = np.maximum(np.sqrt(np.maximum(v, 0)), PISO_LARGURA)

    w = pesos / pesos.sum()
    dentro = (np.abs(X - m) / s * w).sum(1)
    sorteadas = catalogo[rng.integers(len(catalogo), size=n)]
    fora = (np.abs(sorteadas - m) / s * w).sum(1)
    return dentro, fora


def acerto(grupos, catalogo, pesos, semente=SEMENTE):
    """Fracao de vezes em que quem pertence pontuou melhor que quem foi sorteado."""
    rng = np.random.default_rng(semente)
    d, f = [], []
    for X in grupos:
        a, b = escores_do_grupo(X, catalogo, rng, pesos)
        if a is None:
            continue
        d.append(a)
        f.append(b)
    if not d:
        return 0.0
    return float((np.concatenate(d) < np.concatenate(f)).mean())


def main() -> int:
    cabecalho(
        pergunta="Pesar as 9 colunas melhora o escore de pertencimento, ou tratar "
                 "todas igual ja' basta?",
        como="Parte os generos em BUSCA e RESERVA. Procura pesos so' na busca por "
             "tres caminhos (ablacao, importancia do RandomForest, 200 sorteios) e "
             "reporta o acerto na reserva, que a busca nunca viu.",
        ler_assim="O numero que vale e' o da RESERVA. Ganho que aparece na busca e "
                  "some na reserva significa que os pesos decoraram ruido.",
        piso="Peso igual para as 9 colunas — o escore que ja' esta' publicado, com "
             "77,5% de acerto.",
        metricas=["acuracia", "baseline"],
    )
    nota("varredura de pesos do escore de pertencimento",
         sorteios=SORTEIOS, semente=SEMENTE)
    aviso_se_mudou()

    d = por_faixa(carregar()).dropna(subset=F + ["track_genre"])
    Z = ((d[F] - d[F].mean()) / d[F].std())
    dz = d.assign(**{c: Z[c] for c in F})
    catalogo = dz[F].to_numpy()

    tam = dz.groupby("track_genre").size()
    generos = sorted(tam[tam >= MIN_FAIXAS].index.tolist())
    rng = np.random.default_rng(SEMENTE)
    ordem = rng.permutation(len(generos))
    meio = len(generos) // 2
    g_busca = [generos[i] for i in sorted(ordem[:meio])]
    g_reserva = [generos[i] for i in sorted(ordem[meio:])]

    def matrizes(nomes):
        out = []
        for g in nomes:
            sub = dz[dz.track_genre == g]
            if len(sub) > MAX_POR_GRUPO:
                sub = sub.sample(MAX_POR_GRUPO, random_state=SEMENTE)
            out.append(sub[F].to_numpy())
        return out

    B, R = matrizes(g_busca), matrizes(g_reserva)
    igual = np.ones(len(F))

    with rodada("o piso: peso igual para as 9 colunas",
                {"generos_busca": len(B), "generos_reserva": len(R)}) as r:
        pb, pr = acerto(B, catalogo, igual), acerto(R, catalogo, igual)
        r.resultado(acerto_busca=round(pb, 4), acerto_reserva=round(pr, 4))
        r.leitura(f"peso igual acerta {pr:.1%} na reserva. E' contra este numero que "
                  f"todo candidato tem de ser comparado.")
    base_reserva = pr

    # --- candidato 1: ablacao, uma coluna de fora por vez ---
    with rodada("ablacao: alguma coluna esta' ATRAPALHANDO?",
                {"colunas": len(F)}) as r:
        # A ablacao TAMBEM e' uma busca — nove tentativas. Medir so' na metade de
        # busca e reportar o ranking seria o mesmo erro que a divisao existe para
        # evitar. As duas colunas ficam lado a lado, e o que vale e' a RESERVA.
        linhas, res = {}, {}
        for i, col in enumerate(F):
            w = igual.copy()
            w[i] = 0.0
            linhas[col] = round(acerto(B, catalogo, w), 4)
            res[col] = round(acerto(R, catalogo, w), 4)
        r.resultado(**{f"busca_{k}": v for k, v in linhas.items()},
                    **{f"reserva_{k}": v for k, v in res.items()})
        ordem_b = sorted(linhas, key=linhas.get)
        ordem_r = sorted(res, key=res.get)
        concorda = sum(1 for i, c in enumerate(ordem_b) if ordem_r[i] == c)
        r.leitura(f"o ranking da busca e o da reserva concordam em {concorda} de "
                  f"{len(F)} posicoes. Se concordarem pouco, o ranking e' ruido.")
        print(f"\n  {'tirando a coluna':<20} {'busca':>9} {'vs igual':>10} "
              f"{'reserva':>9} {'vs igual':>10}")
        print("  " + "-" * 62)
        for col in sorted(F, key=lambda c: -res[c]):
            print(f"  {col:<20} {linhas[col]:>9.1%} {linhas[col]-pb:>+10.1%} "
                  f"{res[col]:>9.1%} {res[col]-pr:>+10.1%}")
        print(f"\n  ranking igual em {concorda} de {len(F)} posicoes")

    # --- candidato 2: importancia do RandomForest de genero ---
    with rodada("pesos vindos da importancia do RandomForest", {"arvores": 60}) as r:
        sub = dz.sample(min(24_000, len(dz)), random_state=SEMENTE)
        m = RandomForestClassifier(n_estimators=60, n_jobs=-1, random_state=SEMENTE)
        m.fit(sub[F], sub.track_genre)
        w_imp = m.feature_importances_.copy()
        w_imp = w_imp / w_imp.sum() * len(F)
        ab, ar = acerto(B, catalogo, w_imp), acerto(R, catalogo, w_imp)
        r.resultado(acerto_busca=round(ab, 4), acerto_reserva=round(ar, 4),
                    ganho_na_reserva=round(ar - base_reserva, 4),
                    **{c: round(float(x), 3) for c, x in zip(F, w_imp)})
        r.leitura(f"na reserva: {ar:.1%} contra {base_reserva:.1%} do peso igual "
                  f"({ar - base_reserva:+.1%}).")

    # --- candidato 3: sorteio de vetores de peso ---
    with rodada(f"{SORTEIOS} vetores de peso sorteados, melhor da busca",
                {"sorteios": SORTEIOS}) as r:
        rs = np.random.default_rng(SEMENTE)
        melhor_w, melhor_a = igual, pb
        for _ in range(SORTEIOS):
            w = rs.dirichlet(np.ones(len(F))) * len(F)
            a = acerto(B, catalogo, w)
            if a > melhor_a:
                melhor_a, melhor_w = a, w
        ar = acerto(R, catalogo, melhor_w)
        r.resultado(melhor_na_busca=round(melhor_a, 4), acerto_reserva=round(ar, 4),
                    ganho_na_reserva=round(ar - base_reserva, 4),
                    **{c: round(float(x), 3) for c, x in zip(F, melhor_w)})
        r.leitura(f"o melhor da busca tira {melhor_a:.1%} la', e {ar:.1%} na reserva "
                  f"({ar - base_reserva:+.1%} contra o peso igual). A queda entre busca "
                  f"e reserva e' o quanto ele decorou.")
        w_sort, a_sort = melhor_w, ar

    a_imp = acerto(R, catalogo, w_imp)

    print("\n" + "=" * 74)
    print(f"{'candidato':<36} {'busca':>10} {'reserva':>10} {'ganho real':>12}")
    print("-" * 74)
    print(f"{'peso igual (o que esta publicado)':<36} {pb:>10.1%} "
          f"{base_reserva:>10.1%} {'—':>12}")
    print(f"{'importancia do RandomForest':<36} {ab:>10.1%} {a_imp:>10.1%} "
          f"{a_imp - base_reserva:>+12.1%}")
    print(f"{'melhor de ' + str(SORTEIOS) + ' sorteios':<36} {melhor_a:>10.1%} "
          f"{a_sort:>10.1%} {a_sort - base_reserva:>+12.1%}")
    print()
    ganho = max(a_imp, a_sort) - base_reserva
    if ganho < .01:
        print("PESAR NAO PAGA. Nenhum candidato ganha mais que 1 ponto na reserva.")
        print("O escore fica como esta': peso igual para as 9 colunas. Isso e'")
        print("resultado — evita a proxima pessoa refazer a mesma varredura.")
    else:
        print(f"PESAR PAGA: {ganho:+.1%} na reserva. Vale trocar no escore publicado.")
    return 0


if __name__ == "__main__":
    ajuda_se_pedida("p15_pesar_as_features_paga.py — pesar as 9 colunas melhora o escore?")
    sys.exit(main())
