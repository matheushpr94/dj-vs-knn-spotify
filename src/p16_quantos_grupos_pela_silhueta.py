"""Quantos grupos naturais existem nessas musicas, e eles fazem sentido?

Tres perguntas encadeadas, e cada uma so' vale se a
anterior tiver resposta:

  1. Qual o numero de grupos que o dado sustenta? (varredura + silhueta)
  2. O que caracteriza cada grupo? (quais features estao altas e baixas)
  3. As musicas dentro de um grupo sao mesmo parecidas? (olhar com o ouvido)

COMO O K-MEANS FUNCIONA, sem jargao. Voce escolhe QUANTOS grupos quer — ele nao
descobre isso. Ele joga k centros ao acaso, manda cada musica para o centro mais
proximo, move cada centro para a media de quem o escolheu, e repete ate' parar.

A ARMADILHA CENTRAL: o K-Means SEMPRE devolve grupos. Da' ruido puro para ele e
ele devolve k grupos bonitos. Ele nao sabe dizer "nao ha' estrutura aqui". Por
isso todo numero aqui vem com o PISO ao lado — a mesma conta sobre o dado com as
colunas embaralhadas, que preserva as distribuicoes e destroi a correlacao.

POR QUE NAO ESCOLHER O k DE MAIOR SILHUETA: a silhueta cresce sozinha quando k
diminui. Com k=2 quase sempre "ganha", e dois grupos nao dizem nada sobre 89 mil
musicas. O criterio honesto e' a DISTANCIA AO PISO: quanto aquele k supera o
mesmo k no dado embaralhado.

CUSTO: a silhueta compara cada ponto com todos os outros — e' O(n^2). Com 89.741
faixas isso nao cabe em memoria. Amostramos, e o tamanho da amostra e' declarado.

FEATURES: sete, nao nove. Ver o comentario acima da lista `F` — a escolha foi
medida em quatro sementes, e `loudness` e `tempo` saem.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
F9 = ["danceability", "energy", "loudness", "speechiness", "acousticness",
      "instrumentalness", "liveness", "valence", "tempo"]
CONJUNTOS = {
    "7":              [c for c in F9 if c not in ("loudness", "tempo")],
    # Um conjunto curto, de 4 colunas, usado em outra analise do mesmo dataset.
    # Entra aqui para ser medido pelo NOSSO criterio — ganho sobre o piso — em vez
    # de comparar silhuetas cruas, que nao sao comparaveis entre conjuntos diferentes.
    "4-curto":       ["energy", "acousticness", "valence", "instrumentalness"],
    "8-sem-tempo":    [c for c in F9 if c != "tempo"],
    "8-sem-loudness": [c for c in F9 if c != "loudness"],
    "9":              F9,
}
FEATURES = botao("--features", "7", str,
                 "conjunto de colunas: 7 (oficial) | 9 | 8-sem-tempo | 8-sem-loudness | 4-curto")
if FEATURES not in CONJUNTOS:
    raise SystemExit(f"erro: --features espera uma de {sorted(CONJUNTOS)}, veio '{FEATURES}'")
F = CONJUNTOS[FEATURES]

SEMENTE = botao("--semente", 42, int, "sorteio da amostra e do K-Means")
AMOSTRA = botao("--amostra", 12_000, int, "faixas na amostra (silhueta e' O(n^2))")
K_MIN = botao("--k-min", 2, int, "menor numero de grupos testado")
K_MAX = botao("--k-max", 30, int, "maior numero de grupos testado")
MOSTRAR = botao("--mostrar", 6, int, "quantas musicas listar por grupo")
# Piso de k UTIL, e ele e' JULGAMENTO DECLARADO, nao descoberta.
# Medido: k=2 vence os dois criterios (silhueta 0,264, ganho +0,090) e e' artefato
# — com dois grupos voce corta a nuvem ao meio e tudo fica longe da outra metade.
# Dois grupos nao dizem nada sobre 89 mil musicas. O minimo util fica declarado
# aqui, e mexer nele muda a resposta: e' o parametro mais subjetivo do script.
K_UTIL = botao("--k-util", 5, int, "menor k considerado UTIL (julgamento, ver comentario)")


def silhueta(X, k, semente):
    """Silhueta media para k grupos. Devolve tambem os rotulos, para reuso."""
    lab = KMeans(n_clusters=k, n_init=10, random_state=semente).fit_predict(X)
    return float(silhouette_score(X, lab, random_state=semente)), lab


def embaralhar(X, rng):
    """Embaralha CADA COLUNA em separado.

    Isso preserva a distribuicao de cada feature e destroi a relacao entre elas —
    e' o piso certo. Sortear numeros novos mudaria a distribuicao tambem, e ai' a
    comparacao mediria duas coisas ao mesmo tempo.
    """
    Y = X.copy()
    for j in range(Y.shape[1]):
        rng.shuffle(Y[:, j])
    return Y


def nomear(centro, colunas, n=2):
    """Nome legivel a partir das features mais extremas daquele centro.

    O centro esta' em unidades de desvio-padrao (o dado foi padronizado), entao
    +1,2 quer dizer "1,2 desvios acima da media do catalogo".
    """
    ordem = np.argsort(centro)
    baixas = [colunas[i] for i in ordem[:n] if centro[i] < -0.3]
    altas = [colunas[i] for i in ordem[::-1][:n] if centro[i] > 0.3]
    partes = []
    if altas:
        partes.append("+" + " +".join(altas))
    if baixas:
        partes.append("-" + " -".join(baixas))
    return " ".join(partes) if partes else "sem traco marcante"


def main() -> int:
    cabecalho(
        pergunta="Quantos grupos naturais o dado sustenta, e as musicas dentro de "
                 "cada um sao mesmo parecidas?",
        como="Varre k de {} a {}, calcula a silhueta de cada um e a do MESMO k "
             "sobre o dado com as colunas embaralhadas. Escolhe pelo maior ganho "
             "sobre o piso, nao pela maior silhueta.".format(K_MIN, K_MAX),
        ler_assim="Silhueta vai de -1 a 1. Acima de 0,5 e' estrutura clara; abaixo "
                  "de 0,2 e' o algoritmo fatiando uma nuvem continua. O que decide "
                  "nao e' o valor e sim a DISTANCIA ATE' O PISO.",
        piso="O mesmo k sobre o dado com cada coluna embaralhada em separado — "
             "mesmas distribuicoes, correlacao destruida. K-Means devolve grupos "
             "ate' em ruido, entao sem este piso nenhum numero significa nada.",
        metricas=["silhueta", "baseline"],
    )
    aviso_se_mudou()
    nota("varredura de numero de clusters", k_min=K_MIN, k_max=K_MAX,
         amostra=AMOSTRA, semente=SEMENTE, features=FEATURES)

    # dropna sobre as NOVE sempre, mesmo com F menor: se cada conjunto filtrasse
    # pela propria lista, o universo mudaria entre eles e a comparacao --features
    # deixaria de ser pareada.
    d = por_faixa().dropna(subset=F9 + ["track_name", "artists", "track_genre"])
    Z = ((d[F] - d[F].mean()) / d[F].std())
    rng = np.random.default_rng(SEMENTE)
    idx = rng.choice(len(d), min(AMOSTRA, len(d)), replace=False)
    amostra = d.iloc[idx].reset_index(drop=True)
    X = Z.iloc[idx].to_numpy()
    Xr = embaralhar(X, rng)

    ks = list(range(K_MIN, K_MAX + 1))
    with rodada("qual k o dado sustenta?",
                {"k_min": K_MIN, "k_max": K_MAX, "amostra": len(X),
                 "semente": SEMENTE, "features": FEATURES}) as r:
        linhas = []
        for k in ks:
            s_real, _ = silhueta(X, k, SEMENTE)
            s_piso, _ = silhueta(Xr, k, SEMENTE)
            linhas.append({"k": k, "real": s_real, "piso": s_piso,
                           "ganho": s_real - s_piso})
        tab = pd.DataFrame(linhas)
        bruto = tab.loc[tab.ganho.idxmax()]
        # dentro da faixa util, empate desempata pelo MENOR k — dois agrupamentos
        # que explicam igual, o mais simples ganha
        util = tab[tab.k >= K_UTIL]
        melhor = util[util.ganho == util.ganho.max()].iloc[0]
        # A FAIXA importa mais que o pico. Medido com 4 sementes, o k vencedor foi
        # 10, 8, 6 e 8 — tres valores diferentes com ganhos entre +0,068 e +0,080.
        # Anunciar um k exato seria publicar a semente. A faixa plausivel e' todo k
        # cujo ganho fica a menos de 15% do maximo: dentro do ruido do sorteio.
        limiar = util.ganho.max() * 0.85
        faixa = util[util.ganho >= limiar].k.tolist()
        r.resultado(k_escolhido=int(melhor.k), silhueta=round(float(melhor.real), 4),
                    piso=round(float(melhor.piso), 4),
                    ganho=round(float(melhor.ganho), 4),
                    k_util_minimo=K_UTIL,
                    k_de_maior_ganho_bruto=int(bruto.k),
                    k_de_maior_silhueta=int(tab.loc[tab.real.idxmax()].k))
        r.resultado(faixa_plausivel=str(faixa))
        r.leitura(
            f"k={int(bruto.k)} vence os criterios brutos e e' ARTEFATO: com tao "
            f"poucos grupos a silhueta sobe sozinha. Com o minimo util de {K_UTIL}, "
            f"o pico e' k={int(melhor.k)} ({melhor.real:.3f} contra piso "
            f"{melhor.piso:.3f}). MAS o pico troca de lugar conforme a semente — "
            f"medido: 10, 8, 6 e 8 em quatro sorteios. A faixa plausivel e' "
            f"{faixa}, e apontar um k exato dentro dela seria publicar a semente.")

        print(f"\n  {'k':>3} {'silhueta':>10} {'piso':>8} {'ganho':>8}   ")
        print("  " + "-" * 46)
        for _, x in tab.iterrows():
            barra = "#" * int(max(x.ganho, 0) * 200)
            marca = ("  <- pico desta semente" if x.k == melhor.k
                     else "  <- tambem plausivel" if x.k in faixa
                     else "  <- maior ganho bruto, mas artefato" if x.k == bruto.k
                     else "  (abaixo do minimo util)" if x.k < K_UTIL else "")
            print(f"  {int(x.k):>3} {x.real:>10.3f} {x.piso:>8.3f} "
                  f"{x.ganho:>+8.3f} {barra}{marca}")

    K = int(melhor.k)
    with rodada(f"o que caracteriza cada um dos {K} grupos?", {"k": K}) as r:
        km = KMeans(n_clusters=K, n_init=10, random_state=SEMENTE).fit(X)
        amostra["grupo"] = km.labels_
        centros = km.cluster_centers_
        # numeracao estavel: ordena por energia do centro, do mais calmo ao mais
        # agitado. Sem isso o numero do grupo muda a cada execucao e nao serve
        # para conversar — "grupo 3" tem de querer dizer a mesma coisa amanha.
        ordem = np.argsort(centros[:, F.index("energy")])
        mapa = {velho: novo for novo, velho in enumerate(ordem, start=1)}
        amostra["grupo"] = amostra.grupo.map(mapa)
        centros = centros[ordem]

        tam = amostra.grupo.value_counts().sort_index()
        r.resultado(grupos=K, menor=int(tam.min()), maior=int(tam.max()),
                    **{f"g{i}": int(n) for i, n in tam.items()})
        r.leitura(f"grupos numerados por energia crescente: 1 e' o mais calmo, "
                  f"{K} o mais agitado. O menor tem {tam.min()} faixas e o maior "
                  f"{tam.max()}.")

        print(f"\n  {'grupo':>5} {'faixas':>7}  {'o que marca (em desvios-padrao)':<44} "
              f"{'genero mais comum':<20}")
        print("  " + "-" * 84)
        for i in range(1, K + 1):
            sub = amostra[amostra.grupo == i]
            gen = sub.track_genre.mode().iloc[0] if len(sub) else "—"
            print(f"  {i:>5} {len(sub):>7}  {nomear(centros[i-1], F):<44} {gen:<20}")

    with rodada("as musicas dentro de um grupo sao parecidas?",
                {"mostrar": MOSTRAR}) as r:
        # coesao pelo mesmo escore de pertencimento ja' validado em p14_esta_faixa_cabe_na_playlist.py
        coes = {}
        for i in range(1, K + 1):
            sub = amostra[amostra.grupo == i]
            Xi = X[sub.index.to_numpy()] if len(sub) else np.empty((0, len(F)))
            if len(Xi) < 4:
                continue
            c, w = Xi.mean(0), np.maximum(Xi.std(0), 0.10)
            coes[i] = float(np.abs((Xi - c) / w).mean())
        r.resultado(**{f"coesao_g{i}": round(v, 3) for i, v in coes.items()})
        mais, menos = min(coes, key=coes.get), max(coes, key=coes.get)
        r.leitura(f"grupo mais coeso: {mais} ({coes[mais]:.2f} larguras). Mais "
                  f"espalhado: {menos} ({coes[menos]:.2f}). Abaixo de 1,0 as faixas "
                  f"sao do mesmo padrao.")

        print(f"\n  MUSICAS DE CADA GRUPO — confira com o ouvido se fazem sentido\n")
        for i in range(1, K + 1):
            sub = amostra[amostra.grupo == i]
            if not len(sub):
                continue
            print(f"  grupo {i} · {nomear(centros[i-1], F)} · "
                  f"coesao {coes.get(i, float('nan')):.2f}")
            for _, x in sub.head(MOSTRAR).iterrows():
                print(f"      {str(x.track_name)[:42]:<42} {str(x.artists)[:24]:<24} "
                      f"{x.track_genre}")
            print()

    print("=" * 78)
    print("COMO LER ESTE RESULTADO")
    print("=" * 78)
    print(f"  O k NAO E' UM PONTO, E' UMA FAIXA: {faixa}. Com o pipeline ANTERIOR")
    print("  (9 features) o pico mudava com a semente — 10, 8, 6 e 8, ganhos entre")
    print("  +0,068 e +0,080. A varredura de sementes NAO foi refeita sob as 7")
    print("  features; o que se sabe hoje e' que ARI pica em 7 e silhueta em 8 —")
    print("  diferenca dentro do ruido do sorteio. Anunciar um k exato dentro da")
    print("  faixa seria publicar a semente, nao o dado. Confira voce mesmo:")
    print("      uv run python src/p16_quantos_grupos_pela_silhueta.py --semente 7 --k-min 5 --k-max 14")
    print()
    # A LEITURA SAI DO NUMERO. A versao anterior interpolava o valor e emendava
    # "esta' ABAIXO de 0,2" como texto fixo — escrito quando o valor era 0,185.
    # Depois da troca para 7 features o valor virou 0,253 e a frase ficou
    # ARITMETICAMENTE FALSA na saida oficial: comparacao literal ao lado de
    # numero calculado envelhece calada.
    v = float(melhor.real)
    if v < 0.2:
        faixa_txt = ("ABAIXO de 0,2 — os grupos sao fatias de uma nuvem continua, "
                     "nao ilhas separadas")
    elif v < 0.5:
        faixa_txt = ("entre 0,2 e 0,5 — estrutura FRACA. Ha' separacao, e ela nao "
                     "chega a ser clara")
    else:
        faixa_txt = "acima de 0,5 — estrutura clara"
    print(f"  A silhueta escolhida ({v:.3f}) esta' {faixa_txt}.")
    print("  Isso NAO invalida o agrupamento: ele serve para ORGANIZAR o catalogo,")
    print("  e nao para afirmar que existem generos naturais.")
    print(f"  O ganho sobre o piso ({melhor.ganho:+.3f}) e' o que diz que ha' estrutura")
    print("  de verdade — pouca, mas real.")
    return 0


if __name__ == "__main__":
    ajuda_se_pedida("p16_quantos_grupos_pela_silhueta.py — quantos grupos o dado sustenta?")
    sys.exit(main())
