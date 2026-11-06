"""Esta musica cabe nesta playlist?

Esta pergunta e' DIFERENTE da que o MVP responde. O MVP responde
"qual faixa vem depois desta?" — par a par e sequencial. Esta e' pertencimento:
dado um conjunto, uma faixa nova cabe nele?

COMO FUNCIONA, sem jargao:
  toda playlist ocupa uma regiao no espaco das 9 audio features. Ela tem um
  CENTRO (a media de cada coluna) e uma LARGURA (o desvio-padrao de cada coluna).
  O escore de uma faixa e' a distancia dela ate' o centro, medida em numero de
  larguras. Perto de zero, pertence. Longe, destoa.

  Dividir pela largura NAO e' detalhe. Sem isso, playlist de ambient (features
  apertadas) e playlist de festa (features espalhadas) nao sao comparaveis, e o
  escore vira medida da playlist em vez de medida da faixa.

O QUE NAO TEMOS: playlists. O dataset nao traz nenhuma. Usamos `track_genre`
como substituto, e o limite disso e' real — genero e' mais largo que playlist.
`album_name` e' testado junto porque album e' mais estreito que genero, e da'
o outro extremo da regua.

O PISO: uma faixa sorteada do catalogo. Se a faixa que pertence de fato nao
pontuar melhor que a sorteada, nao ha' sinal e isso e' o resultado.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from bancada import nota, rodada
from botoes import ajuda_se_pedida, aviso_se_mudou, botao
from cabecalho import cabecalho
from prepare import AUDIO, carregar, por_faixa

F = ["danceability", "energy", "loudness", "speechiness", "acousticness",
     "instrumentalness", "liveness", "valence", "tempo"]
SEMENTE = botao("--semente", 42, int, "sorteio das faixas de comparacao")
MIN_FAIXAS = botao("--min-faixas", 12, int,
                   "grupo minimo para calcular centro e largura")          # abaixo disso, media e desvio nao se sustentam


def escore(centro, largura, x):
    """Distancia ao centro em numero de larguras, media entre as colunas.

    Zero e' o centro exato. 1,0 quer dizer "uma largura de distancia", que e'
    o quanto uma faixa TIPICA daquele conjunto ja' esta'. Acima de 2 destoa.
    """
    return float(np.mean(np.abs((x - centro) / largura)))


# Piso da largura, e ele NAO e' detalhe de implementacao.
# A versao anterior usava `largura[largura == 0] = 1e-9`, que parece inofensivo e
# nao e': album com uma coluna constante (todas as faixas com instrumentalness 0,
# por exemplo) dividia por 1e-9 e produzia distancias de 674 MILHOES de larguras.
# O catalogo esta' padronizado, entao largura 1,0 e' a do catalogo inteiro. Um piso
# de 0,10 diz: "nenhum grupo conta como mais de 10x mais estreito que o catalogo".
# Sem isso, coluna constante domina o escore inteiro por artefato de divisao.
PISO_LARGURA = 0.10


def avaliar_grupo(d, sub, rng, catalogo):
    """Um teste por grupo: quem pertence pontua melhor que quem foi sorteado?

    Deixa UMA faixa de fora ao calcular centro e largura — senao a faixa que
    estamos pontuando ajudou a definir o proprio alvo, e o escore fica bom de
    graca. E' o mesmo cuidado de nao testar no dado de treino.
    """
    X = sub[F].to_numpy()
    dentro, fora = [], []
    for i in range(len(X)):
        resto = np.delete(X, i, axis=0)
        centro = resto.mean(axis=0)
        largura = np.maximum(resto.std(axis=0), PISO_LARGURA)
        dentro.append(escore(centro, largura, X[i]))
        fora.append(escore(centro, largura, catalogo[rng.integers(len(catalogo))]))
    return np.array(dentro), np.array(fora)


def main() -> int:
    cabecalho(
        pergunta="Da' para dizer se uma musica cabe numa playlist, so' pelas "
                 "audio features?",
        como="Para cada grupo, calcula centro e largura das 9 features SEM a "
             "faixa que esta' sendo pontuada, mede a distancia dela ao centro em "
             "numero de larguras, e compara com uma faixa sorteada do catalogo.",
        ler_assim="O escore e' 'quantas larguras de distancia'. Zero e' o centro. "
                  "O que importa nao e' o valor absoluto e sim a DIFERENCA entre "
                  "quem pertence e quem foi sorteado — e o acerto, que e' quantas "
                  "vezes quem pertence pontuou melhor.",
        piso="Faixa sorteada do catalogo. Sem modelo nenhum, o acerto seria 50%: "
             "meio a meio. Tudo acima de 50% e' sinal.",
        metricas=["baseline", "tamanho de efeito"],
    )
    aviso_se_mudou()
    nota("escore de pertencimento por grupo", semente=SEMENTE, min_faixas=MIN_FAIXAS)

    d = por_faixa(carregar()).dropna(subset=F + ["track_genre", "album_name"])
    # padroniza o catalogo inteiro: sem isso, tempo (0-250) domina as outras oito
    Z = (d[F] - d[F].mean()) / d[F].std()
    dz = d.assign(**{c: Z[c] for c in F})
    catalogo = dz[F].to_numpy()

    resultados = []
    for nome_col, rotulo, amostra in [("track_genre", "genero", 40),
                                      ("album_name", "album", 300)]:
        with rodada(f"o escore separa quem pertence, usando {rotulo} como playlist?",
                    {"agrupador": nome_col, "grupos_testados": amostra,
                     "min_faixas": MIN_FAIXAS, "semente": SEMENTE}) as r:
            rng = np.random.default_rng(SEMENTE)
            tam = dz.groupby(nome_col).size()
            elegiveis = sorted(tam[tam >= MIN_FAIXAS].index.tolist())
            if len(elegiveis) > amostra:
                idx = rng.choice(len(elegiveis), amostra, replace=False)
                elegiveis = [elegiveis[i] for i in sorted(idx)]

            todos_d, todos_f = [], []
            for g in elegiveis:
                sub = dz[dz[nome_col] == g]
                if len(sub) > 120:
                    sub = sub.sample(120, random_state=SEMENTE)
                a, b = avaliar_grupo(dz, sub, rng, catalogo)
                todos_d.append(a)
                todos_f.append(b)
            dentro = np.concatenate(todos_d)
            fora = np.concatenate(todos_f)
            acerto = float((dentro < fora).mean())
            # d de Cohen: a diferenca em numero de desvios-padrao combinados
            dp = np.sqrt((dentro.std() ** 2 + fora.std() ** 2) / 2)
            efeito = float((fora.mean() - dentro.mean()) / dp) if dp else 0.0
            r.resultado(grupos=len(elegiveis), pares=len(dentro),
                        escore_pertence=round(float(dentro.mean()), 3),
                        escore_sorteada=round(float(fora.mean()), 3),
                        acerto=round(acerto, 4), piso=0.5,
                        tamanho_de_efeito=round(efeito, 3))
            r.leitura(
                f"quem pertence tira {dentro.mean():.2f} larguras contra "
                f"{fora.mean():.2f} de quem foi sorteado. Acerta {acerto:.1%} contra "
                f"50% do acaso. Tamanho de efeito {efeito:.2f}.")
            resultados.append({"agrupador": rotulo, "grupos": len(elegiveis),
                               "dentro": dentro.mean(), "fora": fora.mean(),
                               "acerto": acerto, "efeito": efeito})

    print(f"\n{'usando como playlist':<24} {'grupos':>7} {'pertence':>9} "
          f"{'sorteada':>9} {'acerto':>8} {'piso':>6} {'efeito':>8}")
    print("-" * 78)
    for r in resultados:
        print(f"{r['agrupador']:<24} {r['grupos']:>7} {r['dentro']:>9.2f} "
              f"{r['fora']:>9.2f} {r['acerto']:>8.1%} {'50,0%':>6} {r['efeito']:>8.2f}")

    print("\n" + "=" * 78)
    print("COMO LER")
    print("=" * 78)
    print("  'pertence' e 'sorteada' sao distancias ao centro, em numero de larguras.")
    print("  MENOR e' mais parecido. O modelo funciona se 'pertence' < 'sorteada'.")
    print("  'acerto' e' a fracao de vezes em que isso aconteceu, e o piso e' 50%:")
    print("  sem modelo nenhum, seria cara ou coroa.")
    print()
    melhor = max(resultados, key=lambda x: x["acerto"])
    if melhor["acerto"] < .6:
        print("  SINAL FRACO. As audio features quase nao dizem se uma faixa pertence")
        print("  a um grupo. Isso e' resultado, e coerente com o resto da analise:")
        print("  genero nao e' uma regiao limpa no espaco acustico.")
    else:
        print(f"  HA' SINAL: {melhor['acerto']:.1%} usando {melhor['agrupador']}.")
        print("  E ATENCAO ao comparar os dois: album e' mais ESTREITO que genero, entao")
        print("  acerto maior em album nao quer dizer 'album e' melhor playlist' — quer")
        print("  dizer que o problema e' mais facil quando o grupo ja' e' homogeneo.")
    return 0


if __name__ == "__main__":
    ajuda_se_pedida("p14_esta_faixa_cabe_na_playlist.py — esta musica cabe nesta playlist?")
    sys.exit(main())
