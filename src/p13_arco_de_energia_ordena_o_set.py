"""As audio features podem DECIDIR o set, nao so' gerar candidatos?

A critica que motivou este script, e ela procede: "se o DJ usa so' Camelot e
BPM, o dataset quase nao e' aproveitado". Hoje o kNN usa 9 features para GERAR candidatos
e Camelot+BPM para FILTRAR. As 17 colunas geram, 3 colunas decidem.

Existe um uso em que as features decidem, e ele vem do dominio: set de DJ tem
ARCO DE ENERGIA. Nao e' sequencia de pares compativeis — e' uma curva que sobe,
segura e desce. Isso e' `energy`, `valence`, `danceability` e `loudness` fazendo
trabalho de decisao, nao de ranking.

Compara tres montadores sobre as MESMAS restricoes de Camelot e BPM:

  proximidade  escolhe o vizinho acustico mais proximo — o que temos hoje
  arco         escolhe quem esta' mais perto da ENERGIA ALVO daquele passo
  acaso        escolhe qualquer compativel ao acaso — o piso

Se o arco nao superar o acaso, a ideia nao vale e isso e' resultado.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from bancada import nota, rodada
from botoes import ajuda_se_pedida, aviso_se_mudou, botao
from cabecalho import cabecalho
from camelot import bpm_ok, camelot, compativel
from prepare import carregar, por_faixa

F = ["danceability", "energy", "loudness", "speechiness", "acousticness",
     "instrumentalness", "liveness", "valence", "tempo"]
N_SET = botao("--set", 20, int, "quantas faixas o set tem")
MARGEM = botao("--margem", 0.06, float, "quanto o BPM pode diferir (0.06 = 6%)")
TOLERANCIA = botao("--tolerancia", 0.05, float,
                   "quao perto do alvo de energia uma faixa precisa estar para "
                   "entrar no desempate por popularidade (0.05 = 5 pontos de energia)")
SEMENTE = botao("--semente", 42, int, "sorteio das faixas iniciais (teste de robustez)")
SETS = botao("--sets", 60, int, "quantos sets montar para tirar a media")


def pista(d: pd.DataFrame) -> pd.DataFrame:
    """As faixas de pista: dancavel, energetica, com tonalidade detectada."""
    p = d[(d.danceability >= .7) & (d.energy >= .7) & (d.key >= 0)].copy()
    p["cam"] = [camelot(k, m) for k, m in zip(p.key, p["mode"])]
    return p.dropna(subset=["cam", "track_name"]).reset_index(drop=True)


def compativeis(p, i, usados):
    """Indices que respeitam Camelot e BPM a partir de i, ainda nao usados."""
    a = p.iloc[i]
    return [j for j in range(len(p))
            if j not in usados
            and bpm_ok(a.tempo, p.tempo.iat[j], MARGEM)
            and compativel(a.cam, p.cam.iat[j])]


def montar(p, Z, inicio, modo, rng):
    """Constroi o set. O que muda entre os modos e' SO' o criterio de escolha."""
    usados, seq = {inicio}, [inicio]
    e0 = float(p.energy.iat[inicio])
    for passo in range(1, N_SET):
        cand = compativeis(p, seq[-1], usados)
        if not cand:
            break
        if modo == "proximidade":                       # o mais parecido acusticamente
            d = np.linalg.norm(Z[cand] - Z[seq[-1]], axis=1)
            prox = cand[int(np.argmin(d))]
        elif modo in ("arco", "arco+pop"):          # o mais perto da energia alvo
            alvo = e0 + (1.0 - e0) * (passo / (N_SET - 1)) * .8
            erro = np.abs(p.energy.to_numpy()[cand] - alvo)
            if modo == "arco":
                prox = cand[int(np.argmin(erro))]
            else:
                # POPULARIDADE COMO TERCEIRO DESEMPATE, e a ordem importa.
                # Ela NAO compete com o arco: entram no sorteio final so' as
                # candidatas que ja' estao dentro de TOLERANCIA do alvo de energia.
                # Entre essas — que o arco considera equivalentes — ganha a mais
                # conhecida. Se popularidade subisse de posto, a ferramenta viraria
                # ranking de sucesso, que e' a tarefa que esta analise testou e
                # REJEITOU (R2 0,191). Ultimo criterio, nunca o primeiro.
                dentro = np.asarray(cand)[erro <= erro.min() + TOLERANCIA]
                if len(dentro) == 1:
                    prox = int(dentro[0])
                else:
                    prox = int(dentro[np.argmax(p.popc.to_numpy()[dentro])])
        else:                                      # acaso: o piso
            prox = int(rng.choice(cand))
        usados.add(prox)
        seq.append(prox)
    return seq


def avaliar(p, seq):
    """Tres coisas medem se um set presta, e nenhuma delas e' 'parece bom'."""
    s = p.iloc[seq]
    t = s.tempo.to_numpy()
    e = s.energy.to_numpy()
    pos = np.arange(len(e))
    # correlacao entre posicao e energia: e' um ARCO ou um passeio aleatorio?
    subida = float(np.corrcoef(pos, e)[0, 1]) if len(e) > 2 and e.std() > 0 else 0.0
    return {
        "faixas": len(seq),
        "popularidade_media": float(s.popc.mean()),
        "deriva_bpm": float((t[-1] - t[0]) / t[0]),
        "subida_de_energia": subida,
        "amplitude_energia": float(e.max() - e.min()),
        "posicoes_camelot": int(s.cam.nunique()),
    }


def main() -> int:
    cabecalho(
        pergunta="As audio features conseguem DECIDIR a ordem do set, ou so' "
                 "servem para gerar candidatos que a regra do DJ filtra?",
        como="Monta 60 sets de 20 faixas com tres criterios sobre as MESMAS "
             "restricoes de Camelot e BPM: vizinho mais proximo, energia alvo "
             "crescente, e escolha ao acaso.",
        ler_assim="'subida de energia' e' a correlacao entre posicao no set e "
                  "energia. Perto de +1 e' arco que sobe; perto de 0 e' passeio "
                  "aleatorio. E' isso que separa set de playlist embaralhada.",
        piso="O montador ao ACASO, sujeito as mesmas regras. Se o arco nao "
             "superar o acaso, a ideia nao vale.",
        metricas=["baseline"],
    )
    aviso_se_mudou()
    nota("as features decidem o set?", set_len=N_SET, margem=MARGEM, semente=SEMENTE)

    p = pista(por_faixa(carregar()))
    Z = ((p[F] - p[F].mean()) / p[F].std()).to_numpy()

    rng = np.random.default_rng(SEMENTE)
    inicios = rng.choice(len(p), min(SETS, len(p)), replace=False)

    resumo = {}
    for modo in ["acaso", "proximidade", "arco", "arco+pop"]:
        with rodada(f"montador '{modo}': o set tem forma?",
                    {"sets": len(inicios), "faixas_por_set": N_SET,
                     "pista": len(p), "semente": SEMENTE}) as r:
            rng_m = np.random.default_rng(SEMENTE)
            av = [avaliar(p, montar(p, Z, int(i), modo, rng_m)) for i in inicios]
            m = pd.DataFrame(av).mean()
            resumo[modo] = m
            r.resultado(faixas_medias=round(float(m.faixas), 1),
                        subida_de_energia=round(float(m.subida_de_energia), 3),
                        deriva_bpm=round(float(m.deriva_bpm), 4),
                        amplitude_energia=round(float(m.amplitude_energia), 3),
                        posicoes_camelot=round(float(m.posicoes_camelot), 1))
            r.leitura(f"correlacao posicao x energia = {m.subida_de_energia:+.3f}; "
                      f"deriva de BPM {m.deriva_bpm:+.1%}; "
                      f"{m.faixas:.1f} faixas antes de travar.")

    print(f"\n{'montador':<13} {'faixas':>7} {'subida':>9} {'deriva BPM':>11} "
          f"{'ampl. energia':>14} {'posicoes':>9} {'popularid.':>11}")
    print("-" * 69)
    for modo in ["acaso", "proximidade", "arco", "arco+pop"]:
        m = resumo[modo]
        print(f"{modo:<13} {m.faixas:>7.1f} {m.subida_de_energia:>+9.3f} "
              f"{m.deriva_bpm:>+11.1%} {m.amplitude_energia:>14.3f} "
              f"{m.posicoes_camelot:>9.1f} {m.popularidade_media:>11.1f}")

    print("\n" + "=" * 66)
    ganho = resumo["arco"].subida_de_energia - resumo["acaso"].subida_de_energia
    print(f"O arco supera o acaso em subida de energia por {ganho:+.3f}.")
    if ganho > .15:
        print("As features DECIDEM: o mesmo conjunto de restricoes, mudando so' o")
        print("criterio de escolha, produz um set com forma em vez de passeio.")
        print("Isso responde a critica — as 17 colunas nao ficam so' gerando candidato.")
    else:
        print("O arco NAO se distingue do acaso. A restricao de Camelot+BPM e' tao")
        print("apertada que sobra pouca escolha, e o criterio quase nao importa.")
        print("Isso tambem e' resultado: mostra que a regra do DJ domina o problema.")
    return 0


if __name__ == "__main__":
    ajuda_se_pedida("p13_arco_de_energia_ordena_o_set.py — as audio features decidem a ordem do set?")
    sys.exit(main())
