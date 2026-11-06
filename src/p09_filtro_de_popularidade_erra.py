"""O corte `popularity <= 10` remove hit delistado, e em que proporcao?

Correcao de uma leitura anterior: uma primeira medida achou que 56,8% das
faixas cortadas tem gemea viva e concluiu que o filtro remove hits.
A conclusao estava ERRADA: a mediana da gemea e' 8 — tambem abaixo do corte.
Ter gemea viva nao e' o mesmo que ser hit.

A pergunta certa e' outra: quantas cortadas tem gemea ACIMA do corte? So' essas
sao erro do filtro. As demais sao musica de fato pouco tocada, e cortar e'
curadoria legitima.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from bancada import nota, rodada
from cabecalho import cabecalho
from prepare import carregar

CORTE = 10


def main() -> int:
    cabecalho(
        pergunta="Um filtro comum de curadoria, `popularity <= 10`, remove musica pouco tocada "
                 "(curadoria legitima) ou copia delistada de faixa popular (erro)?",
        como="Para cada faixa cortada, procura no arquivo outra com mesmo nome e "
             "artista. Separa por FAIXA de popularidade da gemea, nao so' por "
             "existir gemea.",
        ler_assim="Gemea viva com popularidade 8 nao e' hit — e' a mesma musica "
                  "pouco tocada. So' gemea ACIMA do corte prova erro do filtro.",
        piso="Se o filtro estivesse certo, quase nenhuma cortada teria gemea bem "
             "acima do corte.",
        metricas=["baseline"],
    )
    nota("correcao de leitura: gemea viva != hit", corte=CORTE)

    base = carregar()

    with rodada("quantas cortadas tem gemea ACIMA do corte?", {"corte": CORTE}) as r:
        alvo = base[base.popularity <= CORTE]
        melhor = base.groupby(["track_name", "artists"]).popularity.max().rename("gemea")
        j = alvo.join(melhor, on=["track_name", "artists"])
        faixas = {
            "gemea 0 (tudo zerado)": (j.gemea == 0),
            "gemea 1-10 (tambem cortada)": (j.gemea > 0) & (j.gemea <= CORTE),
            "gemea 11-40 (erro leve)": (j.gemea > CORTE) & (j.gemea <= 40),
            "gemea 41-70 (erro serio)": (j.gemea > 40) & (j.gemea <= 70),
            "gemea 71+ (hit delistado)": (j.gemea > 70),
        }
        r.resultado(cortadas=len(alvo), **{k: int(v.sum()) for k, v in faixas.items()})
        erro = int(((j.gemea > CORTE)).sum())
        r.leitura(f"das {len(alvo):,} cortadas, {erro:,} ({erro/len(alvo):.1%}) tem "
                  f"gemea ACIMA do corte — essas o filtro erra. O resto e' musica "
                  f"pouco tocada de fato, e cortar e' curadoria, nao defeito.")

    with rodada("quem sao os hits que o corte derruba?", {"corte": CORTE}) as r:
        alvo = base[base.popularity <= CORTE]
        melhor = base.groupby(["track_name", "artists"]).popularity.max().rename("gemea")
        j = alvo.join(melhor, on=["track_name", "artists"])
        hits = j[j.gemea > 70].drop_duplicates(["track_name", "artists"])
        r.resultado(hits_derrubados=len(hits),
                    popularidade_mediana=float(hits.gemea.median()) if len(hits) else 0)
        if len(hits):
            print("\n  exemplos de hit que o corte de popularidade derruba:")
            print(f"  {'faixa':<44} {'artista':<26} {'gemea':>6}")
            for _, x in hits.nlargest(10, "gemea").iterrows():
                print(f"  {str(x.track_name)[:44]:<44} {str(x.artists)[:26]:<26} "
                      f"{x.gemea:>6.0f}")
        r.leitura(f"{len(hits):,} musicas distintas com copia de popularidade > 70 "
                  f"sao removidas pelo corte. Poucas em proporcao, mas sao "
                  f"exatamente as que um DJ ou uma playlist quereria.")

    with rodada("e o corte de loudness, erra parecido?", {"corte_db": -16}) as r:
        quietas = base[base.loudness <= -16]
        por_gen = quietas.track_genre.value_counts()
        r.resultado(removidas=len(quietas),
                    generos_afetados=int((por_gen > 0).sum()),
                    **{f"top_{i+1}": f"{g} ({n})" for i, (g, n) in
                       enumerate(por_gen.head(3).items())})
        r.leitura("o corte de loudness nao e' aleatorio: concentra em generos "
                  "silenciosos por natureza. Nao e' defeito, e' timbre.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
