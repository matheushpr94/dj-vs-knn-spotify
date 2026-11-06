"""O que contamina a coluna key. Teste central: fala nao tem tonalidade."""
import pandas as pd
from prepare import carregar, por_faixa
from cabecalho import cabecalho

NOTAS = ["Do","Do#","Re","Re#","Mi","Fa","Fa#","Sol","Sol#","La","La#","Si"]


def main() -> None:
    cabecalho(
        pergunta="O que contamina a coluna key? As deteccoes que falharam viraram 'Do'?",
        como="Fala nao tem tonalidade — entao se houvesse despejo em zero, comedy estaria entupido de key=0. Testa isso.",
        ler_assim="Se fala tem MENOS key=0 que musica, nao houve despejo. A contaminacao esta' em outro lugar.",
        piso="Distribuicao uniforme entre 12 tonalidades = 8,3% cada.",
        metricas=['acaso'],
    )
    bruto = carregar()
    d = por_faixa(bruto)
    print(f"{len(d):,} faixas unicas\n")

    print("=== 1. distribuicao das 12 tonalidades ===")
    v = d.key.value_counts().sort_index()
    base = 100 / len(d)
    for k, n in v.items():
        barra = "#" * int(n * base * 2)
        print(f"  {k:>2} {NOTAS[k]:<4} {n:>6,}  {n*base:5.1f}%  {barra}")
    print(f"\n  marcador oficial de ausencia (key = -1): {(d.key == -1).sum()}")

    print("\n=== 2. TESTE: fala nao tem tonalidade. E comedy? ===")
    for g in ["comedy", "sleep", "ambient", "classical", "pop"]:
        s = d[d.track_genre == g]
        if not len(s):
            continue
        pc0 = 100 * (s.key == 0).mean()
        print(f"  {g:<11} n={len(s):>5,}   key=0 em {pc0:5.1f}%   "
              f"(esperado ~{100/12:.1f}% se uniforme)")

    print("\n  recorte mais duro — so' as faixas que sao quase pura fala:")
    fala = d[d.speechiness > .66]
    print(f"    speechiness > 0,66: {len(fala):,} faixas · key=0 em "
          f"{100*(fala.key == 0).mean():.1f}%")
    musica = d[d.speechiness < .10]
    print(f"    speechiness < 0,10: {len(musica):,} faixas · key=0 em "
          f"{100*(musica.key == 0).mean():.1f}%")

    print("\n=== 3. key=0 coincide com as falhas ja conhecidas? ===")
    quebrado = d[(d.tempo == 0) | (d.time_signature == 0)]
    print(f"  faixas com tempo=0 ou time_signature=0: {len(quebrado):,}")
    print(f"    delas com key=0: {100*(quebrado.key == 0).mean():.1f}%")
    print(f"  catalogo inteiro com key=0            : {100*(d.key == 0).mean():.1f}%")

    print("\n=== 4. tonalidades enarmonicas: o que a coluna nao consegue distinguir ===")
    pretas = [1, 3, 6, 8, 10]
    print(f"  teclas pretas (Do#, Re#, Fa#, Sol#, La#): "
          f"{100*d.key.isin(pretas).mean():.1f}% do catalogo")
    print("  cada uma junta dois nomes — Do# e Reb sao o mesmo numero, e em")
    print("  partitura sao escolhas diferentes. A coluna nao guarda essa diferenca.")


if __name__ == "__main__":
    main()
