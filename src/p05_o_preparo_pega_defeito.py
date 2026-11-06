"""Selftest do ESQUEMA: prova que ele PEGA o que deveria pegar.

Teste que so' passa em dado bom nao prova nada — precisa FALHAR quando tem que
falhar. Este arquivo existe para que uma regressao no `prepare.py` apareca como
erro em vez de virar numero errado num relatorio.

    uv run python src/p05_o_preparo_pega_defeito.py    -> outputs/31-selftest.txt

Codigo de saida 0 se todos os casos passarem, 1 se algum falhar.
"""
import sys

import prepare
from cabecalho import cabecalho

falhas = 0


def caso(nome: str, esperado, obtido) -> None:
    global falhas
    ok = esperado == obtido
    if not ok:
        falhas += 1
    print(f"  [{'ok ' if ok else 'FALHOU'}] {nome:52} esperado {esperado!s:<14} obtido {obtido}")


def main() -> int:
    cabecalho(
        pergunta="O esquema pega o que deveria pegar? E as visoes ainda devolvem o que prometem?",
        como="Injeta defeito de proposito e verifica se a validacao levanta; depois confere a cardinalidade de cada visao.",
        ler_assim="Teste que passa em tudo sem nunca ter falhado nao prova nada — por isso ha casos propositalmente ruins.",
        piso="Cada caso declara o esperado ao lado do obtido.",
    )
    limpo = prepare.carregar(remover_impossiveis=True)

    print("=== 1. defeito e 'fora do cartao' sao coisas DIFERENTES ===")
    print("    o filtro tira falha de extracao comprovada, nao tudo que o cartao proibe")
    caso("nenhum defeito sobra apos a remocao", 0, len(prepare.defeituosas(limpo)))
    caso("mas 1.063 linhas seguem fora do cartao", 1063, len(prepare.fora_do_dominio(limpo)))
    caso("e elas sao musica real, nao lixo (todas com tempo)", True,
         bool((limpo.loc[prepare.fora_do_dominio(limpo)].tempo > 0).all()))

    print("\n=== 2. valor fora do dominio e' pego, coluna por coluna ===")
    provas = [
        ("time_signature", 9),     # fora de 3-7
        ("popularity", 101),       # fora de 0-100
        ("key", 12),               # fora de -1..11
        ("mode", 2),               # nao e' 0 nem 1
        ("danceability", 1.5),     # fora de [0,1]
        ("loudness", 3.2),         # clipping
        ("tempo", 0.0),            # BPM impossivel
        ("duration_ms", 0),        # extracao falhou
    ]
    # A amostra ja' contem linhas legitimas fora do cartao (as 1.063), entao o
    # invariante nao e' "da' 1" — e' "aumenta em exatamente 1".
    amostra = limpo.head(500)
    base = len(prepare.fora_do_dominio(amostra))
    print(f"    a amostra de 500 ja' tem {base} linha(s) fora do cartao; esperado = base + 1")
    for coluna, valor in provas:
        sujo = amostra.copy()
        sujo.iloc[0, sujo.columns.get_loc(coluna)] = valor
        caso(f"{coluna} = {valor}", base + 1, len(prepare.fora_do_dominio(sujo)))

    print("\n=== 3. nulo indevido em coluna nao-nulavel ===")
    sujo = amostra.copy()
    sujo.iloc[0, sujo.columns.get_loc("track_genre")] = None
    caso("track_genre nulo", base + 1, len(prepare.fora_do_dominio(sujo)))

    print("\n=== 4. falha ESTRUTURAL levanta, nao devolve indice vazio ===")
    print("    o caso perigoso: sem a guarda, a tabela errada passaria batido")
    sujo = amostra.rename(columns={"tempo": "bpm"})
    try:
        resto = prepare.fora_do_dominio(sujo)
        caso("coluna renomeada deveria levantar", "ValueError", f"devolveu {len(resto)} indices")
    except ValueError as erro:
        caso("coluna renomeada levanta ValueError", "ValueError", "ValueError")
        print(f"       {str(erro).splitlines()[0]}")


    print("\n=== 5. cardinalidade das visoes ===")
    print("    a chave de dedup ja' inverteu conclusao DUAS vezes nesta analise.")
    print("    se um destes numeros mudar sem ninguem decidir, foi acidente.")
    base = prepare.carregar()
    bi = prepare.carregar(remover_impossiveis=True)
    caso("carregar() — sem as 450 identicas", 113_550, len(base))
    caso("carregar(remover_impossiveis) tira 158", 158, len(base) - len(bi))
    caso("por_faixa_genero = uma linha por (faixa, genero)", len(base), len(prepare.por_faixa_genero(base)))
    caso("por_faixa — uma linha por track_id", 89_741, len(prepare.por_faixa(base)))
    caso("por_musica — so' as vivas, por (nome, artista)", 76_444, len(prepare.por_musica(bi)))
    caso("por_musica < por_faixa (musica agrega gravacoes)", True,
         len(prepare.por_musica(bi)) < len(prepare.por_faixa(bi)))
    caso("por_artista > por_faixa (o ';' desmembra)", True,
         len(prepare.por_artista(bi)) > len(prepare.por_faixa(bi)))

    print("\n=== 6. rotulo duplicado e derivadas ===")
    print("    songwriter == singer-songwriter: 114 rotulos, 113 generos.")
    print("    se o alias sumir do prepare, contagem por genero volta a dobrar mil faixas.")
    a = set(base[base.track_genre == "singer-songwriter"].track_id)
    b = set(base[base.track_genre == "songwriter"].track_id)
    caso("os dois rotulos sao o MESMO conjunto", True, a == b and len(a) == 1_000)
    can = prepare.genero_canonico(base)
    caso("genero_canonico deixa 113 rotulos", 113, can.track_genre.nunique())
    caso("genero_canonico nao perde linha", len(base), len(can))
    der = prepare.com_derivadas(prepare.por_faixa(base))
    caso("instrumental = 18,5% das faixas", 16_633, int(der.instrumental.sum()))
    dentro = (der.tempo_dj >= 70) & (der.tempo_dj < 140)
    caso("tempo_dj: ou cai na janela, ou e' NaN", True,
         bool((dentro | der.tempo_dj.isna()).all()))
    caso("NaN so' nas 157 com tempo 0 (sentinela)", 157, int(der.tempo_dj.isna().sum()))
    caso("nenhum tempo_dj igual a zero", 0, int((der.tempo_dj == 0).sum()))

    print("\n=== 7. a visao independe da ORDEM em que o arquivo chega ===")
    print("    'kind=stable' garante mesma-entrada->mesma-saida, e NAO independencia")
    print("    de ordem. Sem desempate explicito, o genero mudava so' por a linha")
    print("    chegar em outra posicao — e o TAMANHO do estrago depende da visao,")
    print("    porque os universos sao diferentes. Medido no pior de 3")
    print("    embaralhamentos: ~8.800 de 89.741 em `por_faixa` (9,8%) e ~7.000 de")
    print("    81.344 em `por_musica` sem filtrar vivas (8,6%). O numero oscila com")
    print("    o sorteio; o que NAO oscila e' que com desempate total ele e' zero.")
    for nome, fn, idx in [("por_faixa", prepare.por_faixa, ["track_id"]),
                          ("por_musica", prepare.por_musica, ["track_name", "artists"])]:
        ref = fn(base).set_index(idx).track_genre.sort_index()
        piores = max(
            int((ref.to_numpy() != fn(base.sample(frac=1, random_state=s))
                 .set_index(idx).track_genre.sort_index().to_numpy()).sum())
            for s in (1, 7, 42))
        caso(f"{nome} — generos que mudam por ordem", 0, piores)

    print(f"\n=== {'TODOS OS CASOS PASSARAM' if not falhas else f'{falhas} CASO(S) FALHARAM'} ===")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
