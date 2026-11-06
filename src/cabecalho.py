"""Cabecalho didatico dos arquivos de outputs/.

Todo script imprime quatro linhas antes do resultado. O motivo: quem abre um
`.txt` que comeca em `eta^2 (genero) = 0.254` nao sabe o que e' eta^2, contra o
que comparar, nem por que a pergunta importa.

O gabarito e' o de docs/METODO.md — pergunta em portugues, como se faz,
o que a resposta quer dizer, e o piso de comparacao.
"""

LARGURA = 78


def cabecalho(pergunta: str, como: str, ler_assim: str, piso: str = "—",
              metricas: list[str] | None = None) -> None:
    """Imprime o cabecalho de um output. Chamar como primeira linha de main().

    `metricas` lista as chaves do glossario em `metricas.py`. Cada uma vira um
    verbete com o que e', como ler e a armadilha — impresso junto do resultado,
    para nao exigir que o leitor va' procurar em outro arquivo.
    """
    campos = [("PERGUNTA", pergunta), ("COMO", como),
              ("LER ASSIM", ler_assim), ("PISO", piso)]
    print("=" * LARGURA)
    for rotulo, texto in campos:
        prefixo = f"{rotulo:<10} "
        recuo = " " * len(prefixo)
        linha, primeira = prefixo, True
        for palavra in texto.split():
            if len(linha) + len(palavra) + 1 > LARGURA:
                print(linha.rstrip())
                linha, primeira = recuo, False
            linha += palavra + " "
        print(linha.rstrip())
    if metricas:
        from metricas import bloco
        print()
        print(bloco(metricas, LARGURA))
    print("=" * LARGURA)
    print()
