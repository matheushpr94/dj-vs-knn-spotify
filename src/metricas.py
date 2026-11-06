"""Glossario das metricas, para viajar junto do resultado.

O cabecalho de cada output explica a PERGUNTA. Isto explica o JARGAO que aparece
na resposta. Cada verbete tem tres partes: o que e', como ler, e a armadilha.

Uso: `cabecalho(..., metricas=["acaso", "eta2"])`.
"""

GLOSSARIO = {
"acaso": (
 "o que voce acertaria chutando, sem modelo nenhum",
 "todo resultado de acerto so' significa alguma coisa AO LADO dele. Exemplo de conta: "
 "35% parece pouco, mas contra 0,88% de acaso e' 40x. O numero DESTA rodada sai abaixo",
 "resultado sem acaso declarado nao pode ser avaliado. Se alguem mostrar so' a acuracia, "
 "pergunte qual era o acaso"),

"eta2": (
 "eta ao quadrado: que fracao da variacao de um numero e' explicada por uma categoria",
 "vai de 0 a 1. Genero com eta^2 = 0,38 explica 38% da variacao de popularidade",
 "INFLA com muitos grupos — com 30 mil artistas qualquer agrupamento da' valor alto. "
 "Sempre compare com o piso aleatorio de mesmo numero de grupos"),

"r2": (
 "R ao quadrado: quanto da variacao o modelo de regressao consegue explicar",
 "1 e' perfeito, 0 e' inutil. R^2 = 0,20 significa que 80% da variacao continua sem explicacao",
 "R^2 parece grande e engana. Traduza sempre para erro absoluto na unidade original, "
 "e compare com o erro de chutar a mediana. O par medido aqui esta' em 22-correlacoes.txt"),

"mae": (
 "erro medio absoluto: de quanto o modelo erra, em media, na unidade original",
 "MAE de 12 numa escala de 0 a 100 significa errar 12 pontos por faixa, em media",
 "so' significa algo comparado ao MAE de chutar a mediana. Se o modelo nao bate isso, "
 "ele nao serve"),

"pearson": (
 "correlacao linear entre duas colunas numericas, de -1 a +1",
 "0 = sem relacao linear. 0,5 e' relacao moderada. O sinal diz a direcao",
 "so' enxerga RETA. E e' sensivel a massa concentrada num valor — se 34% da coluna e' "
 "zero exato, ele mede a distancia entre dois grupos, nao uma tendencia"),

"spearman": (
 "correlacao de ORDEM: mede se uma sobe quando a outra sobe, mesmo sem ser reta",
 "leia junto do Pearson. Se os dois concordam, a relacao e' solida",
 "quando Pearson e Spearman dao SINAIS OPOSTOS, desconfie: costuma ser inflacao de zero"),

"silhueta": (
 "quao bem separados estao os grupos que uma clusterizacao achou",
 "vai de -1 a 1. Acima de 0,5 e' estrutura clara. Abaixo de 0,2 e' o algoritmo fatiando "
 "uma nuvem continua",
 "K-Means SEMPRE devolve grupos, ate' em ruido puro. Compare com a silhueta do mesmo "
 "dado embaralhado — se a diferenca for pequena, nao ha' estrutura"),

"jaccard": (
 "quanto dois conjuntos se sobrepoem: compartilhado dividido pela uniao",
 "0 = nada em comum, 1 = identicos. Corrige tamanho, entao grupo grande nao ganha de graca",
 "Jaccard 1,000 entre dois rotulos nao quer dizer 'parecidos' — quer dizer que sao o "
 "MESMO conjunto com dois nomes"),

"cramer": (
 "V de Cramer: forca da associacao entre duas colunas de CATEGORIA",
 "0 a 1. Abaixo de 0,1 fraca, 0,3 moderada, acima de 0,5 forte",
 "e' o que correlacao numerica nao alcanca. Foi assim que o par key x mode apareceu — "
 "nenhuma correlacao de Pearson o pegaria"),

"topk": (
 "top-1 e' acertar na primeira tentativa; top-5 e' a resposta certa estar entre 5 palpites",
 "top-5 sempre e' maior. A distancia entre os dois diz se o modelo esta' confuso entre "
 "poucas opcoes ou perdido",
 "compare cada um com SEU proprio acaso: com 114 classes, o acaso do top-1 e' 0,88% e o "
 "do top-5 e' 4,4%"),

"cohen": (
 "d de Cohen: tamanho do efeito, ou seja, a diferenca medida em desvios-padrao",
 "0,2 e' pequeno, 0,5 medio, 0,8 grande. Abaixo de 0,1 e' praticamente nada",
 "existe justamente para separar 'diferenca real' de 'diferenca detectavel'. Com amostra "
 "grande, qualquer diferenca fica 'significativa' — o d diz se ela IMPORTA"),

"transitividade": (
 "a chance de que dois amigos de alguem tambem sejam amigos entre si",
 "0 a 1. Alta significa panelinhas fechadas; baixa significa rede espalhada",
 "so' significa algo contra um MODELO NULO — a mesma rede com as conexoes embaralhadas, "
 "preservando quantas cada um tem"),

"centralidade": (
 "centralidade de intermediacao: quanto alguem aparece no caminho entre dois outros",
 "alta significa PONTE — quem liga comunidades que nao se falariam",
 "nao e' o mesmo que ter muitas conexoes. Mozart tem 149 parceiros e ponte fraca, porque "
 "os 149 sao todos da mesma cena"),

"componente": (
 "componente conexo: um pedaco da rede onde todos alcancam todos, direta ou indiretamente",
 "o 'maior componente' e' a parte principal. Quem fica fora esta' isolado",
 "componente grande nao e' sinal de boa conexao — compare com o modelo nulo. Aqui o real "
 "conecta MENOS que o embaralhado"),

"variancia_explicada": (
 "no PCA, quanto de toda a variacao cada eixo novo captura",
 "PC1 e' o eixo que mais captura. Somando ate' 80% voce sabe quantas dimensoes reais existem",
 "poucos PCs para 80% significa colunas redundantes. Muitos significa que cada coluna "
 "carrega algo proprio — foi o caso aqui, 5 PCs"),

"confianca": (
 "quanta certeza o classificador tem no proprio palpite, de 0 a 1",
 "serve para TRIAR: aceitar so' o que passa de um limiar e mandar o resto para revisao",
 "a curva cobertura x acerto e' o entregavel, nao a acuracia media: exigir mais confianca sobe o acerto e ENCOLHE o acervo coberto. A curva medida esta' em 25-casos-de-uso.txt"),
}


def bloco(chaves: list[str], largura: int = 78) -> str:
    """Monta o bloco COMO LER com os verbetes pedidos."""
    if not chaves:
        return ""
    linhas = ["COMO LER OS NÚMEROS"]
    for c in chaves:
        if c not in GLOSSARIO:
            continue
        oque, ler, cuidado = GLOSSARIO[c]
        LARG_ROT = 12
        for rotulo, texto in (("  " + c, oque), ("     ler", ler), ("  cuidado", cuidado)):
            recuo = " " * LARG_ROT
            if len(rotulo) >= LARG_ROT:       # rotulo longo ganha linha propria
                linhas.append(rotulo)
                linha = recuo
            else:
                linha = f"{rotulo:<{LARG_ROT}}"
            for palavra in texto.split():
                if len(linha) + len(palavra) + 1 > largura:
                    linhas.append(linha.rstrip()); linha = recuo
                linha += palavra + " "
            linhas.append(linha.rstrip())
        linhas.append("")
    return "\n".join(linhas).rstrip()
