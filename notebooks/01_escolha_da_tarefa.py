import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    from prepare import por_musica, por_faixa_genero, AUDIO

    return AUDIO, alt, mo, np, pd, por_faixa_genero, por_musica


@app.cell
def _(mo):
    mo.md("""
    # Escolhendo a tarefa

    Este notebook não é "a exploração". Exploração é **uma etapa** de
    *entendimento dos dados*. O objetivo aqui é outro: **decidir que tarefa de
    machine learning faz sentido** para este dataset, com evidência em vez de palpite.

    Toda análise entra pelo `prepare.py`. Escolher a visão errada já inverteu
    conclusão nesta análise — ver `docs/METODOLOGIA.md`, seção 4.
    """)
    return


@app.cell
def _(por_faixa_genero, por_musica):
    mus = por_musica()
    gen = por_faixa_genero()
    gen = gen[gen.popc > 0]
    (len(mus), len(gen))
    return gen, mus


@app.cell
def _(mo):
    mo.md("""
    ## 1. Comparar qualquer coluna entre qualquer agrupamento
    """)
    return


@app.cell
def _(AUDIO, mo):
    coluna = mo.ui.dropdown(
        options=AUDIO + ["popc"], value="instrumentalness", label="coluna"
    )
    agrup = mo.ui.dropdown(
        options=["gênero", "quadrante sonoro", "faixa de popularidade", "explicit"],
        value="quadrante sonoro", label="agrupar por",
    )
    topn = mo.ui.slider(3, 25, value=8, label="quantos grupos")
    mo.hstack([coluna, agrup, topn], justify="start", gap=2)
    return agrup, coluna, topn


@app.cell
def _(agrup, gen, mus, np, pd):
    def rotular(d):
        if agrup.value == "gênero":
            return d.assign(grupo=d.track_genre)
        if agrup.value == "quadrante sonoro":
            return d.assign(grupo=(
                np.where(d.acousticness >= .5, "orgânico", "produzido") + " / " +
                np.where(d.instrumentalness >= .5, "instrumental", "vocal")))
        if agrup.value == "faixa de popularidade":
            return d.assign(grupo=pd.qcut(
                d.popc, 4, labels=["baixa", "média-baixa", "média-alta", "alta"]).astype(str))
        return d.assign(grupo=np.where(d.explicit, "explicit", "limpo"))

    base = rotular(gen if agrup.value == "gênero" else mus)
    return (base,)


@app.cell
def _(alt, base, coluna, mo, topn):
    resumo = (base.groupby("grupo")[coluna.value]
                  .agg(mediana="median", media="mean", n="size")
                  .reset_index()
                  .sort_values("mediana", ascending=False)
                  .head(topn.value))

    grafico = (
        alt.Chart(resumo)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X("mediana:Q", title=f"{coluna.value} — mediana"),
            y=alt.Y("grupo:N", sort="-x", title=None),
            tooltip=["grupo", "mediana", "media", "n"],
        )
        .properties(height=topn.value * 25)
    )
    mo.vstack([grafico, mo.ui.table(resumo, selection=None)])
    return


@app.cell
def _(mo):
    mo.md("""
    > **Ressalva ao agrupar por gênero:** a amostra tem exatamente 1.000 faixas por
    > gênero *por construção*, e a retenção após limpeza varia de 21% (`latin`) a
    > 99% (`comedy`). Comparar gêneros sem dizer isso é comparar coisas diferentes.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Que tarefa cabe aqui — e quanto sinal cada uma tem

    Já sondamos quase todas. Os números vêm de `outputs/`, não de intuição.
    Baseline sempre ao lado, porque acurácia sem acaso não significa nada.
    """)
    return


@app.cell
def _(mo, pd):
    tarefas = pd.DataFrame([
        ["Classificação", "gênero (114 classes)", "34,0%", "0,88%", "39×", "forte", "test_rules.py"],
        ["Não supervisionado", "agrupamento por confusão", "14 famílias", "—", "—", "forte", "genre_kinship.py"],
        ["Análise de rede", "grafo de colaboração", "41% num componente", "—", "—", "média", "collab_graph.py"],
        ["Classificação", "faixa de popularidade", "44,5%", "25%", "1,78×", "fraca", "rerun_core.py"],
        ["Detecção de anomalia", "faixa com metadado quebrado", "1,20% impossível", "—", "—", "média", "outliers.py"],
        ["Redução de dimensão", "PCA das audio features", "5 PCs p/ 80%", "—", "—", "diagnóstica", "explore_axes.py"],
        ["Regressão", "popularity contínua", "|r| ≤ 0,207", "—", "—", "fraca", "rerun_core.py"],
        ["Recomendação", "similaridade por kNN", "não testado", "—", "—", "?", "—"],
    ], columns=["Família", "Alvo", "Resultado", "Acaso", "Ganho", "Sinal", "Onde"])
    mo.ui.table(tarefas, selection=None)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Leitura

    - **Regressão de `popularity` é a escolha mais óbvia e a mais fraca.** Nenhuma
      audio feature passa de |r| = 0,21. Vira modelo ruim apresentado como resposta.
    - **Classificação de gênero é a de maior sinal**, e o valor não está no acerto:
      está em *onde o modelo erra*. Ele acerta `study` (87%) e `sleep` (84%) e erra
      `brazil` (1%) e `mpb` (8%) — o som identifica função, não cultura.
    - **Recomendação por similaridade é a única não testada**, e tem uma vantagem:
      não depende de `popularity`, então escapa da censura dos 14% zerados.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    # As conclusões que mudaram

    O que vem abaixo foi acrescentado **depois** da primeira versão deste notebook.
    Cada bloco é uma conclusão que estava escrita aqui e caiu.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. A chave de deduplicação inverteu um achado

    Faixas que aparecem em muitos gêneros pareciam ter popularidade **2**. Os exemplos
    eram *Everybody Talks* (Neon Trees) e *LA CANCIÓN* (J Balvin; Bad Bunny) — música
    conhecida marcada com zero.

    **Teste:** metade das faixas zeradas tem uma **gêmea viva** no próprio arquivo, com
    popularidade mediana 67. São cópias delistadas com outro `track_id`.
    """)
    return


@app.cell
def _(mo, pd):
    inversao = pd.DataFrame([
        ["1 gênero", 31, 32],
        ["2–3 gêneros", 44, 46],
        ["4–7 gêneros", 2, 46],
    ], columns=["grupo", "dedup por track_id", "dedup por música"])
    mo.md(f"""
    {mo.ui.table(inversao, selection=None)}

    **Faixa multi-gênero não é impopular — é das mais populares.** Hit atravessa
    categoria. E isso corrige de quebra a popularidade baixa de `latin` e `reggaeton`,
    que era acúmulo de cópias zeradas de hits latinos.

    > A chave de deduplicação não é decisão de limpeza. É **decisão de modelagem**.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. `track_genre` não é uma taxonomia de gênero

    Só **64 dos 114** rótulos são gênero musical. Os outros 50 são outra coisa:

    | Categoria | Quantos | Exemplos |
    |---|---|---|
    | Gênero musical | 64 | rock, jazz, techno, classical |
    | Geografia ou língua | 27 | brazil, mpb, turkish, mandopop |
    | Contexto ou uso | 14 | party, sleep, study, chill |
    | Público ou franquia | 9 | kids, disney, anime, comedy |

    Isso **explica** o achado central: o classificador acerta `study` em 87% e `brazil`
    em 1% porque são coisas de naturezas diferentes na mesma coluna. Nunca foi o
    modelo falhando.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Sinal forte não é produto viável

    O classificador de gênero acerta **39× o acaso**, o que soa pronto para uso.
    Mas a curva de cobertura contra acerto conta outra história.
    """)
    return


@app.cell
def _(alt, mo, pd):
    cobertura = pd.DataFrame([
        [0, 100.0, 33.7], [30, 26.0, 72.5], [40, 16.7, 80.8], [50, 11.6, 87.0],
        [60, 8.4, 90.6], [70, 5.9, 93.5], [80, 4.1, 95.4], [90, 2.8, 97.0],
    ], columns=["confianca_min", "cobertura", "acerto"])
    linha = alt.Chart(cobertura).transform_fold(
        ["cobertura", "acerto"], as_=["medida", "valor"]
    ).mark_line(point=True).encode(
        x=alt.X("confianca_min:Q", title="confiança mínima aceita (%)"),
        y=alt.Y("valor:Q", title="%"),
        color=alt.Color("medida:N", title=None),
    ).properties(height=260)
    mo.vstack([
        mo.ui.altair_chart(linha),
        mo.md("""
        Em acurácia utilizável (93,5%), o modelo cobre **5,9%** do acervo. De 40 mil
        faixas, sobrariam 37.647 para revisão humana. **O entregável honesto é a curva,
        não um número** — o cliente escolhe o ponto conforme o custo de um erro.
        """),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. O erro do detector é de projeto, não aleatório

    Procuramos **contradições lógicas** — casos em que duas colunas afirmam coisas
    incompatíveis. Achamos pouco: **248 faixas, 0,28%**. A maior classe são 234 faixas
    com `liveness` alta em `sleep`, `study` e `ambient`, gêneros que não têm plateia.

    E quando a mesma música tem medidas diferentes, é porque são **gravações
    diferentes** — não porque o detector oscilou.

    **Então o algoritmo é consistente.** Os erros dele estão nas premissas:

    - `mode` força maior/menor em música que não é nem um nem outro (makam turco)
    - `key` colapsa enarmonia e assume 12 notas iguais
    - `time_signature` nunca emite 6 nem 7, e usa o `3` como balde de "sem pulso claro"
    - `instrumentalness` devolve zero exato em 34% — é detector, não medida
    - em 157 faixas a análise rítmica simplesmente não rodou: `tempo`, `danceability`
      e `time_signature` zerados juntos
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Andamento é a coluna que menos define gênero

    Quanto cada coluna é explicada pelo gênero (η²):

    | Coluna | η² | |
    |---|---|---|
    | `acousticness` | 0,498 | mais característica |
    | `speechiness` | 0,466 | |
    | `energy` | 0,464 | |
    | `instrumentalness` | 0,457 | |
    | `valence` | 0,324 | |
    | `duration_ms` | 0,176 | |
    | **`tempo`** | **0,087** | **atravessa gêneros** |

    Gênero é definido por **timbre e textura**, não por velocidade.

    E isso tem consequência prática: a literatura de atmosférica de varejo diz que
    **andamento é a maior alavanca** sobre o comportamento de quem está no ambiente.
    Ou seja, a coluna que mais importa para escolher música de ambiente é justamente a
    que **menos prende você a um gênero**. Dá para acertar o BPM alvo atravessando
    repertórios.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    ## Onde isso continua

    - Writeup completo: `README.md` do repositório
    - Correções e fundamentos: `docs/METODOLOGIA.md`
    - Painéis publicados: auditoria, escolha da tarefa, simulador de ambiente
    - Scripts: `src/`, com a saída de cada um em `outputs/NN-nome.txt`
    """)
    return


if __name__ == "__main__":
    app.run()
