"""Gera as 3 figuras do README a partir das saidas em outputs/ (nada e' recalculado aqui).

    uv run --group figuras python src/figuras.py

1. recusa.png    — por que o DJ recusaria as sugestoes do kNN (11-o-dj-aceitaria-o-knn.txt)
2. arco.png      — correlacao posicao x energia nos tres criterios (13-arco-de-energia-ordena-o-set.txt)
3. silhueta.png  — silhueta do dado real contra o dado embaralhado (16-quantos-grupos-pela-silhueta.txt)

Ler o numero do texto, e nao recalcular, garante que a figura e o README contam a
mesma historia que a saida versionada: se a saida mudar, a figura muda junto.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import altair as alt
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / "outputs"
FIG = RAIZ / "figuras"

AZUL, CINZA, VERMELHO, VERDE = "#3B5BA9", "#9AA0A6", "#C0392B", "#2E8B57"


def ler(nome: str) -> str:
    return (OUT / nome).read_text(encoding="utf-8")


def numero(padrao: str, texto: str) -> float:
    m = re.search(padrao, texto)
    if not m:
        sys.exit(f"padrao nao encontrado: {padrao}")
    return float(m.group(1).replace(",", ""))


def recusa() -> alt.Chart:
    t = ler("11-o-dj-aceitaria-o-knn.txt")
    total = numero(r"sugestões avaliadas\s*:\s*([\d,]+)", t)
    linhas = [
        ("aceitaria", numero(r"o DJ ACEITARIA\s*:\s*([\d,]+)", t), VERDE),
        ("recusaria só por tonalidade", numero(r"só tonalidade\s+([\d,]+)", t), VERMELHO),
        ("recusaria por tom e BPM", numero(r"os dois\s+([\d,]+)", t), "#E67E22"),
        ("recusaria só por BPM", numero(r"só BPM\s+([\d,]+)", t), CINZA),
    ]
    df = pd.DataFrame([(r, n / total, c) for r, n, c in linhas], columns=["motivo", "fracao", "cor"])
    df["rotulo"] = df.fracao.map(lambda x: f"{x:.1%}".replace(".", ","))
    base = alt.Chart(df).encode(
        y=alt.Y("motivo:N", sort=None, title=None),
        x=alt.X("fracao:Q", axis=alt.Axis(format="%", title="fração das sugestões do kNN",
                                          values=[0, .2, .4, .6, .8]),
                scale=alt.Scale(domain=[0, 0.9])),
    )
    barras = base.mark_bar().encode(color=alt.Color("cor:N", scale=None))
    texto = base.mark_text(align="left", dx=4, fontSize=13).encode(text="rotulo:N")
    n = f"{int(total):,}".replace(",", ".")
    return (barras + texto).properties(
        width=520, height=170,
        title=alt.Title(f"O kNN erra o tom, não o BPM ({n} sugestões)",
                        subtitle="kNN nas 9 audio features × regra de DJ (BPM ±6% e Camelot adjacente)"),
    )


def arco() -> alt.Chart:
    t = ler("13-arco-de-energia-ordena-o-set.txt")
    linhas = []
    for modo, nome, cor in [("acaso", "ao acaso (piso)", CINZA),
                            ("proximidade", "por proximidade (o kNN)", VERMELHO),
                            ("arco", "por arco de energia", AZUL)]:
        m = re.search(rf"^{re.escape(modo)}\s+[\d.]+\s+([+-][\d.]+)", t, re.M)
        if not m:
            sys.exit(f"linha do modo {modo} nao encontrada")
        linhas.append((nome, float(m.group(1)), cor))
    df = pd.DataFrame(linhas, columns=["criterio", "r", "cor"])
    df["rotulo"] = df.r.map(lambda x: f"{x:+.3f}".replace(".", ","))
    base = alt.Chart(df).encode(
        y=alt.Y("criterio:N", sort=None, title=None),
        x=alt.X("r:Q", scale=alt.Scale(domain=[-0.2, 1.1]),
                axis=alt.Axis(title="correlação entre posição no set e energia")),
    )
    barras = base.mark_bar().encode(color=alt.Color("cor:N", scale=None))
    texto = base.mark_text(align="left", dx=4, fontSize=13).encode(text="rotulo:N")
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#555").encode(x="x:Q")
    return (barras + texto + zero).properties(
        width=520, height=140,
        title=alt.Title("Mesmas regras de mixagem, só o desempate muda",
                        subtitle="60 sets de 20 faixas; +1 = a energia sobe do começo ao fim"),
    )


def silhueta() -> alt.Chart:
    t = ler("16-quantos-grupos-pela-silhueta.txt")
    linhas = re.findall(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+[+-][\d.]+", t, re.M)
    if len(linhas) < 10:
        sys.exit("tabela de silhueta nao encontrada")
    reg = []
    for k, real, piso in linhas:
        reg += [(int(k), float(real), "dado real"), (int(k), float(piso), "colunas embaralhadas (piso)")]
    df = pd.DataFrame(reg, columns=["k", "silhueta", "serie"])
    return alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("k:Q", title="número de grupos (k)"),
        y=alt.Y("silhueta:Q", title="silhueta média", scale=alt.Scale(zero=False)),
        color=alt.Color("serie:N", title=None,
                        scale=alt.Scale(domain=["dado real", "colunas embaralhadas (piso)"],
                                        range=[AZUL, CINZA]),
                        legend=alt.Legend(orient="top")),
    ).properties(
        width=520, height=220,
        title=alt.Title("K-Means sempre acha grupos; o que importa é a distância ao piso",
                        subtitle="o ganho sobre o dado embaralhado é pequeno (≤ +0,09): nuvem contínua, não tipos naturais"),
    )


def main() -> int:
    FIG.mkdir(exist_ok=True)
    for nome, grafico in [("recusa", recusa), ("arco", arco), ("silhueta", silhueta)]:
        g = grafico().configure_view(stroke=None).configure_axis(labelFontSize=12, titleFontSize=12)
        g.save(str(FIG / f"{nome}.png"), scale_factor=2)
        print(f"  figuras/{nome}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
