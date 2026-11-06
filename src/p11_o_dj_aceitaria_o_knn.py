"""kNN acustico contra a regra de DJ. Quantas sugestoes um DJ recusaria?

O kNN acha faixas que SOAM parecidas. O DJ precisa de duas coisas que o kNN
ignora: BPM compativel (ate ~6%) e tonalidade adjacente na roda de Camelot.
Este script mede o desencontro entre os dois criterios.
"""
import json
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from cabecalho import cabecalho
from camelot import camelot, nome_musical, compativel, bpm_ok, vizinhas
from prepare import carregar, por_musica

F = ["danceability", "energy", "loudness", "speechiness", "acousticness",
     "instrumentalness", "liveness", "valence", "tempo"]
PISTA = ["house", "techno", "edm", "deep-house", "progressive-house", "trance",
         "minimal-techno", "detroit-techno", "chicago-house", "dance", "club"]


def main() -> None:
    cabecalho(
        pergunta="O kNN acustico serve para escolher a proxima musica de um set de DJ?",
        como="Compara os vizinhos acusticos com o que a regra de DJ aceita: BPM ate 6% de diferenca E tonalidade adjacente na roda de Camelot.",
        ler_assim="Taxa de recusa alta significa que soar parecido NAO e' o criterio de quem monta set. As duas regras medem coisas diferentes.",
        piso="Se os criterios fossem equivalentes, a recusa seria proxima de zero.",
        metricas=['acaso'],
    )
    d = por_musica(carregar(remover_impossiveis=True))
    d = d[d.track_genre.isin(PISTA) & d.tempo.between(110, 140) & (d.danceability >= .6)]
    d = d.dropna(subset=["track_name", "artists"]).reset_index(drop=True)
    d["camelot"] = [camelot(k, m) for k, m in zip(d.key, d["mode"])]
    d = d[d.camelot.notna()].reset_index(drop=True)
    print(f"catálogo de pista: {len(d):,} faixas\n")

    Z = StandardScaler().fit_transform(d[F])
    nn = NearestNeighbors(n_neighbors=9).fit(Z)
    _, idx = nn.kneighbors(Z)

    print("=== 1. dos vizinhos que o kNN sugere, quantos o DJ aceitaria? ===")
    aceitos = recusados = 0
    por_motivo = {"só BPM": 0, "só tonalidade": 0, "os dois": 0}
    for i in range(len(d)):
        a = d.iloc[i]
        for j in idx[i, 1:]:
            b = d.iloc[j]
            ok_t, ok_k = bpm_ok(a.tempo, b.tempo), compativel(a.camelot, b.camelot)
            if ok_t and ok_k:
                aceitos += 1
            else:
                recusados += 1
                por_motivo["os dois" if not ok_t and not ok_k
                           else "só BPM" if not ok_t else "só tonalidade"] += 1
    tot = aceitos + recusados
    print(f"  sugestões avaliadas : {tot:,}")
    print(f"  o DJ ACEITARIA      : {aceitos:,} ({aceitos/tot:.1%})")
    print(f"  o DJ RECUSARIA      : {recusados:,} ({recusados/tot:.1%})")
    print("\n  por que recusaria:")
    for k, v in por_motivo.items():
        print(f"    {k:<16} {v:>6,}  ({v/tot:.1%})")

    print("\n=== 2. e o contrario: o DJ tem opcao? ===")
    val = [sum(bpm_ok(d.tempo[i], d.tempo[j]) and compativel(d.camelot[i], d.camelot[j])
               for j in range(len(d)) if j != i) for i in range(min(len(d), 400))]
    val = np.array(val)
    print(f"  faixas validas por semente (amostra de {len(val)}): mediana {np.median(val):.0f}, "
          f"min {val.min()}, max {val.max()}")
    print(f"  sementes sem NENHUMA continuacao valida: {(val == 0).sum()}")

    # --- exporta para o MVP ---
    rng = np.random.default_rng(3)
    sementes = []
    for i in rng.choice(len(d), 12, replace=False):
        a = d.iloc[int(i)]
        knn = [{"nome": str(d.iloc[int(j)].track_name)[:52], "artista": str(d.iloc[int(j)].artists)[:34],
                "bpm": round(float(d.iloc[int(j)].tempo)), "camelot": d.iloc[int(j)].camelot,
                "tom": nome_musical(d.iloc[int(j)].key, d.iloc[int(j)]["mode"]),
                "genero": d.iloc[int(j)].track_genre,
                "bpm_ok": bool(bpm_ok(a.tempo, d.iloc[int(j)].tempo)),
                "tom_ok": bool(compativel(a.camelot, d.iloc[int(j)].camelot))}
               for j in idx[int(i), 1:7]]
        m = d.index[(d.index != i) & [bpm_ok(a.tempo, t) for t in d.tempo]
                    & [compativel(a.camelot, c) for c in d.camelot]]
        dist = np.linalg.norm(Z[m] - Z[int(i)], axis=1) if len(m) else np.array([])
        dj = [{"nome": str(d.iloc[int(j)].track_name)[:52], "artista": str(d.iloc[int(j)].artists)[:34],
               "bpm": round(float(d.iloc[int(j)].tempo)), "camelot": d.iloc[int(j)].camelot,
               "tom": nome_musical(d.iloc[int(j)].key, d.iloc[int(j)]["mode"]),
               "genero": d.iloc[int(j)].track_genre, "bpm_ok": True, "tom_ok": True}
              for j in m[np.argsort(dist)[:6]]] if len(m) else []
        sementes.append({"nome": str(a.track_name)[:52], "artista": str(a.artists)[:34],
                         "bpm": round(float(a.tempo)), "camelot": a.camelot,
                         "tom": nome_musical(a.key, a["mode"]), "genero": a.track_genre,
                         "vizinhas": sorted(vizinhas(a.camelot)), "knn": knn, "dj": dj})

    saida = {"total": len(d), "aceita": aceitos, "recusa": recusados,
             "motivos": por_motivo, "sementes": sementes}
    with open("outputs/mvp.json", "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False)
    print(f"\n  outputs/mvp.json escrito com {len(sementes)} sementes")


if __name__ == "__main__":
    main()
