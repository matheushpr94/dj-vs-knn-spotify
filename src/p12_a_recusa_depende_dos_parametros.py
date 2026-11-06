"""Pre-computa a grade de parametros para o console interativo.

Pagina publicada e' estatica — nao roda Python. Entao a varredura acontece
aqui, e a pagina navega o resultado. Trocar um controle e' consultar a grade,
nao recalcular.
"""
import json, itertools
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from cabecalho import cabecalho
from camelot import camelot, compativel, bpm_ok
from prepare import carregar, por_musica
from registro import registrar

F = ["danceability","energy","loudness","speechiness","acousticness",
     "instrumentalness","liveness","valence","tempo"]
PISTA = ["house","techno","edm","deep-house","progressive-house","trance",
         "minimal-techno","detroit-techno","chicago-house","dance","club"]

MARGENS = [.02, .04, .06, .08, .10, .12]
REGRAS  = ["roda completa", "mesma posição", "ignorar tom"]
KS      = [4, 6, 8, 10]


def main() -> None:
    cabecalho(
        pergunta="Como o veredito muda quando mexemos em cada parametro da regra?",
        como=f"Grade completa: {len(MARGENS)} margens de BPM x {len(REGRAS)} regras de tonalidade x {len(KS)} valores de k.",
        ler_assim="Resultado que muda pouco entre celulas e' robusto. Celula muito diferente das vizinhas merece desconfianca.",
        piso="Com 'ignorar tom' e margem larga, a recusa tende a zero — e' o canto de controle da grade.",
        metricas=['acaso'],
    )
    d = por_musica(carregar(remover_impossiveis=True))
    d = d[d.track_genre.isin(PISTA) & d.tempo.between(110,140) & (d.danceability >= .6)]
    d = d.dropna(subset=["track_name"]).reset_index(drop=True)
    d["cam"] = [camelot(k,m) for k,m in zip(d.key, d["mode"])]
    d = d[d.cam.notna()].reset_index(drop=True)
    Z = StandardScaler().fit_transform(d[F])
    _, idx = NearestNeighbors(n_neighbors=max(KS)+1).fit(Z).kneighbors(Z)
    tempos, cams = d.tempo.to_numpy(), d.cam.to_numpy()
    print(f"catálogo: {len(d):,} faixas · grade de {len(MARGENS)*len(REGRAS)*len(KS)} células\n")

    grade = []
    for k, marg, regra in itertools.product(KS, MARGENS, REGRAS):
        rec = so_tom = so_bpm = tot = 0
        for i in range(len(d)):
            for j in idx[i, 1:k+1]:
                tot += 1
                ok_t = bpm_ok(tempos[i], tempos[j], marg)
                ok_k = (True if regra == "ignorar tom"
                        else cams[i] == cams[j] if regra == "mesma posição"
                        else compativel(cams[i], cams[j]))
                if not (ok_t and ok_k):
                    rec += 1
                    if ok_t: so_tom += 1
                    elif ok_k: so_bpm += 1
        grade.append({"k": k, "margem": marg, "regra": regra,
                      "recusa": round(rec/tot, 4), "so_tom": round(so_tom/tot, 4),
                      "so_bpm": round(so_bpm/tot, 4), "n": tot})
    print(f"  {'k':>3} {'margem':>7}  {'regra':<15} {'recusa':>8}")
    for c in grade:
        if c["k"] == 6:
            print(f"  {c['k']:>3} {c['margem']:>6.0%}  {c['regra']:<15} {c['recusa']:>7.1%}")
    registrar("grade_parametros",
              params={"celulas": len(grade), "faixas": len(d)},
              metricas={"recusa_min": min(c["recusa"] for c in grade),
                        "recusa_max": max(c["recusa"] for c in grade)})
    with open("outputs/grade.json", "w", encoding="utf-8") as f:
        json.dump({"faixas": len(d), "grade": grade,
                   "margens": MARGENS, "regras": REGRAS, "ks": KS}, f, ensure_ascii=False)
    print(f"\n  outputs/grade.json escrito · {len(grade)} células")


if __name__ == "__main__":
    main()
