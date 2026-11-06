"""Registro de execucoes: cada rodada de modelo, com parametros, metricas e commit.

O MLflow resolveria isto, mas exige pandas<3 e este projeto usa 3.0.5 (mesmo
bloqueio do pandas-dq). Isto e' o equivalente minimo: cada rodada vira uma
linha em outputs/execucoes.csv, com parametros e metricas.

Uso:
    from registro import registrar
    registrar("classificacao_genero",
              params={"n_estimators": 250, "min_samples_leaf": 2},
              metricas={"top1": 0.351, "acaso": 0.0088})

Ler depois:
    uv run python src/registro.py            # tabela no terminal
    uv run python src/registro.py --por top1 # ordenado por metrica
"""
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ARQ = Path(__file__).resolve().parents[1] / "outputs" / "execucoes.csv"
CAMPOS = ["quando", "experimento", "commit", "params", "metricas"]


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip() or "—"
    except Exception:
        return "—"


def registrar(experimento: str, params: dict, metricas: dict) -> None:
    """Anexa uma rodada. Nunca sobrescreve — o historico e' o produto.

    `BANCADA_SILENCIO=1` (ligado pelo `rodar_tudo.py`) desliga: reconferir nao
    e' rodada nova.
    """
    if os.environ.get("BANCADA_SILENCIO"):
        return
    ARQ.parent.mkdir(exist_ok=True)
    novo = not ARQ.exists()
    with ARQ.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        if novo:
            w.writeheader()
        w.writerow({
            "quando": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "experimento": experimento, "commit": _commit(),
            "params": json.dumps(params, sort_keys=True, ensure_ascii=False),
            "metricas": json.dumps(metricas, sort_keys=True, ensure_ascii=False),
        })


def mostrar(ordenar_por: str | None = None) -> int:
    if not ARQ.exists():
        print("  nenhuma execucao registrada ainda")
        return 0
    linhas = list(csv.DictReader(ARQ.open(encoding="utf-8")))
    for l in linhas:
        l["_p"] = json.loads(l["params"])
        l["_m"] = json.loads(l["metricas"])
    if ordenar_por:
        linhas = [l for l in linhas if ordenar_por in l["_m"]]
        linhas.sort(key=lambda l: l["_m"][ordenar_por], reverse=True)
        print(f"=== {len(linhas)} execucoes, ordenadas por '{ordenar_por}' ===\n")
    else:
        print(f"=== {len(linhas)} execucoes, em ordem cronologica ===\n")
    for l in linhas:
        p = " · ".join(f"{k}={v}" for k, v in l["_p"].items())
        m = " · ".join(f"{k}={v}" for k, v in l["_m"].items())
        print(f"  {l['quando']}  {l['experimento']}  [{l['commit']}]")
        print(f"    params  {p}")
        print(f"    metrica {m}\n")
    return 0


if __name__ == "__main__":
    arg = sys.argv[1:]
    sys.exit(mostrar(arg[1] if len(arg) > 1 and arg[0] == "--por" else None))
