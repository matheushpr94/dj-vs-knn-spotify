"""Registro de tentativas: toda rodada de experimento vira um evento, inclusive as que falham.

O problema que isto resolve: numa exploracao rapida, quinze experimentos
rodam, quatorze somem, e so' o que deu certo vira codigo. Sem registro das
tentativas, nao ha' como saber quantos caminhos foram testados ate' o resultado. As razoes dos outros quatorze somem junto.

`outputs/` guarda RESULTADO. Esta bancada guarda a TENTATIVA, no instante em que acontece — inclusive a que
falhou, inclusive a que foi abandonada no meio.

Uso:

    from bancada import rodada

    with rodada("a dobra de tempo destrava o DJ?",
                {"margem": 0.06, "pares": 40_000}) as r:
        ...
        r.resultado(aceita_cru=0.1378, aceita_dobrado=0.1713)
        r.leitura("libera +24,3% no eixo BPM, mas o gargalo e' tonalidade")

Excecao dentro do bloco fica registrada como falha, com a mensagem, e sobe.
Cada evento vira uma linha JSON em `outputs/bancada.jsonl` (fora do git).
"""
from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "outputs" / "bancada.jsonl"


def _agora() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _escrever(registro: dict) -> None:
    """Append de uma linha JSON. Append e' atomico o bastante para uma linha
    curta, e nao exige travar arquivo — a pagina pode ler enquanto se escreve.

    `BANCADA_SILENCIO=1` desliga a escrita, e `rodar_tudo.py` liga esse
    interruptor. A razao: reconferencia nao e' experimento. Reexecutar os 36
    scripts para comparar saida acrescentava ~80 linhas de repeticao por rodada
    — 45 perguntas, nenhuma inedita — e sujava o git.
    """
    if os.environ.get("BANCADA_SILENCIO"):
        return
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


class rodada:
    """Contexto que registra inicio, fim, resultado e falha de um experimento."""

    def __init__(self, pergunta: str, parametros: dict | None = None,
                 script: str | None = None):
        self.pergunta = pergunta
        self.parametros = parametros or {}
        self.script = script or Path(traceback.extract_stack()[-2].filename).name
        self._resultado: dict = {}
        self._leitura: str = ""

    def resultado(self, **kv) -> "rodada":
        """Os numeros que sairam. Chave curta, valor cru — a pagina formata."""
        self._resultado.update(kv)
        return self

    def leitura(self, texto: str) -> "rodada":
        """O que o numero quer dizer, em portugues. Sem isto, e' so' numero."""
        self._leitura = texto
        return self

    def __enter__(self) -> "rodada":
        self._t0 = time.time()
        self._id = f"{int(self._t0 * 1000):x}"
        _escrever({"id": self._id, "fase": "inicio", "quando": _agora(),
                   "script": self.script, "pergunta": self.pergunta,
                   "parametros": self.parametros})
        return self

    def __exit__(self, exc_tipo, exc, tb) -> bool:
        base = {"id": self._id, "quando": _agora(), "script": self.script,
                "pergunta": self.pergunta, "parametros": self.parametros,
                "segundos": round(time.time() - self._t0, 2)}
        if exc is not None:
            _escrever({**base, "fase": "falha", "erro": f"{exc_tipo.__name__}: {exc}",
                       "onde": traceback.format_tb(tb)[-1].strip() if tb else ""})
            return False           # a excecao sobe: registrar nao e' engolir
        _escrever({**base, "fase": "fim", "resultado": self._resultado,
                   "leitura": self._leitura})
        return False


def nota(texto: str, **kv) -> None:
    """Marca um ponto sem experimento em volta — decisao, descoberta, abandono."""
    _escrever({"id": f"{int(time.time() * 1000):x}", "fase": "nota",
               "quando": _agora(), "pergunta": texto, "parametros": kv,
               "script": Path(traceback.extract_stack()[-2].filename).name})


if __name__ == "__main__":
    import sys
    if "--limpar" in sys.argv:
        LOG.unlink(missing_ok=True)
        print(f"bancada zerada: {LOG}")
    else:
        n = sum(1 for _ in LOG.open()) if LOG.exists() else 0
        print(f"{n} evento(s) em {LOG}")
        print("acompanhe com: tail -f outputs/bancada.jsonl")
