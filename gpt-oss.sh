#!/usr/bin/env bash
# Instala, serve e testa o gpt-oss-20b com vLLM em Linux com GPU NVIDIA.
#
# Uso:
#   ./gpt-oss.sh install           prepara o ambiente e baixa o modelo
#   ./gpt-oss.sh serve [args]      sobe o servidor em primeiro plano
#   ./gpt-oss.sh test [args]       testa chat, raciocínio e ferramentas
#
# Em "serve" e "test", os argumentos extras vão direto para o "vllm serve".
# As variáveis de ambiente aceitas estão descritas no README.
set -euo pipefail

MODEL="${MODEL:-openai/gpt-oss-20b}"
VLLM_VERSION="${VLLM_VERSION:-0.33.0}"
PYTHON_VERSION="${PYTHON_VERSION:-3.13}"
VENV_DIR="${VENV_DIR:-$HOME/vllm-env}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-600}"
MIN_VRAM_MIB="${MIN_VRAM_MIB:-23000}"
MIN_DISK_GB="${MIN_DISK_GB:-40}"

SERVER_PID=""

# Equivale a ativar o venv: o vLLM procura ferramentas como o ninja no PATH.
export PATH="$VENV_DIR/bin:$PATH"

msg() { printf '==> %s\n' "$*"; }
die() { printf 'ERRO: %s\n' "$*" >&2; exit 1; }

usage() {
  sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 1
}

check_gpu() {
  local vram
  command -v nvidia-smi >/dev/null || die "nvidia-smi não encontrado. Instale o driver NVIDIA."
  vram="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)" \
    || die "O nvidia-smi falhou. Verifique o driver e o acesso à GPU."
  vram="$(printf '%s\n' "$vram" | sort -n | tail -n 1)"
  [ "$vram" -ge "$MIN_VRAM_MIB" ] \
    || die "A maior GPU tem ${vram} MiB; o mínimo é ${MIN_VRAM_MIB} MiB."
  msg "GPU encontrada com ${vram} MiB de VRAM."
}

# O Triton compila kernels com o compilador da máquina ao iniciar o servidor.
check_compiler() {
  [ -n "${CC:-}" ] || command -v gcc >/dev/null || command -v clang >/dev/null \
    || die "Compilador C não encontrado. No Ubuntu: sudo apt install build-essential"
}

check_disk() {
  local dir="${HF_HOME:-$HOME/.cache/huggingface}" free
  mkdir -p "$dir"
  free="$(df -BG --output=avail "$dir" | tail -n 1 | tr -dc '0-9')"
  [ "$free" -ge "$MIN_DISK_GB" ] \
    || die "Há ${free} GB livres em ${dir}; são necessários ${MIN_DISK_GB} GB."
}

ensure_uv() {
  command -v uv >/dev/null && return
  msg "Instalando o uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null || die "Não foi possível instalar o uv."
}

cmd_install() {
  check_gpu
  check_compiler
  [ -x "$VENV_DIR/bin/vllm" ] || check_disk
  ensure_uv

  [ -x "$VENV_DIR/bin/python" ] || uv venv --python "$PYTHON_VERSION" "$VENV_DIR"

  msg "Instalando vLLM ${VLLM_VERSION} em ${VENV_DIR}..."
  uv pip install --python "$VENV_DIR/bin/python" "vllm==$VLLM_VERSION"

  "$VENV_DIR/bin/python" -c \
    'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' \
    || die "O PyTorch instalado não acessa a GPU. Atualize o driver NVIDIA."

  # Sem estes kernels, o FlashInfer tenta compilá-los ao iniciar o servidor e
  # falha em máquinas sem nvcc e ninja.
  msg "Instalando os kernels pré-compilados..."
  "$VENV_DIR/bin/vllm" download-kernels

  # O repositório guarda três cópias dos pesos; só a da raiz é usada pelo vLLM.
  msg "Baixando ${MODEL}..."
  "$VENV_DIR/bin/hf" download "$MODEL" --exclude "original/*" --exclude "metal/*"

  msg "Instalação concluída. Para validar: $0 test"
}

cmd_serve() {
  [ -x "$VENV_DIR/bin/vllm" ] || die "vLLM não instalado. Rode: $0 install"
  [ -z "${API_KEY:-}" ] || export VLLM_API_KEY="$API_KEY"
  # --enable-auto-tool-choice é obrigatório: sem ele, o /v1/chat/completions
  # descarta as chamadas de ferramenta. O vLLM avisa no log que ignora essa
  # flag no gpt-oss, mas o aviso só vale para a Responses API.
  exec "$VENV_DIR/bin/vllm" serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --reasoning-parser openai_gptoss \
    --tool-call-parser openai \
    --enable-auto-tool-choice \
    "$@"
}

stop_server() {
  [ -n "$SERVER_PID" ] || return 0
  msg "Encerrando o servidor de teste..."
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}

server_ready() { curl -fsS -m 2 "$1/health" >/dev/null 2>&1; }

start_test_server() {
  local base="$1" log elapsed=0
  shift
  log="$(mktemp -t gpt-oss-serve.XXXXXX.log)"
  msg "Subindo o servidor (log em ${log})..."
  cmd_serve "$@" >"$log" 2>&1 &
  SERVER_PID=$!
  trap stop_server EXIT

  until server_ready "$base"; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -n 30 "$log" >&2
      SERVER_PID=""
      die "O servidor terminou antes de ficar pronto."
    fi
    [ "$elapsed" -lt "$SERVER_TIMEOUT" ] \
      || die "O servidor não ficou pronto em ${SERVER_TIMEOUT}s. Veja ${log}."
    sleep 2
    elapsed=$((elapsed + 2))
  done
  msg "Servidor pronto em cerca de ${elapsed}s."
}

cmd_test() {
  local client_host="$HOST" base
  [ "$HOST" != "0.0.0.0" ] || client_host=127.0.0.1
  base="http://$client_host:$PORT"

  [ -x "$VENV_DIR/bin/python" ] || die "Ambiente não instalado. Rode: $0 install"
  if server_ready "$base"; then
    msg "Usando o servidor que já está rodando em ${base}."
  else
    start_test_server "$base" "$@"
  fi

  BASE_URL="$base" MODEL="$MODEL" API_KEY="${API_KEY:-}" "$VENV_DIR/bin/python" - <<'PY'
import json
import os
import sys
import time
import urllib.request

BASE_URL = os.environ["BASE_URL"]
MODEL = os.environ["MODEL"]
API_KEY = os.environ["API_KEY"]
failures = 0


def chat(**body):
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    request = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps({"model": MODEL, **body}).encode(),
        headers=headers,
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        data = json.load(response)
    return data, time.perf_counter() - start


def check(name, passed, detail):
    global failures
    failures += not passed
    print(f"[{'OK' if passed else 'FALHOU'}] {name}: {detail}")


data, seconds = chat(
    messages=[{"role": "user", "content": "Quanto é 17 + 25? Responda só com o número."}],
    max_tokens=512,
)
message = data["choices"][0]["message"]
content = message.get("content") or ""
reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
tokens = data["usage"]["completion_tokens"]
check("resposta", "42" in content, repr(content))
check("raciocínio separado", bool(reasoning) and "<|channel|>" not in content, f"{len(reasoning)} caracteres")
print(f"     {tokens} tokens em {seconds:.2f}s ({tokens / seconds:.0f} tokens/s)")

data, _ = chat(
    messages=[{"role": "user", "content": "Como está o clima em São Paulo agora?"}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Consulta o clima atual de uma cidade",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ],
    tool_choice="auto",
    max_tokens=512,
)
calls = data["choices"][0]["message"].get("tool_calls") or []
function = calls[0]["function"] if calls else {}
check(
    "chamada de ferramenta",
    function.get("name") == "get_weather"
    and "Paulo" in json.loads(function.get("arguments") or "{}").get("city", ""),
    f"{function.get('name')}({function.get('arguments')})",
)

sys.exit(1 if failures else 0)
PY
  msg "Todos os testes passaram."
}

case "${1:-}" in
  install | serve | test)
    command="$1"
    shift
    "cmd_$command" "$@"
    ;;
  *) usage ;;
esac
