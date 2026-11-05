# gpt-oss-20b com vLLM

Script para instalar, servir e testar o modelo [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) com o [vLLM](https://github.com/vllm-project/vllm) em uma máquina Linux com GPU NVIDIA. O servidor oferece uma API compatível com a da OpenAI, incluindo raciocínio separado da resposta e chamada de ferramentas.

## Requisitos

- Linux x86_64 com GPU NVIDIA e driver compatível com CUDA 13
- Pelo menos 23 GB de VRAM
- 40 GB livres em disco
- `curl` e um compilador C (`gcc` ou `clang`); no Ubuntu: `sudo apt install curl build-essential`

Python e `uv` não são pré-requisitos: o script instala o que faltar. Ele não usa `sudo` e não altera o Python do sistema, porque tudo fica em um ambiente virtual próprio.

## Uso

```bash
git clone https://github.com/ssmvictor/gpt-oss-vllm-setup.git
cd gpt-oss-vllm-setup

./gpt-oss.sh install   # prepara o ambiente e baixa o modelo
./gpt-oss.sh test      # sobe o servidor, testa e o encerra
./gpt-oss.sh serve     # deixa o servidor rodando em primeiro plano
```

O comando `test` verifica a resposta do modelo, a separação do raciocínio e a chamada de ferramentas. Se já houver um servidor na porta configurada, ele usa esse servidor e não o encerra no final.

Com o servidor no ar, em outro terminal:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"openai/gpt-oss-20b","messages":[{"role":"user","content":"Olá!"}]}'
```

Qualquer cliente da OpenAI funciona apontando para `http://localhost:8000/v1`.

## Configuração

As opções são variáveis de ambiente:

| Variável | Padrão | Função |
| --- | --- | --- |
| `VENV_DIR` | `~/vllm-env` | Pasta do ambiente virtual |
| `HF_HOME` | `~/.cache/huggingface` | Pasta do cache do modelo |
| `HOST` | `127.0.0.1` | Endereço do servidor |
| `PORT` | `8000` | Porta do servidor |
| `API_KEY` | vazio | Exige essa chave nas requisições |
| `VLLM_VERSION` | `0.33.0` | Versão do vLLM |
| `PYTHON_VERSION` | `3.13` | Versão do Python do ambiente |
| `MIN_VRAM_MIB` | `23000` | VRAM mínima exigida na instalação |
| `MIN_DISK_GB` | `40` | Espaço livre mínimo exigido na instalação |
| `SERVER_TIMEOUT` | `600` | Segundos de espera pelo servidor no `test` |

Em `serve` e `test`, os argumentos extras vão direto para o `vllm serve`:

```bash
./gpt-oss.sh serve --gpu-memory-utilization 0.5
```

## Acesso pela rede

Por padrão, o servidor só aceita conexões da própria máquina. Para expô-lo na rede, defina uma chave de API:

```bash
HOST=0.0.0.0 API_KEY=sua-chave ./gpt-oss.sh serve
```

Os clientes passam a enviar o cabeçalho `Authorization: Bearer sua-chave`. A conexão não é criptografada, então use uma rede confiável ou coloque um proxy HTTPS na frente.

## Desempenho medido

Em uma NVIDIA A100 de 80 GB, com 400 tokens por requisição e o servidor já aquecido:

| Requisições simultâneas | Tokens/s no total | Tokens/s por requisição | Primeiro token (mediana) |
| --- | --- | --- | --- |
| 1 | 233 | 234 | 27 ms |
| 8 | 1.179 | 148 | 65 ms |
| 32 | 3.270 | 103 | 126 ms |

Na mesma máquina, a instalação do zero levou menos de 3 minutos, e o servidor subiu em cerca de 2 minutos na primeira vez e 75 segundos nas seguintes. O ambiente ocupa 19 GB e o modelo, 13 GB.

## Comportamentos esperados

- **Muita VRAM em uso:** o vLLM reserva 92% da memória da GPU ao iniciar, cerca de 75 GB em uma A100. O que sobra depois de carregar o modelo vira cache das requisições. Para reservar menos, use `--gpu-memory-utilization`.
- **Primeira resposta mais lenta:** a primeira requisição depois de subir o servidor compila kernels e pode levar alguns segundos.
- **Avisos no log:** `we ignore --enable-auto-tool-choice` vale só para a Responses API; essa flag é necessária em `/v1/chat/completions`. O aviso `Auto-initialization of reasoning token IDs failed` não impediu a separação do raciocínio nos testes.

## Compatibilidade testada

Testado no Ubuntu 24.04 com Python 3.13, driver NVIDIA 580.82.07 e uma A100 de 80 GB, inclusive em um ambiente limpo, com pastas e caches vazios.

O limite de 23 GB foi simulado nessa A100, restringindo a memória disponível ao vLLM, e não em uma placa com essa capacidade. Na simulação de uma GPU de 16 GB, o servidor não subiu, mesmo com contexto reduzido.

## Desinstalação

```bash
rm -rf ~/vllm-env ~/.cache/huggingface/hub/models--openai--gpt-oss-20b ~/.cache/vllm
```
