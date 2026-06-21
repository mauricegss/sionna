# Estimativa de Canal Usando Preâmbulo OFDM com NVIDIA Sionna

> Trabalho acadêmico da disciplina **Tópicos em Redes sem Fio** — UTFPR Ponta Grossa

## Sobre

Este repositório contém o código-fonte, artigo e apresentação do trabalho **"Estimativa de Canal Usando Preâmbulo OFDM: Um Estudo Baseado em Simulações com a Biblioteca NVIDIA Sionna"**.

O trabalho investiga o desempenho de técnicas de estimativa de canal baseadas em preâmbulo (pilotos) em sistemas OFDM, comparando os estimadores **Least Squares (LS)** e **LMMSE** sob diferentes condições de canal (**AWGN** e **CDL-C**), utilizando a biblioteca open-source [NVIDIA Sionna](https://nvlabs.github.io/sionna/).

## Autores

- **Maurice Golin Soares dos Santos**
- **Vinicius Pereira Luz**
- **Natalia Mendes Goes**

**Professor:** Saulo Jorge Beltrão de Queiroz

## Estrutura do Repositório

```
sionna/
├── article/                          # Artigo e apresentação (LaTeX)
│   ├── artigo-final.tex             # Artigo completo final (IEEE format)
│   ├── artigo-etapa1.tex            # Etapa 1: Proposta inicial
│   ├── artigo-etapa2.tex            # Etapa 2: Materiais e métodos
│   ├── artigo-etapa3.tex            # Etapa 3: Fundamentação teórica
│   ├── slides-seminario.tex         # Slides do seminário (Beamer)
│   └── figs/                        # Figuras geradas pela simulação
│       ├── ber_awgn_vs_cdl.png      # Curvas BER × Eb/N₀
│       ├── canal_frequencia_real_vs_estimado.png
│       ├── constelacao_tx_vs_rx.png
│       └── grade_recursos_ofdm.png
│
├── main/                             # Simulação principal
│   ├── ofdm_channel_estimation.py   # Script principal (OFDM + LS + CDL-C)
│   └── output/                      # Saídas da simulação
│       ├── ber_awgn_vs_cdl.png
│       ├── canal_frequencia_real_vs_estimado.png
│       ├── constelacao_tx_vs_rx.png
│       ├── grade_recursos_ofdm.png
│       ├── relatorio_simulacao.txt  # Relatório completo em texto
│       └── resultados_ber.csv       # Dados BER em formato CSV
│
├── experiments/                      # Experimentos auxiliares
│   ├── hello-world.py               # Hello World: LDPC + QAM sobre AWGN
│   └── output/
│       ├── ber_curve.png
│       └── constellation.png
│
├── docs/                             # Documentação auxiliar
│   ├── requirements.md              # Requisitos e cronograma do trabalho
│   └── claude-analysis.md           # Análise dos exemplos oficiais do Sionna
│
├── requirements.txt                  # Dependências Python
└── README.md                         # Este arquivo
```

## Requisitos

- **Python** 3.10 ou superior
- **NVIDIA Sionna** 2.0+ (instala PyTorch automaticamente)
- **Matplotlib** (para geração de gráficos)
- **NumPy**
- **SciPy** (para curvas teóricas no hello-world)

### Hardware

- As simulações funcionam em **CPU** (sem necessidade de GPU)
- GPU NVIDIA com CUDA é opcional (acelera significativamente as simulações)

## Instalação

```bash
# 1. Clonar o repositório
git clone https://github.com/mauricegss/sionna.git
cd sionna

# 2. Criar ambiente virtual (recomendado)
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

# 3. Instalar dependências
pip install sionna
```

> **Nota:** O Sionna instala automaticamente o PyTorch e demais dependências.

## Execução

### Simulação principal (Estimativa de Canal OFDM)

```bash
python main/ofdm_channel_estimation.py
```

**O que faz:**
1. Configura um sistema OFDM completo (76 subportadoras, 14 símbolos/slot, 16-QAM, LDPC R=1/2)
2. Executa Cenário A (AWGN) e Cenário B (CDL-C) com estimativa LS + equalização LMMSE
3. Gera 4 gráficos + relatório TXT + dados CSV em `main/output/`

**Saídas:**
| Arquivo | Descrição |
|---|---|
| `ber_awgn_vs_cdl.png` | Curvas BER × Eb/N₀ para AWGN e CDL-C |
| `canal_frequencia_real_vs_estimado.png` | Resposta em frequência do canal (real vs estimada) |
| `constelacao_tx_vs_rx.png` | Diagrama de constelação 16-QAM (TX vs RX equalizado) |
| `grade_recursos_ofdm.png` | Visualização da grade de recursos OFDM |
| `relatorio_simulacao.txt` | Relatório completo com parâmetros e resultados |
| `resultados_ber.csv` | Dados BER em formato CSV |

### Experimento Hello World

```bash
python experiments/hello-world.py
```

**O que faz:** Demonstra a cadeia básica LDPC + 16-QAM sobre AWGN (sem OFDM), gerando curva BER e diagramas de constelação.

## Parâmetros da Simulação

| Parâmetro | Valor |
|---|---|
| FFT size | 76 |
| Subportadoras efetivas | 64 |
| Símbolos OFDM/slot | 14 |
| Espaçamento entre subportadoras | 30 kHz |
| Prefixo cíclico | 6 amostras |
| Modulação | 16-QAM (4 bits/símbolo) |
| Código FEC | LDPC 5G NR (k=1536, n=3072, R=1/2) |
| Pilotos | Kronecker (símbolos 2 e 11) |
| Batch size | 128 (196.608 bits/ponto) |
| Faixa Eb/N₀ | 0 a 15 dB |
| Canal CDL | Perfil C (NLOS), delay spread 100 ns, 3.5 GHz |

## Compilação do Artigo e Slides

O artigo e os slides estão em formato LaTeX. Para compilar:

```bash
cd article

# Artigo (IEEE format)
pdflatex artigo-final.tex
pdflatex artigo-final.tex   # Segundo passo para referências

# Slides (Beamer)
pdflatex slides-seminario.tex
```

Ou utilize o [Overleaf](https://www.overleaf.com/) para compilação online — basta fazer upload dos arquivos `.tex` e da pasta `figs/`.

## Resultados Principais

- **AWGN:** BER = 0 a partir de Eb/N₀ = 7 dB (efeito *waterfall* do LDPC)
- **CDL-C:** BER ≤ 10⁻³ a partir de Eb/N₀ ≈ 10 dB
- **Penalidade CDL-C vs AWGN:** ≈ 3 dB (consistente com a literatura)

## Referências

1. Hoydis, J. et al. "Sionna: An Open-Source Library for Next-Generation Physical Layer Research." *arXiv:2203.11854*, 2022.
2. Proakis, J. G.; Salehi, M. *Digital Communications*. 5ª ed. McGraw-Hill, 2008.
3. Goldsmith, A. *Wireless Communications*. Cambridge University Press, 2005.
4. van de Beek, J.-J. et al. "On channel estimation in OFDM systems." *Proc. IEEE VTC*, 1995.
5. Edfors, O. et al. "OFDM channel estimation by SVD." *IEEE Trans. Commun.*, 1998.
6. 3GPP TS 38.211 / TR 38.901.

## Licença

Uso acadêmico — UTFPR, 2026.
