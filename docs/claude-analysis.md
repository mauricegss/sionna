# Análise de Exemplos Oficiais — NVIDIA Sionna PHY
## Simulação OFDM em Canal AWGN para Trabalho de Graduação

---

## 1. Visão Geral da Biblioteca Sionna

**Sionna** é uma biblioteca open-source desenvolvida pela NVIDIA para pesquisa em sistemas de comunicação na camada física (PHY). Atualmente na versão **2.0.1**, ela é construída sobre o TensorFlow/Keras, permitindo simulações aceleradas por GPU e diferenciáveis (backpropagation através de toda a cadeia de comunicação).

> [!NOTE]
> **Referência acadêmica principal:**
> Hoydis, J., Cammerer, S., Ait Aoudia, F., et al. "Sionna: An Open-Source Library for Next-Generation Physical Layer Research." arXiv:2203.11854, 2022.
> — [Paper no arXiv](https://arxiv.org/abs/2203.11854)

### Links Essenciais

| Recurso | URL |
|---|---|
| Documentação Oficial | https://nvlabs.github.io/sionna/ |
| Repositório GitHub | https://github.com/NVlabs/sionna |
| Tutoriais PHY (notebooks) | `tutorials/phy/` no repositório GitHub |
| Blog NVIDIA | https://developer.nvidia.com/blog/jumpstart-link-level-simulations-with-nvidia-sionna/ |
| Paper arXiv | https://arxiv.org/abs/2203.11854 |

---

## 2. Conceitos Teóricos Fundamentais

Antes de analisar os exemplos, é importante compreender os conceitos que serão encontrados em todos eles.

### 2.1 OFDM — Orthogonal Frequency-Division Multiplexing

OFDM é uma técnica de modulação multiportadora que divide a banda disponível em **N subportadoras ortogonais**, cada uma transportando dados em paralelo a uma taxa de símbolo mais baixa. Vantagens:

- **Resistência a multipercurso:** O prefixo cíclico (CP) elimina a interferência entre símbolos (ISI)
- **Eficiência espectral:** As subportadoras se sobrepõem sem interferência mútua
- **Implementação eficiente:** Usa IFFT/FFT para modulação/demodulação
- **Base do 4G/5G:** Utilizado em LTE, 5G NR, Wi-Fi (802.11a/n/ac/ax)

**Parâmetros principais:** tamanho da FFT, espaçamento entre subportadoras, comprimento do CP, padrão de pilotos.

### 2.2 AWGN — Additive White Gaussian Noise

O canal AWGN é o modelo de canal mais fundamental em comunicações digitais. Ele adiciona ruído branco gaussiano ao sinal transmitido:

$$y(t) = x(t) + n(t)$$

onde $n(t) \sim \mathcal{N}(0, N_0/2)$

- **Aditivo:** O ruído é somado ao sinal
- **Branco:** Densidade espectral de potência constante em todas as frequências
- **Gaussiano:** Amostras distribuídas segundo uma distribuição normal

### 2.3 SNR e Eb/N₀

- **SNR (Signal-to-Noise Ratio):** Razão entre a potência do sinal e a potência do ruído. Medido em dB
- **Eb/N₀:** Razão entre a energia por bit e a densidade espectral do ruído. Métrica normalizada independente da modulação e da taxa de código
- **Relação:** `SNR = Eb/N₀ × R × log₂(M)`, onde R é a taxa de código e M é a ordem da modulação

### 2.4 BER — Bit Error Rate

Taxa de erro de bit: proporção de bits recebidos incorretamente em relação ao total transmitido.

$$\text{BER} = \frac{\text{Bits errados}}{\text{Total de bits transmitidos}}$$

A curva **BER × Eb/N₀** é o resultado principal de simulações de link-level, permitindo comparar diferentes sistemas, modulações e códigos.

### 2.5 Codificação de Canal — LDPC 5G NR

Códigos LDPC (Low-Density Parity-Check) são usados no padrão 5G NR para os canais de dados. São códigos lineares com excelente desempenho próximo ao limite de Shannon.

### 2.6 Modulação QAM

Quadrature Amplitude Modulation mapeia bits em símbolos complexos no plano I/Q. Ordens comuns: QPSK (4-QAM), 16-QAM, 64-QAM, 256-QAM.

---

## 3. Exemplos Oficiais Identificados

Após análise extensiva da documentação, repositório GitHub, paper arXiv e blog NVIDIA, identifiquei **4 exemplos/cenários oficiais** relevantes para simulação OFDM + AWGN:

---

### 📘 Exemplo 1: Hello World — LDPC sobre AWGN com QAM (BICM Básico)

**Fonte:** Documentação oficial + Blog NVIDIA "Jumpstart Link-Level Simulations"

#### Objetivo da Simulação
Demonstrar a cadeia de comunicação mais básica do Sionna: transmissão de palavras-código LDPC sobre um canal AWGN usando modulação 16-QAM. É o "Hello, World!" da biblioteca.

#### Estrutura do Código

```python
import sionna as sn

# 1. Parâmetros
batch_size = 1024
n = 1000          # Comprimento da palavra-código
k = 500           # Bits de informação
m = 4             # Bits por símbolo (16-QAM)
snr_db = 10

# 2. Componentes (camadas Keras)
constellation = sn.mapping.Constellation("qam", m)
binary_source = sn.utils.BinarySource()
encoder = sn.fec.ldpc.LDPC5GEncoder(k, n)
mapper = sn.mapping.Mapper(constellation=constellation)
awgn = sn.channel.AWGN()
demapper = sn.mapping.Demapper("app", constellation=constellation)
decoder = sn.fec.ldpc.LDPC5GDecoder(encoder)

# 3. Cadeia de transmissão
b = binary_source([batch_size, k])
u = encoder(b)
x = mapper(u)
no = sn.utils.ebnodb2no(snr_db, num_bits_per_symbol=m, coderate=k/n)
y = awgn([x, no])
llr = demapper([y, no])
b_hat = decoder(llr)

# 4. Cálculo do BER
ber = sn.utils.metrics.compute_ber(b, b_hat)
```

#### Conceitos Teóricos Envolvidos
- Codificação de canal LDPC 5G NR
- Modulação QAM (mapeamento de bits para símbolos)
- Canal AWGN
- Demapeamento por APP (A Posteriori Probability)
- Decodificação iterativa LDPC (belief propagation)
- Cálculo de LLR (Log-Likelihood Ratios)
- BER e conversão Eb/N₀ → variância do ruído

#### Resultados e Gráficos Possíveis
- ✅ Curva BER × Eb/N₀ para diferentes modulações (QPSK, 16-QAM, 64-QAM)
- ✅ Comparação codificado vs. não-codificado
- ✅ Impacto da taxa de código (k/n) no desempenho
- ✅ Constelação transmitida vs. constelação recebida (diagrama de dispersão)
- ✅ Comparação com curvas teóricas de BER para QAM em AWGN

#### Avaliação de Dificuldade
| Aspecto | Nível |
|---|---|
| Complexidade do código | ⭐ Baixa (~30 linhas) |
| Conceitos necessários | ⭐⭐ Média (codificação de canal, modulação) |
| Requisitos de hardware | ⭐ Baixa (funciona em CPU e Google Colab) |
| Tempo de execução | ⭐ Baixo (segundos) |
| **Dificuldade geral** | **⭐⭐ Fácil** |

> [!TIP]
> Este é o exemplo **mais simples** para começar. Porém, não inclui OFDM diretamente — é BICM puro sobre AWGN. Para incluir OFDM, são necessários módulos adicionais (ResourceGrid, OFDMModulator, etc.).

---

### 📘 Exemplo 2: Neural Receiver para OFDM SIMO

**Fonte:** `tutorials/phy/Neural_Receiver.ipynb` no repositório GitHub

#### Objetivo da Simulação
Treinar um **receptor neural** (rede neural) que substitui os blocos clássicos de estimação de canal, equalização e demapeamento em um sistema OFDM SIMO (Single-Input Multiple-Output). O tutorial compara o receptor neural com baselines clássicas (LMMSE com CSI perfeita e estimação LS).

#### Estrutura do Código

```
1. Configuração do Sistema
   ├── ResourceGrid (grade de recursos OFDM)
   ├── PilotPattern (padrão de pilotos para estimação)
   ├── Parâmetros: FFT size, CP, número de símbolos OFDM
   └── Modelo de canal: CDL (3GPP) ou AWGN

2. Transmissor
   ├── BinarySource → LDPC5GEncoder
   ├── Mapper (QAM)
   └── ResourceGridMapper → OFDMModulator

3. Canal
   ├── OFDMChannel (aplica o modelo de canal)
   └── AWGN (alternativa simplificada)

4. Receptor Clássico (Baseline)
   ├── OFDMDemodulator
   ├── ChannelEstimator (LS/LMMSE)
   ├── Equalizer (LMMSE)
   ├── Demapper
   └── LDPC5GDecoder

5. Receptor Neural
   ├── Rede neural convolucional
   ├── Entrada: grade de recursos recebida (tempo-frequência)
   └── Saída: LLRs diretamente

6. Treinamento e Avaliação
   ├── Loop de treinamento com SGD
   ├── BER × SNR para diferentes receptores
   └── Comparação de curvas
```

#### Conceitos Teóricos Envolvidos
- **OFDM completo:** Resource Grid, pilotos, subportadoras, prefixo cíclico
- Canal AWGN e modelos 3GPP (CDL)
- Estimação de canal (Least Squares, LMMSE)
- Equalização LMMSE
- Deep Learning aplicado a comunicações (receptor neural)
- Codificação LDPC 5G NR
- Modulação QAM

#### Resultados e Gráficos Possíveis
- ✅ Curva BER × SNR: receptor neural vs. LMMSE (CSI perfeita) vs. LS
- ✅ Visualização da grade de recursos OFDM (pilotos + dados)
- ✅ Constelações recebidas antes/depois da equalização
- ✅ Resposta do canal no domínio da frequência
- ✅ Convergência do treinamento (loss vs. época)

#### Avaliação de Dificuldade
| Aspecto | Nível |
|---|---|
| Complexidade do código | ⭐⭐⭐⭐ Alta (>200 linhas, modelo Keras) |
| Conceitos necessários | ⭐⭐⭐⭐ Alta (OFDM, estimação de canal, Deep Learning) |
| Requisitos de hardware | ⭐⭐⭐ Média-Alta (GPU recomendada para treinamento) |
| Tempo de execução | ⭐⭐⭐ Médio (minutos para treinar) |
| **Dificuldade geral** | **⭐⭐⭐⭐ Difícil** |

> [!WARNING]
> Este tutorial é **excelente e completo**, mas envolve Deep Learning, o que pode ser um desvio da proposta de comunicações digitais. É melhor como extensão avançada do que como trabalho principal.

---

### 📘 Exemplo 3: Sistema OFDM End-to-End sobre AWGN (Construção Personalizada)

**Fonte:** Documentação da API OFDM + Discussões GitHub + Blog NVIDIA

#### Objetivo da Simulação
Construir do zero um sistema OFDM completo sobre canal AWGN, combinando os módulos do Sionna: geração de bits → codificação LDPC → mapeamento QAM → grade de recursos OFDM → modulação OFDM (IFFT + CP) → canal AWGN → demodulação OFDM (FFT) → demapeamento → decodificação → cálculo de BER.

Este cenário combina o Exemplo 1 (BICM + AWGN) com os módulos OFDM do Exemplo 2, resultando em um sistema que atende **exatamente** à proposta do professor.

#### Estrutura do Código

```python
import sionna as sn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# 1. PARÂMETROS DO SISTEMA
# ============================================
batch_size       = 256
fft_size         = 64          # Número de subportadoras
num_ofdm_symbols = 14          # Símbolos OFDM por slot
cyclic_prefix_length = 16      # Comprimento do CP
num_bits_per_symbol = 4        # 16-QAM
k = 500                        # Bits de informação
n = 1000                       # Comprimento da palavra-código
coderate = k / n

# ============================================
# 2. CONFIGURAÇÃO DA GRADE DE RECURSOS
# ============================================
resource_grid = sn.ofdm.ResourceGrid(
    num_ofdm_symbols=num_ofdm_symbols,
    fft_size=fft_size,
    subcarrier_spacing=15e3,     # 15 kHz (LTE/5G)
    cyclic_prefix_length=cyclic_prefix_length
)

# ============================================
# 3. COMPONENTES DO TRANSMISSOR
# ============================================
binary_source = sn.utils.BinarySource()
encoder       = sn.fec.ldpc.LDPC5GEncoder(k, n)
constellation = sn.mapping.Constellation("qam", num_bits_per_symbol)
mapper        = sn.mapping.Mapper(constellation=constellation)
rg_mapper     = sn.ofdm.ResourceGridMapper(resource_grid)
ofdm_mod      = sn.ofdm.OFDMModulator(cyclic_prefix_length)

# ============================================
# 4. CANAL
# ============================================
awgn = sn.channel.AWGN()

# ============================================
# 5. COMPONENTES DO RECEPTOR
# ============================================
ofdm_demod   = sn.ofdm.OFDMDemodulator(fft_size, cyclic_prefix_length)
rg_demapper  = sn.ofdm.ResourceGridDemapper(resource_grid)
demapper     = sn.mapping.Demapper("app", constellation=constellation)
decoder      = sn.fec.ldpc.LDPC5GDecoder(encoder)

# ============================================
# 6. SIMULAÇÃO: BER × Eb/N₀
# ============================================
snr_range = np.linspace(0, 12, 13)
ber_results = []

for snr_db in snr_range:
    # Transmissor
    b = binary_source([batch_size, k])
    u = encoder(b)
    x = mapper(u)
    x_rg = rg_mapper(x)            # Mapeia na grade de recursos
    x_ofdm = ofdm_mod(x_rg)        # Modulação OFDM (IFFT + CP)

    # Canal AWGN
    no = sn.utils.ebnodb2no(snr_db, num_bits_per_symbol, coderate)
    y_ofdm = awgn([x_ofdm, no])

    # Receptor
    y_rg = ofdm_demod(y_ofdm)      # Demodulação OFDM (remove CP + FFT)
    y = rg_demapper(y_rg)           # Extrai elementos de dados
    llr = demapper([y, no])
    b_hat = decoder(llr)

    # BER
    ber = sn.utils.metrics.compute_ber(b, b_hat)
    ber_results.append(ber.numpy())

# ============================================
# 7. GRÁFICO BER × Eb/N₀
# ============================================
plt.figure(figsize=(10, 6))
plt.semilogy(snr_range, ber_results, 'bo-', label='OFDM + LDPC + 16QAM')
plt.xlabel('Eb/N₀ (dB)')
plt.ylabel('BER')
plt.title('Desempenho BER do Sistema OFDM sobre Canal AWGN')
plt.grid(True, which='both', alpha=0.3)
plt.legend()
plt.show()
```

> [!IMPORTANT]
> Este código é uma **estrutura referencial** que combina módulos documentados do Sionna. A API exata pode variar conforme a versão (1.x vs 2.0). Consulte a documentação da versão instalada para verificar nomes de classes e parâmetros.

#### Conceitos Teóricos Envolvidos
- **Todos os conceitos centrais:** OFDM, AWGN, BER, SNR/Eb/N₀
- Grade de recursos (Resource Grid) — estrutura tempo-frequência do OFDM
- Modulação/Demodulação OFDM via IFFT/FFT
- Prefixo cíclico e sua função contra ISI
- Codificação de canal LDPC 5G NR
- Mapeamento QAM e cálculo de LLR
- Conversão Eb/N₀ → variância do ruído

#### Resultados e Gráficos Possíveis
- ✅ **Curva BER × Eb/N₀** — resultado principal
- ✅ Comparação entre modulações (QPSK vs. 16-QAM vs. 64-QAM)
- ✅ Comparação codificado (LDPC) vs. não-codificado
- ✅ Impacto do tamanho da FFT no desempenho
- ✅ Impacto do comprimento do CP
- ✅ Diagrama de constelação transmitido vs. recebido
- ✅ Espectro do sinal OFDM no domínio do tempo e da frequência
- ✅ Visualização da grade de recursos

#### Avaliação de Dificuldade
| Aspecto | Nível |
|---|---|
| Complexidade do código | ⭐⭐ Média (~80 linhas) |
| Conceitos necessários | ⭐⭐⭐ Média (OFDM, codificação, modulação) |
| Requisitos de hardware | ⭐ Baixa (funciona em CPU e Google Colab) |
| Tempo de execução | ⭐ Baixo (segundos a poucos minutos) |
| **Dificuldade geral** | **⭐⭐⭐ Moderada** |

---

### 📘 Exemplo 4: Superimposed Pilots para OFDM

**Fonte:** `tutorials/phy/Superimposed_Pilots.ipynb` no repositório GitHub

#### Objetivo da Simulação
Explorar o conceito de **pilotos sobrepostos** (superimposed pilots) em sistemas OFDM, onde os sinais de referência são transmitidos simultaneamente com os dados (em vez de em subportadoras/slots dedicados). Compara o desempenho com padrões de pilotos convencionais.

#### Conceitos Teóricos Envolvidos
- OFDM com pilotos convencionais vs. sobrepostos
- Estimação de canal
- Trade-off entre overhead de pilotos e eficiência espectral
- Canal AWGN e modelos com desvanecimento

#### Avaliação de Dificuldade
| Aspecto | Nível |
|---|---|
| Complexidade do código | ⭐⭐⭐⭐ Alta |
| Conceitos necessários | ⭐⭐⭐⭐ Alta (pilotos, estimação de canal avançada) |
| **Dificuldade geral** | **⭐⭐⭐⭐ Difícil** |

> [!CAUTION]
> Este exemplo é **muito especializado** para um trabalho de graduação introdutório. Recomendado apenas se o professor direcionar explicitamente para o tema de estimação de canal e pilotos.

---

## 4. Tabela Comparativa dos Exemplos

| Critério | Exemplo 1: Hello World | Exemplo 2: Neural Rx | Exemplo 3: OFDM E2E | Exemplo 4: Pilots |
|---|---|---|---|---|
| **Inclui OFDM?** | ❌ Não | ✅ Sim | ✅ Sim | ✅ Sim |
| **Inclui AWGN?** | ✅ Sim | ✅ Sim | ✅ Sim | ✅ Sim |
| **Gera BER×SNR?** | ✅ Sim | ✅ Sim | ✅ Sim | ✅ Sim |
| **Codificação canal** | ✅ LDPC 5G | ✅ LDPC 5G | ✅ LDPC 5G | ✅ Sim |
| **Dificuldade** | ⭐⭐ Fácil | ⭐⭐⭐⭐ Difícil | ⭐⭐⭐ Moderada | ⭐⭐⭐⭐ Difícil |
| **Deep Learning?** | ❌ Não | ✅ Sim | ❌ Não | ❌ Não |
| **Aderência ao tema** | ⚠️ Parcial | ⚠️ Vai além | ✅ Exata | ⚠️ Tangencial |
| **Executável em Colab** | ✅ Sim | ✅ Sim (GPU) | ✅ Sim | ✅ Sim |

---

## 5. ⭐ Recomendação: Exemplo 3 — Sistema OFDM End-to-End sobre AWGN

> [!IMPORTANT]
> **O Exemplo 3 é o mais adequado para seu trabalho de graduação**, pois:
> 1. Atende **exatamente** à proposta do professor (OFDM + AWGN)
> 2. Tem **complexidade moderada** — desafiador o suficiente para um TCC, mas viável
> 3. Cobre **todos os conceitos da disciplina** (OFDM, AWGN, BER, SNR, codificação, modulação)
> 4. Gera **gráficos publicáveis** (curvas BER, constelações, espectros)
> 5. **Não exige Deep Learning**, mantendo o foco em comunicações digitais
> 6. Pode ser **incrementalmente expandido** para gerar conteúdo adicional

### Estratégia Recomendada

Comece pelo **Exemplo 1** (Hello World) para se familiarizar com o Sionna, e então evolua para o **Exemplo 3** adicionando os módulos OFDM.

---

## 6. Roteiro para Expansão em Artigo Científico

### Título Sugerido
**"Avaliação de Desempenho de um Sistema OFDM com Codificação LDPC em Canal AWGN Utilizando a Biblioteca NVIDIA Sionna"**

### Estrutura do Artigo

#### 1. Introdução (~1 página)
- Contextualização dos sistemas OFDM no cenário de comunicações modernas (4G/5G, Wi-Fi)
- Importância de ferramentas de simulação de link-level para ensino e pesquisa
- Apresentação da biblioteca Sionna e suas vantagens (GPU, diferenciabilidade, open-source)
- Objetivo: avaliar o desempenho BER de um sistema OFDM-LDPC em canal AWGN
- Contribuições: reprodução e extensão de cenários de simulação com uma ferramenta moderna

#### 2. Fundamentação Teórica (~2–3 páginas)
- **2.1 OFDM:** Princípio da multiportadora, IFFT/FFT, prefixo cíclico, grade de recursos
- **2.2 Canal AWGN:** Modelo matemático, caracterização estatística
- **2.3 Modulação QAM:** Mapeamento de bits, diagrama de constelação, BER teórico
- **2.4 Codificação LDPC:** Princípio dos códigos esparsos, grafo de Tanner, decodificação BP
- **2.5 Métricas:** BER, Eb/N₀, SNR, conversões

#### 3. Materiais e Métodos (~1–2 páginas)
- **3.1 Ambiente:** Python 3.x, TensorFlow, Sionna 2.0.x, Google Colab / PC local
- **3.2 Parâmetros da simulação:** tabela completa com todos os parâmetros
- **3.3 Cenários investigados:**
  - Cenário A: Impacto da ordem de modulação (QPSK, 16-QAM, 64-QAM)
  - Cenário B: Impacto da taxa de código LDPC (1/3, 1/2, 3/4)
  - Cenário C: Codificado vs. não-codificado
  - Cenário D: Comparação com curva teórica (BER analítica)
  - Cenário E (bônus): Impacto do tamanho da FFT

#### 4. Resultados e Discussão (~2–3 páginas)
Para cada cenário:
- Gráfico BER × Eb/N₀
- Análise do ganho de codificação
- Comparação com valores teóricos
- Diagramas de constelação em diferentes SNRs
- Tabelas com métricas quantitativas

**Gráficos esperados (mínimo):**
1. BER × Eb/N₀ para QPSK, 16-QAM e 64-QAM (3 curvas no mesmo gráfico)
2. BER × Eb/N₀ codificado vs. não-codificado
3. BER × Eb/N₀ para diferentes taxas de código
4. Diagrama de constelação recebida em SNR baixo, médio e alto
5. Sinal OFDM no domínio do tempo

#### 5. Conclusão (~0.5–1 página)
- Resumo dos resultados obtidos
- Validação contra valores teóricos esperados
- Benefícios da utilização do Sionna para ensino e pesquisa
- Trabalhos futuros: canais com desvanecimento (Rayleigh, CDL), MIMO, receptores neurais

#### 6. Referências
Referências mínimas sugeridas:

| # | Referência |
|---|---|
| 1 | Hoydis, J. et al. "Sionna: An Open-Source Library for Next-Generation Physical Layer Research." arXiv:2203.11854, 2022. |
| 2 | Proakis, J. G.; Salehi, M. *Digital Communications*. 5th ed. McGraw-Hill, 2008. |
| 3 | Goldsmith, A. *Wireless Communications*. Cambridge University Press, 2005. |
| 4 | Haykin, S. *Communication Systems*. 4th ed. Wiley, 2001. |
| 5 | 3GPP TS 38.211. "NR; Physical channels and modulation." |
| 6 | Richardson, T.; Urbanke, R. *Modern Coding Theory*. Cambridge University Press, 2008. |
| 7 | Documentação oficial NVIDIA Sionna — https://nvlabs.github.io/sionna/ |
| 8 | NVIDIA Technical Blog — "Jumpstart Link-Level Simulations with NVIDIA Sionna" |

---

## 7. Dicas de Execução

### Google Colab (Recomendado para Início)
```python
# Primeira célula do notebook no Colab:
!pip install sionna
import sionna
print(f"Sionna version: {sionna.__version__}")
```

### Execução Local
```bash
# Criar ambiente virtual
python -m venv sionna_env
sionna_env\Scripts\activate      # Windows
pip install sionna tensorflow
```

> [!TIP]
> **Para o artigo:** Execute as simulações finais com `batch_size` grande (ex: 4096+) e múltiplas repetições para obter curvas BER suaves e estatisticamente confiáveis. Use `tf.function` para acelerar a execução.

---

## 8. Diagrama da Cadeia de Comunicação

```mermaid
flowchart LR
    A["Fonte Binária"] --> B["Codificador LDPC"]
    B --> C["Mapeador QAM"]
    C --> D["Resource Grid Mapper"]
    D --> E["Modulador OFDM\n(IFFT + CP)"]
    E --> F["Canal AWGN"]
    F --> G["Demodulador OFDM\n(Remove CP + FFT)"]
    G --> H["Resource Grid Demapper"]
    H --> I["Demapeador QAM\n(LLR)"]
    I --> J["Decodificador LDPC"]
    J --> K["Cálculo BER"]

    style A fill:#4CAF50,color:#fff
    style F fill:#f44336,color:#fff
    style K fill:#2196F3,color:#fff
```

---

## 9. Cronograma Sugerido de Execução

| Semana | Atividade |
|---|---|
| 1 | Instalar Sionna, executar Hello World (Exemplo 1) |
| 2 | Estudar módulos OFDM, montar Exemplo 3 básico |
| 3 | Gerar curvas BER × Eb/N₀ para diferentes modulações |
| 4 | Adicionar variações (taxa de código, codificado vs. não-codificado) |
| 5 | Gerar diagramas de constelação e espectros |
| 6 | Escrever introdução + fundamentação teórica |
| 7 | Escrever materiais e métodos + resultados |
| 8 | Revisão, conclusão, preparar apresentação |
