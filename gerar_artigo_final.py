import os
import shutil

with open('article/artigo-etapa3.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update Title and Abstract
text = text.replace(
    r"\title{Estimativa de Canal Usando Preâmbulo OFDM: Um Estudo Baseado em Simulações com a Biblioteca NVIDIA Sionna}",
    r"\title{Estimativa de Canal Usando Preâmbulo OFDM e Receptores Neurais: Um Estudo Baseado em Simulações com a Biblioteca NVIDIA Sionna}"
)

# 2. Add to Methodology (Cenário de Teste)
cenarios_test_anchor = r"\subsection{Cenários de Teste}"
cenario_c = r"""
\subsubsection{Cenário C --- Receptor Neural (Deep Learning)}

Além da abordagem clássica, propõe-se um segundo objeto de estudo: a substituição do equalizador tradicional por um Receptor Neural treinado sob a perspectiva de \textit{Differentiable Communications}. O modelo consiste em um \textit{Multi-Layer Perceptron} (MLP) com camadas densas em PyTorch, que recebe os símbolos estimativos brutos e aprende empiricamente a mitigar o ruído e o \textit{fading} não-linear do canal CDL-C a $10$~dB. O treinamento ocorre de forma end-to-end utilizando a função de perda \textit{Mean Squared Error} (MSE) comparada com os símbolos perfeitos de transmissão.
"""
text = text.replace(cenarios_test_anchor, cenarios_test_anchor + cenario_c)

# 3. Add to Results (Before Conclusion)
conclusion_anchor = r"% ============================================" + "\n" + r"% CONCLUSÃO (ETAPA 4)"
new_results = r"""
\subsection{Desempenho do Receptor Neural e Análise de Custo Computacional}

O segundo objeto de estudo deste trabalho focou na utilização do Sionna para desenvolvimento de um Receptor Neural, contrastando-o com os métodos tradicionais de equalização. O modelo baseado em IA foi treinado por 50 épocas no cenário CDL-C com SNR de $10$~dB, provando que o aprendizado de máquina pode substituir os cálculos rígidos do \textit{PHY Layer}.

A Fig.~\ref{fig:neural_training} ilustra a evolução da curva de perda (MSE) durante o treinamento, evidenciando rápida convergência.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\columnwidth]{figs/treinamento_neural.png}
    \caption{Curva de convergência (MSE) durante o treinamento do Receptor Neural.}
    \label{fig:neural_training}
\end{figure}

A superioridade da técnica é verificada na Fig.~\ref{fig:neural_constellation}, que compara as constelações recuperadas. Enquanto o equalizador clássico apresenta extrema sobreposição inter-simbólica devido ao alto nível de ruído em $10$~dB, o Receptor Neural (em verde) agrupa a constelação nos 4 quadrantes de forma substancialmente superior e robusta.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\columnwidth]{figs/comparacao_neural_receiver.png}
    \caption{Comparação da constelação equalizada: LMMSE Clássico vs. Receptor Neural.}
    \label{fig:neural_constellation}
\end{figure}

Além da resiliência a ruído severo, o \textit{framework} de comunicações diferenciáveis oferece enorme redução do custo computacional. A estimativa clássica (LMMSE) impõe inversão de matrizes de canal com complexidade $O(N^3)$, o que resulta em alta latência para sistemas MIMO com múltiplas antenas. A rede neural, contudo, transfere todo este esforço para a fase de treinamento \textit{off-line}. A inferência durante a operação (na borda) recai em multiplicações matriciais paralelizadas $O(N^2)$, garantindo maior eficiência de bateria para os hardwares do $6$G.

"""
text = text.replace(conclusion_anchor, new_results + "\n" + conclusion_anchor)

# 4. Update Conclusion
conclusao_old = r"Como trabalhos futuros, sugere-se: (i) a implementação do estimador LMMSE completo para a estimativa de canal, quantificando o ganho em relação ao LS; (ii) a avaliação com diferentes ordens de modulação (QPSK, 64-QAM) e taxas de código; (iii) a extensão para configurações MIMO com múltiplas antenas; e (iv) a utilização de estimadores baseados em redes neurais (\textit{neural receivers}), aproveitando a capacidade de diferenciação automática do Sionna."
conclusao_new = r"A biblioteca Sionna provou ser excelente não apenas para replicar conceitos teóricos estabelecidos de telecomunicações, mas para consolidar novos paradigmas. Implementou-se com sucesso um Receptor Neural que superou as limitações da matemática clássica LMMSE sob condições ruidosas no modelo CDL-C. Como trabalhos futuros, sugere-se escalar esta arquitetura neural para cenários \textit{Massive MIMO} e constelações densas de ordem superior, mitigando as restrições energéticas e de banda das redes da próxima geração."
text = text.replace(conclusao_old, conclusao_new)

with open('article/artigo-final.tex', 'w', encoding='utf-8') as f:
    f.write(text)

# Copy the images to the article/figs/ folder
shutil.copyfile('extras/results/comparacao_neural_receiver.png', 'article/figs/comparacao_neural_receiver.png')
shutil.copyfile('extras/results/treinamento_neural.png', 'article/figs/treinamento_neural.png')

print("artigo-final.tex gerado com sucesso e imagens copiadas!")
