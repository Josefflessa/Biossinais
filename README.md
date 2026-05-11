# Aquisição de Biossinais – Projeto da Disciplina

**Universidade Federal do Ceará (UFC)
Departamento de Engenharia de Teleinformática
Curso de Engenharia de Computação**

**Equipe:** José Ferreira Lessa \& Matheus Rocha Gomes da Silva
**Orientador:** Prof. Dr. Victor Hugo C. de Albuquerque
**Semestre:** 2026.1

\---

## Objetivo

Preparar um dataset de biossinais (ECG) para aplicação de técnicas de Reconhecimento de Padrões (RP), percorrendo todo o pipeline de aquisição, pré-processamento, extração de atributos e validação estatística.

## Dataset

**PTB-XL** — Base de Dados de Eletrocardiografia

* 21.799 registros de ECG de 12 derivações (10 s cada)
* 18.869 pacientes
* 71 classes diagnósticas / 5 superclasses (NORM, MI, STTC, CD, HYP)
* Taxa de amostragem utilizada: **100 Hz** (decimação do sinal original de 500 Hz)
* Disponível em: https://physionet.org/content/ptb-xl/1.0.3/

\---

## Pipeline da Disciplina

|#|Etapa|Notebook|Status|
|-|-|-|-|
|0|Estudo MNE + Surveys + Seleção de base|—|✅ Concluído|
|1|Aquisição dos Biossinais|`01\_aquisicao\_biossinais.ipynb`|✅ Concluído|
|2|Avaliação da Qualidade do Sinal (SQI)|`02\_qualidade\_sinal\_SQI.ipynb`|✅ Concluído|
|3|Análise Estatística Inicial|`03\_estatistica\_inicial.ipynb`|✅ Concluído|
|4|Limpeza e Correção dos Dados|`04\_limpeza\_dados.ipynb`|✅ Concluído|
|5|Segmentação (Janelamento)|`05\_segmentacao.ipynb`|✅ Concluído|
|6|Extração de Atributos|`06\_extracao\_features\_completo.ipynb`|✅ Concluído|
|7|Engenharia de Features|`07\_engenharia\_features\_completo.ipynb`|✅ Concluído|
|8|Redução de Dimensionalidade|`08\_reducao\_dimensionalidade\_completo.ipynb`|✅ Concluído|
|9|Seleção de Atributos|`09\_selecao\_atributos\_completo.ipynb`|✅ Concluído|
|10|Dataset Final Validado|`10\_validacao\_estatistica\_final.ipynb`|✅ Concluído|
|F|Relatório Final Integrador (TCC)|—|⬜ Pendente|

\---

## Descrição dos Entregáveis

### Entregável 1 — Aquisição dos Biossinais

Contextualização do dataset PTB-XL e inspeção dos sinais brutos de ECG. Cobre a fundamentação clínica do ECG de 12 derivações, o protocolo original de aquisição (Schiller AG, PTB Berlin, 1989–2001), a justificativa da taxa de amostragem de 100 Hz via Teorema de Nyquist, a análise demográfica (distribuição de idade, sexo, peso e altura), a distribuição de superclasses diagnósticas com suporte a multi-label, a verificação anti-leakage entre folds e a análise espectral dos sinais brutos via método de Welch. Produz `ptbxl\_metadata\_enriched.csv` com metadados enriquecidos e coluna `split` (treino/val/teste).

\---

### Entregável 2 — Avaliação da Qualidade do Sinal (SQI)

Quantificação objetiva da qualidade dos sinais de ECG por meio de métricas SQI (*Signal Quality Index*). Implementa métricas de ordem superior (hosSQI), domínio da frequência (pSQI, basSQI, SNR, entropia espectral), continuidade (flat ratio, qSQI) e multicanal (rinter). Aplica um classificador heurístico em cascata (*Gatekeeper*) que classifica cada registro em quatro categorias: **G** (Excelente), **A** (Aceitável), **P** (Processável) e **U** (Inaceitável). Os limiares são otimizados empiricamente com variação de ±10% em relação aos valores da literatura. Produz `ptbxl\_com\_sqi.csv` e arquivos separados para registros rejeitados.

\---

### Entregável 3 — Análise Estatística Inicial

Análise diagnóstica e metodológica para qualificar os dados e orientar decisões analíticas futuras. Inclui estatística descritiva dos metadados por superclasse (idade, peso, altura), análise temporal do volume de coleta e adoção de laudos automáticos (1989–2001), extração de métricas do sinal (RMS, MAV, variância, P2P, energia, skewness, kurtosis) por derivação e superclasse a partir de amostragem estratificada. Aplica testes de normalidade (Shapiro-Wilk, Kolmogorov-Smirnov) e homocedasticidade (Levene, Bartlett), seguidos de análise de correlação por Pearson e Spearman. Produz `estatistica\_inicial\_resultados.csv`.

\---

### Entregável 4 — Limpeza e Filtragem Digital dos Sinais

Implementação do pipeline de pré-processamento com intervenção direta nos sinais. Aplica filtragem digital Butterworth bidirecional (`filtfilt`) em duas etapas: **passa-alta** (0,5 Hz, ordem 5) para remoção de drift de baseline e **passa-baixa** (40 Hz, ordem 5) para atenuar ruído muscular. A interferência de 50 Hz não requer filtro notch — já foi atenuada pelo processo de downsampling original (anti-aliasing). A **winsorização** por percentis (0,5%–99,5%) por derivação controla outliers de amplitude, com limites calculados exclusivamente nos dados de treino (folds 1–8) para evitar data leakage. A validação por Wilcoxon pareado e Cohen's d confirma redução significativa de energia em todos os grupos com artefato anotado. A reavaliação SQI pós-limpeza mostra redução expressiva da classe P e surgimento de registros G. Parâmetros serializados em `preprocessing\_params.pkl`. Produz `sinais\_bons\_100hz.npy` (G/A) e `sinais\_ruins\_100hz.npy` (P/U).

\---

### Entregável 5 — Segmentação (Janelamento)

Definição e aplicação da estratégia de segmentação dos sinais, adotando uma **arquitetura híbrida em dois níveis**:

* **Nível 1 — Registro de 10 s:** instância principal (1.000 amostras × 12 derivações), preservando a unidade clínica original do PTB-XL, independência amostral e ausência de label inflation.
* **Nível 2 — Batimento (600 ms):** janelas de 60 amostras centradas no pico R, obtidas pelo detector **Pan-Tompkins** (passa-banda 5–15 Hz → derivada → quadratura → janela móvel 150 ms → período refratário adaptativo). Cada batimento cobre 200 ms pré-pico + 400 ms pós-pico, capturando ondas P, QRS e T.

A validação inclui análise de estabilidade intra-janela (ZCR e variância), variância inter-janela (Kruskal-Wallis, p < 0,001), sobreposição visual de batimentos por classe e diagramas de Poincaré dos intervalos RR. Produz `batimentos\_segmentados.npy` (211.442 batimentos × 60 amostras × 12 derivações), `registros\_ids.csv`, `batimentos\_ids.csv` e `metricas\_inter\_janela.csv`.

\---

### Entregável 6 — Extração de Atributos

Transformação dos sinais em vetores numéricos de features distribuídas em cinco domínios complementares:

|Domínio|Técnica|Derivações|Features|
|-|-|-|-|
|**Tempo**|RMS, MAV, variância, P2P, ZCR, skewness, kurtosis|12|84|
|**Frequência**|PSD via Welch (nperseg=256, Hann), potência por banda (PT 0,5–5 Hz, QRS 5–25 Hz), razão QRS/Total, frequência de pico/mediana/centroide|12|84|
|**Morfologia + HRV**|Amplitude R, largura QRS, segmento ST, amplitude T, assimetria QRS (por batimento → agregado), SDNN, RMSSD, meanRR, CV\_RR|DII e V5|\~50|
|**Tempo-Frequência**|DWT Daubechies db4, 4 níveis (A4, D4, D3, D2, D1): energia, energia relativa, entropia de Shannon, razão QRS|DII e V5|\~30|
|**Não-Linear**|Higuchi FD (kmax=10), DFA (expoente α), SampEn (m=2, r=0,2σ), SD1, SD2, razão SD1/SD2|DII e série RR|6|

O sanity check confirma ausência de NaN nas features de sinal e concentração dos valores ausentes (<5%) nas métricas não-lineares da série RR (registros com < 8 batimentos). Produz `features\_raw.parquet` e `feature\_catalog.csv`.

\---

### Entregável 7 — Engenharia de Features

Preparação final do vetor de atributos para modelagem. Realiza quatro frentes:

1. **Limpeza:** remoção de features quasi-constantes (variância < 1e-6) e com NaN > 5%; imputação dos NaN residuais pela mediana da superclasse no treino.
2. **Features de segunda ordem:**

   * *Razões espectrais* — QRS/PT e QRS/Total por derivação (12 × 2 = 24 features adimensionais)
   * *Normalização por baseline NORM* — RMS e MAV divididos pela mediana do grupo NORM no treino (24 features)
   * *Deltas morfológicos* — diferença de amplitude entre derivações complementares (I–aVR, II–aVL, V1–V6, V2–V5) e delta wavelet D3–D4
3. **Normalização robusta:** `RobustScaler` fitado exclusivamente nos folds 1–8 (centraliza pela mediana, escala pelo IQR).
4. **Validação:** ANOVA F-statistic one-vs-rest por classe e análise de redundância (correlação de Pearson, pares com |r| > 0,90).

Produz `features\_engineered.parquet`, `feature\_catalog\_e7.csv` e `scaler\_robust.pkl`.

**Destaques do ranking ANOVA:** `time\_skew\_\*` (CD dominante), `freq\_total\_power\_\*` e `wavelet\_energy\_D3\_\*` (HYP), `morph\_t\_amp\_\*` (STTC/repolarização).

\---

### Entregável 8 — Redução de Dimensionalidade

Compressão do espaço de features por duas técnicas lineares aplicadas em paralelo:

**PCA:** StandardScaler (fit treino) → PCA completo (SVD). Critérios avaliados: cotovelo (\~10–20 PCs), Kaiser (37 PCs) e variância acumulada ≥ 95% **(K = 79, compressão de \~72%)**. Análise de loadings dos 10 primeiros PCs revela que PC1 captura amplitude global, PC2/PC3 contrastes morfológicos de QRS e PCs superiores isolam HRV e complexidade fractal. Projeção 2D com elipses de confiança (1σ) evidencia sobreposição significativa entre classes, confirmando que a separabilidade está distribuída em múltiplos componentes. Validação por RMSE de reconstrução (≈0,21 no espaço padronizado), estável entre treino/validação/teste.

**FastICA:** Aplicado sobre os 20 primeiros PCs. Avaliação dos ICs por ANOVA F-score e não-gaussianidade (kurtosis). ICs com baixo F-score e baixa kurtosis (ex.: IC20, IC13, IC6) são candidatos a descarte.

Produz `features\_pca.parquet` (K = 79 PCs) e `pca\_pipeline.pkl`.

\---

### Entregável 9 — Seleção de Atributos

Seleção formal do subconjunto de features originais com maior poder discriminativo, preservando interpretabilidade clínica. Aplica três famílias de métodos:

**Filter Methods:**

* ANOVA F-score — separação linear entre classes
* Informação Mútua (k-NN Kraskov) — dependências não-lineares
* ReliefF (amostra 5.000 instâncias, k=10) — relevância por vizinhança

**Wrapper Methods:**

* SFS — Sequential Forward Selection (LogisticRegression L2, 5-fold CV, top-50 candidatas → 20 features)
* SBE — Sequential Backward Elimination (mesma configuração; interseção SFS∩SBE = 11 features)

**Embedded Methods:**

* LASSO (LogisticRegressionCV L1, saga, C=0,1 ótimo por CV) — 246 de 282 features ativas
* Random Forest (300 árvores, max\_depth=15, balanced) — importância Gini distribuída (\~221 features para 90% da importância)

**Validação estatística:** Kruskal-Wallis + correção FDR-BH + Eta-quadrado (η²). Todas as 127 features candidatas permaneceram significativas após Bonferroni e FDR-BH. Distribuição por η²: 28 efeitos grandes, 79 médios, 2 pequenos.

**Ranking final:** Borda Count ponderado (LASSO e RF: peso 2,0; filtros, wrapper e η²: peso 1,5–2,0). Subconjunto final: **30 features** (redução de 89,4% em relação às 282 originais), compostas por 13 temporais, 6 derivadas/engenhadas, 5 morfológicas/HRV, 5 espectrais e 1 tempo-frequência.

Produz `features\_selected.parquet`, `features\_selected\_ranking.csv` e `feature\_selection\_pipeline.pkl`.

\---

### Entregável 10 — Validação Estatística Final do Dataset

Verificação de consistência estatística do dataset final antes da fase de Reconhecimento de Padrões. Opera sobre as 30 features selecionadas no E9 e responde a quatro perguntas fundamentais:

1. **Multicolinearidade (VIF):** cálculo iterativo do Variance Inflation Factor com remoção de features com VIF acima do limiar; heatmap de correlação residual.
2. **Separabilidade estatística:** Kruskal-Wallis por feature, comparações par-a-par (Mann-Whitney U) com correção Bonferroni/FDR-BH, effect size (η² e Cohen's d), e projeção LDA 2D para inspeção visual.
3. **Balanceamento de classes:** distribuição absoluta e relativa por split (treino/val/teste) com documentação de estratégias recomendadas para lidar com desbalanceamento (class\_weight, SMOTE, etc.).
4. **Curvas de densidade por classe:** KDE das top features por η², coeficiente de Bhattacharyya para quantificar sobreposição entre pares de classes.

Também entrega uma **tabela final de atributos** com domínio, η², ranking consolidado e interpretação fisiológica, além do dataset definitivo pronto para entrada nos classificadores do RP.

\---

## Estrutura do Repositório

```
Biossinais/
├── README.md
├── docs/
│   ├── research/                    # Literatura de referência (ECG, SQI, HRV, Wavelets…)
│   ├── pipeline.pdf                 # Diagrama do pipeline da disciplina
│   └── Modelo\_de\_Trabalho\_Academico\_UFC.zip
├── entregaveis/
│   ├── entregavel-0/                # MNE + Surveys + Dataset
│   ├── entregavel-1/                
│   │   ├── notebooks/
│   │   │   └── 01\_aquisicao\_biossinais.ipynb
│   │   ├── figuras/
│   │   └── outputs/
│   │       └── ptbxl\_metadata\_enriched.csv
│   ├── entregavel-2/
│   │   ├── notebooks/
│   │   │   └── 02\_qualidade\_sinal\_SQI.ipynb
│   │   ├── figuras/
│   │   └── outputs/
│   │       ├── ptbxl\_com\_sqi.csv
│   │       └── rejected\_ecg\_ids.txt
│   ├── entregavel-3/
│   │   ├── notebooks/
│   │   │   └── 03\_estatistica\_inicial.ipynb
│   │   ├── figuras/
│   │   └── outputs/
│   │       └── estatistica\_inicial\_resultados.csv
│   ├── entregavel-4/
│   │   ├── notebooks/
│   │   │   └── 04\_limpeza\_dados.ipynb
│   │   ├── figuras/
│   │   └── outputs/
│   │       ├── sinais\_bons\_100hz.npy
│   │       ├── sinais\_ruins\_100hz.npy
│   │       ├── ptbxl\_bons\_com\_sqi.csv
│   │       ├── ptbxl\_ruins\_com\_sqi.csv
│   │       └── preprocessing\_params.pkl
│   ├── entregavel-5/
│   │   ├── notebooks/
│   │   │   └── 05\_segmentacao.ipynb
│   │   ├── figuras/
│   │   └── outputs/
│   │       ├── batimentos\_segmentados.npy
│   │       ├── registros\_ids.csv
│   │       ├── batimentos\_ids.csv
│   │       └── metricas\_inter\_janela.csv
│   ├── entregavel-6/
│   │   ├── notebooks/
│   │   │   └── 06\_extracao\_features\_completo.ipynb
│   │   ├── figuras/
│   │   └── outputs/
│   │       ├── features\_raw.parquet
│   │       ├── features\_raw\_sample.csv
│   │       └── feature\_catalog.csv
│   ├── entregavel-7/
│   │   ├── notebooks/
│   │   │   └── 07\_engenharia\_features\_completo.ipynb
│   │   ├── figuras/
│   │   └── outputs/
│   │       ├── features\_engineered.parquet
│   │       ├── features\_engineered\_sample.csv
│   │       ├── feature\_catalog\_e7.csv
│   │       └── scaler\_robust.pkl
│   ├── entregavel-8/
│   │   ├── notebooks/
│   │   │   └── 08\_reducao\_dimensionalidade\_completo.ipynb
│   │   ├── figuras/
│   │   └── outputs/
│   │       ├── features\_pca.parquet
│   │       ├── pca\_pipeline.pkl
│   │       └── pca\_variancia\_componentes.csv
│   ├── entregavel-9/
│   │   ├── notebooks/
│   │   │   └── 09\_selecao\_atributos\_completo.ipynb
│   │   ├── figuras/
│   │   └── outputs/
│   │       ├── features\_selected.parquet
│   │       ├── features\_selected\_ranking.csv
│   │       ├── feature\_selection\_pipeline.pkl
│   │       ├── wrappers/
│   │       └── embedded/
│   └── entregavel-10/
│       ├── notebooks/
│       │   └── 10\_validacao\_estatistica\_final.ipynb
│       ├── figuras/
│       └── outputs/
│           └── dataset\_final\_validado.parquet
├── documento-final/                 # PDF final, figuras e notebooks do TCC
│   ├── figuras/
│   └── notebooks/
├── data/                            # Dataset PTB-XL (baixar conforme instruções)
│   └── ptb-xl/
│       └── ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
└──
```

\---

## Fluxo de Dados entre Entregáveis

```
E1: ptbxl\_metadata\_enriched.csv
        ↓
E2: ptbxl\_com\_sqi.csv  →  rejected\_ecg\_ids.txt
        ↓
E3: estatistica\_inicial\_resultados.csv  (análise, sem novos dados)
        ↓
E4: sinais\_bons\_100hz.npy  +  ptbxl\_bons\_com\_sqi.csv
        ↓
E5: batimentos\_segmentados.npy  +  registros\_ids.csv  +  batimentos\_ids.csv
        ↓
E6: features\_raw.parquet
        ↓
E7: features\_engineered.parquet
       ↙                    ↘
E8: features\_pca.parquet    E9: features\_selected.parquet
       ↘                    ↙
E10: dataset\_final\_validado.parquet  →  RP (Reconhecimento de Padrões)
```

O **E8** e o **E9** são trilhas paralelas que produzem duas representações complementares do mesmo dataset: o E8 gera componentes PCA comprimidos (79 dimensões, prontos para classificadores que operam em espaço reduzido), enquanto o E9 seleciona features originais interpretáveis (30 atributos com identidade fisiológica). O E10 valida a trilha E9 e une as duas no ponto de entrada do RP.

\---

## Como Reproduzir o Pipeline

### 1\. Dados

Baixe o PTB-XL diretamente pelo Python ou via `wget`:

```python
import wfdb
wfdb.dl\_database('ptb-xl/1.0.3',
    dl\_dir='data/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3')
```

Ou consulte `data/README.md` para instruções detalhadas.

### 2\. Ambiente

Instale as dependências listadas em `requirements.txt`:

```bash
pip install -r requirements.txt
```

Pacotes principais: `wfdb`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `joblib`, `tqdm`, `PyWavelets`, `antropy`, `statsmodels`, `skrebate`, `pyarrow`/`fastparquet`.

### 3\. Execução

Execute os notebooks **em ordem**, pois cada um consome o output do anterior:

```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 (paralelo com 09) → 09 → 10
```

Abra cada notebook em `entregaveis/entregavel-X/notebooks/` e execute todas as células. Os arquivos de saída são gravados em `entregaveis/entregavel-X/outputs/`.

> \*\*Aviso de tempo:\*\* o E4 (processamento em lote de \~20.000 sinais), o E6 (extração de features com `joblib`) e o E9 (SBE) são as etapas mais custosas. O E9 persiste os resultados dos wrappers e do LASSO em disco, permitindo reexecução sem recalcular do zero.

### 4\. Relatório Final

O documento final (TCC) é compilado no Overleaf usando o template UFC disponível em `docs/Modelo\_de\_Trabalho\_Academico\_UFC.zip`. Figures e notebooks finais ficam em `documento-final/`.

\---

## Decisões Metodológicas Relevantes

|Decisão|Justificativa|
|-|-|
|Taxa de amostragem 100 Hz|Supera o critério de Nyquist para a banda diagnóstica do ECG (0,05–40 Hz) com margem de 25%; reduz custo computacional em 5× em relação a 500 Hz|
|Filtro notch 50 Hz dispensado|O downsampling original (500 → 100 Hz) já aplicou anti-aliasing; 50 Hz coincide com a frequência de Nyquist, impossibilitando filtro estável|
|Limiar SQI likelihood ≥ 50%|Convenção do PTB-XL para diagnóstico positivo|
|Winsorização por percentil 0,5–99,5%|Controla outliers de amplitude sem remoção de amostras; limites calculados no treino para evitar data leakage|
|Segmentação Pan-Tompkins|Método de referência para detecção de pico R; período refratário adaptativo (60% do RR mediano) permite lidar com taquicardias|
|RobustScaler antes de StandardScaler|RobustScaler (E7) torna o vetor invariante a outliers; StandardScaler adicional (E8/E9) garante variância unitária para PCA, VIF e LDA|
|Testes não-paramétricos|Normalidade e homocedasticidade rejeitadas para todas as variáveis (E3); Kruskal-Wallis, Mann-Whitney e correlação de Spearman são os métodos padrão adotados|
|Borda Count ponderado para ranking final|Método de agregação robusto que não requer comparabilidade de escalas entre diferentes critérios de relevância|

\---

## Referências Principais

* Wagner et al. (2020). *PTB-XL, a large publicly available electrocardiography dataset*. Scientific Data, 7(1), 154.
* Pan, J. \& Tompkins, W.J. (1985). *A real-time QRS detection algorithm*. IEEE Trans. Biomed. Eng.
* Kligfield et al. (2007). *Recommendations for the standardization and interpretation of the electrocardiogram*. JACC.
* Hyvärinen \& Oja (2000). *Independent component analysis: algorithms and applications*. Neural Networks.

