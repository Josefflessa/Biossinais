import json
import os

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_markdown(content):
    source = [line + '\n' for line in content.split('\n')]
    if source:
        source[-1] = source[-1][:-1]
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    })

def add_code(content):
    source = [line + '\n' for line in content.split('\n')]
    if source:
        source[-1] = source[-1][:-1]
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    })

# =====================================================================
# CÉLULA 1 - CABEÇALHO
# =====================================================================
add_markdown("""# Entregável 2: Avaliação da Qualidade do Sinal (SQI)
**Disciplina:** Aquisição de Biossinais
**Autor(es):** José Ferreira Lessa e Matheus Rocha Gomes da Silva
**Data:** Março de 2026""")

# =====================================================================
# CÉLULA 2 - OBJETIVO
# =====================================================================
add_markdown("""## Objetivo
Neste notebook, vamos atribuir a cada registro do PTB-XL um índice objetivo de qualidade de sinal (SQI — *Signal Quality Index*) baseado em múltiplas métricas extraídas diretamente dos sinais brutos. A partir desse score, cada registro será categorizado em níveis de qualidade (A, B ou C), determinando quais seguirão adiante no pipeline e quais serão descartados por ruído irrecuperável.""")

# =====================================================================
# CÉLULA 3 - IMPORTAÇÕES
# =====================================================================
add_markdown("""## 1. Importações e Dependências
Além dos pacotes usuais, importamos `scipy.stats` para kurtosis e skewness, e `tqdm` para acompanhamento da barra de progresso no processamento em lote.""")

add_code("""import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wfdb
import scipy.signal as signal
import scipy.stats as stats
from tqdm.notebook import tqdm
from IPython.display import display, Markdown
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

np.random.seed(42)""")

# =====================================================================
# CÉLULA 4 - CONSTANTES
# =====================================================================
add_markdown("""## 2. Configurações Globais""")

add_code("""PATH_DATA = '../../../data/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
FS = 100
N_LEADS = 12
LEAD_NAMES = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
FOLDS_TREINO = [1, 2, 3, 4, 5, 6, 7, 8]
FOLD_VAL = 9
FOLD_TEST = 10

FIGS_DIR = '../figuras/'
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs('../outputs', exist_ok=True)""")

# =====================================================================
# CÉLULA 5 - LEITURA
# =====================================================================
add_markdown("""## 3. Carregamento dos Metadados Enriquecidos
Partimos do arquivo gerado no Entregável 1, que já contém as colunas de superclasse diagnóstica, split e flags de qualidade.""")

add_code("""print("Carregando metadados do Entregável 1...")
caminho_metadados = '../../entregavel-1/outputs/ptbxl_metadata_enriched.csv'

if not os.path.exists(caminho_metadados):
    raise FileNotFoundError(f"Arquivo não encontrado: {caminho_metadados}\\nExecute o Entregável 1 primeiro.")

df = pd.read_csv(caminho_metadados, index_col='ecg_id')
df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)
df['diagnostic_superclass'] = df['diagnostic_superclass'].apply(ast.literal_eval)

scp_statements = pd.read_csv(os.path.join(PATH_DATA, 'scp_statements.csv'), index_col=0)

print(f"Dataset carregado: {df.shape[0]} registros, {df.shape[1]} colunas.")""")

# =====================================================================
# SEÇÃO 1 - ESTRATÉGIA
# =====================================================================
add_markdown("""---
## Seção 1 — Estratégia de Amostragem para Cálculo do SQI

### 1.1 Processamento em Lote
Calcular métricas como SNR e entropia espectral para todos os registros é viável, mas convém fazer de forma eficiente. Iteraremos sobre todos os registros com `tqdm` para acompanhar o progresso.""")

add_code("""def load_ecg(ecg_id, dataframe, path_base, fs=100):
    \"\"\"Carrega o sinal de ECG a partir do WFDB.\"\"\"
    linha = dataframe.loc[ecg_id]
    file_target = linha['filename_lr'] if fs == 100 else linha['filename_hr']
    signal_arr, _ = wfdb.rdsamp(os.path.join(path_base, file_target))
    return signal_arr""")

add_markdown("""### 1.2 Verificação de Consistência dos Metadados de Qualidade
Antes de calcular as métricas, vale conferir o que os metadados originais do PTB-XL já anotaram sobre problemas de qualidade (baseline drift, ruído estático, etc.).""")

add_code("""qtd_issues = df['has_quality_issues'].value_counts()
pct_limpos = (qtd_issues.get(False, 0) / len(df)) * 100

display(Markdown(f\"\"\"
**Resumo dos metadados de qualidade originais:**
- **{pct_limpos:.2f}%** dos registros não possuem nenhuma anotação de problema de qualidade.
- Os demais têm ao menos um campo de qualidade preenchido (baseline_drift, static_noise, etc.).
\"\"\"))

display(Markdown("**Tabela cruzada: problemas de qualidade por fold:**"))
display(pd.crosstab(df['has_quality_issues'], df['strat_fold'], margins=True))

display(Markdown("**(Espaço para comentário do aluno — dica: verifique se problemas de qualidade se concentram em folds específicos, o que poderia indicar variação entre períodos de coleta.)**"))""")

# =====================================================================
# SEÇÃO 2 - MÉTRICAS SQI
# =====================================================================
add_markdown("""---
## Seção 2 — Implementação das Métricas de SQI

Cada métrica captura um tipo diferente de degradação no sinal.""")

add_code("""def compute_snr(sig, fs=100):
    \"\"\"SNR: filtra o sinal com bandpass 0.5-40Hz como referência de 'sinal limpo'
    e mede a potência do resíduo (ruído).\"\"\"
    nyq = 0.5 * fs
    b, a = signal.butter(4, [0.5/nyq, 40/nyq], btype='bandpass')
    s_clean = signal.filtfilt(b, a, sig, axis=0)
    noise = sig - s_clean
    rms_clean = np.sqrt(np.mean(s_clean**2, axis=0))
    rms_noise = np.sqrt(np.mean(noise**2, axis=0))
    rms_noise = np.where(rms_noise == 0, 1e-10, rms_noise)
    snr_db = 10 * np.log10((rms_clean**2) / (rms_noise**2))
    return np.median(snr_db)

def compute_kurtosis(sig):
    \"\"\"Kurtosis de Fisher (excess kurtosis) mediano das 12 derivações.
    ECG limpo tem kurtosis 5–30: o QRS é fisiologicamente um spike (leptocúrtico).
    kurtosis < 1  → sinal muito plano: eletrodo solto, saturação ou linha isoe-létrica pura.
    kurtosis > 50 → spikes de artefato extremo (impacto mecânico, interferência de RF).\"\"\"
    return np.median(stats.kurtosis(sig, axis=0, fisher=True))

def compute_skewness(sig):
    \"\"\"Assimetria: valores altos (|>5.0|) sugerem artefatos extremos que distorcem a morfologia.\"\"\"
    return np.median(stats.skew(sig, axis=0))

def compute_spectral_entropy(sig, fs=100):
    \"\"\"Entropia espectral normalizada da derivação DII.
    Sinal limpo = espectro concentrado = entropia baixa.
    Ruído branco = espectro plano = entropia ~1.\"\"\"
    dii = sig[:, 1]
    freqs, psd = signal.welch(dii, fs=fs, nperseg=256)
    psd_norm = psd / np.sum(psd)
    H = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    return H / np.log2(len(psd))

def compute_flat_ratio(sig):
    \"\"\"Proporção de amostras consecutivas idênticas.
    Sinal saturado -> muitas amostras com diff == 0.\"\"\"
    diff = np.diff(sig, axis=0)
    flat_pct = np.mean(diff == 0, axis=0)
    return np.max(flat_pct)

def detect_50hz(sig, fs=100):
    \"\"\"Razão da potência na banda 49-51 Hz vs banda cardíaca 10-40 Hz.\"\"\"
    dii = sig[:, 1]
    freqs, psd = signal.welch(dii, fs=fs, nperseg=256)
    idx_50 = np.where((freqs >= 49) & (freqs <= 51))[0]
    idx_cardiac = np.where((freqs >= 10) & (freqs <= 40))[0]
    pow_50 = np.sum(psd[idx_50]) if len(idx_50) > 0 else 0
    pow_cardiac = np.sum(psd[idx_cardiac]) if len(idx_cardiac) > 0 else 1e-10
    return pow_50 / pow_cardiac""")

add_markdown("""### Extração em Lote sobre Todo o Dataset
Aplicamos as 6 métricas em cada registro.

> **Nota sobre a Razão 50 Hz:** A métrica `rede_50hz_ratio` foi calculada para completude, mas é inaplicável ao dataset a 100 Hz: a frequência de Nyquist coincide com a interferência de rede (50 Hz), tornando a detecção espectral nessa banda indefinida. A versão 100 Hz do PTB-XL foi produzida por decimação do sinal a 500 Hz, processo que já remove os conteúdos acima de 50 Hz no anti-aliasing. Por isso, esta métrica foi excluída do score composto.""")

add_code("""results = []

for eid in tqdm(df.index, desc='Calculando métricas SQI', unit=' sinal'):
    try:
        sig = load_ecg(eid, df, PATH_DATA, FS)
        results.append({
            'ecg_id': eid,
            'snr_db': compute_snr(sig, FS),
            'kurtosis': compute_kurtosis(sig),
            'skewness': compute_skewness(sig),
            'entropy_h_norm': compute_spectral_entropy(sig, FS),
            'flat_ratio': compute_flat_ratio(sig),
            'rede_50hz_ratio': detect_50hz(sig, FS)
        })
    except Exception as e:
        print(f"Erro no ECG ID {eid}: {e}")

df_sqi_raw = pd.DataFrame(results).set_index('ecg_id')
print(f"Métricas calculadas para {len(df_sqi_raw)} registros.")""")

add_markdown("""### Distribuição Individual de Cada Métrica
Antes de combinar as métricas em um score único, vale visualizar a distribuição de cada uma para entender o comportamento do dataset e validar os limiares que vamos adotar.""")

add_code("""fig, axes = plt.subplots(2, 3, figsize=(18, 10))

metricas_info = [
    ('snr_db', 'SNR Mediano (dB)', 5, '#2d8f4e'),
    ('kurtosis', 'Kurtosis (Fisher)', (1, 50), '#8a6dab'),
    ('skewness', 'Skewness', 5, '#c2425b'),
    ('entropy_h_norm', 'Entropia Espectral Normalizada', 0.95, '#cc8f3c'),
    ('flat_ratio', 'Flat Ratio', 0.05, '#3a7ca5'),
    ('rede_50hz_ratio', 'Razão Potência 50 Hz', 0.1, '#6b8e6b')
]

for i, (col, titulo, limiar, cor) in enumerate(metricas_info):
    ax = axes[i // 3, i % 3]
    dados = df_sqi_raw[col].dropna()
    sns.histplot(dados, bins=60, kde=True, ax=ax, color=cor, alpha=0.6)
    if isinstance(limiar, tuple):
        ax.axvline(limiar[0], color='red', linestyle='--', linewidth=1.5, label=f'Limiar: {limiar[0]} a {limiar[1]}')
        ax.axvline(limiar[1], color='red', linestyle='--', linewidth=1.5)
    else:
        ax.axvline(limiar, color='red', linestyle='--', linewidth=1.5,
                   label=f'Limiar: {limiar}')
    mediana_val = dados.median()
    ax.axvline(mediana_val, color='black', linestyle=':', linewidth=1,
               label=f'Mediana: {mediana_val:.2f}')
    ax.set_title(titulo)
    ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, 'distribuicao_metricas_sqi.png'), dpi=150, bbox_inches='tight')
plt.show()

display(Markdown("**(Espaço para comentário do aluno — dica: verifique se as medianas do dataset estão acima ou abaixo dos limiares. Se a maioria está acima do limiar de SNR e abaixo do de entropia, o dataset tem boa qualidade geral. Note quais métricas são mais problemáticas.)**"))""")

# =====================================================================
# SEÇÃO 3 - SQI COMPOSTO
# =====================================================================
add_markdown("""---
## Seção 3 — SQI Composto e Critério de Decisão

### 3.1 Cálculo do Score Consolidado
Combinamos as 5 métricas em um score binário simples: cada critério atendido vale 1 ponto, e o total é dividido por 5.

**Critérios de aprovação (cada um vale +1):**
- SNR mediano ≥ 5 dB
- Kurtosis de Fisher entre 1 e 50
- |Skewness| ≤ 5.0
- Entropia espectral normalizada ≤ 0.95
- Flat ratio ≤ 0.05 (≤ 5%)""")

add_code("""score = np.zeros(len(df_sqi_raw))

score += (df_sqi_raw['snr_db'] >= 5).astype(int)
score += ((df_sqi_raw['kurtosis'] >= 1) & (df_sqi_raw['kurtosis'] <= 50)).astype(int)
score += (df_sqi_raw['skewness'].abs() <= 5).astype(int)
score += (df_sqi_raw['entropy_h_norm'] <= 0.95).astype(int)
score += (df_sqi_raw['flat_ratio'] <= 0.05).astype(int)

df_sqi_raw['sqi_score'] = score / 5.0""")

add_markdown("""### 3.2 Categorização em A, B e C
Com base no score, classificamos cada registro:

| Categoria | Score | Significado |
|---|---|---|
| **A** | ≥ 0.8 | Alta qualidade — segue normalmente |
| **B** | 0.6 – 0.8 | Marginal — incluído com ressalvas |
| **C** | < 0.6 | Baixa qualidade — rejeitado do pipeline |""")

add_code("""def categorize_sqi(sqi):
    if sqi >= 0.8:
        return 'A'
    elif sqi >= 0.6:
        return 'B'
    else:
        return 'C'

df_sqi_raw['sqi_category'] = df_sqi_raw['sqi_score'].apply(categorize_sqi)

# Juntando com o DataFrame principal
df = df.join(df_sqi_raw)

display(Markdown("**Distribuição por split e categoria SQI:**"))
display(df.groupby(['split', 'sqi_category']).size().unstack(fill_value=0))

display(Markdown("**(Espaço para comentário do aluno — dica: analise se a proporção de registros na Categoria C é aceitável. Valores muito altos podem indicar problemas nos limiares; valores muito baixos sugerem que o filtro está pouco exigente.)**"))""")

add_markdown("""### 3.3 Validação com Metadados Originais
Cruzamos nosso SQI calculado com as anotações de qualidade originais do dataset para verificar consistência.""")

add_code("""tabela_validacao = pd.crosstab(
    df['has_quality_issues'].replace({True: 'Problema anotado', False: 'Sem problema'}),
    df['sqi_category'],
    normalize='index'
) * 100

display(Markdown("**Porcentagem de cada categoria SQI, separada por presença de problema nos metadados:**"))
display(tabela_validacao.round(1))

# Concordâncias
sem_prob = df[~df['has_quality_issues']]
com_prob = df[df['has_quality_issues']]
taxa_a_limpos = (sem_prob['sqi_category'] == 'A').mean() * 100
taxa_bc_sujos = ((com_prob['sqi_category'] == 'B') | (com_prob['sqi_category'] == 'C')).mean() * 100

display(Markdown(f\"\"\"
**Concordância:**
- Dos registros **sem** problema anotado, **{taxa_a_limpos:.1f}%** foram classificados como A.
- Dos registros **com** problema anotado, **{taxa_bc_sujos:.1f}%** caíram em B ou C.
\"\"\"))

display(Markdown("**(Espaço para comentário do aluno — dica: discuta o grau de concordância. É esperado que não seja perfeita, pois o SQI captura problemas diferentes dos metadados textuais. Falsos positivos e negativos são normais.)**"))""")

# =====================================================================
# SEÇÃO 4 - VISUALIZAÇÕES
# =====================================================================
add_markdown("""---
## Seção 4 — Visualizações Comparativas

### 4.1 Painel Comparativo: Categoria A vs. Categoria C
Visualizamos lado a lado registros de alta e baixa qualidade para entender o que cada categoria significa na prática.""")

add_code("""cat_a = df[df['sqi_category'] == 'A'].sample(3, random_state=42).index
cat_c = df[df['sqi_category'] == 'C'].sample(3, random_state=42).index
amostrados = list(cat_a) + list(cat_c)

fig, axes = plt.subplots(6, 3, figsize=(18, 20))
fig.suptitle('Comparação: Categoria A (verde) vs. Categoria C (vermelho)', fontsize=16)

for i, eid in enumerate(amostrados):
    sig = load_ecg(eid, df, PATH_DATA, FS)
    dii = sig[:, 1]
    t = np.arange(len(dii)) / FS
    cor = '#2d8f4e' if i < 3 else '#c0392b'

    # Coluna 1: Sinal DII
    axes[i, 0].plot(t, dii, color=cor, lw=0.8)
    axes[i, 0].set_title(f'ECG {eid} (SQI {df.loc[eid, "sqi_category"]} = {df.loc[eid, "sqi_score"]:.2f})', fontsize=10)
    axes[i, 0].set_ylabel('mV')

    # Coluna 2: Espectro (Welch)
    freqs, psd = signal.welch(dii, fs=FS, nperseg=256)
    axes[i, 1].plot(freqs, 10 * np.log10(psd + 1e-10), color='k')
    axes[i, 1].axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    axes[i, 1].axvline(40, color='gray', linestyle='--', alpha=0.5)
    axes[i, 1].set_xlim(0, 50)
    axes[i, 1].set_title('Espectro de potência', fontsize=10)

    # Coluna 3: Histograma de amplitude
    sns.histplot(dii, kde=True, ax=axes[i, 2], color=cor, alpha=0.7)
    info = f"SNR={df.loc[eid,'snr_db']:.1f}dB | H={df.loc[eid,'entropy_h_norm']:.2f} | K={df.loc[eid,'kurtosis']:.1f}"
    axes[i, 2].set_title(info, fontsize=9)

plt.tight_layout()
plt.subplots_adjust(top=0.94)
fig.savefig(os.path.join(FIGS_DIR, 'painel_sqi_a_vs_c.png'), dpi=150, bbox_inches='tight')
plt.show()

display(Markdown("**(Espaço para comentário do aluno — dica: compare os sinais verdes (A) com os vermelhos (C). Os registros A devem ter morfologia QRS bem definida e espectro concentrado. Os C provavelmente mostram ruído, espectro plano e histograma irregular.)**"))""")

add_markdown("""### 4.2 Mapa de Calor de Correlação entre Métricas SQI
Métricas muito correlacionadas detectam o mesmo tipo de problema e uma delas poderia ser removida.""")

add_code("""fig = plt.figure(figsize=(8, 6))
cols_sqi = ['snr_db', 'kurtosis', 'skewness', 'entropy_h_norm', 'flat_ratio', 'rede_50hz_ratio']
corr_sqi = df[cols_sqi].corr(method='spearman')

sns.heatmap(corr_sqi, annot=True, cmap='coolwarm', fmt=".2f", center=0,
            mask=np.triu(np.ones_like(corr_sqi, dtype=bool)))
plt.title('Correlação de Spearman entre Métricas SQI')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'correlacao_sqi.png'), dpi=150, bbox_inches='tight')
plt.show()

display(Markdown("**(Espaço para comentário do aluno — dica: verifique se há pares de métricas altamente correlacionados (|r| > 0.7). Métricas independentes são desejáveis, pois cada uma captura um problema diferente.)**"))""")

add_markdown("""### 4.3 Distribuição do SNR por Categoria""")

add_code("""fig = plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='snr_db', hue='sqi_category', bins=50, kde=True,
             palette={'A': '#2d8f4e', 'B': '#e67e22', 'C': '#c0392b'})
plt.title('Distribuição do SNR (dB) por Categoria SQI')
plt.xlabel('SNR Mediano das 12 Derivações (dB)')
plt.axvline(5, color='k', linestyle='--', label='Limiar A (≥ 5 dB)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'snr_por_categoria.png'), dpi=150, bbox_inches='tight')
plt.show()

display(Markdown("**(Espaço para comentário do aluno — dica: observe como as categorias se separam no eixo de SNR. Verifique se o limiar de 5 dB faz boa separação entre sinais limpos e ruidosos.)**"))""")

add_markdown("""### 4.4 SNR por Derivação
Algumas derivações são sistematicamente mais ruidosas que outras — precordiais anteriores (V1, V2) costumam ter mais artefatos de movimento. Verificamos isso calculando o SNR individual de cada derivação em uma amostra de 500 registros.""")

add_code("""# Amostra para cálculo de SNR por derivação
amostra_snr_lead = df.sample(min(500, len(df)), random_state=42).index
snr_por_lead = []

nyq = 0.5 * FS
b_bp, a_bp = signal.butter(4, [0.5/nyq, 40/nyq], btype='bandpass')

for eid in tqdm(amostra_snr_lead, desc='SNR por derivação'):
    sig = load_ecg(eid, df, PATH_DATA, FS)
    s_clean = signal.filtfilt(b_bp, a_bp, sig, axis=0)
    noise = sig - s_clean
    rms_c = np.sqrt(np.mean(s_clean**2, axis=0))
    rms_n = np.sqrt(np.mean(noise**2, axis=0))
    rms_n = np.where(rms_n == 0, 1e-10, rms_n)
    snr_leads = 10 * np.log10((rms_c**2) / (rms_n**2))
    for j, lead in enumerate(LEAD_NAMES):
        snr_por_lead.append({'lead': lead, 'snr_db': snr_leads[j]})

df_snr_leads = pd.DataFrame(snr_por_lead)

fig = plt.figure(figsize=(12, 5))
sns.boxplot(data=df_snr_leads, x='lead', y='snr_db',
            order=LEAD_NAMES, palette='coolwarm')
plt.axhline(5, color='red', linestyle='--', alpha=0.5, label='Limiar 5 dB')
plt.title('SNR por Derivação (amostra de 500 registros)')
plt.xlabel('Derivação')
plt.ylabel('SNR (dB)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'snr_por_derivacao.png'), dpi=150, bbox_inches='tight')
plt.show()

display(Markdown("**(Espaço para comentário do aluno — dica: identifique se alguma derivação é sistematicamente mais ruidosa. V1 e V2 costumam ter SNR menor por estarem mais sujeitas a artefatos de movimento torácico. aVR muitas vezes mostra comportamento diferente por ser a derivação mais 'distante' eletricamente.)**"))""")

# =====================================================================
# SEÇÃO 5 - SALVAMENTO
# =====================================================================
add_markdown("""---
## Seção 5 — Salvamento e Síntese

Exportamos os dados com as métricas SQI e a lista de registros rejeitados. A partir do próximo entregável, trabalharemos apenas com registros de Categoria A e B.""")

add_code("""caminho_final = '../outputs/ptbxl_com_sqi.csv'
df.to_csv(caminho_final)

# Lista de rejeitados
rejeitados = df[df['sqi_category'] == 'C']
cam_rejeitados = '../outputs/rejected_ecg_ids.txt'
with open(cam_rejeitados, 'w') as fh:
    for item in rejeitados.index:
        fh.write(f"{item}\\n")

n_a = len(df[df.sqi_category == 'A'])
n_b = len(df[df.sqi_category == 'B'])
n_c = len(df[df.sqi_category == 'C'])

display(Markdown(f\"\"\"
**Arquivos salvos:**
- Dataset com SQI: `{caminho_final}`
- IDs rejeitados: `{cam_rejeitados}`

**Resumo por categoria:**

| Categoria | Registros | Descrição |
|---|---|---|
| A (Alta qualidade) | {n_a} | Aprovados para o pipeline |
| B (Marginal) | {n_b} | Incluídos com ressalva |
| C (Rejeitados) | {n_c} | Excluídos do pipeline |

**(Espaço para comentário do aluno — dica: comente sobre a proporção de dados que sobreviveram ao filtro SQI e se a taxa de rejeição parece razoável para um dataset clínico real coletado ao longo de 7 anos.)**
\"\"\"))""")

# =====================================================================
# SALVAR NOTEBOOK
# =====================================================================
out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'entregaveis', 'entregavel-2', 'notebooks'))
os.makedirs(out_dir, exist_ok=True)
notebook_path = os.path.join(out_dir, '02_qualidade_sinal_SQI.ipynb')

with open(notebook_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(notebook, indent=2))

print(f"Notebook criado com sucesso: {notebook_path}")
