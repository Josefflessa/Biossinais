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
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_markdown(content):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + '\n' for line in content.split('\n')]
    })

def add_code(content):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in content.split('\n')]
    })

# =====================================================================
# CÉLULA 1 - CABEÇALHO
# =====================================================================
add_markdown("""# Entregável 6: Extração de Atributos do ECG (Feature Extraction)
**Disciplina:** Aquisição de Biossinais
**Equipe:** José Ferreira Lessa e Matheus Rocha Gomes da Silva
**Objetivo:** Transformação de sinais brutos em um dataset de alta dimensão através da extração de atributos temporais, espectrais, morfológicos e não-lineares.""")

# =====================================================================
# CÉLULA 2 - RESUMO ACADÊMICO
# =====================================================================
add_markdown("""## Introdução e Justificativa
A extração de atributos é a etapa onde traduzimos o conhecimento clínico da cardiologia em variáveis numéricas que podem ser processadas por algoritmos de aprendizado de máquina. 

Neste notebook, implementamos uma **Estratégia Híbrida**: 
1.  **Global (10s)**: Capturamos a dinâmica geral do exame em todas as 12 derivações.
2.  **Local (Batimento)**: Analisamos a morfologia fina de cada ciclo cardíaco e agregamos esses valores (via Mediana e Desvio Padrão) para o nível de registro.

Isso nos permite representar tanto alterações de ritmo (HRV) quanto alterações morfológicas (Isquemias, Hipertrofias) de forma robusta.""")

# =====================================================================
# CÉLULA 3 - CONFIGURAÇÕES E INSTALLS
# =====================================================================
add_markdown("""## 1. Instalação e Configuração do Ambiente""")

add_code("""# Instalação de bibliotecas especializadas para análise de biossinais
#%pip install PyWavelets antropy joblib fastparquet pyarrow -q

import os
import ast
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as signal
import pywt
import antropy as ant
from scipy.stats import entropy
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
from IPython.display import display, Markdown

# Configurações de visualização acadêmica
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

FS = 100
LEAD_NAMES = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
np.random.seed(42)""")

# =====================================================================
# CÉLULA 4 - CARREGAMENTO
# =====================================================================
add_markdown("""## 2. Carregamento de Sinais e Metadados
Importamos os sinais limpos (D4) e as segmentações (D5) usando **Memory Mapping** para evitar erros de RAM.""")

add_code("""# Diretórios de Entrada
DIR_IN_D4 = '../../entregavel-4/outputs/'
DIR_IN_D5 = '../../entregavel-5/outputs/'

# Diretório de Saída
DIR_OUT_D6 = '../outputs/'
FIGS_DIR = '../figuras/'
os.makedirs(DIR_OUT_D6, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

print("Carregando bases de dados via Memory Mapping... (Economiza RAM)")
# mmap_mode='r' permite ler o arquivo sem carregá-lo inteiramente na RAM
sinais_10s = np.load(os.path.join(DIR_IN_D4, 'sinais_limpos_100hz.npy'), mmap_mode='r')
batimentos = np.load(os.path.join(DIR_IN_D5, 'batimentos_segmentados.npy'), mmap_mode='r')

df_reg_ids = pd.read_csv(os.path.join(DIR_IN_D5, 'registros_ids.csv'), index_col='ecg_id')
df_beat_ids = pd.read_csv(os.path.join(DIR_IN_D5, 'batimentos_ids.csv'))

print(f"Dataset 10s: {sinais_10s.shape}")
print(f"Dataset Batimentos: {batimentos.shape}")""")

# =====================================================================
# SEÇÃO 1 - DOMÍNIO DO TEMPO
# =====================================================================
add_markdown("""---
## Seção 1 — Domínio do Tempo (Registro Inteiro)

Extraímos estatísticas descritivas que capturam a distribuição de amplitude e a energia do sinal em todas as 12 derivações.""")

add_code("""def extract_time_domain(sig_12l):
    \"\"\"Calcula estatísticas temporais para as 12 derivações.\"\"\"
    feats = {}
    for i, lead in enumerate(LEAD_NAMES):
        s = sig_12l[:, i]
        feats[f'time_rms_{lead}'] = np.sqrt(np.mean(s**2))
        feats[f'time_mav_{lead}'] = np.mean(np.abs(s))
        feats[f'time_var_{lead}'] = np.var(s)
        feats[f'time_p2p_{lead}'] = np.ptp(s)
        feats[f'time_zcr_{lead}'] = np.sum(np.diff(np.sign(s)) != 0) / len(s)
        
        # Estatísticas de forma (usando pandas para skew/kurt)
        ser = pd.Series(s)
        feats[f'time_skew_{lead}'] = ser.skew()
        feats[f'time_kurt_{lead}'] = ser.kurt()
    return feats

print("Processando Domínio do Tempo em paralelo...")
time_feats = Parallel(n_jobs=-1)(delayed(extract_time_domain)(s) for s in tqdm(sinais_10s))
df_time = pd.DataFrame(time_feats, index=df_reg_ids.index)
# Limpeza de memória
del time_feats
gc.collect()
display(df_time.head(3))""")

# =====================================================================
# SEÇÃO 2 - DOMÍNIO DA FREQUÊNCIA
# =====================================================================
add_markdown("""---
## Seção 2 — Domínio da Frequência (PSD via Welch)

Analisamos o conteúdo espectral do ECG. As bandas foram escolhidas para isolar componentes morfológicas (P, QRS, T).""")

add_code("""def extract_freq_domain(sig_12l, fs=100):
    \"\"\"Calcula potência em bandas e descritores espectrais via Welch PSD.\"\"\"
    feats = {}
    for i, lead in enumerate(LEAD_NAMES):
        s = sig_12l[:, i]
        # Welch com janela de 2.56s (nperseg=256) para boa resolução
        f, psd = signal.welch(s, fs=fs, nperseg=256, noverlap=128)
        
        # Integração em bandas fisiológicas (Regra Trapezoidal)
        feats[f'freq_pt_band_{lead}'] = np.trapz(psd[(f >= 0.5) & (f <= 5)], f[(f >= 0.5) & (f <= 5)])
        feats[f'freq_qrs_band_{lead}'] = np.trapz(psd[(f >= 5) & (f <= 25)], f[(f >= 5) & (f <= 25)])
        feats[f'freq_total_pwr_{lead}'] = np.trapz(psd[(f >= 0.5) & (f <= 40)], f[(f >= 0.5) & (f <= 40)])
        
        # Descritores de forma da PSD
        feats[f'freq_peak_{lead}'] = f[np.argmax(psd)]
        feats[f'freq_median_{lead}'] = f[np.where(np.cumsum(psd) >= np.sum(psd)/2)[0][0]]
        feats[f'freq_centroid_{lead}'] = np.sum(f * psd) / np.sum(psd)
    return feats

print("Processando Domínio da Frequência...")
freq_feats = Parallel(n_jobs=-1)(delayed(extract_freq_domain)(s) for s in tqdm(sinais_10s))
df_freq = pd.DataFrame(freq_feats, index=df_reg_ids.index)
# Limpeza de memória
del freq_feats
gc.collect()
display(df_freq.head(3))""")

# =====================================================================
# SEÇÃO 3 - MORFOLOGIA E HRV
# =====================================================================
add_markdown("""---
## Seção 3 — Morfologia Fina e Variabilidade (HRV)

Trabalhamos agora sobre os **batimentos segmentados**. Extraímos métricas clínicas e agregamos os resultados para o nível do registro.""")

add_code("""def extract_morph_per_beat(beat_12l):
    \"\"\"Extrai morfologia de um único batimento (60 amostras).\"\"\"
    feats = {}
    # Focamos nas derivações mais diagnósticas para morfologia: II e V5
    for l_idx, l_name in [(1, 'II'), (10, 'V5')]:
        s = beat_12l[:, l_idx]
        feats[f'morph_r_amp_{l_name}'] = np.max(s)
        # QRS Duration (estimação por cruzamento de limiar 15% do pico)
        peak_idx = np.argmax(s)
        thresh = s[peak_idx] * 0.15
        try:
            qrs_pts = np.where(s > thresh)[0]
            feats[f'morph_qrs_width_{l_name}'] = (qrs_pts[-1] - qrs_pts[0]) * 10 # em ms
        except:
            feats[f'morph_qrs_width_{l_name}'] = 100 # Default fallback
            
        # Segmento ST (Média entre 35-45 amostras - pós-QRS)
        feats[f'morph_st_amp_{l_name}'] = np.mean(s[35:45])
        # Onda T (Pico na janela final)
        feats[f'morph_t_amp_{l_name}'] = np.max(s[40:])
        
    return feats

print("Processando Morfologia Individual dos Batimentos...")
morph_list = Parallel(n_jobs=-1)(delayed(extract_morph_per_beat)(b) for b in tqdm(batimentos))
df_morph_all = pd.DataFrame(morph_list)

# Agregação: Vinculamos os batimentos aos seus IDs de registro original
df_morph_meta = pd.concat([df_beat_ids[['ecg_id', 'rr_interval_ms']], df_morph_all], axis=1)

# Atributos de HRV (Variabilidade) por Registro
df_hrv = df_morph_meta.groupby('ecg_id')['rr_interval_ms'].agg([
    ('hrv_meanRR', 'mean'),
    ('hrv_sdRR', 'std'),
    ('hrv_rmssd', lambda x: np.sqrt(np.mean(np.diff(x.dropna())**2)) if len(x.dropna())>1 else np.nan),
    ('hrv_cvRR', lambda x: (np.std(x)/np.mean(x)) if np.mean(x) != 0 else 0)
])

# Atributos Morfológicos Agregados (Mediana e STD)
agg_rules = {col: ['median', 'std'] for col in df_morph_all.columns}
df_morph_agg = df_morph_meta.groupby('ecg_id').agg(agg_rules)
df_morph_agg.columns = ['_'.join(col).strip() for col in df_morph_agg.columns.values]

df_beat_final = pd.concat([df_hrv, df_morph_agg], axis=1)
# Limpeza
del morph_list, df_morph_all
gc.collect()
print(f"Features de Morfologia/HRV agregadas: {df_beat_final.shape}")""")

# =====================================================================
# SEÇÃO 4 - DOMÍNIO TEMPO-FREQUÊNCIA E NÃO-LINEAR
# =====================================================================
add_markdown("""---
## Seção 4 — Domínio Tempo-Frequência (Wavelets) e Dinâmica Não-Linear

Para capturar a complexidade do sinal, usamos a **Transformada Wavelet Discreta (DWT)** com a família **db4** (Daubechies) e métricas como **DFA** e **Entropia de Amostra**.""")

add_code("""def extract_complex_features(sig_10s_lead, rr_series):
    '''Calcula DWT e métricas não-lineares para uma única derivação (DII).'''
    
    def calculate_shannon_entropy(x):
        '''Helper para calcular entropia de Shannon via histograma.'''
        counts, _ = np.histogram(x, bins='auto')
        return entropy(counts, base=2)

    feats = {}
    
    # 1. DWT (db4, Level 4)
    coeffs = pywt.wavedec(sig_10s_lead, 'db4', level=4)
    for i, c in enumerate(coeffs):
        level_name = f'D{4-i+1}' if i>0 else 'A4'
        energy = np.sum(c**2)
        feats[f'wavelet_energy_{level_name}'] = energy
        feats[f'wavelet_entropy_{level_name}'] = calculate_shannon_entropy(c)
        
    # 2. Não-Linear (no sinal DII)
    feats[f'nonlin_higu_fd'] = ant.higuchi_fd(sig_10s_lead)
    feats[f'nonlin_dfa_alpha'] = ant.detrended_fluctuation(sig_10s_lead)
    
    # 3. Não-Linear (na série RR - se houver batimentos suficientes)
    if len(rr_series) > 8:
        feats['nonlin_sampen_rr'] = ant.sample_entropy(rr_series)
        # Poincaré Plot descriptors
        sd_diff = np.diff(rr_series)
        feats['nonlin_sd1'] = np.sqrt(0.5 * np.var(sd_diff))
        feats['nonlin_sd2'] = np.sqrt(2*np.var(rr_series) - 0.5*np.var(sd_diff))
    else:
        feats['nonlin_sampen_rr'] = np.nan
        feats['nonlin_sd1'] = np.nan
        feats['nonlin_sd2'] = np.nan
        
    return feats

print("Processando Domínio Complexo (Wavelet + Não-Linear) em DII...")
rr_groups = df_beat_ids.dropna(subset=['rr_interval_ms']).groupby('ecg_id')['rr_interval_ms'].apply(list)

# Por ser muito custoso, rodamos o processamento em DII e séries RR
complex_feats = []
for i, eid in enumerate(tqdm(df_reg_ids.index, desc='Complex Analysis')):
    r_series = rr_groups.get(eid, [])
    # Extraímos na Lead II (DII)
    res = extract_complex_features(sinais_10s[i, :, 1], r_series)
    complex_feats.append(res)

df_complex = pd.DataFrame(complex_feats, index=df_reg_ids.index)

# Limpeza final de sinais da memória
del sinais_10s, batimentos
gc.collect()""")

# =====================================================================
# SEÇÃO 5 - CONSOLIDAÇÃO
# =====================================================================
add_markdown("""---
## Seção 5 — Consolidação e Geração do Dataset Final (Features Raw)

Unificamos todos os domínios em um único DataFrame otimizado (formato Parquet).""")

add_code("""# Unificação total
df_features_raw = df_reg_ids.join([df_time, df_freq, df_beat_final, df_complex], how='left')

# Salvamento robusto
# Parquet é preferível aqui pelo volume de colunas (> 200)
df_features_raw.to_parquet(os.path.join(DIR_OUT_D6, 'features_raw.parquet'), index=True)
df_features_raw.to_csv(os.path.join(DIR_OUT_D6, 'features_raw_sample.csv'), index=True)

display(Markdown(f\"\"\"
### Síntese da Extração:
- **Total de Características**: {len(df_features_raw.columns) - len(df_reg_ids.columns)}
- **Nº de Registros**: {len(df_features_raw)}
- **Distribuição de Domínios**:
    - Tempo: {len(df_time.columns)}
    - Frequência: {len(df_freq.columns)}
    - Morfologia/HRV: {len(df_beat_final.columns)}
    - Complexo (Wavelet/Não-Linear): {len(df_complex.columns)}
\"\"\"))""")

# =====================================================================
# SEÇÃO 6 - VISUALIZAÇÃO ACADÊMICA
# =====================================================================
add_markdown("""## 6. Visualização e Sanity Check""")

add_code("""# Plot 1: Distribuição de Energia RMS por Superclasse (Lead II)
df_plot = df_features_raw.copy()

# Correção: Converter string para lista se necessário e extrair o primeiro rótulo
if isinstance(df_plot['diagnostic_superclass'].iloc[0], str):
    import ast
    df_plot['diagnostic_superclass'] = df_plot['diagnostic_superclass'].apply(ast.literal_eval)

df_plot['label'] = df_plot['diagnostic_superclass'].apply(lambda x: x[0] if len(x)>0 else 'UNKNOWN')

plt.figure(figsize=(12, 6))
# Ordenação alfabética para consistência
order = sorted(df_plot['label'].unique())
sns.boxplot(x='label', y='time_rms_II', data=df_plot, hue='label', palette='Spectral', order=order, legend=False)

plt.title('Distribuição da Energia RMS (Lead II) por Categoria Diagnóstica')
plt.yscale('log')
plt.xlabel('Superclasse')
plt.ylabel('RMS (mV)')
plt.savefig(os.path.join(FIGS_DIR, 'box_rms_superclass.png'))
plt.show()

# Plot 2: Poincaré Plot (Exemplo HRV para registros NORM)
plt.figure(figsize=(8, 8))
# Filtragem: Pegamos os dados do registro NORM para representar a base saudável
df_norm = df_plot[df_plot['label'] == 'NORM']

plt.scatter(df_norm['nonlin_sd1'], df_norm['nonlin_sd2'], c='blue', alpha=0.1, label='Base Saudável (NORM)')
plt.xlabel('SD1 (Variabilidade de Curto Prazo)')
plt.ylabel('SD2 (Variabilidade de Longo Prazo)')
plt.title('Dispersão do Balanço HRV (Poincaré) - Registros Normais')
plt.legend()
plt.tight_layout()
plt.show()""")

# =====================================================================
# SEÇÃO 7 - SÍNTESE E CONEXÃO
# =====================================================================
add_markdown("""---
## Seção 7 — Síntese e Conexão

Neste entregável, consolidamos a transformação dos sinais brutos em um dataset de alta dimensão. A extração cobriu os domínios temporal, espectral, morfológico e não-linear, resultando em uma representação rica da atividade cardíaca.

**Principais Conclusões:**
1.  **Representatividade**: O dataset final contém atributos que capturam tanto a dinâmica global (10s) quanto a morfologia fina dos batimentos (segmentação).
2.  **Integridade**: Foram tratadas as falhas de cálculo (NaNs) em séries RR curtas e garantida a consistência das colunas.
3.  **Próximos Passos**: No **Entregável 7 (Engenharia de Atributos)**, realizaremos a normalização robusta e a criação de features de segunda ordem (razões entre bandas) para refinar a discriminabilidade do conjunto para o Reconhecimento de Padrões.
""")

add_code("""# Verificação final de sanidade
print(f"Dimensão final do dataset: {df_features_raw.shape}")
print(f"Valores nulos totais: {df_features_raw.isnull().sum().sum()}")
if df_features_raw.isnull().sum().sum() > 0:
    print("Aviso: Existem valores nulos que serão tratados no Entregável 7 (Normalização/Imputação).")
""")

# =====================================================================
# SALVAR O SCRIPT GERADOR
# =====================================================================
out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'entregaveis', 'entregavel-6', 'notebooks'))
os.makedirs(out_dir, exist_ok=True)
notebook_path = os.path.join(out_dir, '06_extracao_features.ipynb')

with open(notebook_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(notebook, indent=2))

print(f"Notebook '06_extracao_features.ipynb' finalizado com sucesso!")
print(f"Número total de features planejadas: ~250.")
