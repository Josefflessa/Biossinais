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
add_markdown("""# Entregável 5: Estratégia de Segmentação de Sinais ECG
**Disciplina:** Aquisição de Biossinais
**Equipe:** José Ferreira Lessa e Matheus Rocha Gomes da Silva
**Objetivo:** Transformação de sinais em instâncias de análise (10s e Batimentos) com validação de estabilidade.""")

# =====================================================================
# CÉLULA 2 - INTRODUÇÃO
# =====================================================================
add_markdown("""## Introdução e Metodologia
Nesta etapa, estruturamos a forma como os dados de ECG (10 segundos) serão segmentados para a extração de atributos no Entregável 6. 

Diferentes de aplicações de *Deep Learning* que costumam utilizar janelas curtas com sobreposição para aumentar o volume de dados (N), para a extração de atributos clássicos (SVM, RF, etc.), essa prática pode introduzir correlação artificial e redudância excessiva (*label inflation*). 

**Decisão de Projeto:** 
Adotamos os **10 segundos como instância principal (Opção A)** para preservar a unidade clínica original do dataset. Paralelamente, implementamos a **Segmentação por Batimento (Opção C)** apenas como uma etapa intermediária para calcular atributos de morfologia e variabilidade, que serão agregados e devolvidos ao nível do registro de 10 segundos.""")

# =====================================================================
# CÉLULA 3 - IMPORTAÇÕES
# =====================================================================
add_markdown("""## 1. Configuração do Ambiente""")

add_code("""import os
import ast
import gc
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as signal
from tqdm.notebook import tqdm
from IPython.display import display, Markdown
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Estética
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

np.random.seed(42)""")

# =====================================================================
# CÉLULA 4 - CONSTANTES
# =====================================================================
add_markdown("""## 2. Parâmetros e Diretórios""")

add_code("""# Parâmetros
FS = 100
LEAD_NAMES = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Diretórios
DIR_OUT_D4 = '../../entregavel-4/outputs/'
DIR_OUT_D5 = '../outputs/'
FIGS_DIR = '../figuras/'

os.makedirs(DIR_OUT_D5, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)""")

# =====================================================================
# CÉLULA 5 - LEITURA
# =====================================================================
add_markdown("""## 3. Leitura do Dataset Processado""")

add_code("""print("Carregando sinais limpos e metadados...")
npy_path = os.path.join(DIR_OUT_D4, 'sinais_limpos_100hz.npy')
csv_path = os.path.join(DIR_OUT_D4, 'ptbxl_filtrados_sqi_indices_ordenados.csv')

sinais = np.load(npy_path)
df = pd.read_csv(csv_path, index_col='ecg_id')
df['diagnostic_superclass'] = df['diagnostic_superclass'].apply(ast.literal_eval)

print(f"Dataset carregado: {len(sinais)} registros de 10s (1000 amostras cada).")""")

# =====================================================================
# SEÇÃO 1 - OPÇÃO A (10S)
# =====================================================================
add_markdown("""---
## Seção 1 — Estruturação do Dataset de Nível Superior (10 segundos)

Conforme decidido, cada registro de 10 segundos constitui uma instância única. O objetivo nesta seção é apenas propagar os metadados e validar a ordem dos registros antes da extração de features globais.""")

add_code("""# Propagação de identificadores e superclasses diagnósticas
df_instancias_10s = df[['patient_id', 'strat_fold', 'sqi_category', 'sqi_score', 'diagnostic_superclass']].copy()

# Validação final do balanceamento
resumo_folds = df_instancias_10s['strat_fold'].value_counts().sort_index()
display(Markdown("**Distribuição de Registros por Fold:**"))
display(resumo_folds)

df_instancias_10s.to_csv(os.path.join(DIR_OUT_D5, 'registros_ids.csv'))""")

# =====================================================================
# SEÇÃO 2 - OPÇÃO C (BATIMENTOS)
# =====================================================================
add_markdown("""---
## Seção 2 — Segmentação por Batimento (Análise Morfológica)

### 2.1 Algoritmo de Detector de Picos R (Pan-Tompkins Nativo)
Implementamos um detector baseado na lógica de Pan-Tompkins para identificar o momento exato de cada sístole ventricular (onda R).
1. **Filtro Digital**: Passa-banda entre 5-15 Hz.
2. **Derivada e Quadratura**: Enfatiza a inclinação característica do QRS.
3. **Integração**: Suaviza picos múltiplos para detecção de energia volumétrica.
4. **Localização**: Identifica os picos e aplica período refratário de 200ms.""")

add_code("""def detector_pan_tompkins(sinal_v, fs=100):
    \"\"\"Implementação nativa do detector Pan-Tompkins para detecção de picos R.\"\"\"
    # 1. Bandpass 5-15 Hz
    nyq = 0.5 * fs
    b, a = signal.butter(2, [5/nyq, 15/nyq], btype='bandpass')
    f = signal.filtfilt(b, a, sinal_v)
    
    # 2. Derivada e Quadratura
    d = np.diff(f)
    s = d**2
    
    # 3. Moving Average (150ms)
    ma = np.convolve(s, np.ones(int(0.15 * fs))/int(0.15 * fs), mode='same')
    
    # 4. Busca de picos
    picos, _ = signal.find_peaks(ma, distance=int(0.2 * fs), prominence=np.max(ma)*0.05)
    return picos

def extrair_janelas_beat(sig_12l, picos, fs=100):
    \"\"\"Extrai janelas de 600ms (200ms pre, 400ms post) centradas no pico R.\"\"\"
    bt, meta = [], []
    pre, post = int(0.2 * fs), int(0.4 * fs)
    
    for i, r_idx in enumerate(picos):
        if r_idx > pre and r_idx < (1000 - post):
            bt.append(sig_12l[r_idx - pre : r_idx + post, :])
            rr = (picos[i+1] - r_idx) * (1000/fs) if i < (len(picos)-1) else np.nan
            meta.append({'r_peak_idx': r_idx, 'rr_interval_ms': rr})
            
    return np.array(bt, dtype=np.float32), meta""")

# =====================================================================
# SEÇÃO 3 - EXTRAÇÃO EM LOTE
# =====================================================================
add_markdown("""### 2.2 Processamento do Dataset Completo""")

add_code("""lista_bt_final = []
lista_meta_bt = []

for i, eid in enumerate(tqdm(df.index, desc='Extraindo batimentos')):
    curr_sig = sinais[i]
    pks = detector_pan_tompkins(curr_sig[:, 1], FS) # Lead II como referência
    janelas, meta_janelas = extrair_janelas_beat(curr_sig, pks, FS)
    
    if len(janelas) > 0:
        lista_bt_final.append(janelas)
        for m in meta_janelas:
            m.update({'ecg_id': eid, 'patient_id': df.loc[eid, 'patient_id'], 
                      'strat_fold': df.loc[eid, 'strat_fold'],
                      'diagnostic_superclass': df.loc[eid, 'diagnostic_superclass']})
            lista_meta_bt.append(m)

# OTIMIZAÇÃO DE MEMÓRIA: Liberar os sinais de 10s após a extração
print("Liberando memória (sinais de 10s)...")
del sinais
gc.collect()

# Concatenação eficiente
matriz_bt = np.concatenate(lista_bt_final, axis=0)
df_bt_ids = pd.DataFrame(lista_meta_bt)
del lista_bt_final
gc.collect()

print(f"Total de batimentos extraídos: {len(matriz_bt)} ({matriz_bt.shape})")""")

# =====================================================================
# SEÇÃO 4 - VALIDAÇÃO (ESTABILIDADE E RITMO)
# =====================================================================
add_markdown("""---
## Seção 3 — Validação da Segmentação e Estabilidade

### 3.1 Métricas de Ritmo e Frequência cardíaca (FC)
Validamos se os batimentos detectados condizem com as superclasses esperadas. Um detector preciso deve mostrar maior variabilidade (SDNN) em casos de Arritmia (AFIB).""")

add_code("""# FC e SDNN por registro
df_rr_calc = df_bt_ids.dropna(subset=['rr_interval_ms'])
stats_ritmo = df_rr_calc.groupby('ecg_id')['rr_interval_ms'].agg(['mean', 'std']).rename(columns={'mean': 'mean_RR', 'std': 'sd_RR'})
stats_ritmo['fc_bpm'] = 60000 / stats_ritmo['mean_RR']

# Cruzamento com labels (extração segura do primeiro label ou 'NONE')
def get_safe_label(x):
    if isinstance(x, list) and len(x) > 0:
        return x[0]
    return 'NONE'

df_val = stats_ritmo.join(df['diagnostic_superclass'].apply(get_safe_label), how='inner')

display(Markdown("**Estatísticas de Ritmo por Superclasse (Mediana):**"))
display(df_val.groupby('diagnostic_superclass')[['fc_bpm', 'sd_RR']].median().round(1))""")

add_markdown("""### 3.2 Estabilidade do Sinal de 10 Sedungos
Calculamos o Zero Crossing Rate (ZCR) e a Variância para garantir que as janelas não contenham sinais mortos ou artefatos de amplitude irreal.""")

add_code("""# Nota: Como 'sinais' foi deletado, recarregamos de forma dinâmica via mmap para validação estatística
sinais_mmap = np.load(npy_path, mmap_mode='r')

# Cálculo de estabilidade na Lead II para uma amostra de 20% do dataset
sample_idx = np.random.choice(len(df), size=int(len(df)*0.2), replace=False)
zcr, var_s = [], []

for idx in sample_idx:
    s_test = sinais_mmap[idx, :, 1]
    zcr.append(np.sum(np.diff(np.sign(s_test)) != 0) / 1000)
    var_s.append(np.var(s_test))

df_stab = pd.DataFrame({'zcr': zcr, 'var': var_s})
display(Markdown(f"**Validação de Estabilidade (Amostra 20%):** ZCR Médio: {np.mean(zcr):.4f} | Variância Média: {np.mean(var_s):.4f}"))
del sinais_mmap, zcr, var_s""")

# =====================================================================
# SEÇÃO 5 - VISUALIZAÇÕES
# =====================================================================
add_markdown("""---
## Seção 4 — Visualizações Adicionais e Alinhamento

### 4.1 Validação de Alinhamento de Batimentos (Poincaré & Sobreposição)
Verificamos a precisão do detector de Pan-Tompkins através da sobreposição dos batimentos extraídos de um registro de controle.""")

add_code("""# Busca por um registro NORM com boa detecção
eid_norm = df_val[df_val['diagnostic_superclass'] == 'NORM'].index[2]
indices_norm_beats = df_bt_ids[df_bt_ids['ecg_id'] == eid_norm].index

plt.figure(figsize=(10, 6))
t = np.linspace(-200, 400, 60)
for idx in indices_norm_beats:
    plt.plot(t, matriz_bt[idx, :, 1], color='#3498db', alpha=0.3)

plt.plot(t, np.mean(matriz_bt[indices_norm_beats, :, 1], axis=0), color='#e74c3c', lw=3, label='Batimento Médio')
plt.title(f'Validação de Alinhamento (DII) — Registro {eid_norm} (NORM)')
plt.xlabel('Tempo (ms) relativo ao pico R')
plt.ylabel('mV')
plt.legend()
plt.savefig(os.path.join(FIGS_DIR, 'sobreposicao_beats.png'), dpi=150)
plt.show()""")

add_markdown("""### 4.2 Painel de Comparação por Superclasse (15 Exemplos)
Abaixo, apresentamos uma visão panorâmica de 3 registros para cada uma das 5 superclasses principais, permitindo uma inspeção visual rápida da qualidade dos sinais que seguem para a extração de features.""")

add_code("""sinais_tmp = np.load(npy_path, mmap_mode='r')
classes = ['NORM', 'MI', 'CD', 'STTC', 'HYP']
fig, axes = plt.subplots(len(classes), 3, figsize=(18, 15), sharex=True)

for i, c in enumerate(classes):
    # Seleção segura: pega até 3 exemplos, ou os que estiverem disponíveis
    c_examples = df_val[df_val['diagnostic_superclass'] == c].index
    n_show = min(len(c_examples), 3)
    
    for j in range(3):
        if j < n_show:
            eid = c_examples[j]
            loc = df.index.get_loc(eid)
            axes[i, j].plot(np.arange(1000)/100, sinais_tmp[loc, :, 1], color='black', lw=0.6)
            axes[i, j].set_title(f'{c} - ID {eid}', fontsize=10)
        else:
            axes[i, j].text(0.5, 0.5, 'N/A', ha='center', va='center')
            axes[i, j].set_axis_off()
        
        if j == 0: axes[i, j].set_ylabel('mV')
        if i == len(classes)-1: axes[i, j].set_xlabel('Tempo (s)')

plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'painel_15_exemplos.png'), dpi=200)
plt.show()
del sinais_tmp""")

# =====================================================================
# SEÇÃO 6 - EXPORTAÇÃO
# =====================================================================
add_markdown("""---
## Seção 5 — Exportação de Artefatos e Síntese Final

Concluímos a segmentação garantindo dois níveis de dados para a próxima etapa (Extração de Features):
1. **Nível Registro (10s):** Matriz original preservada para features globais.
2. **Nível Batimento (600ms):** Segmentos alinhados para features de morfologia.

**Arquivos Salvos em:** `outputs/`""")

add_code("""# Salvamento final
np.save(os.path.join(DIR_OUT_D5, 'batimentos_segmentados.npy'), matriz_bt)
df_bt_ids.to_csv(os.path.join(DIR_OUT_D5, 'batimentos_ids.csv'), index=False)

summary = f\"\"\"
- **Mapeamento 10s**: `registros_ids.csv` ({len(df_instancias_10s)} instâncias)
- **Matriz Batimentos**: `batimentos_segmentados.npy` ({matriz_bt.shape})
- **Metadados Batimentos**: `batimentos_ids.csv`
\"\"\"
display(Markdown(summary))""")

# =====================================================================
# SALVAR
# =====================================================================
out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'entregaveis', 'entregavel-5', 'notebooks'))
os.makedirs(out_dir, exist_ok=True)
notebook_path = os.path.join(out_dir, '05_segmentacao.ipynb')

with open(notebook_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(notebook, indent=2))

print(f"Notebook finalizado com sucesso: {notebook_path}")
print("Implementação completa seguindo Opção A + C e validações de estabilidade.")
