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
add_markdown("""# Entregável 1: Aquisição de Biossinais e Caracterização do Dataset PTB-XL
**Disciplina:** Aquisição de Biossinais
**Autor(es):** José Ferreira Lessa e Matheus Rocha Gomes da Silva
**Data:** Março de 2026""")

# =====================================================================
# CÉLULA 2 - OBJETIVO
# =====================================================================
add_markdown("""## Objetivo
Este notebook tem como objetivo documentar as condições originais de aquisição do dataset PTB-XL, caracterizar a base de dados de forma demográfica e clínica, e realizar inspeções visuais do sinal bruto de ECG. Aqui não fazemos nenhuma transformação nos sinais — a saída é um arquivo de metadados enriquecido (`ptbxl_metadata_enriched.csv`) que servirá de base para todos os notebooks subsequentes do pipeline.""")

# =====================================================================
# CÉLULA 3 - IMPORTAÇÕES
# =====================================================================
add_markdown("""## 1. Importações e Dependências
Bibliotecas necessárias: `pandas` e `numpy` para manipulação de dados, `matplotlib` e `seaborn` para visualizações, `wfdb` para leitura dos sinais no formato PhysioNet, e `scipy` para análise espectral.""")

add_code("""import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wfdb
import scipy.signal as signal
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
add_markdown("""## 2. Configurações Globais
Definimos aqui as constantes do projeto. Optamos pela amostragem de **100 Hz** pois cobre a banda diagnóstica do ECG (0,05–40 Hz) com margem, respeitando o critério de Nyquist (`fs ≥ 80 Hz`), e reduz o custo computacional em 5× comparado aos 500 Hz.""")

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
# CÉLULA 5 - LEITURA DOS DADOS
# =====================================================================
add_markdown("""## 3. Carregamento dos Metadados Base
O PTB-XL organiza seus metadados em dois arquivos CSV principais. O `ptbxl_database.csv` contém informações de cada registro (idade, sexo, diagnósticos) e o `scp_statements.csv` traz o dicionário de códigos diagnósticos SCP-ECG.""")

add_code("""print("Carregando ptbxl_database.csv...")
df = pd.read_csv(os.path.join(PATH_DATA, 'ptbxl_database.csv'), index_col='ecg_id')

# Convertendo a coluna scp_codes de string para dicionário Python
df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)

print("Carregando scp_statements.csv...")
scp_statements = pd.read_csv(os.path.join(PATH_DATA, 'scp_statements.csv'), index_col=0)

print(f"Dataset carregado: {df.shape[0]} registros e {df.shape[1]} colunas originais.")""")

# =====================================================================
# SEÇÃO 1 - CONTEXTUALIZAÇÃO
# =====================================================================
add_markdown("""---
## Seção 1 — Contextualização do Dataset

### 1.1 O Sinal de ECG e sua Relevância Clínica
O Eletrocardiograma (ECG) registra a atividade elétrica do coração ao longo do tempo através de eletrodos posicionados na superfície corporal. O padrão clínico utiliza 12 derivações que oferecem perspectivas complementares do mesmo fenômeno elétrico:

- **Derivações de membros** (I, II, III, aVL, aVR, aVF): visualizam o plano frontal do coração.
- **Derivações precordiais** (V1–V6): visualizam o plano horizontal.

Cada derivação mostra a mesma sequência — despolarização atrial (onda P), despolarização ventricular (complexo QRS) e repolarização ventricular (onda T) — sob ângulos diferentes. Essa redundância é necessária: certos diagnósticos só aparecem em derivações específicas. Por exemplo, infarto anterior se manifesta em V1–V4; infarto inferior aparece em II, III e aVF.""")

add_code("""# 1.2 Protocolo de Aquisição Original
n_registros = len(df)
n_pacientes = df['patient_id'].nunique()
n_devices = df['device'].nunique()
n_sites = df['site'].nunique()

display(Markdown(f\"\"\"
### 1.2 Protocolo de Aquisição Original

| Parâmetro | Valor |
|---|---|
| **Equipamento** | Dispositivos Schiller AG (fabricante suíço) |
| **Período de coleta** | Outubro de 1989 a Junho de 1996 |
| **Instituição** | Physikalisch-Technische Bundesanstalt (PTB), Berlim, Alemanha |
| **Registros** | {n_registros} registros de 10 segundos |
| **Pacientes** | {n_pacientes} pacientes distintos |
| **Dispositivos** | {n_devices} modelos diferentes em {n_sites} sites |
| **Derivações** | 12 padrão (I, II, III, aVL, aVR, aVF, V1–V6) |
| **Freq. original** | 400 Hz → reamostrado p/ 500 Hz (interpolação cúbica) |
| **Freq. downsampled** | 100 Hz (decimação do sinal de 500 Hz) |
| **Precisão** | 16 bits, resolução de 1 µV/LSB |
| **Formato** | WFDB (PhysioNet), arquivos `.dat` + `.hea` |
\"\"\"))""")

add_markdown("""### 1.3 Justificativa do Critério de Nyquist
Pelo Teorema de Nyquist-Shannon, para reconstruir um sinal com componente máxima `f_max`, a taxa de amostragem precisa ser `fs ≥ 2 × f_max`. A banda de interesse diagnóstica do ECG vai de 0,05 Hz até cerca de 40 Hz, o que exige no mínimo 80 Hz. Portanto, a taxa de 100 Hz que adotamos satisfaz o critério com folga, e os 500 Hz só seriam necessários para análise de micro-potenciais tardios ou deep learning com resolução temporal fina.""")

add_code("""# 1.4 Processo de Anotação dos Labels
pct_human = df['validated_by_human'].value_counts(normalize=True).get(True, 0) * 100
pct_auto = df['initial_autogenerated_report'].value_counts(normalize=True).get(True, 0) * 100

display(Markdown(f\"\"\"
### 1.4 Processo de Anotação dos Labels
O processo de rotulagem do PTB-XL seguiu múltiplas etapas: laudo inicial (por cardiologista ou gerado automaticamente), conversão para códigos SCP-ECG, revisão por segundo cardiologista, e anotação de qualidade do sinal.

- **{pct_human:.1f}%** dos registros possuem validação humana confirmada (`validated_by_human = True`).
- **{pct_auto:.1f}%** tiveram o laudo inicial gerado automaticamente pelo dispositivo ECG.
\"\"\"))

display(df['validated_by_human'].value_counts())
display(df['initial_autogenerated_report'].value_counts())""")

add_markdown("""### 1.5 Verossimilhança dos Diagnósticos (Likelihood)
A coluna `scp_codes` armazena um dicionário onde cada chave é um código SCP e o valor é a verossimilhança (0–100%), que reflete o grau de certeza do cardiologista:

| Likelihood | Significado |
|---|---|
| 0% | Statement de forma/ritmo (binário) ou certeza mínima |
| 15% | "Não se pode excluir" |
| 50% | "Possivelmente", "talvez" |
| 80% | "Compatível com" |
| 100% | Diagnóstico confirmado |

Para classificação, adotaremos o limiar de **likelihood ≥ 50** para considerar um diagnóstico como positivo.""")

add_code("""# 1.6 Estrutura do Arquivo de Metadados — visão geral das colunas
display(Markdown("### 1.6 Estrutura do Arquivo de Metadados"))
display(Markdown("**Amostra dos dados brutos:**"))
display(df[['patient_id', 'age', 'sex', 'height', 'weight', 'report', 'scp_codes', 'validated_by_human', 'strat_fold']].head(5))

# Completude dos campos
pct_h = df['height'].notna().mean() * 100
pct_w = df['weight'].notna().mean() * 100
pct_ax = df['heart_axis'].notna().mean() * 100

display(Markdown(f\"\"\"
**Completude dos campos opcionais:**
- Altura: {pct_h:.1f}% preenchidos
- Peso: {pct_w:.1f}% preenchidos
- Eixo cardíaco: {pct_ax:.1f}% preenchidos
\"\"\"))""")

# =====================================================================
# SEÇÃO 2 - PANORAMA ESTATÍSTICO
# =====================================================================
add_markdown("""---
## Seção 2 — Panorama Estatístico do Dataset

### 2.1 Distribuição Demográfica""")

add_code("""# Limpeza da idade (valores de 300 representam pacientes >= 90 anos, anonimizados)
df['age_clean'] = df['age'].replace(300, np.nan)

mediana_idade = df['age_clean'].median()
iqr_idade = df['age_clean'].quantile(0.75) - df['age_clean'].quantile(0.25)

# --- Plot 1: Idade por Sexo + Pizza ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.histplot(data=df, x='age_clean', hue='sex', bins=30,
             kde=True, ax=axes[0], alpha=0.5, palette=['#4e82c7', '#c2425b'])
axes[0].set_title('Distribuição de Idade por Sexo')
axes[0].set_xlabel('Idade (anos)')
axes[0].set_ylabel('Frequência')
axes[0].axvline(mediana_idade, color='black', linestyle='--', label=f'Mediana: {mediana_idade:.0f}')
axes[0].legend()

sex_pct = df['sex'].value_counts(normalize=True)
axes[1].pie(sex_pct, labels=['Masculino (0)', 'Feminino (1)'], autopct='%1.1f%%',
            colors=['#4e82c7', '#c2425b'], explode=(0, 0.04), shadow=True, startangle=90)
axes[1].set_title('Distribuição por Sexo')

plt.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, 'distribuicao_demografica.png'), dpi=150, bbox_inches='tight')
plt.show()""")

add_code("""# --- Plot 2: Peso e Altura ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

pct_w = df['weight'].notna().mean() * 100
pct_h = df['height'].notna().mean() * 100

sns.histplot(data=df, x='weight', kde=True, ax=axes[0], color='#8a6dab')
axes[0].set_title(f'Peso ({pct_w:.1f}% preenchidos)')
axes[0].set_xlabel('Peso (kg)')

sns.histplot(data=df, x='height', kde=True, ax=axes[1], color='#328059')
axes[1].set_title(f'Altura ({pct_h:.1f}% preenchidos)')
axes[1].set_xlabel('Altura (cm)')

plt.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, 'peso_altura.png'), dpi=150, bbox_inches='tight')
plt.show()""")

add_code("""# --- Plot 3: Dispositivos ---
fig = plt.figure(figsize=(12, 5))
df_devices = df['device'].value_counts()
sns.barplot(x=df_devices.index, y=df_devices.values, palette='crest')
plt.title('Registros por Modelo de Dispositivo')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Quantidade de registros')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'dispositivos.png'), dpi=150, bbox_inches='tight')
plt.show()

display(Markdown("**(Espaço para comentário do aluno — dica: comente sobre o balanceamento demográfico do dataset, se a representação masculino/feminino é adequada, e se a distribuição de idade faz sentido para um dataset clínico cardíaco.)**"))""")

# --- SUPERCLASSES ---
add_markdown("""### 2.2 Criação das Colunas de Superclasse Diagnóstica
O PTB-XL utiliza códigos SCP-ECG detalhados, mas para o nosso pipeline de classificação é mais prático agrupá-los em 5 superclasses diagnósticas: **NORM** (normal), **MI** (infarto do miocárdio), **CD** (distúrbio de condução), **STTC** (alterações ST/T) e **HYP** (hipertrofia). A informação de mapeamento vem do próprio `scp_statements.csv`.""")

add_code("""df_diagnostic = scp_statements[scp_statements.diagnostic == 1]
scp_to_superclass = dict(zip(df_diagnostic.index, df_diagnostic.diagnostic_class))

def get_superclasses(scp_dict, threshold=50):
    \"\"\"Retorna lista de superclasses cujos diagnósticos têm likelihood >= threshold.\"\"\"
    superclasses = set()
    for scp, likelihood in scp_dict.items():
        if scp in scp_to_superclass and likelihood >= threshold:
            superclasses.add(scp_to_superclass[scp])
    return list(superclasses)

df['diagnostic_superclass'] = df['scp_codes'].apply(get_superclasses)
df['n_superclasses'] = df['diagnostic_superclass'].apply(len)

multi_label = df['n_superclasses'].value_counts().sort_index()
display(Markdown("**Distribuição multi-label** (quantidade de superclasses simultâneas por registro):"))
display(pd.DataFrame({'N Superclasses': multi_label.index, 'Registros': multi_label.values}))

display(Markdown("**(Espaço para comentário do aluno — dica: discuta o que significa ter registros com 0, 1, 2 ou mais superclasses. Registros com 0 superclasses geralmente possuem apenas statements de forma/ritmo.)**"))""")

# --- LABELS ---
add_markdown("""### 2.3 Distribuição de Labels Diagnósticos""")

add_code("""all_superclasses = [cls for array in df['diagnostic_superclass'] for cls in array]
freq_super = pd.Series(all_superclasses).value_counts()

fig = plt.figure(figsize=(10, 5))
sns.barplot(x=freq_super.values, y=freq_super.index, palette='magma')
plt.title('Frequência Absoluta das Superclasses Diagnósticas (Multi-Label)')
plt.xlabel('Quantidade de registros')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'superclasses_frequencia.png'), dpi=150, bbox_inches='tight')
plt.show()

display(pd.DataFrame({'Registros': freq_super, '% do total': (freq_super / len(df) * 100).round(1)}))

display(Markdown("**(Espaço para comentário do aluno — dica: identifique qual é a classe dominante, se existe desbalanceamento entre classes, e o que isso pode implicar para o treinamento dos modelos.)**"))""")

# --- RITMO / FORMA ---
add_markdown("""### 2.4 Distribuição de Statements de Forma e Ritmo
Além dos diagnósticos, o PTB-XL rotula os registros com códigos de **ritmo** (e.g., ritmo sinusal, AFIB) e **forma** (e.g., QRS anormal). Esses não possuem graduação de likelihood — são presença/ausência.""")

add_code("""# Ritmos
df_rhythm = scp_statements[scp_statements.rhythm == 1]
def get_rhythms(scp_dict):
    return [scp for scp in scp_dict.keys() if scp in df_rhythm.index]

all_rhythms = pd.Series([r for subset in df['scp_codes'].apply(get_rhythms) for r in subset]).value_counts()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

sns.barplot(x=all_rhythms.head(8).values, y=all_rhythms.head(8).index, palette='viridis', ax=axes[0])
axes[0].set_title('Top 8 Ritmos Cardíacos')
axes[0].set_xlabel('Ocorrências')

# Formas
df_form = scp_statements[scp_statements.form == 1]
def get_forms(scp_dict):
    return [scp for scp in scp_dict.keys() if scp in df_form.index]

all_forms = pd.Series([f for subset in df['scp_codes'].apply(get_forms) for f in subset]).value_counts()

sns.barplot(x=all_forms.head(8).values, y=all_forms.head(8).index, palette='crest', ax=axes[1])
axes[1].set_title('Top 8 Statements de Forma')
axes[1].set_xlabel('Ocorrências')

plt.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, 'ritmo_forma.png'), dpi=150, bbox_inches='tight')
plt.show()

display(Markdown("**(Espaço para comentário do aluno — dica: note que o ritmo sinusal normal (SR) domina amplamente. AFIB é a arritmia mais comum. Discuta a relevância clínica disso.)**"))""")

# --- MULTIPLICIDADE ---
add_markdown("""### 2.5 Multiplicidade de Registros por Paciente
Um paciente pode ter feito mais de um exame ao longo dos anos. É fundamental que todos os registros de um mesmo paciente estejam no mesmo fold do cross-validation para evitar data leakage.""")

add_code("""records_per_patient = df.groupby('patient_id').size().value_counts().sort_index()
display(pd.DataFrame({'Registros por paciente': records_per_patient.index, 'Quantidade de pacientes': records_per_patient.values}))

# Verificação de integridade: todos os registros de um paciente estão no mesmo fold?
leakage_check = df.groupby('patient_id')['strat_fold'].nunique().max()
if leakage_check == 1:
    display(Markdown("✅ **Verificação anti-leakage aprovada:** todos os registros de um mesmo paciente estão contidos no mesmo fold. Sem risco de contaminação entre treino e teste."))
else:
    display(Markdown("⚠️ **Atenção:** encontrados pacientes em mais de um fold. Investigar!"))""")

# =====================================================================
# SEÇÃO 3 - SINAL BRUTO
# =====================================================================
add_markdown("""---
## Seção 3 — Carregamento e Visualização do Sinal Bruto

### 3.1 Função de Carregamento com WFDB""")

add_code("""def load_ecg(ecg_id, dataframe, path_base, fs=100):
    \"\"\"Carrega o sinal de ECG usando a biblioteca wfdb.\"\"\"
    linha = dataframe.loc[ecg_id]
    file_target = linha['filename_lr'] if fs == 100 else linha['filename_hr']
    signal_arr, fields = wfdb.rdsamp(os.path.join(path_base, file_target))
    return signal_arr, fields""")

add_code("""# 3.2 Seleção de Registros Representativos
df['is_pure_norm'] = df['diagnostic_superclass'].apply(lambda x: x == ['NORM'])
df['has_quality_issues'] = df[['baseline_drift', 'static_noise', 'burst_noise', 'electrodes_problems']].notna().any(axis=1)

# Buscando exemplares variados por filtros programáticos
try:
    id_norm = df[(df['is_pure_norm']) & (df['validated_by_human'] == True) & (~df['has_quality_issues'])].index[0]
except IndexError:
    id_norm = df[df['is_pure_norm']].index[0]

try:
    id_mi = df[df['diagnostic_superclass'].apply(lambda x: 'MI' in x)].index[0]
except IndexError:
    id_mi = df.index[1]

try:
    id_afib = df[df['scp_codes'].apply(lambda d: 'AFIB' in d)].index[0]
except IndexError:
    id_afib = df.index[2]

try:
    id_baseline = df[df['baseline_drift'].notna()].index[0]
except IndexError:
    id_baseline = df.index[3]

registros_selecionados = {
    'NORM (sinal limpo)': id_norm,
    'MI (infarto)': id_mi,
    'AFIB (fibrilação atrial)': id_afib,
    'Baseline drift': id_baseline
}

display(Markdown("**Registros selecionados para visualização:**"))
for desc, eid in registros_selecionados.items():
    display(Markdown(f"- **{desc}**: ECG ID {eid} | Superclasse: {df.loc[eid, 'diagnostic_superclass']}"))""")

add_code("""# 3.3 Visualização das 12 Derivações
def plot_clinical_ecg(ecg_id, titulo_extra=""):
    sig, header = load_ecg(ecg_id, df, PATH_DATA, FS)
    t = np.arange(sig.shape[0]) / FS

    fig, axes = plt.subplots(2, 6, figsize=(20, 8), sharex=True, sharey=True)
    fig.suptitle(f'ECG 12 Derivações — ID: {ecg_id} | {titulo_extra}', fontsize=14)

    for i in range(12):
        row, col = i // 6, i % 6
        ax = axes[row, col]
        ax.plot(t, sig[:, i], color='black', linewidth=0.8)
        ax.set_title(LEAD_NAMES[i], loc='left', fontweight='bold')
        ax.set_ylim(-2, 2)
        ax.grid(alpha=0.3)
        if row == 1:
            ax.set_xlabel('Tempo (s)')
        if col == 0:
            ax.set_ylabel('Amplitude (mV)')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, f'ecg_12leads_{ecg_id}.png'), dpi=150, bbox_inches='tight')
    plt.show()

for desc, eid in registros_selecionados.items():
    plot_clinical_ecg(eid, titulo_extra=desc)

display(Markdown("**(Espaço para comentário do aluno — dica: compare visualmente as morfologias. No registro NORM, espere ondas P, QRS e T bem definidas. No MI, procure por alterações de ST. No AFIB, note a ausência de onda P. No baseline drift, observe a oscilação lenta do eixo.)**"))""")

# --- ESPECTRO ---
add_markdown("""### 3.4 Análise Espectral Visual do Sinal Bruto (Derivação DII)
Usando o método de Welch, visualizamos a distribuição de energia no espectro de frequências. Isso nos dá uma visão do "antes" que será comparada com o "depois" da filtragem no Entregável 4.""")

add_code("""fig = plt.figure(figsize=(12, 5))

for desc, eid in registros_selecionados.items():
    sig, _ = load_ecg(eid, df, PATH_DATA, FS)
    dii = sig[:, 1]
    freqs, psd = signal.welch(dii, fs=FS, nperseg=256)
    psd_db = 10 * np.log10(psd + 1e-10)
    plt.plot(freqs, psd_db, label=f'ID {eid} ({desc})')

plt.axvline(0.5, color='gray', linestyle=':', alpha=0.7, label='Passa-alta (0.5 Hz)')
plt.axvline(40, color='gray', linestyle='--', alpha=0.7, label='Passa-baixa (40 Hz)')
plt.axvline(50, color='red', linestyle='-.', alpha=0.4, label='Rede elétrica (50 Hz)')

plt.xlim(0, 50)
plt.xlabel('Frequência (Hz)')
plt.ylabel('PSD (dB)')
plt.title('Espectro de Potência — Derivação DII (Welch)')
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'espectro_welch_bruto.png'), dpi=150, bbox_inches='tight')
plt.show()

display(Markdown("**(Espaço para comentário do aluno — dica: observe como a energia se concentra abaixo de 30 Hz para a maioria dos registros. Verifique se algum registro apresenta pico próximo de 50 Hz, indicando interferência de rede.)**"))""")

# =====================================================================
# SEÇÃO 4 - ENRIQUECIMENTO E SALVAMENTO
# =====================================================================
add_markdown("""---
## Seção 4 — Enriquecimento do Metadado e Salvamento

### 4.1 Adição de Colunas Calculadas""")

add_code("""def get_split(fold):
    if fold in FOLDS_TREINO:
        return 'train'
    elif fold == FOLD_VAL:
        return 'val'
    elif fold == FOLD_TEST:
        return 'test'
    return 'unknown'

df['split'] = df['strat_fold'].apply(get_split)

# Limpando colunas auxiliares
df.drop(columns=['age_clean'], inplace=True, errors='ignore')""")

add_code("""# 4.2 Salvamento
caminho_resultado = '../outputs/ptbxl_metadata_enriched.csv'
df.to_csv(caminho_resultado)

n_train = len(df[df.split == 'train'])
n_val = len(df[df.split == 'val'])
n_test = len(df[df.split == 'test'])
n_problemas = df.has_quality_issues.sum()

display(Markdown(f\"\"\"
**Arquivo salvo em:** `{caminho_resultado}`

| Split | Registros |
|---|---|
| Treino (folds 1–8) | {n_train} |
| Validação (fold 9) | {n_val} |
| Teste (fold 10) | {n_test} |

Registros com problemas de qualidade anotados: **{n_problemas}**
\"\"\"))""")

# =====================================================================
# SEÇÃO 5 - SÍNTESE
# =====================================================================
add_markdown("""---
## Seção 5 — Síntese e Conexão

Neste primeiro entregável, o dataset PTB-XL foi carregado, inspecionado e documentado. Realizamos a caracterização demográfica e clínica, criamos o mapeamento para superclasses diagnósticas, verificamos a integridade dos folds contra data leakage, e visualizamos o sinal bruto em múltiplos cenários clínicos.

Os metadados foram enriquecidos com as colunas calculadas e salvos em `ptbxl_metadata_enriched.csv`. O próximo passo (Entregável 2) irá quantificar objetivamente a qualidade dos sinais usando métricas como SNR, kurtosis e entropia espectral.

**(Espaço para comentário do aluno — dica: faça um resumo pessoal das descobertas mais relevantes deste entregável: distribuição demográfica, presença de multi-label, qualidade geral do dataset.)**
""")

# =====================================================================
# SALVAR NOTEBOOK
# =====================================================================
out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'entregaveis', 'entregavel-1', 'notebooks'))
os.makedirs(out_dir, exist_ok=True)
notebook_path = os.path.join(out_dir, '01_aquisicao_biossinais.ipynb')

with open(notebook_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(notebook, indent=2))

print(f"Notebook criado com sucesso: {notebook_path}")
