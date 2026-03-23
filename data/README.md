# Dados – PTB-XL

## Download

O dataset PTB-XL está disponível publicamente no PhysioNet:

**URL:** https://physionet.org/content/ptb-xl/1.0.3/

### Opção 1: Via `wfdb` (recomendado)

```bash
pip install wfdb
```

```python
import wfdb
wfdb.dl_database('ptb-xl', dl_dir='ptb-xl')
```

### Opção 2: Download manual

1. Acesse https://physionet.org/content/ptb-xl/1.0.3/
2. Baixe o arquivo completo (~2.4 GB)
3. Extraia para esta pasta (`data/ptb-xl/`)

## Estrutura do Dataset

```
ptb-xl/
├── records100/         # Sinais a 100 Hz (pasta por grupo)
├── records500/         # Sinais a 500 Hz (alta resolução)
├── ptbxl_database.csv  # Metadados clínicos
├── scp_statements.csv  # Descrição dos diagnósticos SCP
└── *.hea / *.dat       # Formato WFDB (header + dados binários)
```

## Formato dos Arquivos

Os sinais estão em formato **WFDB** (.hea + .dat):
- `.hea` — Header com metadados (canais, taxa de amostragem, ganho)
- `.dat` — Dados binários do sinal

**NÃO são .csv!** Usar a biblioteca `wfdb` em Python para leitura.

## Referência

Wagner, P. et al. "PTB-XL, a large publicly available electrocardiography dataset."
*Scientific Data*, v. 7, n. 1, p. 154, 2020.
