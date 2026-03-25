# Aquisição de Biossinais – Projeto da Disciplina

**Universidade Federal do Ceará (UFC)**  
**Departamento de Engenharia de Teleinformática**  
**Curso de Engenharia de Computação**

**Equipe:** José Lessa & Matheus Rocha  
**Orientador:** Prof. Dr. Victor Hugo C. de Albuquerque  
**Semestre:** 2026.1

---

## Objetivo

Preparar um dataset de biossinais (ECG) para aplicação de técnicas de Reconhecimento de Padrões (RP), percorrendo todo o pipeline de aquisição, pré-processamento, extração de atributos e validação estatística.

## Dataset

**PTB-XL** — Base de Dados de Eletrocardiografia  
- 21.799 registros de ECG de 12 derivações (10s cada)
- 18.869 pacientes
- 71 classes diagnósticas / 5 superclasses
- Disponível em: https://physionet.org/content/ptb-xl/1.0.3/

## Pipeline da Disciplina

| # | Etapa | Entregável | Status |
|---|-------|------------|--------|
| 0 | Estudo MNE + Surveys + Seleção de base | Entregável 0 | ✅ Concluído |
| 1 | Aquisição dos Biossinais | Entregável 1 | 🔄 Em andamento |
| 2 | Avaliação da Qualidade do Sinal (SQI) | Entregável 2 | ⬜ Pendente |
| 3 | Análise Estatística Inicial | Entregável 3 | ⬜ Pendente |
| 4 | Limpeza e Correção dos Dados | Entregável 4 | ⬜ Pendente |
| 5 | Segmentação (Janelamento) | Entregável 5 | ⬜ Pendente |
| 6 | Extração de Atributos | Entregável 6 | ⬜ Pendente |
| 7 | Engenharia de Features | Entregável 7 | ⬜ Pendente |
| 8 | Redução de Dimensionalidade | Entregável 8 | ⬜ Pendente |
| 9 | Seleção de Atributos | Entregável 9 | ⬜ Pendente |
| 10 | Dataset Final Validado | Entregável 10 | ⬜ Pendente |
| F | Relatório Final Integrador (TCC) | Entregável Final | ⬜ Pendente |

## Estrutura do Repositório

```
Biossinais/
├── README.md                    # Este arquivo
├── docs/                        # Documentos de referência
│   ├── research/                # Documentos que serviram de base ou compuseram a pesquisa deste projeto ( nem todos foram usados, alguns são só curiosidade )
│   ├── pipeline.pdf             # Pipeline da disciplina
│   └── Modelo_de_Trabalho_Academico_UFC.zip
├── entregaveis/                 # Entregáveis individuais
│   ├── entregavel-0/            # ✅ MNE + Surveys + Dataset
│   ├── entregavel-1/            # 🔄 Aquisição dos Biossinais
│   │   ├── notebooks/
│   │   └── figuras/
│   ├── entregavel-2/ ... entregavel-10/
├── documento-final/             # PDF final, figuras e notebooks do TCC
│   ├── figuras/
│   └── notebooks/
├── data/                        # Dados do dataset PTB-XL
└── scripts/                     # Scripts utilitários
```

## Como Usar

1. **Dados:** Siga as instruções em `data/README.md` para baixar o PTB-XL
2. **Notebooks:** Abra os Jupyter Notebooks em cada `entregaveis/entregavel-X/notebooks/`
3. **Relatórios:** Os PDFs dos entregáveis ficam em cada pasta de entregável
4. **Documento Final:** Compilado no Overleaf usando template UFC
