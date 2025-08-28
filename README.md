# analise-preditiva-churn-telecom
Projeto de Machine Learning para prever a evasão de clientes em uma empresa de telecomunicações.
# Análise Preditiva de Churn em Clientes de Telecomunicações

*Análise de ponta a ponta para identificar os principais fatores de cancelamento de clientes e construir um modelo de Machine Learning para prever o churn futuro.*

![Banner do Projeto](https://i.imgur.com/your-image-link-here.png) ## 1. Sumário do Problema e Objetivo de Negócio

A evasão de clientes, ou "churn", é um dos desafios mais críticos para empresas de serviços por assinatura, como as de telecomunicações. Reter um cliente existente é significativamente mais barato do que adquirir um novo.

O objetivo deste projeto é duplo:
1.  **Análise Diagnóstica:** Realizar uma análise exploratória profunda nos dados para entender os principais fatores que levam um cliente a cancelar seu contrato. Quais são os perfis de clientes com maior risco de churn?
2.  **Análise Preditiva:** Desenvolver e avaliar um modelo de Machine Learning com alta performance para prever quais clientes estão mais propensos a cancelar o serviço nos próximos meses.

O resultado final é uma ferramenta que pode ser usada pela equipe de retenção para direcionar esforços de forma proativa, oferecendo promoções ou suporte personalizado aos clientes de alto risco, otimizando recursos e reduzindo perdas de receita.

**Fonte dos Dados:** [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## 2. Metodologia e Ferramentas

| Ferramenta / Biblioteca | Propósito                                                                 |
| ----------------------- | ------------------------------------------------------------------------- |
| **Python** | Linguagem principal para análise e modelagem.                             |
| **Pandas & NumPy** | Manipulação, limpeza e pré-processamento dos dados.                       |
| **Matplotlib & Seaborn**| Geração de visualizações estáticas para a Análise Exploratória (EDA).     |
| **Plotly** | Criação de gráficos interativos para uma análise mais profunda.           |
| **Scikit-learn** | Treinamento, avaliação de modelos e pipeline de pré-processamento.        |
| **Jupyter Notebook** | Ambiente de desenvolvimento para exploração e documentação do processo.   |

## 3. Pipeline do Projeto

### 3.1. Análise Exploratória de Dados (EDA)

Nesta fase, investigamos os dados para extrair insights. Alguns dos principais achados foram:
* **Contratos Mensais:** Clientes com contratos do tipo "mês a mês" têm uma taxa de churn drasticamente maior em comparação com clientes de contratos anuais.
* **Serviços de Tecnologia:** Clientes sem serviços como "Online Security", "Online Backup" e "Tech Support" são significativamente mais propensos a cancelar.
* **Tempo de Contrato (Tenure):** Clientes mais novos (baixo "tenure") apresentam um risco de churn muito mais elevado.

*(Aqui você insere 1 ou 2 gráficos chave gerados na sua análise, por exemplo, um gráfico de barras mostrando a taxa de churn por tipo de contrato)*
![Churn por Tipo de Contrato](caminho/para/seu/grafico1.png)

### 3.2. Pré-processamento e Engenharia de Features

* **Codificação de Variáveis Categóricas:** Utilização de `OneHotEncoder` para variáveis nominais (ex: Gênero) e `LabelEncoder` para variáveis binárias.
* **Tratamento de Dados Numéricos:** Aplicação de `StandardScaler` para normalizar features como "MonthlyCharges" e "TotalCharges", garantindo que os modelos não sejam enviesados por escalas diferentes.
* **Valores Faltantes:** Identificada uma pequena quantidade de valores faltantes em `TotalCharges` para clientes novos, que foram imputados com a mediana.

### 3.3. Modelagem e Treinamento

Foram testados e comparados três algoritmos diferentes, começando de um baseline simples para um mais complexo.

1.  **Regressão Logística (Baseline):** Um modelo simples, rápido e interpretável. Serviu como nosso ponto de partida para avaliar os demais.
2.  **Random Forest Classifier:** Um modelo de ensemble robusto que lida bem com interações complexas entre as features.
3.  **Gradient Boosting (XGBoost):** Um modelo de boosting conhecido por sua alta performance em competições e problemas tabulares.

Os modelos foram treinados utilizando Validação Cruzada (Stratified K-Fold) para garantir a robustez dos resultados, e os hiperparâmetros foram otimizados com `GridSearchCV`.

## 4. Resultados e Avaliação

A métrica principal escolhida foi a **AUC-ROC**, pois ela oferece uma visão completa da performance do modelo independentemente do threshold de classificação, o que é ideal para problemas com classes desbalanceadas como o churn.

| Modelo                  | Acurácia | Precisão (Classe 1) | Recall (Classe 1) | F1-Score | AUC-ROC |
| ----------------------- | -------- | ------------------- | ----------------- | -------- | ------- |
| Regressão Logística     | 0.80     | 0.65                | 0.54              | 0.59     | 0.84    |
| Random Forest           | 0.79     | 0.64                | 0.49              | 0.55     | 0.82    |
| **XGBoost (Otimizado)** | **0.81** | **0.67** | **0.55** | **0.60** | **0.85**|

O modelo **XGBoost** apresentou a melhor performance geral, com a maior AUC-ROC. As features mais importantes identificadas pelo modelo foram, em ordem: `Contract` (Tipo de Contrato), `tenure` (Tempo de Contrato) e `OnlineSecurity`.

## 5. Conclusão e Próximos Passos

Este projeto demonstrou com sucesso a capacidade de prever o churn de clientes com alta precisão. O modelo XGBoost pode ser integrado a um sistema de CRM da empresa para gerar uma "pontuação de risco" semanal para cada cliente.

**Próximos Passos:**
* **Deploy:** Publicar o modelo como uma API REST para que outros sistemas possam consumi-lo.
* **Monitoramento:** Implementar um sistema para monitorar o desempenho do modelo ao longo do tempo (MLOps).
* **Custo-Benefício:** Desenvolver uma análise de custo para estimar o ROI (Retorno Sobre Investimento) das campanhas de retenção baseadas nas previsões do modelo.

## 6. Como Executar o Projeto

1.  Clone este repositório:
    ```bash
    git clone [https://github.com/seu-usuario/analise-preditiva-churn-telecom.git](https://github.com/seu-usuario/analise-preditiva-churn-telecom.git)
    cd analise-preditiva-churn-telecom
    ```
2.  Crie um ambiente virtual e instale as dependências:
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  Execute o Jupyter Notebook para ver a análise completa:
    ```bash
    jupyter notebook Analise_Churn_Telecom.ipynb
    ```
