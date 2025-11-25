# Predição de Diabetes Mellitus Tipo 2 com Inteligência Artificial  
### Base: Vigitel 2023 (Ministério da Saúde)  
**Autor:** Heitor Augusto Freire Borges  
**Orientação:** Marcos Lopes e Silvia Brandão
**Curso:** [Sistemas de Informação] – Centro Universitário UNA  
**Projeto apresentado na EXPOUNA 2025.2**

## Descrição do Projeto
Desenvolvimento de um modelo capaz de prever o risco de **Diabetes Mellitus Tipo 2** utilizando apenas dados autorreferidos do **Vigitel 2023** (Ministério da Saúde).

O modelo utiliza 5 variáveis de fácil coleta:
- Idade, IMC, Hipertensão, Escolaridade, Atividade física.

Ideal para triagem populacional de baixo custo em UBS, aplicativos de saúde ou vigilância epidemiológica.

## Resultados Alcançados (Vigitel 2023 – n ≈ 21.689)
| Modelo              | AUC-ROC | Acurácia | F1-score |
|---------------------|---------|----------|----------|
| Gradient Boosting   | 0.857   | 83,4%    | 0,84     |
| Random Forest       | 0.870  | 81,9%    | 0,82     |
| Rede Neural         | 0.813   | 80,5%    | 0,81     |
| Regressão Logística | 0.787   | 77,1%    | 0,77     |

## Como Executar

1. **Baixe os dados do Vigitel 2023**  
   Link oficial: https://svs.aids.gov.br/download/Vigitel/  
   → Baixe o arquivo `Vigitel-2023-peso-rake.xlsx` e coloque na pasta raiz (ou use o link direto no código).

2. **Instale as dependências**
```bash
pip install -r requirements.txt
