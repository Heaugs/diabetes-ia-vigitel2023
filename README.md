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
| Gradient Boosting   | 0.857   | 77,8%    | 0,79     |
| Random Forest       | 0.870   | 80,1%    | 0,81     |
| Rede Neural         | 0.813   | 76,6%    | 0,77     |
| Regressão Logística | 0.787   | 74,5%    | 0,75     |

## Como Executar

1. **Baixe os dados do Vigitel 2023**  
   Link oficial: https://svs.aids.gov.br/download/Vigitel/  
   → Baixe o arquivo `Vigitel-2023-peso-rake.xlsx` e coloque na pasta raiz (ou use o link direto no código).

2. **Instale as dependências**
```bash
pip install -r requirements.txt
