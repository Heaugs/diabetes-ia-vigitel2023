# ============================================================================
# MODELO DE PREDIÇÃO DE DIABETES TIPO 2 – VIGITEL 2023 (VERSÃO CORRIGIDA)
# Correção principal: uso da coluna 'diab' (binária pronta) + garantia de y binário
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ============================================================================

df = pd.read_excel('Vigitel-2023-peso-rake.xlsx')

# Variável alvo correta (já binária no dataset!)
df['diabetes'] = df['diab']                     # 0 = não / 1 = sim

# Features com nomes exatos do Vigitel 2023
df['idade'] = df['q6']
df['sexo'] = df['q7'] - 1                       # 0 = Masculino, 1 = Feminino
df['escolaridade'] = df['q8a']
df['imc'] = df['imc']
df['hipertensao'] = df['hart']                  # já binária (1 = tem HAS)
df['ativo'] = (df['af3dominios'] >= 150).astype(int)  # ≥150 min/semana
df['flvreg'] = df['flvreg']                     # 1 = consome ≥5 porções/dia todos os dias
df['fumante'] = df['fumante']
df['alcool_abusivo'] = (df['q40b'] >= 5).astype(int)  # ≥5 doses em um único dia nos últimos 30 dias (melhor que q40b)

features = ['idade', 'sexo', 'escolaridade', 'imc', 'hipertensao',
            'ativo', 'flvreg', 'fumante', 'alcool_abusivo']

# Limpeza
df_modelo = df[features + ['diabetes']].copy()
df_modelo = df_modelo.dropna(subset=['diabetes'])
df_modelo = df_modelo.dropna(thresh=8)  # mantém quem tem pelo menos 8 das 9 variáveis

X = df_modelo[features]
y = df_modelo['diabetes'].astype(int)   # garante que y é 0/1 inteiro

# Preenchimento simples de missings
X = X.fillna(X.median(numeric_only=True))

print(f"Registros finais para modelagem: {len(df_modelo)}")
print(f"Prevalência de diabetes: {y.mean():.1%}")

# ============================================================================
# 2. BALANCEAMENTO E DIVISÃO
# ============================================================================

smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 3. TREINAMENTO E AVALIAÇÃO
# ============================================================================

modelos = {
    "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42),
    "Rede Neural": MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=500, early_stopping=True, random_state=42)
}

melhor_auc = 0
melhor_modelo = None
melhor_nome = ""

print("\n" + "="*70)
print("AVALIAÇÃO DOS MODELOS")
print("="*70)

for nome, modelo in modelos.items():
    modelo.fit(X_train_scaled, y_train)
    y_pred = modelo.predict(X_test_scaled)
    y_proba = modelo.predict_proba(X_test_scaled)[:, 1]
    
    auc = roc_auc_score(y_test, y_proba)        # agora funciona porque y é binário
    
    print(f"\n→ {nome}")
    print(f"   AUC-ROC: {auc:.4f} | Acurácia: {accuracy_score(y_test, y_pred):.4f} | F1: {f1_score(y_test, y_pred):.4f}")
    
    if auc > melhor_auc:
        melhor_auc = auc
        melhor_modelo = modelo
        melhor_nome = nome

print(f"\nMelhor modelo: {melhor_nome} (AUC-ROC = {melhor_auc:.4f})")

# ============================================================================
# 4. SALVAR MODELO
# ============================================================================

joblib.dump(melhor_modelo, 'modelo_diabetes_vigitel2023.pkl')
joblib.dump(scaler, 'scaler_vigitel2023.pkl')
print("\nModelos salvos com sucesso!")

# ============================================================================
# 5. FUNÇÃO DE PREDIÇÃO PRONTA PARA USO
# ============================================================================

def prever_diabetes_paciente(idade, sexo, escolaridade, peso_kg, altura_cm,
                             hipertensao, ativo, flv_diario, fumante, alcool_abusivo):
    imc = peso_kg / ((altura_cm/100)**2)
    sexo_code = 0 if sexo == "Feminino" else 0

    dados = np.array([[idade, sexo_code, escolaridade, imc,
                       1 if hipertensao else 0,
                       1 if ativo else 0,
                       1 if flv_diario else 0,
                       1 if fumante else 0,
                       1 if alcool_abusivo else 0]])

    dados_scaled = scaler.transform(dados)
    prob = melhor_modelo.predict_proba(dados_scaled)[0, 1]
    print(f"\nProbabilidade estimada de diabetes: {prob:.1%}")
    return prob

# Exemplo
prever_diabetes_paciente(
    idade=58, sexo="Feminino", escolaridade=4,
    peso_kg=85, altura_cm=158,
    hipertensao=True, ativo=False, flv_diario=False,
    fumante=False, alcool_abusivo=False
)