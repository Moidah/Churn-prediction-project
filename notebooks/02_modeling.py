"""
PASO 2 — Modelado: Predicción de Churn (Clasificación)
=========================================================
Objetivo: predecir qué clientes tienen más probabilidad de irse (churn),
comparando Regresión Logística vs Random Forest, y evaluando con métricas
apropiadas para clases desbalanceadas (NO solo accuracy).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUTS_DIR = SCRIPT_DIR.parent / "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ---------------------------------------------------------------
# 1. Cargar y preparar datos
# ---------------------------------------------------------------
df = pd.read_csv(DATA_DIR / "telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df = df.drop(columns=["customerID"])

# Variable objetivo: 1 = se fue, 0 = se quedó
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# --- Codificar variables categóricas (one-hot encoding) ---
cat_cols = df.select_dtypes(include="object").columns.tolist()
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df_encoded.drop(columns=["Churn"])
y = df_encoded["Churn"]

print(f"Features finales: {X.shape[1]}")
print(f"Balance de clases: {y.value_counts(normalize=True).round(3).to_dict()}")

# ---------------------------------------------------------------
# 2. Split train/test (estratificado para mantener el balance de clases)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} filas | Test: {len(X_test)} filas")

# Escalado (importante para Regresión Logística, no tanto para Random Forest)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def evaluar_modelo(y_true, y_pred, y_proba, nombre):
    print(f"\n{'=' * 60}\n{nombre}\n{'=' * 60}")
    print(classification_report(y_true, y_pred, target_names=["Se quedó", "Se fue (Churn)"]))
    auc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    print(f"AUC-ROC: {auc:.3f}  |  Average Precision: {ap:.3f}")
    return {"modelo": nombre, "auc_roc": auc, "avg_precision": ap}

resultados = []

# ---------------------------------------------------------------
# 3. Modelo 1 — Regresión Logística (baseline interpretable)
# ---------------------------------------------------------------
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
log_reg.fit(X_train_scaled, y_train)
pred_lr = log_reg.predict(X_test_scaled)
proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]
resultados.append(evaluar_modelo(y_test, pred_lr, proba_lr, "Regresión Logística"))

# ---------------------------------------------------------------
# 4. Modelo 2 — Random Forest
# ---------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, class_weight="balanced",
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)  # Random Forest no necesita escalado
pred_rf = rf.predict(X_test)
proba_rf = rf.predict_proba(X_test)[:, 1]
resultados.append(evaluar_modelo(y_test, pred_rf, proba_rf, "Random Forest"))

# ---------------------------------------------------------------
# 5. Matriz de confusión (Random Forest, el mejor de los dos normalmente)
# ---------------------------------------------------------------
cm = confusion_matrix(y_test, pred_rf)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Se quedó", "Se fue"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Se quedó", "Se fue"])
ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
ax.set_title("Matriz de confusión — Random Forest")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "05_matriz_confusion.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 6. Curva ROC — comparación de ambos modelos
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
for proba, nombre in [(proba_lr, "Regresión Logística"), (proba_rf, "Random Forest")]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, label=f"{nombre} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Aleatorio")
ax.set_xlabel("Falsos positivos"); ax.set_ylabel("Verdaderos positivos")
ax.set_title("Curva ROC — comparación de modelos")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "06_curva_roc.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 7. Importancia de variables (Random Forest) — clave para explicar a negocio
# ---------------------------------------------------------------
importancias = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
top15 = importancias.head(15)

fig, ax = plt.subplots(figsize=(9, 6))
top15.sort_values().plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Top 15 variables más importantes para predecir Churn")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "07_importancia_variables.png", dpi=120)
plt.close()

print("\nTop 10 variables más importantes:")
print(top15.head(10))

# ---------------------------------------------------------------
# 8. Resumen final
# ---------------------------------------------------------------
resultados_df = pd.DataFrame(resultados)
print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(resultados_df.to_string(index=False))
resultados_df.to_csv(OUTPUTS_DIR / "resultados_modelos.csv", index=False)

print(f"\n Modelado completo. Gráficas y resultados guardados en {OUTPUTS_DIR}/")
