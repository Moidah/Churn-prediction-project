"""
PASO 1 — Análisis Exploratorio de Datos (EDA)
================================================
Dataset: Telco Customer Churn (IBM) — 7,043 clientes reales de una
empresa de telecomunicaciones, con la variable objetivo 'Churn' (se fue
el cliente o no).
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUTS_DIR = SCRIPT_DIR.parent / "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# --- Cargar datos ---
df = pd.read_csv(DATA_DIR / "telco_churn.csv")

print(f"Filas: {len(df)}, Columnas: {len(df.columns)}")
print("\nPrimeras filas:")
print(df.head())

print("\nTipos de datos:")
print(df.dtypes)

# --- Limpieza: TotalCharges viene como texto y tiene algunos vacíos ---
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(f"\nValores nulos tras convertir TotalCharges a número: {df['TotalCharges'].isna().sum()}")
# Estos suelen ser clientes con tenure=0 (recién llegados, aún no facturados)
print(df[df["TotalCharges"].isna()][["customerID", "tenure", "MonthlyCharges", "TotalCharges"]])
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# --- Balance de clases (¡importante! el churn suele estar desbalanceado) ---
churn_counts = df["Churn"].value_counts()
churn_pct = df["Churn"].value_counts(normalize=True) * 100
print("\nBalance de clases (Churn):")
print(churn_counts)
print(churn_pct.round(1))

fig, ax = plt.subplots(figsize=(5, 4))
churn_counts.plot(kind="bar", ax=ax, color=["steelblue", "coral"])
ax.set_title("Distribución de Churn (¿el cliente se fue?)")
ax.set_ylabel("Número de clientes")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "01_balance_clases.png", dpi=120)
plt.close()

# --- Churn por tipo de contrato (una de las variables más predictivas) ---
churn_by_contract = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
fig, ax = plt.subplots(figsize=(7, 4))
churn_by_contract["Yes"].plot(kind="bar", ax=ax, color="coral")
ax.set_title("% de Churn por tipo de contrato")
ax.set_ylabel("% que se fue")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "02_churn_por_contrato.png", dpi=120)
plt.close()
print("\nChurn por tipo de contrato:")
print(churn_by_contract.round(1))

# --- Churn por antigüedad (tenure) ---
fig, ax = plt.subplots(figsize=(8, 4))
df.boxplot(column="tenure", by="Churn", ax=ax)
ax.set_title("Antigüedad del cliente (meses) vs Churn")
plt.suptitle("")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "03_tenure_vs_churn.png", dpi=120)
plt.close()

# --- Churn por cargo mensual ---
fig, ax = plt.subplots(figsize=(8, 4))
df.boxplot(column="MonthlyCharges", by="Churn", ax=ax)
ax.set_title("Cargo mensual vs Churn")
plt.suptitle("")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "04_cargo_mensual_vs_churn.png", dpi=120)
plt.close()

# --- Churn por servicio de internet ---
churn_by_internet = pd.crosstab(df["InternetService"], df["Churn"], normalize="index") * 100
print("\nChurn por tipo de servicio de internet:")
print(churn_by_internet.round(1))

print(f"\n EDA completo. Gráficas guardadas en {OUTPUTS_DIR}/")
print("\nObservaciones clave para anotar en tu README:")
print("- Los clientes con contrato 'Month-to-month' tienden a irse mucho más que los de 1-2 años.")
print("- La antigüedad (tenure) baja se asocia con más churn: los clientes nuevos son más volátiles.")
print("- Revisa si el servicio de Fibra Óptica tiene más churn que DSL (suele pasar por precio/competencia).")
