import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Créer le dossier pour les graphiques
os.makedirs('Article/plots', exist_ok=True)

# Générer des données synthétiques
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X.flatten() + np.random.randn(100) * 2

# Modèle de régression
model = LinearRegression()
model.fit(X, y.reshape(-1, 1))
y_pred = model.predict(X)

# 1. Graphique de régression
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Données')
plt.plot(X, y_pred, color='red', label='Droite de régression')
plt.xlabel('Variable indépendante')
plt.ylabel('Variable dépendante')
plt.title('Régression linéaire')
plt.legend()
plt.grid(True)
plt.savefig('Article/plots/regression_plot.png')
plt.close()

# 2. Graphique des résidus
residuals = y - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, color='blue', alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Valeurs prédites')
plt.ylabel('Résidus')
plt.title('Graphique des résidus')
plt.grid(True)
plt.savefig('Article/plots/residuals_plot.png')
plt.close()

# 3. Courbe d'apprentissage
train_sizes, train_scores, test_scores = learning_curve(
    LinearRegression(), X, y, cv=5, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10)
)
train_scores_mean = -train_scores.mean(axis=1)
test_scores_mean = -test_scores.mean(axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, label='Erreur d\'entraînement')
plt.plot(train_sizes, test_scores_mean, label='Erreur de test')
plt.xlabel('Taille de l\'échantillon')
plt.ylabel('MSE')
plt.title('Courbe d\'apprentissage')
plt.legend()
plt.grid(True)
plt.savefig('Article/plots/learning_curve.png')
plt.close()

# 4. Exemple pratique : Prédiction des prix immobiliers
data = pd.DataFrame({
    'Taille_m2': np.random.rand(100) * 200,
    'Prix_euros': np.random.rand(100) * 100000 + 50000
})
data['Prix_euros'] = 2000 * data['Taille_m2'] + np.random.randn(100) * 10000

X = data[['Taille_m2']]
y = data['Prix_euros']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f} €")
print(f"RMSE: {rmse:.2f} €")
print(f"R²: {r2:.2f}")