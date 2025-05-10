# Évaluation des Modèles de Régression

Ce dépôt contient un article et des scripts pour évaluer la performance des modèles de régression en data science. L'article couvre les métriques clés (MAE, MSE, RMSE, R², R² ajusté, MAPE), l'analyse des résidus, et les courbes d'apprentissage, avec des exemples pratiques.

## Structure du dépôt

- `Article/`
  - `article.md` : Article détaillé sur l'évaluation des modèles de régression.
  - `plots/` : Dossier contenant les graphiques générés :
    - `regression_plot.png` : Nuage de points avec droite de régression.
    - `residuals_plot.png` : Graphique des résidus.
    - `learning_curve.png` : Courbe d'apprentissage.
- `generate_plots.py` : Script Python pour générer les graphiques et exécuter un exemple de prédiction des prix immobiliers.
- `README.md` : Ce fichier.

## Prérequis

- Python 3.8+
- Bibliothèques Python : `numpy`, `pandas`, `matplotlib`, `scikit-learn`
  Installez-les avec :
  ```bash
  pip install numpy pandas matplotlib scikit-learn
  ```

## Comment utiliser

1. Clone ce dépôt :
   ```bash
   git clone https://github.com/ton-compte/regression-article.git
   cd regression-article
   ```

2. Exécute le script pour générer les graphiques et voir l'exemple :
   ```bash
   python generate_plots.py
   ```

3. Consulte l'article dans `Article/article.md` et les graphiques dans `Article/plots/`.

## Contribution

N'hésite pas à ouvrir une issue ou une pull request pour suggérer des améliorations !