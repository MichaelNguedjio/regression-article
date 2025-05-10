# Comment évaluer la performance de ton modèle de régression

Dans le monde de la data science, construire un modèle de régression, c'est bien, mais savoir s'il est performant, c'est encore mieux ! Que tu prédises des prix immobiliers, des ventes ou des températures, évaluer la qualité de ton modèle est essentiel pour garantir des prédictions fiables. Dans cet article, nous allons explorer les métriques clés pour évaluer une régression, analyser les résidus, et utiliser des courbes d'apprentissage pour détecter des problèmes comme le surapprentissage. Prêt à plonger dans l'univers des métriques ? C'est parti !

## 1. Les métriques essentielles pour évaluer une régression

Les métriques de régression permettent de quantifier l'écart entre les prédictions de ton modèle et les valeurs réelles. Voici les plus courantes, avec leurs forces et faiblesses.

### 1.1. Erreur Absolue Moyenne (MAE)
La **MAE** (Mean Absolute Error) mesure l'écart moyen absolu entre les prédictions et les valeurs réelles. Elle est robuste aux valeurs aberrantes (outliers).

**Formule :**
\[
\text{MAE} = \frac{1}{n} \sum_{i=1}^n |\hat{y}_i - y_i|
\]
où \( \hat{y}_i \) est la prédiction, \( y_i \) la valeur réelle, et \( n \) le nombre d'observations.

**Exemple** : Si ton modèle prédit des prix immobiliers et que la MAE est de 5 000 €, en moyenne, tes prédictions s'écartent de 5 000 € de la réalité.

**Avantages** : Facile à interpréter, moins sensible aux outliers.
**Limites** : Ne pénalise pas fortement les grosses erreurs.

### 1.2. Erreur Quadratique Moyenne (MSE)
La **MSE** (Mean Squared Error) calcule la moyenne des erreurs au carré. Elle pénalise davantage les grandes erreurs, ce qui peut être utile pour détecter des problèmes graves.

**Formule :**
\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2
\]

**Exemple** : Une MSE élevée indique que ton modèle fait de grosses erreurs sur certaines prédictions.

**Avantages** : Sensible aux grandes erreurs, utile pour l'optimisation.
**Limites** : Sensible aux outliers, unité au carré (difficile à interpréter directement).

### 1.3. Racine de l'Erreur Quadratique Moyenne (RMSE)
La **RMSE** (Root Mean Squared Error) est la racine carrée de la MSE. Elle est exprimée dans la même unité que la variable cible, ce qui la rend plus interprétable.

**Formule :**
\[
\text{RMSE} = \sqrt{\text{MSE}}
\]

**Exemple** : Une RMSE de 7 000 € signifie que, en moyenne, tes prédictions s'écartent de 7 000 €, avec une pénalisation des grandes erreurs.

**Avantages** : Interprétable, équilibre entre MAE et MSE.
**Limites** : Toujours sensible aux outliers.

### 1.4. Coefficient de détermination (R²)
Le **R²** mesure la proportion de la variance de la variable cible expliquée par le modèle. Il varie entre 0 et 1 (parfois négatif pour de mauvais modèles).

**Formule :**
\[
R^2 = 1 - \frac{\sum_{i=1}^n (\hat{y}_i - y_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}
\]
où \( \bar{y} \) est la moyenne des valeurs réelles.

**Exemple** : Un R² de 0,85 signifie que 85 % de la variance des prix immobiliers est expliquée par ton modèle.

**Avantages** : Sans unité, facile à comparer entre modèles.
**Limites** : Peut être trompeur si le modèle est surajusté.

### 1.5. R² ajusté
Le **R² ajusté** pénalise l'ajout de variables inutiles, contrairement au R² classique. Il est idéal pour comparer des modèles avec un nombre différent de prédicteurs.

**Formule :**
\[
R^2_{\text{ajusté}} = 1 - \left(1 - R^2\right) \frac{n-1}{n-p-1}
\]
où \( p \) est le nombre de prédicteurs.

**Exemple** : Si le R² est de 0,85 mais le R² ajusté tombe à 0,80, cela suggère que certaines variables n'apportent pas d'information utile.

**Avantages** : Évite le surajustement.
**Limites** : Moins intuitif pour les débutants.

### 1.6. Erreur en Pourcentage Absolu Moyen (MAPE)
La **MAPE** (Mean Absolute Percentage Error) mesure l'erreur relative en pourcentage, utile lorsque l'échelle de la variable cible varie beaucoup.

**Formule :**
\[
\text{MAPE} = \frac{100}{n} \sum_{i=1}^n \left| \frac{\hat{y}_i - y_i}{y_i} \right|
\]

**Exemple** : Une MAPE de 10 % signifie que tes prédictions s'écartent en moyenne de 10 % de la valeur réelle.

**Avantages** : Intuitif pour les variables à grande échelle.
**Limites** : Problématique si \( y_i = 0 \).

## 2. Analyse des résidus : un outil clé

Les métriques ne racontent pas toute l'histoire. L'**analyse des résidus** (différence entre valeurs prédites et réelles) permet de vérifier si ton modèle respecte les hypothèses de la régression linéaire :
- Les résidus doivent être distribués aléatoirement autour de 0.
- Ils doivent suivre une distribution normale.
- Leur variance doit être constante (homoscédasticité).

Un graphique de résidus (résidus vs valeurs prédites) peut révéler des problèmes comme une non-linéarité ou des outliers. Voir le script `generate_plots.py` pour générer ce graphique.

## 3. Visualiser la régression : Nuage de points et droite ajustée

Un graphique simple mais puissant pour évaluer une régression est le nuage de points avec la droite de régression. Ce graphique est généré par le script `generate_plots.py` et sauvegardé dans `Article/plots/regression_plot.png`.

## 4. Courbes d'apprentissage : Détecter le surapprentissage

Les **courbes d'apprentissage** permettent de visualiser la performance du modèle sur les données d'entraînement et de test en fonction de la taille de l'échantillon. Elles aident à détecter le surapprentissage (overfitting) ou le sous-apprentissage (underfitting). Ce graphique est généré par `generate_plots.py` et sauvegardé dans `Article/plots/learning_curve.png`.

## 5. Conseils pratiques pour choisir les bonnes métriques

- **Si les outliers sont un problème** : Privilégie la MAE, moins sensible aux valeurs extrêmes.
- **Si les grandes erreurs sont critiques** : Utilise la MSE ou la RMSE pour pénaliser les écarts importants.
- **Si tu compares des modèles complexes** : Le R² ajusté est ton allié pour éviter les modèles surajustés.
- **Si l'échelle varie beaucoup** : La MAPE est idéale pour interpréter les erreurs en pourcentage.
- **Combine les métriques** : Ne te fie pas à une seule métrique. Par exemple, un R² élevé avec une RMSE importante peut indiquer des problèmes.

N'oublie pas de valider ton modèle avec une **validation croisée** pour t'assurer que les performances sont robustes sur différentes parties des données.

## 6. Exemple pratique : Prédiction des prix immobiliers

Le script `generate_plots.py` inclut un exemple de prédiction des prix immobiliers avec un jeu de données synthétique, calculant les métriques MAE, RMSE, et R².

## 7. Conclusion

Évaluer la performance d’un modèle de régression, c’est comme vérifier la solidité d’une maison : il faut examiner plusieurs aspects (métriques, résidus, courbes d’apprentissage) pour s’assurer qu’elle tient debout. En combinant des métriques comme la MAE, RMSE, et R², en analysant les résidus, et en utilisant des courbes d’apprentissage, tu pourras non seulement évaluer ton modèle, mais aussi l’améliorer. Explore le code dans `generate_plots.py` pour reproduire les graphiques et tester sur tes propres données !