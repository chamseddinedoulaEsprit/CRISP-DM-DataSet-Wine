# Étape 1 : Business Understanding
# (Voir crisp-dm.txt pour détails)

# Étape 2 : Data Understanding
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA  # Ajout pour visualisation de features complexes

# Charger le dataset
wine = load_wine()
df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
df['class'] = wine.target  # Classes 0,1,2

# Mapper pour lisibilité (optionnel)
class_names = ['class_0', 'class_1', 'class_2']  # Correspond à cultivars italiens

# Statistiques descriptives
print(df.describe())

# Vérifier les valeurs manquantes
print(df.isnull().sum())  # Devrait être zéro

# Distribution des classes (légèrement déséquilibré)
print(df['class'].value_counts())

# Visualisations : Pairplot avec PCA pour réduire à 2D (car 13 features)
pca = PCA(n_components=2)
df_pca = pd.DataFrame(pca.fit_transform(df.drop('class', axis=1)), columns=['PC1', 'PC2'])
df_pca['class'] = df['class']
sns.pairplot(df_pca, hue='class')
plt.show()

# Corrélations (plus complexes qu'Iris)
corr = df.drop('class', axis=1).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

# Boxplots pour outliers
for feature in wine.feature_names:
    sns.boxplot(x='class', y=feature, data=df)
    plt.show()

# Étape 3 : Data Preparation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Features et target
X = df.drop('class', axis=1)
y = df['class']

# Détection outliers basique (IQR) et suppression optionnelle
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
outliers = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
print(f"Nombre d'outliers détectés : {outliers.sum()}")
# Option : df = df[~outliers]  # Supprimer si besoin, mais pour Wine, peu d'outliers

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling (essentiel pour features à échelles variées)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Taille train :", X_train.shape)
print("Taille test :", X_test.shape)

# Étape 4 : Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

models = {
    'Logistic Regression': LogisticRegression(max_iter=200),  # Augmenté pour convergence
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier()
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    results[name] = scores.mean()
    print(f"{name}: {scores.mean():.4f}")

# Tuning pour le meilleur (supposons Random Forest) - Ajout pour complexité
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_
print(f"Meilleurs params : {grid_search.best_params_}")

# Étape 5 : Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Prédictions
y_pred = best_model.predict(X_test_scaled)

# Métriques (focus sur macro pour déséquilibre)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=class_names))

# Visualisation confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.show()

# Étape 6 : Deployment
import joblib

# Sauvegarder
joblib.dump(best_model, 'wine_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Exemple de prédiction
def predict_wine(features_list):  # features_list : liste de 13 valeurs
    input_data = np.array([features_list])
    scaled = scaler.transform(input_data)
    pred = best_model.predict(scaled)
    return class_names[pred[0]]

# Test (exemple d'un échantillon de class_0)
test_features = [13.74, 1.67, 2.25, 16.4, 118, 2.6, 2.9, 0.21, 1.62, 5.85, 0.92, 3.2, 1060]  # Valeurs typiques
print(predict_wine(test_features))  # Devrait être 'class_0'