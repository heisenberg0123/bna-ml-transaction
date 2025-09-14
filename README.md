# BNA ML Transaction Classifier

Ce projet implémente un modèle de classification des transactions (urgent, standard, high_value) pour la banque BNA, intégré à l'application bancaire via une API Flask.

## Fonctionnalités
- Classification des transactions avec un modèle entraîné (ex. : DecisionTreeClassifier).
- Entraînement sur des données historiques de transactions.
- API REST pour fournir des prédictions en temps réel.
- Intégration avec le backend Spring Boot pour les requêtes de classification.

## Prérequis
- Python 3.8+
- Environnement virtuel (recommandé : `venv`).
- Backend Spring Boot pour les données de transactions.

## Installation
1. Clonez le dépôt :
   ```bash
   git clone git@github.com:votre-utilisateur/bna-ml-transaction.git
   cd bna-ml-transaction
