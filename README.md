# 🎓 Certification IA Developer - Alyra 

> Projets de certification  
> **Promotion Douglas Adams** | 

---

## 📋 Projets

| Projet | Type | Objectif | Résultat |
|--------|------|----------|----------|
| **ML-Diabetes_prediction** | Machine Learning | Prédiction du risque de diabète | F1-score : **0.80** |
| **DL-NutrIA** | Deep Learning | Reconnaissance d'aliments par image | Accuracy : **74.84%** |

---

## 🩺 Projet 1 : Prédiction du diabète

Outil d'aide au repérage des patients à risque de diabète à partir de données cliniques (âge, IMC, glycémie, HbA1c...).

- **Dataset** : Kaggle Diabetes Prediction (~100k observations)
- **Modèles** : Logistic Regression, SVM, Random Forest
- **Meilleur modèle** : Random Forest (F1 = 0.80)

---

## 🍔 Projet 2 : NutrIA

Application permettant d'identifier un plat à partir d'une photo et d'estimer son apport calorique.

- **Dataset** : Food-101 (101 catégories, 101k images)
- **Architecture** : MobileNetV2 + Transfer Learning
- **Déploiement** : FastAPI + Streamlit + Docker

---

## 📁 Arborescence

```
├── DL-NutrIA/
│   ├── Deploy/
│   │   ├── api/
│   │   │   ├── Dockerfile
│   │   │   └── main.py
│   │   ├── front/
│   │   │   ├── app.py
│   │   │   └── Dockerfile
│   │   ├── model/
│   │   ├── docker-compose.yml
│   │   └── requirements.txt
│   └── notebooks/
│       ├── experiments/
│       │   ├── NutrIA_Food101_DL_run_2_baseline.ipynb
│       │   ├── NutrIA_Food101_DL_run_4_FullFT_v2.ipynb
│       │   ├── NutrIA_RunA_FE_PartialFT.ipynb
│       │   └── NutrIA_RunB_FE_FullFT.ipynb
│       └── NutrIA_Food101_FINAL.ipynb
│
└── ML-Diabetes_prediction/
    ├── diabetes_prediction_dataset.csv
    └── diabetes_prediction_FINAL.ipynb
```

---

## 🚀 Installation

```bash
git clone https://github.com/Svvmy/Alyra_Projet_Certif.git
cd Alyra_Projet_Certif

# Environnement virtuel
python -m venv venv
source venv/bin/activate

# Dépendances
pip install -r requirements.txt
```

