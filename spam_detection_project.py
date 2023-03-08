import os
import numpy as np

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split, cross_val_score

# Le code de la ligne 12 à 35 a été récupéré. Il permet uniquement d'accéder au dataset

DATA_DIR = 'enron' # Nom du dossier où se situe le dataset
target_names = ['ham', 'spam']

def get_data(DATA_DIR):
    subfolders = ['enron%d' % i for i in range(1,7)]

    data = []
    target = []
    for subfolder in subfolders:
        # spam
        spam_files = os.listdir(os.path.join(DATA_DIR, subfolder, 'spam'))
        for spam_file in spam_files:
            with open(os.path.join(DATA_DIR, subfolder, 'spam', spam_file), encoding='ascii', errors='ignore') as f:
                data.append(f.read())
                target.append(1)
                
        # ham --> mail non spam
        ham_files = os.listdir(os.path.join(DATA_DIR, subfolder, 'ham'))
        for ham_file in ham_files:
            with open(os.path.join(DATA_DIR, subfolder, 'ham', ham_file), encoding='ascii', errors='ignore') as f:
                data.append(f.read())
                target.append(0)
                
    target = np.array(target)
    return(data, target)

# On attribut X à la variable 'data' de la ligne 17 --> Cette variable est une matrice ayant pour chaque ligne un mail et pour chaque colonne un mot composant le mail
# On attribut y à la variable 'target' de la ligne 18 --> Matrice de dimension 1 composée uniquement de 0 et 1 (0 -> Non Spam, 1 -> Spam) --> ligne 19 à 32
# Ces deux matrices permettront d'entraîner l'algorithme dans la détection de spam
X, y = get_data(DATA_DIR)

# A l'aide de la méthode 'train_test_split' :
# On va séparer les données en deux parties -> Une partie pour entraîner l'algorithme (X_train, y_train) et une seconde pour tester l'algorithme (X_test, y_test)
# Le paramètre 'test_size=0.10' signifie que nous allouons 10% du dataset pour la seconde phase (test de l'algorithme)
# Le valeur du paramètre 'random_state' n'a pas réellement d'importance, le nombre 42 est souvent utilisé en tant que random
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# CountVectorizer() à pour but de compter la fréquence d'apparition d'un mot dans chaque vecteur (ligne de la matrice)
# Si le mot est présent l'algorithme ajoute 1 sinon rien
# A travers la variable 'X_train_counts', l'objectif est d'obtenir une matrice indiquant le nombre d'apparition d'un mot pour chaque mail
count_vec = CountVectorizer()
X_train_counts = count_vec.fit_transform(X_train)

# Lorsque la différence de taille entre les mails (nombre de mots) est trop importante, certains mots peuvent apparaître trop souvent et créer un déséquilibre :
# Nous appliquons la formule TfidfTransformer() pour pondérer la fréquence des mots.
# Cela revient à faire --> Tf (Term Frequency) * IDF (Inverse Document Frequency)
# Le résultat obtenu est compris entre 0 et 1 et donc plus facile à analyser
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# On utilise donc la matrice 'X_train_tfidf' pour alimenter notre classificateur et ainsi faire des prédictions à partir de cette matrice


# La méthode 'fit' permet d'entraîner le classificateur 
clf = MultinomialNB().fit(X_train_tfidf, y_train)
# Nous pouvons également entraîner le classificateur BernoulliNB() qui s'est révélé légèrement moins précis que MultinominalNB()
# clf = BernoulliNB().fit(X_train_tfidf, y_train)

# On applique le même schéma réalisé précédemment (ligne 52 et 59) pour les données de test
X_test_counts = count_vec.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# On réalise la prédiction sur X_test_tfidf (matrice dimension 1 contenant les vecteurs avec des valeurs entre 0 et 1) 
# et pas X_test (matrice contenant un mail par ligne et un mot par colonne)
# car on souhaite utiliser l'algorithme sur des valeurs numériques et non sur des chaînes de caractères
y_pred = clf.predict(X_test_tfidf)

# Permet de retourner un rapport détaillé pour les spam et non spam
print(metrics.classification_report(y_test, y_pred, target_names=target_names))

# Permet d'afficher la précision de l'algorithme 
print(metrics.classification.accuracy_score(y_test, y_pred))