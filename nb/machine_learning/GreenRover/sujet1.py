#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import matplotlib.pyplot as plt
import os
path = os.getcwd() + "/"


# # Question 1
# 
# En analysant pas_herbes.bin, on voit que ce fichier a été écrit en binaire. Les images me semblent être en noir et blanc, je les traite comme tel.

# In[39]:


f = open(path + "pas_herbes.bin", "rb") # ouverture en binaire
him = f.read()
him1 = him[24:len(him)//2] # on enleve les en-têtes des images
him2 = him[len(him)//2+24:]

def hex_to_im(hex_im):
    im = np.zeros((len(hex_im)), dtype = np.uint8) 
    for i in range(len(im)):
        im[i] = np.uint8(hex_im[i]) # conversion hexadécimal -> uint8
    return im.reshape((480, 640))

im1, im2 = hex_to_im(him1), hex_to_im(him2)
plt.imsave(path + "photo1.png", im1, cmap = "gray")
plt.imsave(path + "photo2.png", im2, cmap = "gray")

plt.imshow(im2, cmap = "gray")


# Remarque: peut-être que l'idée était plutôt d'utiliser ImageReader de OpenMV, je n'en suis pas sûr.... 

# # Question 2
# J'utilise 71raw_04062020.png pour extraire la feuille et la texture de l'herbe.
# L'extraction de la feuille peut se faire avec n'importe quel algorithme de segmentation d'image, par exemple un graph cut.
# Le code ci-dessous a été adapté depuis https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html :

# In[3]:


import cv2
img = cv2.imread(path + '71raw_04062020.png')

mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (300,150,160,160) # la feuille est dans rect
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

img = img[:, :, 0] # image en noir et blanc

feuille_mask = np.where((mask==0)|(mask==2),0,1).astype('uint8')
feuille_mask = feuille_mask[:,:]
feuille = img*feuille_mask

feuille = feuille[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] # pour n'avoir que la feuille
feuille_mask = feuille_mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]


# In[4]:


plt.imshow(feuille_mask, cmap = "binary")


# Pour la génération de la texture d'herbe, je récupère des parties de 71raw_04062020.png que je sais être de l'herbe que j'assemble aléatoirement. J'utilise trois textures d'herbes différentes (de tailles 128x96), mais on pourrait en utiliser plus.

# In[23]:


def gazon():
    n, m = 96, 128 
    herbes = [img[:n,:m], img[:n, -m:], img[-n:, -m:]]
    res = np.zeros((480, 640), dtype = np.uint8)
    for i in range(0, 480, n):
        for j in range(0, 640, m):
            i2, j2 = min(i+n, 479), min(j+m, 639)
            res[i:i+96, j:j+128] = herbes[np.random.randint(0, 3)]
    return res

gazon_ex = gazon()
plt.imshow(gazon_ex, cmap = "gray") # exemple de texture générée


# Je rajoute ensuite une feuille en la plaçant aléatoirement puis en appliquant une symétrie horizontale (resp. verticale) avec probabilité 0.5, ce qui permet d'obtenir les 4 orientations possibles uniformément au hasard.

# In[37]:


def add_feuille(im):
    global feuille, feuille_mask
    dx, dy = feuille.shape[0], feuille.shape[1]
    xf = np.random.randint(0, 480 - dx)
    yf = np.random.randint(0, 640 - dy)
    if np.random.random() < 0.5:
        feuille = feuille[::-1, :] # symétrie verticale
        feuille_mask = feuille_mask[::-1, :]
    if np.random.random() < 0.5:
        feuille = feuille[:, ::-1] # symétrie horizontale
        feuille_mask = feuille_mask[:, ::-1]
    im[xf:xf+dx, yf:yf+dy] = (1 - feuille_mask)*im[xf:xf+dx, yf:yf+dy] + feuille_mask*feuille # ajoute la feuille à l'image

add_feuille(gazon_ex)
plt.imshow(gazon_ex, cmap = "gray")


# Créons ensuite un jeu de 1000 images. Malheureusement, des images 640x480 étaient trop grosses pour être utilisées par mon réseau de neurones. J'ai donc décidé de redimensionner les images en 128x96 (hauteur et largeur divisées par 5).

# In[30]:


def creer_jeu(n):
    jeu = [gazon() for i in range(n)]
    Y = np.zeros((len(jeu))) # Y[i] vaudra 1 si jeu[i] contient au moins une feuille, 0 sinon
    for i in range(len(jeu)):
        pas_de_feuille = np.random.random() # on choisit de générer autant d'image avec feuilles que d'image sans feuille
        if pas_de_feuille > 0.5: Y[i] = 0
        else:
            nb_feuilles = np.random.randint(1, 5) # ajouter entre 1 et 4 feuilles
            for _ in range(nb_feuilles): add_feuille(jeu[i])
            Y[i] = 1 
        jeu[i] = cv2.resize(jeu[i], (128, 96)) # redimensionnement
    return jeu, Y

np.random.seed(0) # juste pour avoir des résultats reproductibles 
jeu, Y = creer_jeu(1000)
X = np.array(jeu)[:, :, :, np.newaxis]/255 # rajout d'une dimension pour l'input du CNN + normalisation
X_test, X_train = X[:500], X[500:]
Y_test, Y_train = Y[:500], Y[500:]
plt.imshow(jeu[7], cmap = "gray") # exemple d'image générée


# Nous allons utiliser 500 images pour entraîner notre réseau de neurones et 500 images pour le tester. De plus, les images sont normalisées (valeurs entre 0 et 1 plutôt que 0 et 255), ce qui est conseillé lorsqu'un réseau de neurones est utilisé (et de manière plus générale en ML). <br>
# Vérifions que nos données sont bien de la bonne forme:

# In[8]:


X_train.shape, Y_train.shape


# In[9]:


plt.hist(Y_train)


# J'ai décidé de générer (à peu près) autant d'images sans feuille que d'images avec feuilles.
# 
# Utilisons le réseau de neurones convolutionnel (modèle adapté à la vision par ordinateur) suivant: 

# In[10]:


from keras import layers
from keras import models
from keras import optimizers 

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 128, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


# L'architecture du CNN a été choisie de façon "standard", et pourrait être affinée expérimentalement. <br>
# Un CNN a la propriété intéressante d'être "invariant par translation" grâce aux filtres de convolutions, ce qui est adapté à notre problème (la feuille peut être n'importe où sur l'image). <br>
# Les MaxPooling permettent, entre autres, de réduire la taille de l'image. <br>
# Le DropOut est une des techniques permettant d'éviter le phénomène d'overfitting.<br>
# On peut se permettre d'augmenter le nombre de channels dans les couches de convolutions lorsqu'on entre plus profondément dans le réseau, car il y a moins de paramètres (grâce aux MaxPooling).
# Le nombre de paramètres total, 437k, semble raisonnable.
# 
# Entraînons notre CNN:

# In[11]:


model.fit(X_train, Y_train, epochs=15, verbose = 2)


# In[12]:


Y_pred = np.round(model.predict(X_test).flatten())
print(model.predict(X_test).flatten()[:20])
print(Y_test[:20]) # on regarde quelques prédictions


# In[13]:


from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, Y_pred)


# 17 fois, le modèle a prévu qu'il n'y avait pas de feuille (Y_pred[i] vaut 0) alors qu'il y en avait (Y_test[i] vaut 1)

# In[14]:


model.evaluate(X_test, Y_test, verbose = 2)


# Le modèle a une accuracy élevée et ne semble *a priori* pas être sujet à l'overfitting, vu l'accuracy sur le jeu de test.

# Remarque: si on souhaite non pas seulement savoir s'il y a une feuille mais en plus compter le nombre de feuilles (ce qui est un problème plus difficile), il faudrait encoder les labels Y sous forme de *one-hot* vecteurs (de taille 5) avec, par exemple la fonction `to_categorical` de keras. Les MaxPooling pourraient alors poser problème car il y a un risque de fusionner plusieurs feuilles en une seule.

# # Question 3
# On souhaite, dans une image, détecter les zones contenant un objet de celles ne contenant que de l'herbe. <br>
# C'est un problème de segmentation d'image: extraire les objets (feuilles). On peut donc encore utiliser l'algorithme de graph cut ou un algorithme de classification ou clustering (par exemple avec KMeans).
# 
# Je propose une autre méthode réutilisant le réseau de neurones de la question précédente: 
# - Parcourir l'image par fenêtre de taille 128x96 (méthode de la fenêtre glissante)
# - Appliquer la fonction process ci-dessous pour obtenir une image du même "format" (normaliser et générer une image d'une façon similaire) que celles utilisées pour le CNN
# - Utiliser le CNN pour prédire si il y a une feuille dans la fenêtre

# In[15]:


def process(im):
    im2 = np.ones((480, 640))
    for i in range(0, 480, 96):
        for j in range(0, 640, 128):
            im2[i:i+96, j:j+128] = im
    return cv2.resize(im2, (128, 96))/255


# In[31]:


im = cv2.imread(path + '71raw_04062020.png')
fenetre = process(im[:96, :128, 0])
plt.imshow(fenetre, cmap = "gray") # exemple d'image générée à partir d'une fenêtre


# In[17]:


def detecter_feuilles(im):
    L_feuilles = [] # liste des feuilles détectées
    for i in range(0, 480, 96):
        for j in range(0, 640, 128):
            fenetre = process(im[i:i+96, j:j+128, 0])
            if model.predict(np.array([fenetre[:, :, np.newaxis]]))[0] > 0.5:
                L_feuilles.append(im[i:i+96, j:j+128, 0])
    return L_feuilles


# In[18]:


def afficher_feuilles(im):
    L = detecter_feuilles(im)
    n = len(L)
    fig=plt.figure(figsize=(96, 128))
    for i in range(n):
        fig.add_subplot(1, n+1, i+1)
        plt.imshow(L[i], cmap = "gray")


# Testons maintenant sur les images données en exemple:

# In[19]:


afficher_feuilles(cv2.imread(path + '71raw_04062020.png'))


# In[20]:


afficher_feuilles(cv2.imread(path + '79raw_04062020.png'))


# In[21]:


afficher_feuilles(cv2.imread(path + '75raw_04062020.png'))


# In[22]:


afficher_feuilles(cv2.imread(path + '114raw_26052020.png'))


# On remarque qu'il y a quelques faux positifs, mais la grande majorité des prédictions (positives ou négatives) semblent correctes. La capacité à généraliser de notre CNN semble donc assez bonne. <br>
# Cependant, l'objet en bas à gauche des images (rebord du robot?) est détecté à tord comme une feuille. Pour éviter ce problème, on pourrait rajouter cet objet aux données d'entraînement X_train du CNN. 
