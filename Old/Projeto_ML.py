import cv2
import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from glob import glob
from skimage import io, color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score
from sklearn.model_selection import cross_val_score, RepeatedKFold, cross_validate, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import friedmanchisquare, f_oneway
from statsmodels.stats.multitest import multipletests as mt

#IMPRESSÃO DA ACURÁCIA, DESVIO PADRÃO E MEDIANA
def accuracy(mean, std, median):
  print("Accuracy: %0.2f - std: (+/- %0.2f) - med: %0.2f" % (mean, std, median))

#EXECUTAR CLASSIFICADORES
def execute(X, y):
  acc = []
  fit_time = []
  classifier = []
  std = []
  median = []
  
  y = y.astype('str')
  rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=100)

  #DECISION TREE
  dt = DecisionTreeClassifier()
  cvs_dt = cross_val_score(dt, X, y, cv=rkf)
  print("DT")
  classifier.append('DT')
  accuracy(cvs_dt.mean(), cvs_dt.std(), np.median(cvs_dt))
  acc.append(cvs_dt.mean())
  std.append(cvs_dt.std())
  median.append(np.median(cvs_dt))
  cv_reports = cross_validate(dt, X, y, cv=rkf, return_train_score=False)
  fit_time.append(cv_reports['fit_time'].mean())

  #NAIVE_BAYES
  naive_bayes = GaussianNB()
  cvs_nb = cross_val_score(naive_bayes, X, y, cv=rkf)
  print("NB")
  classifier.append('NB')
  accuracy(cvs_nb.mean(), cvs_nb.std(), np.median(cvs_nb))
  acc.append(cvs_nb.mean())
  std.append(cvs_nb.std())
  median.append(np.median(cvs_nb))
  cv_reports = cross_validate(naive_bayes, X, y, cv=rkf, return_train_score=True)
  fit_time.append(cv_reports['fit_time'].mean())

  #KNN
  k = [3, 5, 7]
  for n in k:
    knn = KNeighborsClassifier(n_neighbors = n, metric='euclidean')
    cvs_knn = cross_val_score(knn, X, y, cv=rkf, scoring='accuracy')
    print("KNN", n)
    classifier.append('KNN ' + str(n))
    accuracy(cvs_knn.mean(), cvs_knn.std(), np.median(cvs_knn))
    acc.append(cvs_knn.mean())
    std.append(cvs_knn.std())
    median.append(np.median(cvs_knn))
    cv_reports = cross_validate(knn, X, y, cv=rkf, return_train_score=True)
    fit_time.append(cv_reports['fit_time'].mean())

  #WKNN
  for n in k:
    for weights in ['uniform', 'distance']:
      wknn = KNeighborsClassifier(n_neighbors = n, metric='euclidean', weights=weights)
      cvs_wknn = cross_val_score(wknn, X, y, cv=rkf)
      print(weights, " KNN", n)
      classifier.append(weights + ' WKNN ' + str(n))
      accuracy(cvs_wknn.mean(), cvs_wknn.std(), np.median(cvs_wknn))
      acc.append(cvs_wknn.mean())
      std.append(cvs_wknn.std())
      median.append(np.median(cvs_wknn))
      cv_reports = cross_validate(wknn, X, y, cv=rkf, return_train_score=True)
      fit_time.append(cv_reports['fit_time'].mean())

  #SVM
  '''vm = svm.SVC(kernel='linear')
  print("SVM Linear")
  classifier.append('SVM LINEAR')
  cvs_svm_lnr = cross_val_score(vm, X, y, cv=rkf)
  print('Fim SVM LINEAR')
  accuracy(cvs_svm_lnr.mean(), cvs_svm_lnr.std(), np.median(cvs_svm_lnr))
  acc.append(cvs_svm_lnr.mean())
  std.append(cvs_svm_lnr.std())
  median.append(np.median(cvs_svm_lnr))  
  cv_reports = cross_validate(vm, X, y, cv=rkf, return_train_score=True)
  fit_time.append(cv_reports['fit_time'].mean())'''

  vm = svm.SVC(kernel='rbf', gamma='auto')
  print("SVM RBF")
  classifier.append('SVM RBF')
  cvs_svm_rbf = cross_val_score(vm, X, y, cv=rkf)
  acc.append(cvs_svm_rbf.mean())
  std.append(cvs_svm_rbf.std())
  median.append(np.median(cvs_svm_rbf))
  accuracy(cvs_svm_rbf.mean(), cvs_svm_rbf.std(), np.median(cvs_svm_rbf))
  cv_reports = cross_validate(vm, X, y, cv=rkf, return_train_score=True)
  fit_time.append(cv_reports['fit_time'].mean())

  #LOGISTIC REGRESSION
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cvs_logR = cross_val_score(LogisticRegression(), X, y, cv=rkf)  
    print("LR")
    classifier.append('LR')
    accuracy(cvs_logR.mean(), cvs_logR.std(), np.median(cvs_logR))
    acc.append(cvs_logR.mean())
    std.append(cvs_logR.std())
    median.append(np.median(cvs_logR))  
    cv_reports = cross_validate(LogisticRegression(), X, y, cv=rkf, return_train_score=True)
    fit_time.append(cv_reports['fit_time'].mean())

  #TABLE MEAN
  df_table = pd.DataFrame()
  df_table['Classificador'] = classifier
  df_table['Acurácia'] = acc
  df_table['Desvio Padrão'] = std
  df_table['Mediana'] = median
  print(df_table)
  df_table.to_csv('tabela_de_medias.csv')

  #Test de Hyphoteses ANOVA / HolmSidak
  reject, p_vals, sidak, bonf = mt(acc)
  df = pd.DataFrame()
  df['classifier'] = classifier
  df['reject'] = reject
  df['p_values'] = p_vals
  df['accuracy'] = acc
  df['fit_time'] = fit_time
  df = df.sort_values(by='accuracy', ascending=True)
  df.to_csv('tabela_estatistica.csv')
  df.plot(kind='barh', x='classifier', y='accuracy', color='blue')
  plt.title('Média de Acurácia')
  plt.show()
  df = df.sort_values(by='fit_time', ascending=True)
  df.plot(kind='barh', x='classifier', y='fit_time', color='orange')
  plt.title('Média de Tempo de Classificação')
  plt.show()
  
  

#REMOÇÃO DO BACKGROUND
def removeBackground(image):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  retval, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
  _, cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  cnts = sorted(cnts, key=cv2.contourArea)
  for cnt in cnts:
    if cv2.contourArea(cnt) > 100:
      break        
  mask = np.zeros(image.shape[:2],np.uint8)
  cv2.drawContours(mask, [cnt],-1, 255, -1)
  dst = cv2.bitwise_and(image, image, mask=mask)
  return dst

distances = [1]
angles = [0]
#list of features
properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

glcm_feats = []
path = 'Plant_Leaves_Dataset'
listpath = os.listdir(path)

for label in listpath:
  #read image
  data = glob(path + '/' + label + '/' + '/*.jpg')
  images = [cv2.imread(img) for img in data]

  #transform gray levels
  for image in images:
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      img = removeBackground(image)
      gray_image = img_as_ubyte(color.rgb2gray(image))

    #texture features
    glcm = greycomatrix(gray_image,
                        distances=distances,
                        angles=angles,
                        symmetric=True,
                        normed=True)  
    feats = np.hstack([greycoprops(glcm, prop).ravel() for prop in properties])
    
    #insert entropy
    feats = np.append(feats, shannon_entropy(gray_image))
    
    #insert color features color mean, standard deviation, skewness, kurtosis
    #data in RGB
    (means, stds) = cv2.meanStdDev(image)
    feats = np.append(feats, means)
    feats = np.append(feats, stds)

    #include class plants
    feats = np.append(feats, label)

    #insert array features
    glcm_feats.append(feats)

#Go to classification
dataset = pd.DataFrame(glcm_feats)
dataset.to_csv('base_pre_processada.txt', header=None, index=False)
X = dataset.values[:,0:12]  #features
y = dataset.values[:,13]    #class
execute(X, y)
