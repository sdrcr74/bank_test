import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

url = 'https://raw.githubusercontent.com/sdrcr74/bank_nov23/main/bank.csv'
bank = pd.read_csv(url)
st.markdown("<img src='https://raw.githubusercontent.com/sdrcr74/bank_test/main/datascientest_logo.png' width='100' style='display: block; margin: 0 auto;'>" , unsafe_allow_html=True)
st.markdown("<h1 style='text-align : center;'>Prédiction du succès d une campagne de Marketing d’une banque</h1>", unsafe_allow_html=True)
st.subheader("NOV23_CONTINU_DA - Datascientest", divider='blue')

st.sidebar.title("Sommaire")
pages=["Le projet","Le jeu de données", "Datavisualisation","Préparation des données","Modélisation","Conclusion"]
page=st.sidebar.radio("Aller à la page:", pages)
st.sidebar.title("Auteurs")
with st.sidebar:
        st.write("Maxence MALHERRE")
        st.write("Sophie DORCIER")
        st.write("Stéphane LASCAUX")
        st.write("Van-Anh HA")
if page==pages[0]:
  st.header("Description du projet")
  st.subheader("L'objectif :")
  st.write("Ce projet a été mené dans le cadre de notre formation Data Analyst avec Datascientest.")
  st.write("L’objectif du projet est d’établir un modèle permettant de prédire le succès d’une campagne marketing d’une banque.")
  st.write("Concrétement il s'agit de prédire, sur la base des données démographiques du client, sa situation financière et son précédent contact avec la banque, s'il va souscrire ou non au produit Dépôt à terme.")
  st.write("Le jeu de données qui nous a été mis à disposition s’appelle 'Bank Marketing Dataset'. Ce jeu de données est disponible librement sur [Kaggle] (https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset)")
  st.write("Ce Streamlit illustre notre approche, allant de l'exploration des données à la création du modèle prédictif.")
  st.image("https://raw.githubusercontent.com/sdrcr74/bank_nov23/main/banking.jpg")
elif page==pages[1]:
  st.subheader("Le jeu de données :")
  st.write("Description du contenu : Données personnelles issues des campagnes de marketing direct d’une banque portugaise.")
  st.write("Périmètre temporel : 2012")
  st.write("Source : UC Irvine Machine Learning Repository, mise à disposition sur Kaggle")
  st.write("Dimension : 11 162 lignes & 17 colonnes")
  st.write("Définition des variables :")
  url2 = 'https://raw.githubusercontent.com/sdrcr74/bank_nov23/main/Liste%20variable.csv'
  liste_variable = pd.read_csv(url2, sep =";", index_col = None)
  st.write(liste_variable)
  st.write("Dans un premier temps, nous étudierons les différentes variables à travers les visualisations. Dans un deuxième temps, nous procéderons aux préparations de données nécessaires permettant de les modéliser par la suite.")
  st.write('Exploration des données')
  st.write("Avant d'explorer les données du dataset, il nous a semblé pertinent de comprendre les différentes variables présentes dans le jeu de données.") 
  st.write("Pour la plupart, l'intitulé des variables était clair et compréhensible. Nous allons cependant clarifier certaines variables:") 
  st.write("-balance: montant du compte en banque")
  st.write("-housing: prêt immobilier") 
  st.write("-loan: autre prêt") 
  st.write("-contact: moyen de contact") 
  st.write("-day & month: jour et mois du contact")
  st.write("-duration: durée du contact") 
  st.write("-campaign: nombre de contact durant la campagne marketing") 
  st.write("-pdays: nombre de jours de contact avant la campagne")
  st.write("-previous: nombre de contact avant la campagne")
  st.write("-poutcome: résultat de la dernière campagne")
  st.write("Aperçu de notre dataset")
  st.dataframe(bank.head())
  st.write('Dimensions du Dataframe')
  st.write(bank.shape)
  if st.checkbox("Afficher le nombre de doublons"):
    st.dataframe(bank.duplicated())
  if st.checkbox("Afficher les valeurs manquantes"):
    st.dataframe(bank.isna().sum())
  if st.checkbox("Répartition de la variable deposit"):
    st.dataframe(bank['deposit'].value_counts())
  if st.checkbox("Répartition en % par résultat de la dernière campagne via la variable poutcome"):
    st.dataframe(bank['poutcome'].value_counts()/len(bank))
  if st.checkbox("Pourcentage du nombre de contact lors de la dernière campagne égal à 0"):
    st.dataframe((bank['previous'] == 0).value_counts()/len(bank))
  if st.checkbox("Pourcentage de -1 dans la variable pdays"):
    st.dataframe((len(bank[bank['pdays'] == -1]) / len(bank)))
  if st.checkbox("Nombre de chiffres négatifs dans la variable balance"):
    st.dataframe(len(bank[bank['balance'] < 0]))
