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

elif page==pages[2]:
  st.write("Dans cette partie nous avons sélectionné les quelques visusalisations qui permettent, selon nous, une meilleure appréhension du jeu de données. Dans un premier temps nous avons fait une analyse de la distribution des variables et ensuite nous avons visualisé la répartition des données en fonction de la variable cible.")
  st.header("Distribution des variables")
  st.subheader("Variables principales")
  Graphique_sélectionné=st.selectbox(label="Graphiques",options=['Répartition par âge','Répartition par métier','Répartition par mois','Répartition du nombre de contact de la dernière campagne'])
  if Graphique_sélectionné =='Répartition par âge':   
    fig=px.histogram(bank, x='age')
    st.plotly_chart(fig, key="bank", on_select="rerun")
    st.write("La tranche d'âge la plus représentée est les 30-40 ans")
  if Graphique_sélectionné =='Répartition par métier':    
    fig1=px.histogram(bank, x='job')
    st.plotly_chart(fig1, key="bank", on_select="rerun")
    st.write("Les catégories de métier les plus représentées sont management, blue-collar et technician.")
  if Graphique_sélectionné =='Répartition par mois': 
    fig2=px.histogram(bank,x='month').update_xaxes(categoryorder='total descending')
    st.plotly_chart(fig2, key="bank", on_select="rerun",)
    st.write("Les trimestres 2 et 3 sont ceux qui sont le plus représentés dans le jeu de donnée. Sans surprise durant le mois de décembre il y a eu très peu de contact.")
  if Graphique_sélectionné =='Répartition du nombre de contact de la dernière campagne': 
    fig3=px.histogram(bank,x='previous')
    st.plotly_chart(fig3, key="bank", on_select="rerun")
    st.write("Les clients sont en général contactés moins de 8 fois.")
  st.subheader("Variables secondaires")
  Graphique_sélectionné1=st.selectbox(label="Graphiques", options=['Répartition par statut marital','Répartition par éducation','Répartition par défauts de paiement', 'Répartition par prêt immobilier','Répartition des prêts à la conso','Répartition par type de contact','Résultat sur la dernière campagne marketing'])
  if Graphique_sélectionné1 =='Répartition par statut marital': 
    fig4=px.histogram(bank, x='marital')
    st.plotly_chart(fig4, key="bank")
  if Graphique_sélectionné1 =='Répartition par éducation': 
    fig6=px.histogram(bank, x='education')
    st.plotly_chart(fig6, key="bank")
  if Graphique_sélectionné1 =='Répartition par défauts de paiement': 
    fig7=px.histogram(bank,x='default')
    st.plotly_chart(fig7, key="bank")
  if Graphique_sélectionné1 =='Répartition par prêt immobilier': 
    fig8=px.histogram(bank, x='housing')
    st.plotly_chart(fig8, key="bank")
  if Graphique_sélectionné1 =='Répartition des prêts à la conso': 
    fig9=px.histogram(bank,x='loan')  
    st.plotly_chart(fig9, key="bank")
  if Graphique_sélectionné1 =='Répartition par type de contact': 
    fig10=px.histogram(bank, x='contact', histnorm='percent')
    st.plotly_chart(fig10, key="bank")
  if Graphique_sélectionné1 =='Résultat sur la dernière campagne marketing': 
    fig11=px.histogram(bank, x='poutcome', histnorm = 'percent')
    st.plotly_chart(fig11, key="bank")
  st.subheader("Variable cible")
  st.write("Notre variable cible est ‘deposit’, elle renseigne si le client a fait un dépôt à terme. Les classes pour cette variable sont bien équilibrées, ce qui est positif pour la suite de nos prédictions.")    
  fig12=px.histogram(bank, x='deposit', histnorm = 'percent')
  st.plotly_chart(fig12, key="bank")
  st.header("Répartition des données en fonction de la variable cible")
  st.write("La variable qui aura certainement l'impact le plus important dans la modélisation est la durée du contact (duration) car elle est la plus corrélée avec notre variable cible.")
  deposit = lambda x:1 if x=='yes' else 0
  bank['deposit_num'] = bank['deposit'].apply(deposit)
  default = lambda x:1 if x=='yes' else 0
  bank['default_num'] = bank['default'].apply(default)
  housing = lambda x:1 if x=='yes' else 0
  bank['housing_num'] = bank['housing'].apply(housing)
  loan = lambda x:1 if x=='yes' else 0
  bank['loan_num'] = bank['loan'].apply(loan)
  matrice = bank.corr(numeric_only = callable )
  fig, ax = plt.subplots(figsize = (10,10))
  sns.heatmap(matrice, annot=True, cmap = 'rainbow', ax = ax);
  st.write(fig)
  st.write("Aussi nous observons que la durée de contact médian est plus élevée pour les personnes ayant souscrit un dépôt à terme.")
  fig13=px.box(bank, x='deposit', y='duration')
  st.plotly_chart(fig13, key="bank")
  st.write("Nous avons identifié d'autres variables explicatives qui pourraient avoir un impact sur la modélisation:")
  if st.checkbox("Mois"):
          st.write("Nous remarquons des pics de refus ou de contrat accepté selon les mois.")
          st.write("Beaucoup de personne ont été contactées au mois de mai mais les refus représentent une part importante.Alors que sur les mois de mars, octobre et novembre il y a eu beaucoup moins de contact mais avec plus de résultat.")
          fig18=px.histogram(bank,x='month', color = 'deposit', barnorm = 'percent')
          st.plotly_chart(fig18, key="bank")
  if st.checkbox("Métier"):
          st.write("La catégorie professionnelle a un impact modéré sur le jeu de donnéees. Les catégories Student, Retired et unemployment ont d'avantage souscrit à un dépôt ) terme dans leur catégorie.")
          fig19=px.histogram(bank,x='job',color = 'deposit', barnorm = 'percent').update_xaxes(categoryorder='total descending')
          st.plotly_chart(fig19, key="bank")
  if st.checkbox("Age et solde du compte"):
          st.write("L'âge et le solde sur le compte influent également sur notre variable cible. Plus ils sont élevés plus il y a de souscription.")
          fig20=px.scatter(bank,x='balance',y="age", color='deposit')
          st.plotly_chart(fig20, key="bank")
          st.write("Cependant l’âge aura certainement un impact un peu moins important étant donné que comme nous pouvons le voir l’âge médian est similaire entre les personnes qui ont souscrit ou non.")
          fig21=px.box(bank, x='deposit', y='age')
          st.plotly_chart(fig21, key="bank")
          fig22=px.box(bank, x='deposit', y='balance')
          st.plotly_chart(fig22, key="bank")
  st.write("Nous avons aussi identifié certaines variables qui selon nous sont moins pertinentes, dû notamment à la distribution des données. Nous avons notamment identifié 4 variables avec des valeurs “unknown” plus ou moins représentées.")
  if st.checkbox("Variable Contact"):
          st.write("Elle comporte principalement la classe ‘cellular contact’. Cette variable n’apporte donc que peu de valeur ajoutée.")
          fig23=px.histogram(bank, x='contact', histnorm='percent')
          st.plotly_chart(fig23, key="bank")
  if st.checkbox("Variable pdays"):
          st.write("Cette variable comporte presque 75% de valeur négative (-1) que nous ne sommes pas sûrs de bien interpréter.")
  if st.checkbox("Variable poutcome"):
          st.write("Cette variable contient près de 80% de valeur unknown + others (certainement interprétable comme du unknown.")
          fig24=px.histogram(bank, x='poutcome', histnorm = 'percent')
          st.plotly_chart(fig24, key="bank")
          
elif page==pages[3]:
  st.write("Dans cette partie, nous allons voir quels sont les traitements que nous avons effectués sur le JDD afin de le « nettoyer » et de la préparer pour la suite.")
  
