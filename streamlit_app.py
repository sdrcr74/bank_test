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
          fig20 = px.scatter(bank, y="count", x="month",color="deposit")
          st.plotly_chart(fig20, key="bank")
  if st.checkbox("Métier"):
          st.write("La catégorie professionnelle a un impact modéré sur le jeu de donnéees. Les catégories Student, Retired et unemployment ont d'avantage souscrit à un dépôt ) terme dans leur catégorie.")
          fig19=px.histogram(bank,x='job',color = 'deposit', histnorm = 'percent',histfunc='sum').update_xaxes(categoryorder='total descending')
          st.plotly_chart(fig19, key="bank")
  st.write("-Age")
  st.write("-Job")
  st.write("-Month")
  st.write("-Balance")
  
elif page==pages[3]:
  st.write("DataViz")
  st.write("Notre variable cible est ‘deposit’, elle renseigne si le client a fait un dépôt à terme. Les classes pour cette variable sont bien équilibrées, ce qui est positif pour la suite de nos prédictions.")
  st.write("Selon nos analyses du jeu de données, nous avons remarqué que la variable la plus pertinente est la durée du contact (duration).")
  st.write("Les autres variables qui nous semblent les plus pertinentes sont :")
  st.write("-Age")
  st.write("-Job")
  st.write("-Month")
  st.write("-Balance")

  st.write("Notre variable cible est ‘deposit’, elle renseigne si le client a fait un dépôt à terme. Les classes pour cette variable sont bien équilibrées, ce qui est positif pour la suite de nos prédictions.")         
  st.write("Nous avons pu identifier des relations entre certaines variables explicatives et notre variable cible :")
  st.write("La durée de contact (duration) aura un impact important dans la modélisation. En effet, nous remarquons que cette variable est la plus corrélée avec notre variable cible.")
  Graphique=st.selectbox(label="",options=['Heatmap',"Dépôt à terme en fonction de la durée du contact","Répartition par mois en fonction du dépôt à terme",'Relation Age, balance et Deposit',"Dépôt à terme en fonction de l'age"])
  if Graphique=='Heatmap':
    st.write("La durée de contact (duration) aura un impact important dans la modélisation. En effet, nous remarquons que cette variable est la plus corrélée avec notre variable cible.")
   
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
   
  if Graphique=="Dépôt à terme en fonction de la durée du contact": 
    st.write("Aussi nous observons que la durée de contact médian est plus élevée pour les personnes ayant souscrit un dépôt à terme.")
    fig13=sns.catplot(x='deposit', y='duration', data=bank, kind = 'box')
    plt.title("Dépôt à terme en fonction de la durée du contact")
    st.pyplot(fig13.fig)
  

  if Graphique=='Relation Age, balance et Deposit': 
    st.write("L’âge et la balance influent également sur le deposit. Plus ils sont élevés, plus il y a de dépôt.")
    fig12=px.scatter(bank,x="balance",y="age", color='deposit', title='Relation Age, balance et Deposit')
    st.plotly_chart(fig12) 
  if Graphique=="Répartition par mois en fonction du dépôt à terme":
    st.write("Entre Deposit et Month où l’on remarque des pics de refus ou de contrat accepté selon les mois.")
    st.write('Le mois va avoir un impact, nous remarquons notamment qu’au mois de mai le volume de client contacté est le plus élevé. Le nombre de personnes n’ayant pas souscrit à un dépôt à terme est plus important sur ce mois.')
    fig18=sns.catplot(x='month', kind='count', data=bank, hue = 'deposit')
    plt.title("Répartition par mois en fonction du dépôt à terme");
    st.pyplot(fig18.fig) 
  if Graphique=="Dépôt à terme en fonction de l'age":
    st.write("Cependant l’âge aura certainement un impact un peu moins important étant donné que comme nous pouvons le voir l’âge médian est similaire entre les personnes qui ont souscrit ou non.")
    fig17=sns.catplot(x='deposit', y='age', data=bank, kind = 'box')
    plt.title("Dépôt à terme en fonction de l'age");
    st.pyplot(fig17.fig)

  bank_cleaned = bank.drop(bank.loc[bank["job"] == "unknown"].index, inplace=True)
  bank_cleaned = bank.drop(bank.loc[bank["education"] == "unknown"].index, inplace=True)
  bank_cleaned = bank.drop(['contact', 'pdays'], axis = 1)
  
  
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  from sklearn.preprocessing import LabelEncoder
  from sklearn.preprocessing import OneHotEncoder
  from sklearn.metrics import f1_score
  from imblearn.over_sampling import RandomOverSampler
  from sklearn.compose import ColumnTransformer
  from sklearn.pipeline import Pipeline
  from sklearn.impute import SimpleImputer
  feats = bank_cleaned.drop(['deposit'], axis = 1)
  target = bank_cleaned['deposit']
  X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state=42)
  scaler = StandardScaler()
  cols = ['age','balance','day','campaign','previous','duration']

  X_train[cols] = scaler.fit_transform(X_train[cols])
  X_test[cols] = scaler.transform(X_test[cols])
  
  def replace_yes_no(x):
    if x == 'no':
      return 0
    if x == 'yes':
      return 1

  X_train['default'] = X_train['default'].apply(replace_yes_no)
  X_test['default'] = X_test['default'].apply(replace_yes_no)

  X_train['housing'] = X_train['housing'].apply(replace_yes_no)
  X_test['housing'] = X_test['housing'].apply(replace_yes_no)

  X_train['loan'] = X_train['loan'].apply(replace_yes_no)
  X_test['loan'] = X_test['loan'].apply(replace_yes_no)

  def replace_month(x):
    if x == 'jan':
      return 1
    if x == 'feb':
      return 2
    if x == 'mar':
      return 3
    if x == 'apr':
      return 4
    if x == 'may':
      return 5
    if x == 'jun':
      return 6
    if x == 'jul':
      return 7
    if x == 'aug':
      return 8
    if x == 'sep':
      return 9
    if x == 'oct':
      return 10
    if x == 'nov':
      return 11
    if x == 'dec':
      return 12

  X_train['month'] = X_train['month'].apply(replace_month)
  X_test['month'] = X_test['month'].apply(replace_month)
  X_train = pd.get_dummies(X_train, dtype = 'int')
  X_test= pd.get_dummies(X_test, dtype = 'int')
  le = LabelEncoder()

  y_train = le.fit_transform(y_train)
  y_test = le.transform(y_test)
  from sklearn.linear_model import LogisticRegression
  reglog = LogisticRegression(random_state=42)
  reglog.fit(X_train, y_train)
  print('Accuracy score du Logistic regression (train) : ',reglog.score(X_train, y_train))
  from sklearn.ensemble import RandomForestClassifier

  forest = RandomForestClassifier(random_state=42)
  forest.fit(X_train, y_train)
  print('Accuracy score du Random Forest (train) : ',forest.score(X_train, y_train))
  from sklearn.tree import DecisionTreeClassifier

  treecl = DecisionTreeClassifier(random_state=42)
  treecl.fit(X_train,y_train)

  print('Accuracy score du Decision Tree (train) : ',treecl.score(X_train, y_train))

elif page==pages[4]:
  st.write('Modélisation')
  modèle_sélectionné=st.selectbox(label="Modèle", options=['Régression logistique','Decision Tree','Random Forest'])

  if modèle_sélectionné=='Régression logistique':
    st.metric(label="accuracy", value=reglog.score(X_train, y_train))
  
  if modèle_sélectionné=='Decision Tree':
    st.metric(label="accuracy", value= treecl.score(X_train, y_train))

  if modèle_sélectionné=='Random Forest':
    st.metric(label="accuracy", value=forest.score(X_train, y_train))
  fig12=px.scatter(bank,x="balance",y="age", color='deposit', title='Relation Age, balance et Deposit')
  st.plotly_chart(fig12)
