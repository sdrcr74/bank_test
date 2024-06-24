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
pages=["Le projet","Exploration du JDD", "Analyse et visualisation du JDD","Préparation des données","Modélisation","Conclusion"]
page=st.sidebar.radio("Aller à la page:", pages)
st.sidebar.title("Auteurs")
with st.sidebar:
        st.write("Maxence MALHERRE")
        st.write("Sophie DORCIER")
        st.write("Stéphane LASCAUX")
        st.write("Van-Anh HA")
if page==pages[0]:
  st.header("Généralités")
  st.markdown("- En marketing, l'analyse prédictive permet une personnalisation plus précise et des campagnes mieux ciblés.")
  st.markdown("- Application en projet de formation Data Analyst : prédiction du succès de la campagne marketing du produit Dépôt à terme d’une banque.")
  st.markdown("- Ce Streamlit illustre notre approche, allant de l'exploration des données à la création du modèle prédictif.")
elif page==pages[1]:
  st.subheader("Le jeu de données")
  st.markdown("- Le jeu de données nous a été fourni, il s'agit d'un fichier CSV s'appelant 'Bank Marketing Dataset', disponible librement sur [Kaggle](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset).")
  st.markdown("- Description du contenu : Données personnelles (informations démographiques, situation financière, contact précédent avec la banque) issues de campagnes d'appel télémarketing d’une banque portugaise.")
  st.markdown("- Périmètre temporel : 2012")
  st.markdown("- Source : UC Irvine Machine Learning Repository")
  st.markdown("- Dimension : 11 162 lignes & 17 colonnes (16 variables explicatives & 1 variable cible")
  st.markdown("- Définition des variables :")
  url2 = 'https://raw.githubusercontent.com/sdrcr74/bank_nov23/main/Liste%20variable.csv'
  liste_variable = pd.read_csv(url2, sep =";", index_col = None)
  st.write(liste_variable)
  st.markdown("- Qualité de données : à première vue, la base de données nous semble propre :")
  if st.checkbox("Nombre de doublons :"):
    st.write(bank.duplicated().sum())
  if st.checkbox("Nombre de valeurs manquantes :"):
    st.write(bank.isna().any(axis = 0))
  st.write("Dans la partie suivante, nous allons explorer les données de façon plus approfondie à travers de la datavisualisation.")
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
  st.write("La particularité de ce jeu de donnée est que nous remontons beaucoup d'outliers sur les variables numériques. Dans la partie suivante de pre-processing nous devrons choisir de les garder, les modifier ou les supprimer.")

elif page==pages[3]:
  st.write("Dans cette partie, nous allons voir quels sont les traitements que nous avons effectués sur le JDD afin de le « nettoyer » et de le préparer pour la suite.")
  st.header("Preprocessing des données bancaires")
  def preprocess_data(bank):
    st.write("Suite à ces analyses nous pouvons passer au pré-processing du jeu de données.")
    
    st.write("Comme vu précédemment, le jeu de données est propre car il ne contient aucun doublon ni valeurs manquantes. Il dispose malgré tout de nombreuses valeurs insignifiantes telles que 'unknown' (11239) et 'others' (537). Nous avons décidé de supprimer la valeur 'unknown' des variables 'job' et 'education' car cela n'impactera pas le dataset au vu du faible volume de cette valeur.")
    st.write("### Nombre total de lignes avant nettoyage:")
    st.write(bank.shape[0])
    code = """
bank_cleaned = bank.drop(bank.loc[bank["job"] == "unknown"].index, inplace=False)
bank_cleaned = bank_cleaned.drop(bank_cleaned.loc[bank_cleaned["education"] == "unknown"].index, inplace=False)
    """
    st.code(code, language='python')
    bank_cleaned = bank.drop(bank.loc[bank["job"] == "unknown"].index, inplace=False)
    bank_cleaned = bank_cleaned.drop(bank_cleaned.loc[bank_cleaned["education"] == "unknown"].index, inplace=False)
    st.write(bank_cleaned.head()) 
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Distribution de Job après nettoyage:")
        st.write(bank_cleaned['job'].value_counts())
    with col2:
        st.write("### Distribution de Education après nettoyage:")
        st.write(bank_cleaned['education'].value_counts())
    st.write("Nous avons eu une réflexion pour certaines variables :")
    st.write("- **poutcome**")
    st.write("Nous avons réfléchi à 3 options :")
    st.write("1. Soit nous gardons cette variable dans le dataset et nous supprimons les lignes 'unknown'. Cela a pour conséquence de réduire considérablement la taille de notre dataset. Mais nous serons certainement amenés à le réduire dans tous les cas par la suite.")
    st.write("2. Soit nous la gardons telle quelle. Nous pouvons choisir un modèle qui peut être entraîné avec ce type de donnée, et nous verrons l’impact.")
    st.write("3. Soit nous supprimons complètement cette colonne car la distribution pourrait impacter négativement notre modèle.")
    st.write("Nous sommes plutôt partis sur la deuxième solution, car outre les 'unknown' et 'other', la distribution de la variable est plutôt bonne.")
    st.write("- **contact**")
    st.write("Nous avons décidé de supprimer cette colonne car sa distribution n’est pas représentative.")
    st.write("- **pdays**")
    st.write("Nous avons décidé de supprimer cette colonne à cause de la valeur -1 sur-représentée et que nous ne sommes pas sûrs de bien interpréter.")
    code = """
bank_cleaned = bank_cleaned.drop(['contact', 'pdays'], axis=1)
    """
    st.code(code, language='python')
    bank_cleaned = bank_cleaned.drop(['contact', 'pdays'], axis=1)
    st.write(bank_cleaned.head())
    st.write("Nous avons également transformé la durée en minute sur Duration.")
    code = """
bank_cleaned['duration'] = bank_cleaned['duration'] // 60
    """
    st.code(code, language='python')
    bank_cleaned['duration'] = bank_cleaned['duration'] // 60
    st.write(bank_cleaned.head())
    st.write("### Nombre total de lignes après nettoyage:")
    st.write(bank_cleaned.shape[0])
    st.write("### Aperçu des premières lignes des données nettoyées:")
    st.write(bank_cleaned.head())
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Outliers sur la colonne 'previous':")
        st.write(bank_cleaned['previous'].describe())
    with col2:
        st.write("### Outliers sur la colonne 'duration':")
        st.write(bank_cleaned['duration'].describe())
    st.write("Le jeu de données contient de nombreux outliers, mais étant donné qu'il ne s'agit pas de données aberrantes, nous avons décidé de le conserver tel quel.")
    return bank_cleaned
  bank_cleaned = preprocess_data(bank)
elif page==pages[4]:
        import streamlit as st
        import pandas as pd
        import plotly.express as px
        import matplotlib.pyplot as plt
        import seaborn as sns

        url = 'https://raw.githubusercontent.com/sdrcr74/bank_nov23/main/bank.csv'
        bank = pd.read_csv(url)
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import LabelEncoder
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.metrics import f1_score
        from imblearn.over_sampling import RandomOverSampler
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        bank_cleaned = bank.drop(bank.loc[bank["job"] == "unknown"].index, inplace=True)
        bank_cleaned = bank.drop(bank.loc[bank["education"] == "unknown"].index, inplace=True)
        bank_cleaned = bank.drop(['contact', 'pdays'], axis = 1)
        st.subheader('Modélisation')
        st.write('Le modèle choisi sera la classification en raison des valeurs discrètes de la variable cible deposit')
       
        st.write(bank_cleaned['deposit'].head())
        feats = bank_cleaned.drop(['deposit'], axis = 1)
        target = bank_cleaned['deposit']
        st.write('Le jeu de données sera donc séparé en 2 dataframes: "feats" et "target"')

      
        st.dataframe(feats)
     
        st.dataframe(target)
        st.write("1. Nous allons procéder à la séparation du jeu de données en jeu d'entrainement X_train et test X_test avec la répartition 80 et 20%")
        X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state=42)
      
        st.code('X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state=42')
        st.write("2. Puis nous allons dans un deuxième temps appliquer la standardisation des variables numériques")
        cols = ['age','balance','day','campaign','previous','duration']
        scaler = StandardScaler()
        cols =['age','balance','day','campaign','previous','duration']
        cols1 = bank_cleaned[['age','balance','day','campaign','previous','duration']]
      
        st.code("X_train[cols] = scaler.fit_transform(X_train[cols]")
        st.code("X_test[cols] = scaler.transform(X_test[cols]")
       
        st.dataframe(cols1)
        X_train[cols]=scaler.fit_transform(X_train[cols])
        X_test[cols] = scaler.transform(X_test[cols])
        
        st.write("3. Ensuite nous encoderons les variables explicatives Housing, Default et Loan de valeur booléenne avec la formule")
       
        st.code('def replace_yes_no(x)')
        st.code("if x=='no':")
        st.code("  return 0")
        st.code("if x=='yes':")
        st.code("  return 1")
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
        st.write('4. Nous utiliserons également une définition pour la variable month en remplaçant le mois de janvier par 1, février par 2 etc')
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
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import LabelEncoder
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.metrics import f1_score
        from imblearn.over_sampling import RandomOverSampler
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        X_train['month'] = X_train['month'].apply(replace_month)
        X_test['month'] = X_test['month'].apply(replace_month)
        st.write('5. Nous nous servirons de la fonction get.dummies pour les variables de chaîne de caractères')
      
        st.code("X_train = pd.get_dummies(X_train, dtype = 'int')")
        st.code("X_test= pd.get_dummies(X_test, dtype = 'int')")
        
        X_train = pd.get_dummies(X_train, dtype = 'int')
        X_test= pd.get_dummies(X_test, dtype = 'int')
      
        st.write(X_train.head())
        st.write("6. Et pour la dernière étape, nous procéderons à l'encodage de la variable cible avec LabelEncoder")
        le = LabelEncoder()
     
        st.code('y_train= le.fit_transform(y_train)')
        st.code('le.transform(y_test)')
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        from sklearn.linear_model import LogisticRegression
        reglog = LogisticRegression(random_state=42)
        reglog.fit(X_train, y_train)
        print("Accuracy score du Logistic regression (train) : ',reglog.score(X_train, y_train)")
        
        from sklearn.ensemble import RandomForestClassifier
        
        forest = RandomForestClassifier(random_state=42)
        forest.fit(X_train, y_train)
        print('Accuracy score du Random Forest (train) : ',forest.score(X_train, y_train))
        from sklearn.tree import DecisionTreeClassifier
        
        treecl = DecisionTreeClassifier(random_state=42)
        treecl.fit(X_train,y_train)
        
        print('Accuracy score du Decision Tree (train) : ',treecl.score(X_train, y_train))
        
        
        st.subheader('Résultats du modèle')
        modèle_sélectionné=st.selectbox(label="Modèle", options=['Régression logistique','Decision Tree','Random Forest'])
        
        if modèle_sélectionné=='Régression logistique':
            st.metric(label="accuracy y_train", value=round(reglog.score(X_train, y_train),2))
            st.metric(label="accuracy y_test", value=round(reglog.score(X_test, y_test),2))
            
        if modèle_sélectionné=='Decision Tree':
            st.metric(label="accuracy y_train", value=round( treecl.score(X_train, y_train),2))
            st.metric(label="accuracy y_test", value=round(treecl.score(X_test, y_test),2))
        if modèle_sélectionné=='Random Forest':
            st.metric(label="accuracy y_train", value=round(forest.score(X_train, y_train),2))
            st.metric(label="accuracy y_test", value=round(forest.score(X_test, y_test),2))
        st.write("Le modèle RandomForest est donc le meilleur modèle au vu des résultats mais nous constatons un problème d'overfitting")
        st.write('Afin d’évaluer la précision de notre modèle, nous avons vérifié sa volatilité avec la technique de validation croisée sur le modèle RandomForest. Celle-ci étant peu volatile [0.77762106 0.74424071 0.78232252 0.83921016 0.82267168] , nous pouvons considérer que le modèle est fiable via un train_test_split.')
        st.write("Techniques utilisées pour baisser l'overfitting")
        techniques=st.selectbox(label='Techniques', options=['Bagging','RandomOverSampler','GridSearchCV','Importance_feature','Suppression variable Duration'])
        
        feat_importances = pd.DataFrame(forest.feature_importances_, index=X_test.columns, columns =['Importance'] )
        feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
        feat_importances.plot(kind='bar', figsize=(8,6))
        if techniques=='Bagging':
            st.write("La méthode Bagging permet d'améliorer la performance et la stabilité des algorithmes en réduisant la variance et en limitant l'overfitting")
            if st.button('accuracy y_train et y_test Bagging'):
               st.button('1 & 0.84')
               st.write('Nous n’avons pas observé de différence en utilisant le Bagging sur l’overfitting')
         if techniques=='GridSearchCV':
            if st.button('Best Hyperparameter'):
               st.write("{'max_depth': 10, 'n_estimators': 1000}")
            if st.button('accuracy y_train et y_test avec max_depth:10'):
               st.button("0.87 & 0.83")
        if techniques=='RandomOverSampler':
            if st.button('accuracy y_train et y_test avec SMOTE'):
               st.button('0.83 & 0.83')
               st.write("Le résultat a été concluant avec un résultat de 0,83 sur le modèle d'entraînement.") 
               st.write('Quant au modèle test, nous obtenons également un très bon score avec 0.83.')
        if techniques=='Importance_feature':
           st.pyplot(feat_importances.plot(kind='bar').figure)
           st.write('Nous remarquons que la variable Duration prédomine de manière significative sur toutes les autres variables')
           st.write("Suite à l'analyse de l'importance des variables, nous allons réduire notre jeu de données à 5 et 9 variables:")
           if st.button('accuracy y_train et y_test à 5 variables'):                   
              st.button('0.87 & 0.81')   
              st.write("Le fait de ne garder que les 5 variables les plus importantes réduit notre score")                                                                                              
           if st.button('accuracy y_train et y_test à 9 variables'):
              st.button('0.89 & 0.84')
              st.write("Le fait de ne garder que les 9 variables les plus importantes améliore sensiblement notre score")
        
        if techniques=='Suppression variable Duration':
            if st.button('accuracy y_train et y_test sans la variable Duration'): 
               st.button('0.79 & 0.71')
               st.write('L’overfitting sur la random forest a empiré en faisant baisser le score de notre jeu de test.') 
               st.write("En conclusion, l'hyperparamètre max_depth 10 semble être le meilleure solution avec un jeu de données réduit aux 9 variables les plus importantes pour éviter l'overfitting et conserver un modèle prédictif fiable et robuste.")
elif page==pages[5]:
  st.subheader("Conclusion")
  st.markdown("- Ce rapport offre à une entreprise des perspectives précieuses sur les schémas comportementaux des clients et les déterminants influençant leurs choix, offrant ainsi la possibilité d'optimiser les stratégies de marketing et de prendre des décisions stratégiques mieux éclairées.")
  st.markdown("- Notre travail a permis d'identifier les variables les plus significatives tout en surmontant des obstacles tels que la signification des variables.")
  st.markdown("- Des améliorations supplémentaires auraient pu être envisagées afin de peaufiner davantage notre modèle (analyse plus approfondie de l'importance des variables, application de techniques avancées de réduction de la dimensionnalité)")      
