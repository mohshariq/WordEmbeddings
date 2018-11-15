import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec

k=pd.read_csv('csvfile.csv')
k.columns

Index(['Unnamed: 0', 'event', 'event_sub', 'url', 'ind_link', 'ind_head',
       'Unnamed: 5'],
      dtype='object')


#1)CREATING TERM-DOCUMENT MATRIX USING PYTHON OF THE COLUMN 'TITLE'

TITLE=k['ind_head']
TITLE

0          The Finite String, Volume 15, Numbers 1-5, 1978
1          The Finite String, Volume 15, Numbers 1-5, 1978
2          The Finite String, Volume 15, Numbers 1-5, 1978
3          The Finite String, Volume 15, Numbers 1-5, 1978
4          The Finite String, Volume 15, Numbers 1-5, 1978
5          The Finite String, Volume 14, Numbers 1-7, 1977
6          The Finite String, Volume 14, Numbers 1-7, 1977
7          The Finite String, Volume 14, Numbers 1-7, 1977
8          The Finite String, Volume 14, Numbers 1-7, 1977
9          The Finite String, Volume 14, Numbers 1-7, 1977
10         The Finite String, Volume 14, Numbers 1-7, 1977
11         The Finite String, Volume 14, Numbers 1-7, 1977
12         The Finite String, Volume 13, Numbers 1-8, 1976
13         The Finite String, Volume 13, Numbers 1-8, 1976
14         The Finite String, Volume 13, Numbers 1-8, 1976
15         The Finite String, Volume 13, Numbers 1-8, 1976
16         The Finite String, Volume 13, Numbers 1-8, 1976
17         The Finite String, Volume 13, Numbers 1-8, 1976
18         The Finite String, Volume 13, Numbers 1-8, 1976
19         The Finite String, Volume 13, Numbers 1-8, 1976
20         The Finite String, Volume 12, Numbers 1-6, 1975
21         The Finite String, Volume 12, Numbers 1-6, 1975
22       Mechanical Translation and Computational Lingu...
23       Mechanical Translation and Computational Lingu...
24       Mechanical Translation and Computational Lingu...
25       Mechanical Translation and Computational Lingu...
26       Mechanical Translation and Computational Lingu...
27       Mechanical Translation and Computational Lingu...
28       Mechanical Translation and Computational Lingu...
29       Mechanical Translation and Computational Lingu...
                               ...                        
19611    Proceedings of the Third Workshop on Computati...
19612    Proceedings of the Third Workshop on Computati...
19613    Proceedings of the 7th Workshop on Computation...
19614    Proceedings of the 7th Workshop on Computation...
19615    Proceedings of the 7th Workshop on Computation...
19616    Proceedings of the 7th Workshop on Computation...
19617    Proceedings of the 7th Workshop on Computation...
19618    Proceedings of the 7th Workshop on Computation...
19619    Proceedings of the 2nd Workshop on the Use of ...
19620    Proceedings of the 2nd Workshop on the Use of ...
19621    Proceedings of the 2nd Workshop on the Use of ...
19622    Proceedings of the 8th Workshop on Cognitive M...
19623    Proceedings of the 8th Workshop on Cognitive M...
19624    Proceedings of the 8th Workshop on Cognitive M...
19625    Proceedings of the 8th Workshop on Cognitive M...
19626    Proceedings of the 8th Workshop on Cognitive M...
19627    Proceedings of the 8th Workshop on Cognitive M...
19628    Proceedings of the 8th Workshop on Cognitive M...
19629    Proceedings of the 8th Workshop on Cognitive M...
19630    Proceedings of the 8th Workshop on Cognitive M...
19631    Proceedings of the 8th Workshop on Cognitive M...
19632    Proceedings of the Fourth International Worksh...
19633    Proceedings of the Fourth International Worksh...
19634    Proceedings of the Fourth International Worksh...
19635    Proceedings of the Fourth International Worksh...
19636    Proceedings of the Fourth International Worksh...
19637    Proceedings of the Fourth International Worksh...
19638    Proceedings of the Fourth International Worksh...
19639    Proceedings of the Fourth International Worksh...
19640    Proceedings of the Fourth International Worksh...
Name: ind_head, Length: 19641, dtype: object

vec = CountVectorizer()

X = vec.fit_transform(TITLE)
X
<19641x623 sparse matrix of type '<class 'numpy.int64'>'
	with 219874 stored elements in Compressed Sparse Row format>

df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
df
       00  02  03  04  06  08  10  10th  11  11th  ...   word  wordnet  work  \
0       0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
1       0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
2       0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
3       0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
4       0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
5       0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
6       0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
7       0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
8       0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
9       0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
10      0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
11      0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
12      0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
13      0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
14      0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
15      0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
16      0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
17      0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
18      0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19      0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
20      0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
21      0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
22      0   0   0   0   0   0   0     0   1     0  ...      0        0     0   
23      0   0   0   0   0   0   0     0   1     0  ...      0        0     0   
24      0   0   0   0   0   0   0     0   1     0  ...      0        0     0   
25      0   0   0   0   0   0   0     0   1     0  ...      0        0     0   
26      0   0   0   0   0   0   0     0   1     0  ...      0        0     0   
27      0   0   0   0   0   0   0     0   1     0  ...      0        0     0   
28      0   0   0   0   0   0   0     0   1     0  ...      0        0     0   
29      0   0   0   0   0   0   0     0   1     0  ...      0        0     0   
...    ..  ..  ..  ..  ..  ..  ..   ...  ..   ...  ...    ...      ...   ...   
19611   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19612   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19613   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19614   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19615   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19616   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19617   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19618   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19619   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19620   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19621   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19622   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19623   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19624   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19625   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19626   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19627   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19628   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19629   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19630   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19631   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19632   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19633   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19634   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19635   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19636   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19637   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19638   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19639   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   
19640   0   0   0   0   0   0   0     0   0     0  ...      0        0     0   

       workshop  world  writing  xi  xml  xv  york  
0             0      0        0   0    0   0     0  
1             0      0        0   0    0   0     0  
2             0      0        0   0    0   0     0  
3             0      0        0   0    0   0     0  
4             0      0        0   0    0   0     0  
5             0      0        0   0    0   0     0  
6             0      0        0   0    0   0     0  
7             0      0        0   0    0   0     0  
8             0      0        0   0    0   0     0  
9             0      0        0   0    0   0     0  
10            0      0        0   0    0   0     0  
11            0      0        0   0    0   0     0  
12            0      0        0   0    0   0     0  
13            0      0        0   0    0   0     0  
14            0      0        0   0    0   0     0  
15            0      0        0   0    0   0     0  
16            0      0        0   0    0   0     0  
17            0      0        0   0    0   0     0  
18            0      0        0   0    0   0     0  
19            0      0        0   0    0   0     0  
20            0      0        0   0    0   0     0  
21            0      0        0   0    0   0     0  
22            0      0        0   0    0   0     0  
23            0      0        0   0    0   0     0  
24            0      0        0   0    0   0     0  
25            0      0        0   0    0   0     0  
26            0      0        0   0    0   0     0  
27            0      0        0   0    0   0     0  
28            0      0        0   0    0   0     0  
29            0      0        0   0    0   0     0  
...         ...    ...      ...  ..  ...  ..   ...  
19611         1      0        0   0    0   0     0  
19612         1      0        0   0    0   0     0  
19613         1      0        0   0    0   0     0  
19614         1      0        0   0    0   0     0  
19615         1      0        0   0    0   0     0  
19616         1      0        0   0    0   0     0  
19617         1      0        0   0    0   0     0  
19618         1      0        0   0    0   0     0  
19619         1      0        0   0    0   0     0  
19620         1      0        0   0    0   0     0  
19621         1      0        0   0    0   0     0  
19622         1      0        0   0    0   0     0  
19623         1      0        0   0    0   0     0  
19624         1      0        0   0    0   0     0  
19625         1      0        0   0    0   0     0  
19626         1      0        0   0    0   0     0  
19627         1      0        0   0    0   0     0  
19628         1      0        0   0    0   0     0  
19629         1      0        0   0    0   0     0  
19630         1      0        0   0    0   0     0  
19631         1      0        0   0    0   0     0  
19632         1      0        0   0    0   0     0  
19633         1      0        0   0    0   0     0  
19634         1      0        0   0    0   0     0  
19635         1      0        0   0    0   0     0  
19636         1      0        0   0    0   0     0  
19637         1      0        0   0    0   0     0  
19638         1      0        0   0    0   0     0  
19639         1      0        0   0    0   0     0  
19640         1      0        0   0    0   0     0  

[19641 rows x 623 columns]



#2)PERFORMING SVD ON THE ABOVE MATRIX USING PYTHON 

u,s,v=np.linalg.svd(m)
print(u)

array([[-4.85564751e-03, -4.17739321e-02, -2.02792394e-01, ...,
         4.40896547e-17,  4.64737278e-17,  1.66132869e-01],
       [-4.85564751e-03, -4.17739321e-02, -2.02792394e-01, ...,
         2.39520815e-17,  2.50852598e-17, -7.41503292e-02],
       [-4.85564751e-03, -4.17739321e-02, -2.02792394e-01, ...,
         2.03274606e-16,  2.11594759e-16,  6.61573655e-01],
       ...,
       [-1.24545152e-01,  3.36760990e-02, -3.90346604e-03, ...,
        -1.49967429e-02, -1.49967429e-02, -3.02704992e-03],
       [-1.24545152e-01,  3.36760990e-02, -3.90346604e-03, ...,
         9.85003257e-01, -1.49967429e-02, -3.02704992e-03],
       [-1.24545152e-01,  3.36760990e-02, -3.90346604e-03, ...,
        -1.49967429e-02,  9.85003257e-01, -3.02704992e-03]])


print(s)



array([2.36179922e+01, 1.74589783e+01, 1.07416083e+01, 5.03162003e+00,
       4.56862622e+00, 3.87274779e+00, 3.65784739e+00, 3.34880090e+00,
       2.33467868e+00, 2.18165296e+00, 8.52270156e-15, 1.72053603e-15,
       1.72053603e-15, 1.72053603e-15, 1.72053603e-15, 1.72053603e-15,
       1.72053603e-15, 1.72053603e-15, 1.72053603e-15, 1.72053603e-15,
       1.72053603e-15, 1.72053603e-15, 1.72053603e-15, 1.72053603e-15,
       1.72053603e-15, 1.72053603e-15, 1.72053603e-15, 1.72053603e-15,
       1.72053603e-15, 1.72053603e-15, 1.72053603e-15, 1.72053603e-15,
       1.72053603e-15, 1.72053603e-15, 1.72053603e-15, 1.72053603e-15,
       1.72053603e-15, 1.72053603e-15, 1.72053603e-15, 1.72053603e-15,
       1.72053603e-15, 1.72053603e-15, 1.72053603e-15, 1.72053603e-15,
       1.72053603e-15, 1.72053603e-15, 1.72053603e-15, 1.72053603e-15,
       1.72053603e-15, 1.72053603e-15, 1.72053603e-15, 1.72053603e-15,
       1.72053603e-15, 1.72053603e-15, 1.72053603e-15, 1.72053603e-15,
       1.72053603e-15, 1.72053603e-15, 1.72053603e-15, 1.72053603e-15,
       1.72053603e-15, 1.72053603e-15, 1.72053603e-15, 1.72053603e-15,
       1.72053603e-15, 1.72053603e-15, 1.72053603e-15, 1.72053603e-15,
       1.72053603e-15, 1.72053603e-15, 1.72053603e-15, 1.72053603e-15,
       1.72053603e-15, 1.72053603e-15, 1.72053603e-15, 1.72053603e-15,
       1.72053603e-15, 1.72053603e-15, 1.72053603e-15, 1.72053603e-15,
       1.72053603e-15, 1.72053603e-15, 1.72053603e-15, 1.72053603e-15,
       1.72053603e-15, 1.72053603e-15, 1.72053603e-15, 1.72053603e-15,
       1.72053603e-15, 1.72053603e-15, 1.72053603e-15, 1.72053603e-15,
       1.72053603e-15, 1.72053603e-15, 1.72053603e-15, 1.72053603e-15,
       1.72053603e-15, 1.72053603e-15, 1.72053603e-15, 1.59833770e-15])



print(v)



array([[ 4.46620199e-17, -1.11022302e-16, -1.11022302e-16, ...,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [ 8.76483496e-17, -1.11022302e-16,  5.55111512e-17, ...,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [ 3.85616596e-17,  1.52655666e-16, -5.55111512e-17, ...,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       ...,
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
         1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
         0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
         0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])




#3)CONSTRUCTION OF WORD2VEC EMBEDDINGS FOR EACH WORD


example=a['Unnamed:5']
tokenized_sents = [word_tokenize(i) for i in example]
model = Word2Vec(tokenized_sents, min_count=1)
list(model.wv.vocab)

SATURATED', 'PARTITIONS', 'German-English', 'periphrasing', 'Sur', 'quelques', 'preprietes', 'communes', 'catagories', 'semantiques', 'procedures', 'generatrices', 'trois', 'modeles', 'synthese', 'processus', 'TA', 'resume', 'government', 'Transcoding', 'structurelle', 'grouses', 'nominaux', 'anglais', 'Ein', 'Programm', 'zur', 'automatischen', 'Synthese', 'englischer', 'Satze', 'HINDI', '63', '64', '65', 'DISKONTINUIERLICHE', 'KONSTITUENTEN', '66', 'INTUITVE', '67', 'DIE', 'SEMANTISCHE', 'DER', 'KOMMUNIKATIVEN', 'GRAMMATIK', 'AUF', 'EDV-ANLAGEN', '68', 'MULTI-INDEX', '69', 'Isaiah', 'Disputed', 'ADOUT', 'VECTORIAL', 'SYTACTIC', 'MARKERS', 'DOC', '71', 'METHODOLOGICAL', 'MALARINSELN', 'IHRE', 'SEHENSWURDIGKEITEN', 'Declarations', 'Half', 'COMMENTARY', 'MEY', 'PAPER', 'Post-print', 'METAPRINT', 'HALF', 'Alphabetic', 'LEXICOLOGICAL', 'BALTIC', 'corrigenda', 'addenda', 'pre-prints', 'Preprints', 'BULLETIN', 'MONDAY', 'SEPTEMBER', 'NEIGHBOURHOOD', 'MATHEMATIQUE', "D'ANALYSE", 'SELON', 'Z.', 'HARRIS', 'CONSTRUCTIBLE', 'METALANGUAGE', 'SEMANTICAL', 'WORKING', 'T.G.T.-SYSTEM', 'JOYCE', 'FRIEDMAN', 'OBLING', 'TESTER', 'ALGORITHMISATION', 'STRUCTURAL-PROBABILISTIC', 'IN-COMPUTERIZED', 'OLD', 'CORNISH', 'VERBAL', 'BEHAVIOUR', 'UNIQUE', 'DECIPHERABILITY', 'SYLLABIFICATION', 'TURKISH', 'DECIPHERING', 'OBTAINED', 'CONSONNES', 'DESTINEE', 'LEURS', 'COMBINAISONS', 'DICTIONNAIRE', 'DICTIONNAIRE-MACHINE', 'HYPOTHESE', 'RECONSTRUCTING', 'PREHISTORIC', 'TRIUMPH', 'NEOGRAMMARIAN', 'FILE', 'IMPROVEMENT', 'CHINESE/ENGLISH', 'HERITAGE', 'HUNGARIAN', 'TRAVAUX', 'MISE', 'POINT', "D'UN", 'LEXIQUE', 'EN', 'VUE', 'TRANSCRIPTION', 'FRANCAISE', 'OU', 'SEMI-AUTOMATIQUE', 'TEXTE', 'GREC', "D'ARISTOTE", 'RECHERCHE', 'DERIVATION', 'ITALIEN', 'FEASIBILITY', 'LEXICOGRAPHIC', 'COMPUTER-AIDED', 'EDITING', 'REFINEMENTS', 'REGIONAL', 'AGE', 'SYNTAGMS', 'ANALYTICITY', 'ASPIRATION', 'PSYCHOTIC', 'PSYCHONEUROTIC', 'PATIENTS', 'KAZAKH', 'SYNTAGMATIC', 'POLISH', 'HIPPO', 'AUTOMATISATION', 'DONNEES', 'STYLOMETRIQUES', 'PARTIR', "L'OEUVRE", "D'HIPPOCRATE", 'PROPORTION', 'NUMBER', 'WORKS', 'THOMISTICUS', 'DETERMINATIONS', 'DIALINGUISTIQUES', 'GENRE', 'LITTERAIRE', 'CONSTITUTION', 'PROGRAMME', "D'APPRENTISSAGE", 'POUR', 'ALGORITHMES', 'SIMPLES', 'CONVERSING', 'MAKING', 'UTILISANT', 'CHAINE', 'ANALYSEUR', 'SYNTAXIQUE', 'INTERACTIF', 'HOMME-MACHINE', 'UNIVERSALS', 'GROUP', 'STEP', 'STEM-SUFFIX', 'DISCRIMINATION', 'SYSTEME', 'A.T.E.F', 'MORPHOLOGIQUE', 'RUSSES', 'ANCIENT', 'REQUEST', 'PAR', "D'ETATS", 'FINIS', 'NUMBERS', 'NUMERALS', 'VICE', 'VERSA', 'BUILT', 'HOMOGRAPH', 'WRITTEN', 'SGS', 'MECHANICAL', 'PREDICATIONAL', 'PERCEPTUAL', 'FUZZY', 'EAST-', 'WEST-GERMAN', 'HIERARCHICAL', 'TOPOLOGICAL', 'TIME-RELATED', 'ACTIONS', 'PT-CHART', 'TEMPS', 'PASSE', 'peut', 'appliquer', 'entre', 'nom', 'matiere', 'comptable', 'aux', 'temps', 'verbe', 'UDERSTANDING', 'PICTURE', 'METAPHOR', 'WHY-QUESTIONS', 'EXPRESSIVENESS', 'SACRED', 'LEGENDS', 'SPEECHES', 'MARTIN', 'LUTHER', 'KING', 'JR', 'EMBEDDED', 'ADAPTATION', 'MONTAGUE', 'QUESTION-ANSWERING', 'THEORETIC', 'MANY-PURPOSE', 'INTENSIONAL', 'CONNOTATION', 'MAXIM', 'TOPIC-COMMENT', 'KEYBOARD', 'ASSIGNMENT', 'KEY-CODES', 'ERROR', 'FROFF', 'SCIENCE', 'STROKE', 'SEQUENCE', 'KANJI', 'INTERNAL', 'STOPAGE', 'CONSIDERATIONS', 'KEYS', 'HIGH-SPEED', 'DISPLAY', 'NON-SEGMENTED', 'KANA', 'KANJI-KANA', 'DIFFERENT', 'COMPILATION', 'CONCORDANCES', 'THAI', 'THREE', 'SEALS', 'LAW', 'AGAINST', 'SCHEMATA', 'DEAL', 'AMBIGUITIES', 'WHILE', 'EXAM', 'LONGMAN', 'UNIT-TO-UNIT', 'PREDICATE', 'ATTRIBUTE', 'ORDERINGS', 'PARADIGMS', 'AUTOMATIZED', 'RUSSIAN-FRENCH', 'GETA', 'DETAILED', 'EXAMPLE', 'ENGLISH-JAPANESE', 'CASE-STRUCTURE', 'CONVERSION', 'TERMINOLOGY', 'BANKS', 'HIGH-QUALITY', 'TRIAL', 'IMPATIENT', 'ATNS', 'USED', 'MODULARITY', 'intelligent', 'digester', 'processings', 'PRATIQUE', "D'UTILISATION", 'LINGUISTIQUE', "D'INFORMATION", 'BILAN', 'APPLICATIVE', 'IDEOGRAPHIC', 'ALPHABETIC', 'ATTEMPT', 'NATURAL-ARTIFICIAL', 'GOAL', 'ORIENTED', 'NATURLICHSPRACHIGE', 'PROBLEMBESCHREIBUNG', 'ALS', 'VERFAHREN', 'FUR', 'DEN', 'BURGERNAHEN', 'ZUGANG', 'ZU', 'DOKUMENTATIONSSYSTEMEN', 'GUIDED', 'ANSWER', 'ZAPSIB', 'MANIPULATION', 'BAHASA', 'MALAYSIA', 'REDUCE', 'VOCABULARY-TEXT', 'GLAPS', 'verbalism', 'VISION', 'ARIANE', '78.4', 'FORMULAS', 'DISCRETELY', 'UTTERED', 'DEFINED', 'RECHERCHES', 'SUR', 'CONNAISSANCES', 'ARCHES', 'FRAME', 'THEME', 'CONTINUITY', 'FORWARD', 'BACKWARD', 'ABSTRACTING', 'DIALOGIC', 'ATTRIBUTES', 'ACTIVATION', 'Order-Free', 'MULTILAYERED', 'FORMATION', 'LESNIEWSKIAN', 'INTRA-SENTENTIAL', 'CODE-SWITCHING', 'MARKEDNESS', 'LOGICALLY', 'ISOMORPHIC', 'RESEDA', 'TULIPS-2', 'Anatomy', 'ARBUS', 'DEVELOPING', 'MULTI-LINGUAL', 'PARAPHRASING', 'SCIENTIFIC', 'ZUM', 'WIEDERAUFFINDEN', 'VON', 'INFORMATIONEN', 'AUTOMATISCHEN', 'WORTERBOCHERN', 'TERMSERVICE', 'SERVICES', 'ENGLISH-INTO-JAPANESE', 'SERVICE', 'WAYS', 'CZECH', 'MESSAGE-PASSING', 'D-Trees', 'CONNOTATIVE', 'ANALOGICAL', 'tense', 'IMPROVED', 'LEFT-CORNER', 'PARALLELISM', 'LAGUAGE', 'MANDARIN', 'Phase', 'UTILISATION', 'PARALLELISME', 'AUTOMATISEE', 'ORDINATEUR', 'NATURAL-LANGUAGE-ACCESS', 'Justifying', 'MAN-ASSISTED', 'TEST-SCORE', 'METHODES', 'MORPHOSYNTAXIQUE', 'LEXICALE-SEMANTIQUE', 'LANGUE', 'ESPAGNOLE', 'BASIC', 'KEYWORD', 'SUBORDINATE', 'BELIEF', 'MODES', 'COMMENTATOR', 'SUBSTANTIONAL', 'ADAPTIVE', 'MECHANIZED', 'SPEZIELLES', 'FRAGE', 'ANTWORT-BEZIEHUNGEN', 'FRAGEN', 'PLURALISCHEN', 'ANALYSER', 'LDVLIB', 'LEM', 'LEMMATIZING', 'MERGING', 'ART', 'SINGLE', 'REVISING', 'COLLOCATIONAL', 'CONTRASTIVE', 'ARTICLES', 'REALISATIONS', 'BUREAU', 'CANADIEN', 'TRADUCTIONS', 'DOMAINE', "L'AUTOMATISATION", 'LOGIC-ORIENTED', 'RHEME', 'AMBIGUITATENPROBLEM', 'BEI', 'KOORDINATIVER', 'VERBINDUNGEN', 'CONJUNCTION', 'DOMAIN-INDEPENDENT', 'WHY', 'MUST', 'OVER', 'ABOVE', 'ANY', 'ADVERBIAL', 'TIBAQ', 'VERBOSITY', 'SOLVERS', 'INFLEXIONAL', 'PRIVILEGE', 'GRAPHEME-TO-PHONEME', 'TRANSFORNATION', 'STRATIFIKATIVE', 'SFRACHBESCHREIBUNG', 'INKORPORIERENDER', 'BEDEUTUNGSKOMPONENTE', 'ELEMENTE', 'EINES', 'ENTWURFS', 'PEARL', 'LOCALITY', 'PHENOMENON', 'NATURE', 'CONVERSATIONS', 'PET', 'BULGARIAN', 'PROCESSORS', 'COMPRESSION', 'EXPLICIT', 'POSSESSIVITY', 'HUMAN-LIKE', 'BUNDLES', 'ADVERBS', 'FUNCTIONS', 'SECRET', 'GRAMMATIC', 'NORMATIVITY', 'ESSAY', 'DERIVED', 'REAL', 'LESK', 'SYNTHETIZING', 'BEDE', 'MICROPROCESSOR-BASED', 'ADVANCEMENT', 'INFORMATIONAL', 'ANALYSERS', 'POOLING', 'THREADING', 'UNDERLYING', 'COGITOLOGY', 'AUTHORS', 'Mu-Project', 'SCALE', 'EXPRESSIVE', 'OPERA', 'DISPARITY', 'sensitivity', 'DIAIOGUE', 'INTERFACILE', 'PROJECTIVITY', 'Plurals', 'Cardinalities', 'MODIFIED', 'PREARRANGED', 'D-PATR', 'Verifiability', 'Mu', 'Element', 'Idiosyncratic', 'Tough', 'Structure-bound', 'MT-oriented', 'versions', 'Valency', 'Lexicase', 'Lexicon-driven', 'Mu-MACHI', 'NE', 'TRANSLATI', 'ROMANCE', 'LAGUAGES', 'lexicon-grammars', 'Reconnaissance-Attack', 'Ready', 'LanguageCraft', 'XCALIBUR', 'experiences', 'Already', 'PP-Fronting', 'cognitive', 'SITUATIONAL', 'PRESUPPISITION', 'LINKING', 'COHESION', 'DEGREES', 'Parenthesis-Free', 'STRATA', 'NON-CF', 'PARSERS', 'KIND', 'DCKR', 'ELEMENTARY', 'CONTRACTS', 'TRIAD', 'ELEMENT', 'TBMS', 'READER', 'SUMMARIZATION', 'Synergy', 'minimum', 'Kana-Kanji', 'Non-Segmented', 'ANALOGY', 'DEVELOPMENTS', 'SINCE', 'THEORETICALLY', 'COMMITTED', 'EUROTRA-D', 'NARA', 'Two-way', 'methodological', 'study-', 'experience', 'Collative', 'Locative', 'Reopened', 'Non-Singular', 'CLINICAL', 'NARRATIVES', 'PeriPhrase', 'SCSL', 'specification', 'Friendly', 'APE', 'grammaticales', 'un', 'modele', 'Arlane', 'SYSTEMES', 'TRANSFORMATIONNELS', 'CRITAC', 'PROOFREADING', 'STORING', 'INTEGER', 'BetaText', 'motivations', 'organisation', 'INDEXAGE', 'MT-DIRECTED', 'BANK', 'Shakespeare', 'Kowledge', 'MANIPULATING', 'Movement-Rules', 'LFG-Parser', 'C-', 'F-Structure', 'LFG-Proposal', 'DISCOURSE-ORIENTED', 'Honorifics', 'ASSERTION', 'FORMALIZATIONS', 'BUILDRS', 'DR', 'Government-Binding', 'Evaluation-', 'SYNTHESIZING', 'WEATHER', 'FORECASTS', 'FORMATTED', 'teaching', 'various', 'Traffic', 'PARAPHRASES', 'functional', 'DIVIDED', 'VALENCY-ORIENTED', 'UNDSTANDING', 'Semantic-Representation-to-Speech', 'Tile', 'procedure', 'construct', 'predictor', 'task-specific', 'MERGED', 'PIVOT', 'VESPRA', 'LUTE', 'Stride', 'Malay', 'EXTRACTS', 'PPOCESSTNG', 'MANUALS', 'modular', 'Mariko', 'talks', 'Siegfried', 'Japanese/German', 'Project-', 'Feasible', 'accounts', 'MT-SYSTEM', 'LFG-Parsers', 'Table-Lookup', 'Isomorphic', 'UCGs', 'E-Framework', 'CREATING', 'WORDCLASS', 'LABELLED', 'TEXT-CORPORA', 'CO-ORDINATIVE', 'RESTORATION', 'TWO-COMPONENT', 'UNDERSTANDS', 'CORRECTS', 'MISTAKES', 'SPEECH-RATE', 'DURATION', 'Multi-Strategy', 'RUG', 'IMT/EC', 'SCHEMES', 'GRAFON', 'PEP', 'THOROUGHLY', 'DESCENDANT', 'WEP', 'Manager', 'Orderings', 'Aktionsarten', 'morpho-syntactic', 'quantifier', 'Sequencing', 'severely', 'corrupted', 'PSI/PHI', 'Demonstrative', 'Unbounded', 'FILLER', 'PRINCIPLE', 'CROSSING', 'COREFERENCE', 'Failure', 'failure', 'logics', 'NL-semantics', 'CRITTER', 'agricultural', 'market', 'reports', 'Bidirectionality', 'Concretion', 'Assumption-Based', 'LOCALLY', 'DEPENDECNCY', 'RECONNAISSANCE-ATTACK', 'VOCNETS', 'VOCABULARIES', 'Ancient', 'Accadian', 'Coocurrence', 'TRADITIONAL', 'Finnish', 'RECENTLY', 'INSTATIATIONS', 'OPTIONAL', 'ACTANTS', 'SAGE', 'Incomplete', 'NOSVO', 'testing', 'phases', 'PROCESS-ACTIVATION', 'Functorial', 'assertions', 'presuppositions', 'Rock', 'DIAGNOSING', 'TRAINING', 'Frame-Adverbials', 'EFFECTIVE', 'MT-SYSTEMS', 'Bottle', 'Neck', 'Grammarians', 'Skeptical', 'Implementors', 'Wait', 'Theses', 'why', 'care', 'Ordinary', 'Discontinuities', 'Postediting', 'look-ahead', 'on-line', 'OUTPUT', 'Denoting', 'HINTING', 'INSTRUCTION', 'Sourcebook', 'Intersection', 'intermediate', 'YES-NO', 'FEATURING', 'Providing', "'Lexicalized", 'Output-', 'IMPLICITNESS', 'GUIDING', 'SENSITIVE', 'Directing', 'Island', 'LangLAB', 'paradigm', 'interlaces', 'knowledge-based', 'Lexicon-Driven', 'Phrase-Structure-Based', 'PRORGRAMS', 'MASSIVE', 'breakdown','peer-support', 'Data61-CSIRO', 'Severity', 'ReachOut', 'radial', 'Distress', 'LT3', 'GW/UMD', 'ReachOut.com', 'forum', 'emergencies', 'talking', 'ToWork', 'Rumor', 'Arousal', 'Facebook', 'Purity', 'Homophily', 'Odawa', 'power', 'times', 'uncertainty', 'gradient', 'certainty', 'weights', 'activation', 'connected', 'learn', 'continuity', 'SOM-based', 'harvester', 'Sound-aligned', 'Udmurt', 'dialectal', 'Finno-Ugric', 'entries', 'Uralic']




x=model[model.wv.vocab]
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
result = pca.fit_transform(x)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))

#WORD2VEC EMBEDDINGS FOR EACH WORD::

Text(-0.329525,-0.0328367,'Desk')
Text(-0.0936722,0.00972406,'Noen')
Text(-0.0230605,0.0131622,'problemer')
Text(0.0190407,0.0404647,'ved')
Text(0.0142469,0.0493471,'opbygging')
Text(0.0126291,0.0938168,'av')
Text(-0.0587143,0.0648439,'datamaskinelt')
Text(-0.128439,0.0114468,'morfem-lexikon')
Text(-0.0675735,0.0294409,'problems')
Text(-0.16183,-0.000888386,'computerized')
Text(-0.34054,-0.00686881,'MONOTONICITY')
Text(-0.339895,-0.00148765,'HEADEDNESS')
Text(-0.267189,-0.0357589,'Inherently')
Text(-0.273666,-0.0366957,'Computability')
Text(-0.388314,-0.0183194,'REVERSIBILITY')
Text(-0.366,0.0393175,'DERIVING')
Text(-0.294957,-0.00481793,'FORMALISM-INDEPENDENT')
Text(-0.320002,-0.0212699,'CHARACTERISATION')
Text(-0.241423,0.01171,'LINE')
Text(-0.389217,0.00130115,'WHATEVER')
Text(-0.338128,-0.023293,'COMPILING')
Text(-0.301803,-0.0372864,'Ontology-Driven')
Text(-0.291588,0.024318,'TERMINOLOGY-INTENSIVE')
Text(-0.25759,0.0237533,'SO')
Text(-0.260984,0.00950017,'MANY')
Text(-0.325492,-0.04116,'non-lexical')
Text(-0.187136,0.000703899,'meaning')
Text(-0.350409,-0.0540764,'Tense/Aspect')
Text(-0.290664,-0.0420473,'Autonomy')
Text(-0.350232,-0.0519514,'Suit')
Text(-0.287754,-0.0416358,'Lexical/Semantic')
Text(-0.291947,-0.0383912,'Standardizing')
Text(-0.299064,-0.0387958,'Journey')
Text(-0.307554,-0.0342389,'LDBs')
Text(-0.252598,-0.00960102,'Sufficient')
Text(-0.25788,-0.0235613,'Rhetoric')
Text(-0.335743,-0.0390804,'Goal-Based')
Text(-0.260993,-0.0453502,'Two-Medium')
Text(-0.275511,-0.0301366,'Defense')
Text(-0.224849,-0.00454231,'promptly')
Text(-0.207518,0.00828119,'god')
Text(-0.236241,-0.0120545,'happy')
Text(-0.263142,-0.0139067,'Domain-Dependent')
Text(-0.288695,-0.0418458,'Intention-Based')
Text(-0.311614,-0.0774157,'Intentionally-Based')
Text(-0.265575,-0.0324102,'Interdependencies')
Text(-0.291581,-0.03373,'RElations')
Text(-0.275986,-0.0118975,'Could')
Text(-0.22415,-0.0325733,'Multiagent')
Text(-0.256724,-0.032972,'Universes')
Text(-0.258258,-0.0198222,'Text-Analytic')
Text(-0.24276,-0.0195204,'Lever')
Text(-0.27706,-0.0326471,'Intentionality')
Text(-0.308194,-0.0370198,'NPtool')
Text(-0.282368,-0.0519597,'V-N-Collocations')
Text(-0.25604,-0.00904938,'sign-based')
Text(-0.266029,-0.0225141,'Puzzle')
Text(-0.342526,-0.0483028,'Multra')
Text(-0.2771,-0.0398097,'London-Lund')
Text(-0.305777,-0.0326232,'Designs')
Text(-0.218831,-0.00227838,'Braying')
Text(-0.311345,-0.022201,'AMALGAM')
Text(-0.173857,0.000252014,'Lexico-Grammatical')
Text(-0.289622,-0.0547863,'Approches')
Text(-0.22765,-0.022361,'Statistical/Symbolic')
Text(-0.275913,-0.0311199,'Captioned-Information')
Text(-0.288051,-0.0198719,'Morpho-phonology')
Text(-0.322501,-0.0380207,'Exceptions')
Text(-0.259183,-0.0221159,'Linearly')
Text(-0.247415,-0.0212719,'Ordered')
Text(-0.278437,-0.0329565,'Vowels')
Text(-0.276043,-0.0343023,'Phonotactics')
Text(-0.28858,-0.0160104,'DPOCL')
Text(-0.249315,-0.0287439,'Moving')
Text(-0.349369,-0.0612151,'Cornerstone')
Text(-0.340494,-0.0332735,'Sage')
Text(-0.32105,-0.0821349,'Causation')
Text(-0.277418,-0.0177408,'Orders')
Text(-0.288773,-0.0200365,'CHoices')
Text(-0.272974,-0.00217963,'Appeared')
Text(-0.329752,-0.0459032,'Digressive')
Text(-0.280202,-0.0127374,'Yes-No')
Text(-0.306608,-0.0405216,'NL-SOAR')
Text(-0.297678,-0.0351753,'Compromises')
Text(-0.278217,-0.0515751,'Quasi-Logical')
Text(-0.320288,-0.0273908,'CORECT')
Text(-0.338831,-0.0231528,'Semanitic')
Text(-0.282182,-0.0812768,'Sign-Language')
Text(-0.29352,-0.0500222,'ZARDOZ')
Text(-0.221146,-0.0192369,'Groupings')
Text(-0.319472,-0.0227425,'IDF')
Text(-0.319571,-0.0199527,'Deviations')
Text(-0.319197,-0.0268904,'Poisson')
Text(-0.262824,-0.0577559,'Suggestion')
Text(-0.281284,-0.0631978,'Predefined')
Text(-0.274695,-0.0369137,'Teleman')
Text(-0.289421,-0.0173856,'Tree-shaped')
Text(-0.281327,-0.0366363,'Inside-out')
Text(-0.286426,-0.0573654,'Triumphs')
Text(-0.30438,-0.0474482,'Tribulations')
Text(-0.261907,-0.0420595,'Teams')
Text(-0.280405,-0.027903,'Toolkits')
Text(-0.280116,-0.0263239,'Corpus-driven')
Text(-0.311021,-0.0199062,'Hunting')
Text(-0.284146,-0.00562923,'Elusive')
Text(-0.320078,-0.0373471,'Timestamped')
Text(-0.282042,-0.0282919,'Diffusion')
Text(-0.298995,-0.0381108,'Min-Cuts')
Text(-0.319225,-0.0413593,'DLSITE-2')
Text(-0.27113,-0.0328838,'Develop')
Text(-0.278163,-0.0191123,'Spell-checker')
Text(-0.292457,-0.0667133,'Troubleshooting')
Text(-0.282302,-0.0332873,'real-world')
Text(-0.27672,-0.0239049,'POMDP-based')
Text(-0.283077,-0.0301086,'Dashboard')
Text(-0.280452,-0.0359993,'Olympus')
Text(-0.251912,-0.0122899,'open-source')
Text(-0.274309,-0.0416062,'Practices')
Text(-0.257524,-0.0385668,'Reconciling')
Text(-0.174191,-0.0222956,'3000')
Text(-0.183241,-0.0215937,'Agency')
Text(-0.240589,-0.0147339,'academic')
Text(-0.14192,0.0181417,'industrial')
Text(-0.27427,-0.030639,'In-Service')
Text(-0.301923,-0.0480863,'Call-Routing')
Text(-0.314718,-0.0507297,'AdaRTE')
Text(-0.205177,-0.000446191,'Multi-slot')
Text(-0.18616,0.0101212,'natural-language')
Text(-0.19765,0.00427565,'routing')
Text(-0.219253,-0.0184037,'commercial')
Text(-0.213482,-0.0145997,'grammar-based')
Text(-0.320281,-0.0247973,'WIRE')
Text(-0.302096,-0.0180244,'Wearable')
Text(-0.272007,-0.0302135,'chatbot')
Text(-0.313361,-0.0428803,'Chunk-Level')
Text(-0.255459,-0.0156462,'Target-side')
Text(-0.28939,-0.00801889,'divergence')
Text(-0.331913,-0.0486099,'Transfer-Based')
Text(-0.244904,-0.0193958,'Phon')
Text(-0.230701,-0.00923756,'1.2')
Text(-0.261814,-0.031537,'High-accuracy')
Text(-0.267218,-0.0309014,'CHILDES')
Text(-0.300799,-0.0386942,'Conundrum')
Text(-0.329648,-0.0467178,'Revolution')
Text(-0.285677,-0.0210535,'Triumph')
Text(-0.267868,-0.0233775,'Hope')
Text(-0.2748,-0.0170059,'Variance')
Text(-0.274612,-0.0411281,'D-Confidence')
Text(-0.204019,-0.032564,'which')
Text(-0.212594,-0.0164842,'Identifies')
Text(-0.318929,-0.0198994,'Staleness')
Text(-0.243121,-0.0296304,'Folding')
Text(-0.266755,-0.021588,'Care')
Text(-0.27689,-0.0214938,'Evoked')
Text(-0.311294,-0.0299566,'Text-driven')
Text(-0.288193,-0.029217,'Propositional')
Text(-0.291876,-0.0131825,'lambda')
Text(-0.201962,-0.00587377,'tractable')
Text(-0.200646,-0.00214931,'intractable')
Text(-0.228195,0.00269827,'reciprocal')
Text(-0.282508,-0.0474701,'Cross-lingual/Interlingual')
Text(-0.233145,-0.0115018,'specifying')
Text(-0.280513,-0.0336236,'underquantification')
Text(-0.297456,-0.0277492,'images')
Text(-0.312038,-0.0446054,'Elaborating')
Text(-0.288231,-0.0398983,'Confinement')
Text(-0.206096,-0.0049595,'determiner')
Text(-0.232476,-0.00374883,'virtual')
Text(-0.262304,0.000564295,'Maximalization')
Text(-0.22957,0.0164914,'witness')
Text(-0.219825,-0.0281779,'N-V')
Text(-0.2527,-0.00998045,'DISCUSS')
Text(-0.213298,-0.0123719,'move')
Text(-0.243179,-0.0144683,'layered')
Text(-0.223441,0.00471675,'Connotational')
Text(-0.241441,-0.0149426,'Drifts')
Text(-0.30334,-0.0417833,'Component-Based')
Text(-0.340409,-0.0531306,'BALLGAME')
Text(-0.278471,-0.0527034,'art')
Text(-0.296887,-0.0191129,'Sibling')
Text(-0.305582,-0.0276245,'Evaluativity')
Text(-0.287655,-0.0382003,'MMIL')
Text(-0.264839,-0.039816,'MEDIA')
Text(-0.305983,-0.077301,'Picture')
Text(-0.216326,-0.0112986,'Edge')
Text(-0.189703,-0.00387155,'dependent')
Text(-0.176495,0.00934349,'pathway')
Text(-0.132157,0.0257089,'scoring')
Text(-0.146607,-0.00215723,'calculating')
Text(-0.273248,-0.02701,'equal')
Text(-0.257002,-0.0457267,'Phenotype')
Text(-0.268413,-0.0250644,'EVEX')
Text(-0.241452,-0.0129836,'PubMed-Scale')
Text(-0.230884,-0.00426263,'Homology-Based')
Text(-0.244254,-0.0246114,'assignment')
Text(0.179988,0.0509049,'Triage')
Text(-0.305707,-0.0267612,'Biocuration')
Text(-0.311998,-0.0737246,'Comparaison')
Text(-0.255216,-0.0349898,'Bio-Medical')
Text(-0.20339,-0.0027922,'frame-based')
Text(-0.251493,0.014747,'ontological')
Text(-0.293849,-0.0372973,'Coreference-Annotated')
Text(-0.315028,-0.0485107,'Biochemistry')
Text(-0.265027,-0.00926698,'Hospital')
Text(-0.265954,0.00112227,'Discharge')
Text(-0.329221,-0.0326316,'Locations')
Text(-0.240519,-0.025354,'deposition')
Text(-0.240645,-0.0145938,'statements')
Text(-0.245911,-0.0122489,'where')
Text(-0.270494,-0.0192217,'Pathways')
Text(-0.230092,-0.0156891,'Biomolecular')
Text(-0.25965,-0.0185978,'Exhaustive')
Text(-0.302554,-0.0267113,'Modifications')
Text(-0.28974,-0.0251083,'Full-Text')
Text(-0.295505,-0.0372863,'SimSem')
Text(-0.270315,-0.0322065,'Based-on')
Text(-0.284034,-0.0138949,'Positively')
Text(-0.225552,-0.00630649,'Unlocking')
Text(-0.231607,-0.0127524,'Non-Ontology')
Text(-0.246212,-0.0351445,'co-training')
Text(-0.277078,-0.0182984,'ThaiHerbMiner')
Text(-0.223837,-0.0165564,'Herbal')
Text(-0.314348,0.00311004,'Vocalization')
Text(-0.30361,-0.0182811,'rabic')
Text(-0.278804,-0.0466572,'Infant')
Text(-0.299233,-0.0440044,'Spatiotemporal')
Text(-0.270198,-0.0303576,'Actionable')
Text(-0.157009,0.0163347,'aitian')
Text(-0.228895,-0.00525099,'reyol')
Text(-0.29674,-0.0340127,'Meaning-Preserving')
Text(-0.32786,-0.0341039,'ULISSE')
Text(-0.265562,-0.0331604,'Second-order')
Text(-0.267544,-0.0231245,'k-Nearest')
Text(-0.299834,-0.0152811,'L_0')
Text(-0.282759,-0.00964071,'-norm')
Text(-0.306332,-0.0354399,'Normalized-Cut')
Text(-0.253947,-0.0314642,'onto')
Text(-0.284621,-0.0131196,'Web-scale')
Text(-0.269404,-0.0236399,'Rival')
Text(-0.227146,-0.0220533,'OWL/DL')
Text(-0.20059,-0.0186082,'MULTEXT-East')
Text(-0.186588,-0.0235334,'morphosyntactic')
Text(-0.232758,-0.015932,'specifications')
Text(-0.197089,0.00371726,'scaleable')
Text(-0.118832,-0.00471141,'automated')
Text(-0.138308,0.00721649,'assurance')
Text(-0.186175,-0.00272477,'proposition')
Text(-0.202647,-0.0132208,'banks')
Text(-0.268134,-0.0354843,'usability')
Text(-0.287866,-0.0388109,'Tamil')
Text(-0.301959,-0.027649,'MAE')
Text(-0.23349,0.00163347,'MAI')
Text(-0.227538,-0.0109587,'Adjudication')
Text(-0.298036,-0.0506812,'It-Timeml')
Text(-0.302495,-0.0464785,'Ita-TimeBank')
Text(-0.317042,-0.0413043,'Discourse-constrained')
Text(-0.267093,-0.0212981,'Interesting')
Text(-0.305762,-0.0275195,'SciSumm')
Text(-0.283774,-0.0128486,'wrote')
Text(-0.282162,-0.0258068,'WikiTopics')
Text(-0.195016,0.0133496,'Explicitly')
Text(-0.215922,0.00205992,'Contain')
Text(-0.23511,-0.0263515,'Recalibration')
Text(-0.265612,-0.0330509,'MCFGs')
Text(-0.259133,-0.0183325,'MGs')
Text(-0.25068,-0.0148186,'Universals')
Text(-0.217418,-0.0363006,'WM')
Text(-0.192425,-0.0221684,'Load')
Text(-0.2956,-0.0432038,'Chameleons')
Text(-0.280535,-0.047482,'Imagined')
Text(-0.290048,-0.0330817,'Atypical')
Text(-0.178121,-0.0184477,'Autism')
Text(-0.298532,-0.0482591,'Colourful')
Text(-0.308144,-0.0322382,'Word-Colour')
Text(-0.281505,-0.0338133,'Survival')
Text(-0.289149,-0.0406665,'Fixation')
Text(-0.298731,-0.0348773,'CMDA')
Text(-0.265663,-0.0115763,'such')
Text(-0.281482,-0.0239854,'pushes')
Text(-0.253447,-0.0175054,'buttons')
Text(-0.228376,-0.019588,'comment')
Text(-0.260092,-0.0171598,'blog')
Text(0.0832016,0.0673209,'posts')
Text(-0.276313,-0.000768067,'SXSW')
Text(-0.271268,-0.00894248,'trending')
Text(-0.273034,-0.0169393,'reflection')
Text(-0.265335,-0.0192451,'socialization')
Text(-0.290202,-0.0208444,'communities')
Text(-0.293827,-0.0548975,'NV')
Text(-0.335479,-0.0382786,'Decreasing')
Text(-0.289283,-0.0186967,'Tree-Rewriting')
Text(-0.304291,-0.0359148,'MWU-Aware')
Text(-0.293498,-0.0143921,'PERSON')
Text(-0.296462,-0.0121732,'Berners-Lee')
Text(-0.325678,-0.00902047,'African-Americans')
Text(-0.249634,-0.02219,'Handle')
Text(-0.2924,-0.038039,'Stepwise')
Text(-0.266244,-0.0140533,'jMWE')
Text(-0.246836,-0.00686896,'FipsCoView')
Text(-0.200858,-0.0456114,'Visualisation')
Text(-0.251938,0.00432372,'StringNet')
Text(-0.267585,-0.00850119,'Knowledgebase')
Text(-0.318551,-0.02796,':NSP')
Text(-0.28084,-0.0263579,'Ngrams')
Text(-0.294688,-0.0303007,'mwetoolkit')
Text(-0.302164,-0.0241177,'Event-Sentiment')
Text(-0.323337,-0.0336537,'VigNet')
Text(-0.283996,-0.0134969,'Desperately')
Text(-0.285955,-0.0252913,'Coercive')
Text(-0.294636,-0.0523818,'Biparsing')
Text(-0.283962,-0.022825,'Multi-Stage')
Text(-0.277965,-0.0290129,'Target-Side')
Text(-0.261865,-0.0249062,'SCFG-Based')
Text(-0.342774,-0.0757774,'Max-margin')
Text(-0.277009,-0.026185,'Wikipedia-based')
Text(-0.303464,-0.0404179,'GrawlTCQ')
Text(-0.313104,-0.070297,'Feature-Weight')
Text(-0.254877,-0.00999918,'ranked')
Text(-0.207424,0.0075187,'two-stage')
Text(-0.262532,-0.0225769,'Copiale')
Text(-0.28328,-0.0183802,'Cipher')
Text(-0.269136,-0.0543196,'Dependency-parsing')
Text(-0.296381,-0.0289285,'Metasearch')
Text(-0.258568,-0.0367968,'Machine-Translated')
Text(-0.20637,0.0141399,'Pointwise')
Text(-0.335508,-0.0547039,'2011')
Text(-0.307236,-0.0312104,'Frustratingly')
Text(-0.287821,-0.0365674,'Word-Space')
Text(-0.243961,-0.0284955,'Perceived')
Text(-0.24927,-0.0257268,'Peer-Review')
Text(-0.227571,-0.0143643,'Varied')
Text(-0.2776,-0.0193133,'Exercises')
Text(-0.241418,-0.0235946,'Elicited')
Text(-0.202298,-0.00296628,'OPI')
Text(-0.261295,-0.0156488,'Comparability')
Text(-0.278641,-0.00703334,'PRESEMT')
Text(-0.221697,-0.0103435,'Recognition-based')
Text(-0.279563,-0.0396896,'PLUTO')
Text(-0.316481,-0.0304497,'ATLAS')
Text(-0.270077,-0.00554714,'clocks')
Text(-0.252442,-0.0192587,'striking')
Text(-0.267691,-0.00632649,'surprising')
Text(-0.293,-0.0155052,'Linguistically-Augmented')
Text(-0.257488,-0.0185534,'Bulgarian-to-English')
Text(-0.331535,-0.0261556,'Sense-labeled')
Text(-0.295927,-0.0286422,'visualization')
Text(-0.186123,-0.00555595,'Princeton')
Text(-0.223732,-0.00284011,'Visualising')
Text(-0.206482,0.00774891,'Plotting')
Text(-0.201225,-2.83845e-06,'WALS')
Text(-0.249831,-0.025637,'Heat')
Text(-0.276432,-0.0281908,'Academic')
Text(-0.22949,-0.0207069,'dialect')
Text(-0.207789,-0.00804202,'geography')
Text(-0.22534,-0.0119803,'unaligned')
Text(-0.303074,-0.0435422,'Shibboleths')
Text(-0.273217,-0.0212476,'visualizing')
Text(-0.207502,-0.00607169,'weighted')
Text(-0.228886,-0.0135903,'force-directed')
Text(-0.25248,-0.0205273,'layout')
Text(-0.234491,-0.00171617,'creole')
Text(-0.280394,-0.0190558,'dynamics')
Text(-0.250814,-0.0202775,'kinship')
Text(-0.27188,-0.0229497,'AustKin')
Text(-0.256847,-0.0128786,'etymological')
Text(-0.26166,-0.00980755,'sound')
Text(-0.312956,-0.0490049,'LexStat')
Text(-0.278615,-0.0452535,'Wordlists')
Text(-0.295886,-0.0296985,'Inputlog')
Text(-0.300044,-0.0181103,'Drafting')
Text(-0.284085,-0.00156749,'Guideline')
Text(-0.244348,-0.0173651,'Legislative')
Text(-0.237457,-0.00842863,'Verbnet')
Text(-0.346673,-0.0319798,'Clarifying')
Text(-0.247522,-0.00971768,'grounding')
Text(-0.304163,-0.0245475,'Sound-based')
Text(-0.288124,-0.04238,'Aggregates')
Text(-0.295976,-0.0248853,'Curse')
Text(-0.225123,-0.0124903,'Boon')
Text(-0.226323,-0.0133029,'Mood')
Text(-0.235444,-0.0183117,'Darcy')
Text(-0.221912,-0.0101569,'Toad')
Text(-0.193371,-0.00424989,'gentlemen')
Text(-0.250574,-0.0124731,'kinds')
Text(-0.303026,-0.0257201,'Feeling')
Text(-0.299101,-0.0386337,'Perceptually-Grounded')
Text(-0.269206,-0.0138154,'Rejection')
Text(-0.26026,-0.0308429,'Interval')
Text(-0.272989,-0.0379968,'Assertion')
Text(-0.28848,-0.0104413,'hard')
Text(-0.304683,-0.0500039,'Schema-agnostic')
Text(-0.246192,-0.00254538,'annotate')
Text(-0.247006,-0.0232163,'GraphAnno')
Text(-0.267341,-0.0187924,'lightweight')
Text(-0.249958,-0.0101863,'multi-level')
Text(-0.271682,-0.0271519,'Implicitation')
Text(-0.272026,-0.0298486,'ISO')
Text(-0.336724,-0.0352008,'24617-8')
Text(-0.28949,-0.0278137,'transcripts')
Text(-0.248959,-0.0124403,'hedging')
Text(-0.212305,-0.0291605,'focused')
Text(-0.227931,-0.0307728,'epistemic')
Text(-0.289636,-0.02577,'informal')
Text(-0.306367,-0.0339338,'modals')
Text(-0.186357,-0.00950084,'Cartesian')
Text(-0.205299,-0.0521089,'N-ary')
Text(-0.269319,-0.0346994,'culture')
Text(-0.281912,-0.0281951,'Trimming')
Text(-0.248726,-0.0060347,'consistent')
Text(-0.270039,0.00957363,'relying')
Text(-0.264003,-0.0108906,'Lying')
Text(-0.237172,-0.0134242,'Genetics')
Text(-0.309609,-0.0156387,'Editorials')
Text(-0.301948,-0.02677,'Toulmin')
Text(-0.288448,-0.0334412,'Claim')
Text(-0.309563,-0.0462622,'Counter-considerations')
Text(-0.283859,-0.0307671,'Prominent')
Text(-0.220914,-0.0227102,'difficulty')
Text(-0.301238,-0.0409596,'Response-to-Text')
Text(-0.231532,-0.0192582,'Picture-based')
Text(-0.255331,-0.0259056,'Narration')
Text(-0.26515,-0.0238395,'AESW')
Text(-0.244631,-0.0205163,'Lark')
Text(-0.214983,-0.00289967,'Trills')
Text(-0.164485,0.00707372,'Drills')
Text(-0.175943,0.0143469,'Text-to-speech')
Text(-0.270007,-0.0129298,'learners')
Text(-0.304436,-0.0291728,'Jinan')
Text(-0.282136,-0.0297159,'Efforts')
Text(-0.298227,-0.0169429,'Embarrassed')
Text(-0.253549,-0.0165808,'Awkward')
Text(-0.27765,-0.0241986,'Wording')
Text(-0.301747,-0.0455935,'RevUP')
Text(-0.297909,-0.0483429,'Gap-Fill')
Text(-0.303763,-0.0353365,'PEGWriting')
Text(-0.21763,-0.00755253,'Eighth-Grade')
Text(-0.253682,-0.0225991,'Quasi-Experimental')
Text(-0.263819,-0.0154875,'Pedagogic')
Text(-0.267224,-0.0426419,'Gap-fill')
Text(-0.28814,-0.0212602,'Task-Independent')
Text(-0.255524,-0.0445668,'Watson')
Text(-0.229423,-0.0137152,'Advisor')
Text(-0.201043,-0.00890474,'Question-answering')
Text(-0.34445,-0.064935,'Enquirer')
Text(-0.218373,-0.0149833,'Distorted')
Text(-0.223921,-0.00607498,'Skull')
Text(-0.218002,-0.00609916,'Lies')
Text(-0.272264,-0.0288958,'Paintings')
Text(-0.308931,-0.0418926,'Heights')
Text(-0.296068,-0.0355283,'Gender-Distinguishing')
Text(-0.308831,-0.0185077,'Manuscripts')
Text(-0.317722,-0.0453564,'Chiasmus')
Text(-0.294902,-0.0135699,'Hafez')
Text(-0.244319,-0.026267,'late-life')
Text(-0.230431,-0.0244535,'depression')
Text(-0.279508,-0.0343614,'Dementia')
Text(-0.330447,-0.0399131,'Self-Reflective')
Text(-0.335352,-0.0442037,'Movies')
Text(-0.275192,-0.0406565,'Psychotherapy')
Text(-0.252679,-0.0440136,'Crazy')
Text(-0.229672,-0.0394942,'Nutters')
Text(0.0378852,0.0291896,'mental')
Text(0.06296,0.0570021,'health')
Text(-0.207266,-0.00751962,'Spectrum')
Text(-0.233078,-0.0177865,'Disorders')
Text(-0.284394,-0.0304024,'Clinically')
Text(-0.240141,-0.0305516,'Life-Changing')
Text(-0.176225,-0.0126785,'Misunderstood')
Text(-0.132883,-0.0343143,'Suicide')
Text(0.145918,0.0571467,'CLPsych')
Text(-0.151194,-0.0196397,'Triaging')
Text(-0.263581,-0.017996,'peer-support')
Text(-0.371754,-0.0278384,'Data61-CSIRO')
Text(-0.232032,-0.00741757,'Severity')
Text(-0.269091,-0.0129052,'ReachOut')
Text(-0.212856,0.0117472,'radial')
Text(-0.276693,-0.0164845,'Distress')
Text(-0.267419,-0.0167487,'LT3')
Text(-0.275645,-0.0351594,'GW/UMD')
Text(-0.240092,-0.0318305,'ReachOut.com')
Text(-0.0980888,0.00862419,'forum')
Text(-0.162763,-0.00722649,'emergencies')
Text(-0.319261,-0.0256963,'talking')
Text(-0.319155,-0.0206434,'ToWork')
Text(-0.321131,-0.0299709,'Rumor')
Text(-0.242698,0.00282123,'Arousal')
Text(-0.244233,-0.000142064,'Facebook')
Text(-0.285317,-0.0231458,'Purity')
Text(-0.290475,-0.0195859,'Homophily')
Text(-0.339663,-0.0412495,'Odawa')
Text(-0.262634,-0.0324167,'power')
Text(-0.259068,-0.0148066,'times')
Text(-0.209901,-0.0280196,'uncertainty')
Text(-0.22655,-0.0322846,'gradient')
Text(-0.266176,-0.00347271,'certainty')
Text(-0.266917,0.000790596,'weights')
Text(-0.283039,-0.0169902,'activation')
Text(-0.243408,-0.0135573,'connected')
Text(-0.21886,0.0143098,'learn')
Text(-0.182625,-0.00203618,'continuity')
Text(-0.179043,-0.015005,'SOM-based')
Text(-0.231174,-0.0115131,'harvester')
Text(-0.273027,-0.0295863,'Sound-aligned')
Text(-0.198994,-0.014808,'Udmurt')
Text(-0.207132,-0.00808143,'dialectal')
Text(-0.253685,-0.0267831,'Finno-Ugric')
Text(-0.227613,-0.0348178,'entries')
Text(-0.237143,-0.0298362,'Uralic')






