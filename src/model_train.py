import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Load processed data
df = pd.read_csv('data/processed/combined_data.csv')
df = df.dropna() # Safety check

x_train, x_test, y_train, y_test = train_test_split(df['content'], df['label'], test_size=0.2, random_state=42)

# Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# Model Training
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Evaluation
y_pred = pac.predict(tfidf_test)
print(f'Model Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%')

# Save Model and Vectorizer
pickle.dump(pac, open('models/pac_model.pkl', 'wb'))
pickle.dump(tfidf_vectorizer, open('models/tfidf_vectorizer.pkl', 'wb'))
print("Model and Vectorizer saved in models/ folder.")