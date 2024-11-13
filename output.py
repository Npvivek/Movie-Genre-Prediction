# Importing essential libraries
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import joblib

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('wordnet')

# Loading the 'movie_metadata.tsv' dataset
movie_metadata = pd.read_csv('movie_metadata.tsv', sep='\t', header=None)

# Loading the 'plot_summaries.tsv' dataset
plot_summary = pd.read_csv('plot_summaries.tsv', sep='\t', header=None)

movie_metadata.head()

movie_metadata.shape

movie_metadata.dtypes

plot_summary.head()

plot_summary.shape

plot_summary.dtypes

# Renaming the required columns
movie_metadata.columns = ["movie_id", 1, "movie_name", 3, 4, 5, 6, 7, "genre"]

# Renaming the required columns
plot_summary.columns = ["movie_id", "plot"]

# Merging both dataframes
df = pd.merge(movie_metadata[['movie_id', 'movie_name', 'genre']], plot_summary, on='movie_id')

df.shape

df.head()

# Cleaning the genre column
df['genre'][0]

json.loads(df['genre'][0]).values()

df['genre'] = df['genre'].apply(lambda x: list(json.loads(x).values()))
df.head()

# Remove rows with 0 genre tags
print("Before removing the rows: {}".format(df.shape))
df = df[df['genre'].apply(lambda x: False if len(x)==0 else True)]
print("After removing the rows: {}".format(df.shape))

# Flatten the list of genres
temp = [genre for sublist in df['genre'] for genre in sublist]

# Calculate unique genres
all_genre = set(temp)
print('Total number of unique genres are: {}'.format(len(all_genre)))

# Using FreqDist to calculate the frequency of all the genres in the dataset
from nltk.probability import FreqDist
temp_with_count = FreqDist(temp)

# Creating a dataframe of genre_count
df_genre_count = pd.DataFrame({'Genre': list(temp_with_count.keys()),
                               'Count': list(temp_with_count.values())})

# Plotting the top 50 genres
genre_top_50 = df_genre_count.sort_values(by=['Count'], ascending=False).iloc[0:50, :]
plt.figure(figsize=(7,15))
sns.barplot(x='Count', y='Genre', data=genre_top_50)
plt.xlabel('Frequency Count')
plt.ylabel('Genres')

def plot_clean(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers
    text = ' '.join(text.split())
    return text

lemmatizer = WordNetLemmatizer()

def plot_lemmatization(text):
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha()]
    text = ' '.join(words)
    return text

# Convert Genre into Target variables using MultiLabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genre'])

# Use the raw 'plot' data and apply all preprocessing in the pipeline
X = df['plot']

# Update train_test_split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

# Update LogisticRegression in the pipeline
pipeline = Pipeline([
    ('cleaning', FunctionTransformer(lambda x: [plot_clean(text) for text in x], validate=False)),
    ('lemmatization', FunctionTransformer(lambda x: [plot_lemmatization(text) for text in x], validate=False)),
    ('tfidf', TfidfVectorizer(
        max_df=0.8,
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english'
    )),
    ('clf', OneVsRestClassifier(LogisticRegression(random_state=42)))
])
# Fit the pipeline on the raw training data
pipeline.fit(X_train_raw, y_train)

# Predict on the raw test data
y_pred_ovr = pipeline.predict(X_test_raw)


# Evaluate using the default threshold
print("F1 Score with default threshold:")
print(f1_score(y_test, y_pred_ovr, average="micro"))
print("\nClassification Report with default threshold:")
print(classification_report(y_test, y_pred_ovr, target_names=mlb.classes_))

# Transform X_test_raw up to the TF-IDF step
X_test_transformed = pipeline.named_steps['tfidf'].transform(
    pipeline.named_steps['lemmatization'].transform(
        pipeline.named_steps['cleaning'].transform(X_test_raw)
    )
)

# Get predicted probabilities for each class
y_pred_prob = np.array([
    estimator.predict_proba(X_test_transformed)[:, 1]
    for estimator in pipeline.named_steps['clf'].estimators_
]).T

# Adjust the threshold
threshold = 0.2  # Adjust as needed
y_pred_new = (y_pred_prob >= threshold).astype(int)

# Evaluate the model with the new threshold
print(f"\nF1 Score with threshold {threshold}:")
print(f1_score(y_test, y_pred_new, average="micro"))
print("\nClassification Report with adjusted threshold:")
print(classification_report(y_test, y_pred_new, target_names=mlb.classes_))


def predict_genre_tags(text):
    # Use the pipeline directly
    text_pred = pipeline.predict([text])
    return mlb.inverse_transform(text_pred)

# Prediction 1
movie_name = 'Titanic (1998)'
plot_summary = "After winning a trip on the RMS Titanic during a dockside card game, American Jack Dawson spots the society girl Rose DeWitt Bukater who is on her way to Philadelphia to marry her rich snob fiancé Caledon Hockley. Rose feels helplessly trapped by her situation and makes her way to the aft deck and thinks of suicide until she is rescued by Jack. Cal is therefore obliged to invite Jack to dine at their first-class table where he suffers through the slights of his snobbish hosts. In return, he spirits Rose off to third-class for an evening of dancing, giving her the time of her life. Deciding to forsake her intended future all together, Rose asks Jack, who has made his living making sketches on the streets of Paris, to draw her in the nude wearing the invaluable blue diamond Cal has given her. Cal finds out and has Jack locked away. Soon afterwards, the ship hits an iceberg and Rose must find Jack while both must run from Cal even as the ship sinks deeper into the freezing water."
actual_genre = ['Drama', 'Romance']
predicted_genre = predict_genre_tags(plot_summary)
print('Movie: {}\nPredicted genres: {}\nActual genres: {}'.format(movie_name, predicted_genre[0], actual_genre))

# Prediction 2
movie_name = 'Avatar (2009)'
plot_summary = "On the lush alien world of Pandora live the Na'vi, beings who appear primitive but are highly evolved. Because the planet's environment is poisonous, human/Na'vi hybrids, called Avatars, must link to human minds to allow for free movement on Pandora. Jake Sully (Sam Worthington), a paralyzed former Marine, becomes mobile again through one such Avatar and falls in love with a Na'vi woman (Zoe Saldana). As a bond with her grows, he is drawn into a battle for the survival of her world."
actual_genre = ['Action', 'Adventure', 'Fantasy']
predicted_genre = predict_genre_tags(plot_summary)
print('Movie: {}\nPredicted genres: {}\nActual genres: {}'.format(movie_name, predicted_genre[0], actual_genre))

# Prediction 3
movie_name = 'Conjuring (2013)'
plot_summary = "In 1971, Carolyn and Roger Perron move their family into a dilapidated Rhode Island farm house and soon strange things start happening around it with escalating nightmarish terror. In desperation, Carolyn contacts the noted paranormal investigators, Ed and Lorraine Warren, to examine the house. What the Warrens discover is a whole area steeped in a satanic haunting that is now targeting the Perron family wherever they go. To stop this evil, the Warrens will have to call upon all their skills and spiritual strength to defeat this spectral menace at its source that threatens to destroy everyone involved."
actual_genre = ['Horror', 'Thriller']
predicted_genre = predict_genre_tags(plot_summary)
print('Movie: {}\nPredicted genres: {}\nActual genres: {}'.format(movie_name, predicted_genre[0], actual_genre))

# Prediction 4
movie_name = 'The Hangover (2009)'
plot_summary = "Three buddies wake up from a bachelor party in Las Vegas, with no memory of the previous night and the bachelor missing. They make their way around the city in order to find their friend before his wedding."
actual_genre = ['Comedy']
predicted_genre = predict_genre_tags(plot_summary)
print('Movie: {}\nPredicted genres: {}\nActual genres: {}'.format(movie_name, predicted_genre[0], actual_genre))

# Prediction 5
movie_name = 'La La Land (2016)'
plot_summary = "The story of aspiring actress Mia and dedicated jazz musician Sebastian, who struggle to make ends meet while pursuing their dreams in a city known for destroying hopes and breaking hearts. With modern-day Los Angeles as the backdrop, this musical about everyday life explores what more important: a once-in-a-lifetime love or the spotlight."
actual_genre = ['Comedy', 'Drama', 'Music']
predicted_genre = predict_genre_tags(plot_summary)
print('Movie: {}\nPredicted genres: {}\nActual genres: {}'.format(movie_name, predicted_genre[0], actual_genre))



