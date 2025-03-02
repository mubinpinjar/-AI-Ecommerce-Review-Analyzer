print("AI-Powered E-commerce Review Analyzer is ready!")
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

print("All libraries are working correctly!")
import pandas as pd

# Load the dataset from reviews.csv
df = pd.read_csv("reviews.csv")

# Display the first few rows
print(df.head())
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER sentiment analysis tool (only needed once)
nltk.download('vader_lexicon')

# Load the dataset
df = pd.read_csv("reviews.csv")

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to analyze sentiment
def get_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df["Sentiment"] = df["Review Text"].apply(get_sentiment)

# Print results
print(df[["Review Text", "Sentiment"]])
import matplotlib.pyplot as plt
import seaborn as sns

# Count the number of each sentiment type
sentiment_counts = df["Sentiment"].value_counts()

# Plot the sentiment distribution
plt.figure(figsize=(6, 4))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="coolwarm")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.title("Sentiment Distribution of Product Reviews")
plt.show()
# Save the results to a new CSV file
df.to_csv("sentiment_analysis_results.csv", index=False)
print("Sentiment analysis results saved to sentiment_analysis_results.csv")


