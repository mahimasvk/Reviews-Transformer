!pip install -q requests pandas matplotlib scikit-learn transformers torch torchvision

import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error
from transformers import pipeline

# choosing professors that I had
profs = [
    "Nora Burkhauser",
    "Larry Herman",
    "Maksym Morawski", 
    "Clyde Kruskal",
    "Nelson Padua-Perez" 
]

APIURL = "https://planetterp.com/api/v1/"
def getReviews(name):
    url = APIURL + "professor"

    params = {
        "name": name,
        "reviews": "true"
    }

    response = requests.get(url, params = params)

 # if the request fails, return an empty list
    if response.status_code != 200:
        print("Cant get", name, response.status_code)
        return []

    data = response.json()

    # if the professor has no reviews, return an empty list
    if "reviews" not in data:
        print("No reviews for:", name)
        return []

    rows = []

    # go through each review and save the important parts
    for review in data["reviews"]:
        rows.append({
            "professor": name,
            "review": review.get("review", ""),
            "stars": review.get("rating", None),
            "course": review.get("course", None)
        })

    return rows

# searches through professor names
def searchProf(name):
    url = APIURL + "professors"

    params = {
        "name": name,
        "limit": 10
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print("Search failed for", name, response.status_code)
        return []

    data = response.json()

    for prof in data:
        print(prof.get("name"))

# this stores reviews from all professors
reviews = []

# get reviews for each professor and add them to the big list
for prof in profs:
    profreviews = getReviews(prof)
    reviews.extend(profreviews)

# turn the list into a dataframe
df = pd.DataFrame(reviews)

# remove rows with missing reviews or missing stars
df = df.dropna(subset=["review", "stars"])

# remove blank reviews
df = df[df["review"].str.strip() != ""]

# make stars whole numbers
df["stars"] = df["stars"].astype(int)

# check how many reviews we have
print("Number of reviews:", len(df))

# install transformer
!pip install transformers torch -q
from transformers import pipeline

# model predicts labels like "1 star", "2 stars", etc.
model = pipeline(
    "text-classification",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# define predict stars
def predictStars(text):
    result = model(text[:512])[0]
    label = result["label"]
    return int(label[0])

df["predicted"] = df["review"].apply(predictStars)

# import sklearn metrics for accuracy and mae
from sklearn.metrics import accuracy_score, mean_absolute_error

# create dataframes for correct and mean error
df["correct"] = df["stars"] == df["predicted"]
df["error"] = abs(df["stars"] - df["predicted"])

# score accuracy and mean absolute error 
# between actual and predicted
accuracy = accuracy_score(df["stars"], df["predicted"])
mae = mean_absolute_error(df["stars"], df["predicted"])
offbyone = (df["error"] <= 1).mean()

# print results
print("Exact accuracy:", accuracy)
print("Mean absolute error:", mae)
print("Off by one accuracy:", offbyone)

# create summary chart
summary = df.groupby("professor").agg(
    reviews=("review", "count"),
    actualavg=("stars", "mean"),
    predictedavg=("predicted", "mean"),
    accuracy=("correct", "mean"),
    avgerror=("error", "mean")
).reset_index()

# display summary chart !!
summary
