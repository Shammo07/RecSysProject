import os
import pandas as pd


def loadData():
    return getBooks(), getUsers(), getRates()


# itemId,Book-Title,Book-Author,Year-Of-Publication,Publisher,Image-URL-S,Image-URL-M,Image-URL-L,Book-Description
def getBooks():
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/book_info.csv"
    df = pd.read_csv(path)
    df = df.drop(columns='Image-URL-S')
    df = df.drop(columns='Image-URL-L')
    df = df[['itemId', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-M', 'Book-Description']]
    return df


# user id, location, age
def getUsers():
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/Users.csv"
    df = pd.read_csv(path)
    return df


# user id, item id, rating
def getRates():
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/book_data.csv"
    df = pd.read_csv(path, delimiter=",", names=["userId", "itemId", "rating"])
    df = df[['userId', 'itemId', 'rating']]

    return df

def ratesFromUser(rates):
    itemID = []
    userID = []
    rating = []

    for rate in rates:
        userID.append(rate['userId'])
        itemID.append(rate['itemId'])
        rating.append(rate['rating'])

    ratings_dict = {
        "userId": userID,
        "itemId": itemID,
        "rating": rating,
    }

    return pd.DataFrame(ratings_dict)