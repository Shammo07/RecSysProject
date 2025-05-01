from flask import (
    Blueprint, render_template, request, jsonify
)

from .tools.data_tool import *  

from surprise import Reader
from surprise import KNNWithMeans
from surprise import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict

bp = Blueprint('main', __name__, url_prefix='/')

books, users, rates = loadData()

@bp.route('/', methods=('GET', 'POST'))
def index():
    return render_template('index.html')

@bp.route('/api/initial-suggestions', methods=['POST'])
def get_initial_suggestions():
    data = request.json
    age = int(data['age'])
    location = data['location']

    # Compute location similarity
    location_similarities = compute_location_similarity(location, users['Location'])

    # Compute age similarity: 1.0 for same age, linearly decreasing to 0.0 at 10+ years difference
    age_diffs = abs(users['Age'] - age).clip(upper=10)
    age_similarities = 1 - (age_diffs / 10)

    # Combine age and location similarity with equal weights
    combined_similarity = 0.5 * age_similarities + 0.5 * location_similarities

    # Select top 10% most similar users
    threshold = combined_similarity.quantile(0.90)
    similar_users = users[combined_similarity >= threshold]
    print(len(similar_users))

    # Get top-rated books from similar users
    if len(similar_users) > 0:
        top_books = (
            rates[rates['userId'].isin(similar_users['User-ID'])]
            .groupby('itemId')['rating']
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        selected_books = books[books['itemId'].isin(top_books.index)]
    else:
        selected_books = pd.DataFrame()

    # Fill with random books if fewer than 10
    if len(selected_books) < 10:
        additional_books = books[~books['itemId'].isin(selected_books['itemId'])].sample(n=10 - len(selected_books))
        selected_books = pd.concat([selected_books, additional_books])

    return jsonify(selected_books.to_dict('records'))

def preprocess_location(location):
    return location.replace(',', '')  # Remove commas, keep whitespace and casing

def compute_location_similarity(input_location, user_locations):
    # Apply preprocessing to both input and user locations
    processed_user_locations = user_locations.apply(preprocess_location)
    input_processed = preprocess_location(input_location)

    # Combine input location with user locations for vectorization
    all_locations = pd.Series([input_processed])._append(processed_user_locations, ignore_index=True)

    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_locations)

    # Compute cosine similarity between input and all users
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return similarities

@bp.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    user_rates = request.json['ratings']
    recommendations = collaborativeFilteringRatings(user_rates)
    return jsonify(recommendations)


def collaborativeFilteringRatings(user_rates):
    results = []
    if len(user_rates) > 0:
        # Initialize a reader with rating scale from 1 to 10
        reader = Reader(rating_scale=(1, 10))
        # Define the algorithm
        algo = KNNWithMeans(sim_options={'name': 'pearson', 'user_based': True})
        # Convert the user's ratings (stored in "user_rates") to the Dataset format
        user_rates = ratesFromUser(user_rates)
        # Add the user’s rating information into the Movielens dataset
        training_rates = pd.concat([rates, user_rates], ignore_index=True)
        # Load the combined data as a training dataset 
        training_data = Dataset.load_from_df(training_rates, reader=reader)
        # Build a full training set from the dataset
        trainset = training_data.build_full_trainset()
        # Fit the algorithm using the trainset
        algo.fit(trainset)
        all_item_ids = books['itemId'].unique()
        # Predict ratings for all movies for the specified user (assuming user ID 278859)
        user_id = 278859 
        rated_item_ids = user_rates[user_rates['userId'] == user_id]['itemId'].tolist()
        predictions = [algo.predict(user_id, item_id) for item_id in all_item_ids if item_id not in rated_item_ids]
        top_predictions = [pred for pred in predictions]
        # sort predicted ratings in a descending order
        top_predictions.sort(key=lambda x: x.est, reverse=True)
        # Select the top-K items (e.g., 18)
        top_item_ids = [pred.iid for pred in top_predictions[:18]]
        results = books[books['itemId'].isin(top_item_ids)]

        return results.to_dict('records')
    

@bp.route('/api/similar-books', methods=['POST'])
def get_similar_books():
    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=10000,  # Limit vocabulary size
            ngram_range=(1, 2)  # Include unigrams and bigrams
        )
        
        books['Book-Description'] = books['Book-Description'].fillna("").astype(str)
    
        # Fit and transform all descriptions
        tfidf_matrix = vectorizer.fit_transform(books['Book-Description'])
        
        data = request.json
        book_description = data.get('bookDescription', '')
        
        if not book_description:
            return jsonify({"error": "No book description provided"}), 400
        
        # Transform input description to TF-IDF
        input_tfidf = vectorizer.transform([book_description])
        
        # Calculate similarities
        similarities = cosine_similarity(
            input_tfidf,
            tfidf_matrix
        ).flatten()
        
        # Get indices of top 3 most similar books
        top_indices = np.argsort(similarities)[-4:-1][::-1]  # Get 3rd to 5th most similar
        
        # Prepare results
        similar_books = []
        for idx in top_indices:
            book = books.iloc[idx]
            similar_books.append({
                'itemId': book['itemId'],
                'Book-Title': book['Book-Title'],
                'Book-Author': book['Book-Author'],
                'Image-URL-M': book.get('Image-URL-M', ''),
                'Book-Description': book['Book-Description'],
                'similarity_score': float(similarities[idx])
            })
        
        return jsonify(similar_books)
    
    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500
    
# @bp.route('/api/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         # Debug: Log incoming request data
#         print("Request JSON:", request.json)

#         # 1. Get user's actual ratings
#         actual_ratings = {int(r['itemId']): r['actualRating'] for r in request.json['ratings']}
#         print("Actual Ratings:", actual_ratings)

#         # 2. Get system's predicted ratings for these books
#         predicted_ratings = get_predictions_for_items(actual_ratings.keys())
#         print("Predicted Ratings:", predicted_ratings)

#         # 3. Calculate RMSE and MAE
#         errors = []
#         for item_id, actual in actual_ratings.items():
#             predicted = predicted_ratings.get(item_id)
#             if predicted is not None:  # Only compare if prediction exists
#                 errors.append({
#                     'itemId': item_id,
#                     'actual': actual,
#                     'predicted': predicted,
#                     'squared_error': (predicted - actual) ** 2,
#                     'absolute_error': abs(predicted - actual)
#                 })

#         if not errors:
#             return jsonify({'error': 'No valid predictions available for evaluation'}), 400

#         # 4. Compute metrics
#         rmse = (sum(e['squared_error'] for e in errors) / len(errors)) ** 0.5
#         mae = sum(e['absolute_error'] for e in errors) / len(errors)

#         return jsonify({
#             'rmse': round(rmse, 2),
#             'mae': round(mae, 2),
#             'details': errors  # For debugging or detailed results
#         })
#     except Exception as e:
#         return jsonify({'error': f'Internal server error: {str(e)}'}), 500




    

@bp.route('/api/evaluate', methods=['POST'])
def evaluate():
    try:
        print("Request JSON:", request.json)
        actual_ratings = {int(r['itemId']): r['actualRating'] for r in request.json['ratings']}
        print("Actual Ratings:", actual_ratings)

        # Get predictions using both methods
        knn_predictions = get_predictions_for_items(actual_ratings.keys(), request.json.get('likedBooks', []))
        hybrid_predictions = get_hybrid_predictions(actual_ratings.keys(), request.json.get('likedBooks', []))

        # Calculate metrics for both methods
        knn_errors = calculate_errors(actual_ratings, knn_predictions)
        hybrid_errors = calculate_errors(actual_ratings, hybrid_predictions)

        if not knn_errors or not hybrid_errors:
            return jsonify({'error': 'No valid predictions available for evaluation'}), 400

        return jsonify({
            'knn': calculate_metrics(knn_errors),
            'hybrid': calculate_metrics(hybrid_errors)
        })

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500
    
def get_predictions_for_items(item_ids, liked_books=None):
    """
    Get predicted ratings for specific items using the KNN model.
    """
    try:
        # Extract initial ratings from the request JSON
        initial_ratings_raw = request.json.get('initialRatings', {}).get('_rawValue', {})
        initial_ratings = {int(itemId): rating for itemId, rating in initial_ratings_raw.items()}
        print("Initial Ratings:", initial_ratings)

        # Convert books['itemId'] to integers, handling cases where 'X' is present
        def convert_item_id(item_id):
            if isinstance(item_id, str) and item_id.endswith('X'):
                return int(item_id[:-1])  # Remove 'X' and convert to integer
            return int(item_id)  # Convert directly to integer if no 'X'

        books['itemId'] = books['itemId'].apply(convert_item_id)

        # Convert item_ids to integers to match the books['itemId'] type
        item_ids = [int(item_id) for item_id in item_ids]

        # Debug: Log item IDs and their data types
        print("Item IDs:", item_ids)
        print("Data type of item_ids:", type(item_ids[0]) if item_ids else "Empty list")

        # Debug: Log books['itemId'] and its data type
        print("Books Item IDs:", books['itemId'].values)
        print("Data type of books['itemId']:", type(books['itemId'].iloc[0]) if not books.empty else "Empty DataFrame")

        # Initialize the reader and algorithm with the same parameters as recommendations
        reader = Reader(rating_scale=(1, 10))
        algo = KNNWithMeans(sim_options={'name': 'pearson', 'user_based': True})

        # Use existing ratings data for training
        training_rates = rates.copy()

        # Add initial ratings to the training data
        if initial_ratings:
            initial_training_data = pd.DataFrame([{
                'userId': 278859,  # Dummy user ID for initial ratings
                'itemId': itemId,
                'rating': rating
            } for itemId, rating in initial_ratings.items()])
            training_rates = pd.concat([training_rates, initial_training_data], ignore_index=True)

        # Add liked books to the training data (if provided)
        if liked_books:
            liked_training_data = pd.DataFrame([{
                'userId': 278859,  # Same dummy user ID
                'itemId': convert_item_id(book['itemId']),
                'rating': 8.0  # Assign a realistic rating for liked books
            } for book in liked_books])
            training_rates = pd.concat([training_rates, liked_training_data], ignore_index=True)

        # Ensure training_rates['itemId'] is also an integer
        training_rates['itemId'] = training_rates['itemId'].apply(convert_item_id)

        # Load training data
        training_data = Dataset.load_from_df(training_rates[['userId', 'itemId', 'rating']], reader=reader)
        trainset = training_data.build_full_trainset()
        algo.fit(trainset)

        # Get predictions for all requested items including liked books
        predictions = {}
        user_id = 278859
        for item_id in item_ids:
            if item_id in books['itemId'].values:
                pred = algo.predict(user_id, item_id)
                predictions[item_id] = pred.est

        print("KNN Predictions:", predictions)
        return predictions

    except Exception as e:
        print(f"Error in get_predictions_for_items: {str(e)}")
        return {}

def calculate_errors(actual_ratings, predictions):
    errors = []
    for item_id, actual in actual_ratings.items():
        predicted = predictions.get(item_id)
        if predicted is not None:
            errors.append({
                'itemId': item_id,
                'actual': actual,
                'predicted': predicted,
                'squared_error': (predicted - actual) ** 2,
                'absolute_error': abs(predicted - actual)
            })
    return errors

def calculate_metrics(errors):
    rmse = (sum(e['squared_error'] for e in errors) / len(errors)) ** 0.5
    mae = sum(e['absolute_error'] for e in errors) / len(errors)
    return {
        'rmse': round(rmse, 2),
        'mae': round(mae, 2),
        'details': errors
    }

def get_hybrid_predictions(item_ids, liked_books=None):
    """
    Get predictions using a hybrid approach (KNN + Content-Based)
    """
    try:
        # Get KNN predictions
        knn_predictions = get_predictions_for_items(item_ids, liked_books)
        
        # Get Content-Based predictions
        cb_predictions = get_content_based_predictions(item_ids, liked_books)
        
        # Combine predictions (simple average)
        hybrid_predictions = {}
        for item_id in item_ids:
            knn_pred = knn_predictions.get(item_id, 5.0)
            cb_pred = cb_predictions.get(item_id, 5.0)
            hybrid_predictions[item_id] = (0.7* knn_pred) + (0.3 * cb_pred)
        print("\nFinal Hybrid Predictions:", hybrid_predictions)

        return hybrid_predictions
        

    except Exception as e:
        print(f"Error in get_hybrid_predictions: {str(e)}")
        return {}

def get_content_based_predictions(item_ids, liked_books=None):
    """
    Get predictions using TF-IDF and cosine similarity
    """
    try:
        # Initialize TF-IDF vectorizer
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))

        # Combine title, description, and author for better features
        book_descriptions = books[['itemId', 'Book-Title', 'Book-Description', 'Book-Author']].copy()
        book_descriptions['text'] = (
            book_descriptions['Book-Title'].fillna('') + ' ' +
            book_descriptions['Book-Description'].fillna('') + ' ' +
            book_descriptions['Book-Author'].fillna('')
        )

        # Create TF-IDF matrix
        tfidf_matrix = tfidf.fit_transform(book_descriptions['text'])
        cosine_sim = cosine_similarity(tfidf_matrix)

        # Define rating scale and similarity bounds
        min_rating = 1
        max_rating = 10
        min_similarity = 0  # Cosine similarity minimum
        max_similarity = 1  # Cosine similarity maximum

        # Convert books['itemId'] to integers, handling cases where 'X' is present
        def convert_item_id(item_id):
            if isinstance(item_id, str) and item_id.endswith('X'):
                return int(item_id[:-1])  # Remove 'X' and convert to integer
            return int(item_id)  # Convert directly to integer if no 'X'

        book_descriptions['itemId'] = book_descriptions['itemId'].apply(convert_item_id)

        predictions = {}
        for item_id in item_ids:
            if liked_books:
                # Get average similarity with liked books
                similarities = []
                for liked_book in liked_books:
                    liked_idx = book_descriptions[book_descriptions['itemId'] == convert_item_id(liked_book['itemId'])].index
                    target_idx = book_descriptions[book_descriptions['itemId'] == convert_item_id(item_id)].index

                    # Check if both indices are valid
                    if not liked_idx.empty and not target_idx.empty:
                        similarities.append(cosine_sim[liked_idx[0], target_idx[0]])

                # If no valid similarities, assign default rating
                if similarities:
                    avg_similarity = np.mean(similarities)
                    # Apply the formula to calculate predicted rating
                    predicted_rating = min_rating + ((avg_similarity - min_similarity) / (max_similarity - min_similarity)) * (max_rating - min_rating)
                    predictions[item_id] = predicted_rating
                else:
                    predictions[item_id] = 5.0  # Default rating
            else:
                predictions[item_id] = 5.0  # Default rating

        print("\nFinal Content-Based Predictions:", predictions)
        return predictions

    except Exception as e:
        print(f"Error in get_content_based_predictions: {str(e)}")
        return {}
    
@bp.route('/api/liked-books', methods=['POST'])
def get_liked_book():
    try:
        data = request.json
        item_id = data.get('itemId', None)

        if item_id is None:
            return jsonify({'error': 'No itemId provided'}), 400

        # Find the book with the given itemId
        liked_book = books[books['itemId'] == item_id]

        if liked_book.empty:
            return jsonify({'error': 'Book not found'}), 404

        # Convert to dictionary and return
        return jsonify(liked_book.iloc[0].to_dict())

    except Exception as e:
        return jsonify({'error': 'Internal server error'}), 500
    

# Add new route for search:

@bp.route('/api/search-books', methods=['POST'])
def search_books():
    try:
        query = request.json.get('query', '').lower()
        if not query:
            return jsonify([])
            
        results = books[
            (books['Book-Title'].str.lower().str.contains(query)) |
            (books['Book-Author'].str.lower().str.contains(query))
        ].head(5).to_dict('records')
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
