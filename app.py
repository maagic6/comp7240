import os
import gc
from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import tensorflow as tf

# import recommender class
try:
    from fusion_predict import AdvancedMusicRecommender
    RECOMMENDER_AVAILABLE = True
except ImportError as e:
    print(f"Could not import AdvancedMusicRecommender from Fusion_predict.py: {e}")
    AdvancedMusicRecommender = None
    RECOMMENDER_AVAILABLE = False
except Exception as e:
     print(f"An unexpected error occurred during import: {e}")
     AdvancedMusicRecommender = None
     RECOMMENDER_AVAILABLE = False


# flask app setup
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))

# initialize recommender globally
recommender = None
if RECOMMENDER_AVAILABLE and AdvancedMusicRecommender:
    print("Initializing music recommender")
    recommender = AdvancedMusicRecommender()
    if recommender.model is None:
        print("Recommender initialization failed (check logs above)")
        recommender = None
else:
     print("Flask app running without recommender initialization")

# routes

@app.route('/', methods=['GET'])
def index():
    """Displays the initial search page"""
    # clear previous results from session
    session.pop('search_results', None)
    session.pop('search_query', None)
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handles the song/artist search query"""
    if recommender is None:
        return render_template('index.html', error="Sorry, the recommendation model is currently unavailable")

    query = request.form.get('query', '').strip() # get query, default to empty string, strip whitespace
    if not query:
        return redirect(url_for('index'))

    # split query by comma, search for each non-empty term
    search_terms = [term.strip() for term in query.split(',') if term.strip()]
    if not search_terms:
         return redirect(url_for('index'))

    all_results = {}
    found_any = False

    print(f"Flask: Searching for terms: {search_terms}")
    for term in search_terms:
         results_df = recommender.search_songs(term, top_k=5) # limit results per term
         if not results_df.empty:
              # convert df to list of dicts
              all_results[term] = results_df.to_dict('records')
              found_any = True
         else:
              all_results[term] = []

    # store results and original query in memory
    session['search_results'] = all_results
    session['search_query'] = query

    # clear tf session memory
    tf.keras.backend.clear_session()
    gc.collect()

    if not found_any:
         return render_template('search_results.html', results=all_results, query=query, error="No matching songs found for your query.")
    else:
         return render_template('search_results.html', results=all_results, query=query)


@app.route('/recommend', methods=['POST'])
def recommend():
    """Generates recommendations based on selected songs"""
    if recommender is None:
        return render_template('index.html', error="Sorry, the recommendation model is currently unavailable")

    selected_song_ids = request.form.getlist('selected_songs')
    original_search_results = session.get('search_results')
    original_query = session.get('search_query')

    if not selected_song_ids:
        if original_search_results:
             return render_template('search_results.html',
                                    results=original_search_results,
                                    query=original_query,
                                    error="Please select at least one song to get recommendations")
        else:
             return redirect(url_for('index'))


    print(f"Flask: Selected song IDs for recommendation: {selected_song_ids}")

    try:
        print("Flask: Calling recommender.get_recommendations_from_ids...")
        recommendations_df = recommender.get_recommendations_from_ids(selected_song_ids, top_n=10)
        print(f"Flask: Received {len(recommendations_df)} recommendations")

        # clear tf session memory after rec
        tf.keras.backend.clear_session()
        gc.collect()

        if recommendations_df.empty:
             return render_template('recommendations.html',
                                    recommendations=None,
                                    error="Could not generate recommendations based on your selection")

        recommendations_list = recommendations_df.to_dict('records')
        return render_template('recommendations.html', recommendations=recommendations_list)

    except ValueError as e: # catch errors by recommener
         print(f"Flask: Value error during recommendation: {e}")
         if original_search_results:
              return render_template('search_results.html',
                                     results=original_search_results,
                                     query=original_query,
                                     error=f"Recommendation error: {e}")
         else:
              return redirect(url_for('index')) # fallback

    except RuntimeError as e: # catch recommender not initialized errors
          print(f"Flask: Runtime Error during recommendation: {e}")
          return render_template('index.html', error="Sorry, the recommendation engine encountered an internal state error")

    except Exception as e:
        # catch any other errors
        print(f"Flask: An unexpected error occurred during recommendation: {e}")
        import traceback
        traceback.print_exc()

        # generic error message on the index page
        return render_template('index.html', error="An unexpected error occurred while generating recommendations. Please try again later")


if __name__ == '__main__':
    # check if recommender loaded successfully
    if recommender is None and RECOMMENDER_AVAILABLE:
         print("Recommender failed to initialize during startup")
    elif not RECOMMENDER_AVAILABLE:
         print("Could not import Recommender class")


    # set debug=True for development
    # set debug=False for production/sharing
    app.run(debug=True, host='127.0.0.1', port=5000)