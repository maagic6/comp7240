import os
import gc
from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import tensorflow as tf
import traceback

# import recommender class
try:
    from predict import AdvancedMusicRecommender
    RECOMMENDER_AVAILABLE = True
except ImportError as e:
    print(f"Could not import AdvancedMusicRecommender from predict.py: {e}")
    AdvancedMusicRecommender = None
    RECOMMENDER_AVAILABLE = False
except Exception as e:
     print(f"An unexpected error occurred during import: {e}")
     AdvancedMusicRecommender = None
     RECOMMENDER_AVAILABLE = False


# setup
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))

# global recommender initialization
recommender = None
if RECOMMENDER_AVAILABLE and AdvancedMusicRecommender:
    print("Initializing music recommender for Flask")
    # options: 'fusion', 'gmf', 'mlp'
    model_choice = 'fusion'
    print(f"Loading model type: {model_choice}")
    recommender = AdvancedMusicRecommender(model_type=model_choice)
    if not hasattr(recommender, 'model') or recommender.model is None:
        print("Recommender initialization failed (check logs above)")
        recommender = None
else:
     print("Flask app running without recommender")

# helper function for fetching song details
def get_song_details(song_ids):
    """Fetches title and artist for a list of song IDs."""
    if not song_ids or recommender is None or not hasattr(recommender, 'metadata') or recommender.metadata.empty:
        print("Recommender not ready or no song IDs provided")
        return []
    try:
        required_cols = ['song_id', 'title', 'artist_name']
        if not all(col in recommender.metadata.columns for col in required_cols):
             print(f"Metadata missing one or more columns: {required_cols}")
             cols_to_fetch = [col for col in required_cols if col in recommender.metadata.columns]
             if 'song_id' not in cols_to_fetch: return []
        else:
            cols_to_fetch = required_cols

        mask = recommender.metadata['song_id'].isin(song_ids)
        details = recommender.metadata.loc[mask, cols_to_fetch].to_dict('records')

        for item in details:
            item.setdefault('title', 'N/A')
            item.setdefault('artist_name', 'N/A')

        print(f"Fetched details for {len(details)} out of {len(song_ids)} input IDs")
        return details
    except Exception as e:
        print(f"Error fetching song details: {e}")
        traceback.print_exc()
        return []

# routes
@app.route('/', methods=['GET'])
def index():
    """Displays the initial search page."""
    session.pop('search_results', None)
    session.pop('search_query', None)
    session.pop('current_selection_ids', None)
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handles the song/artist search query."""
    if recommender is None:
        return render_template('index.html', error="Sorry, the recommendation engine is currently unavailable")

    query = request.form.get('query', '').strip()
    if not query:
        return redirect(url_for('index'))

    search_terms = [term.strip() for term in query.split(',') if term.strip()]
    if not search_terms:
         return redirect(url_for('index'))

    all_results = {}
    found_any = False

    print(f"Flask: Searching for terms: {search_terms}")
    for term in search_terms:
         results_df = recommender.search_songs(term, top_k=5)
         if not results_df.empty:
              all_results[term] = results_df.to_dict('records')
              found_any = True
         else:
              all_results[term] = []

    session['search_results'] = all_results
    session['search_query'] = query

    tf.keras.backend.clear_session()
    gc.collect()

    if not found_any:
         return render_template('search_results.html', results=all_results, query=query, error="No matching songs found for your query")
    else:
         return render_template('search_results.html', results=all_results, query=query)


@app.route('/recommend', methods=['POST'])
def recommend():
    """Generates recommendations based on selected songs."""
    if recommender is None:
        return render_template('index.html', error="Sorry, the recommendation engine is currently unavailable")

    selected_song_ids = request.form.getlist('selected_songs')
    original_search_results = session.get('search_results')
    original_query = session.get('search_query')

    if not selected_song_ids:
        if original_search_results: return render_template('search_results.html', results=original_search_results, query=original_query, error="Please select at least one song")
        else: return redirect(url_for('index'))


    print(f"Flask: Selected song IDs for recommendation: {selected_song_ids}")
    session['current_selection_ids'] = selected_song_ids

    try:
        print("Flask: Calling recommender.get_recommendations_from_ids...")
        recommendations_df = recommender.get_recommendations_from_ids(selected_song_ids, top_n=10)
        print(f"Flask: Received {len(recommendations_df)} recommendations")

        input_songs_details = get_song_details(selected_song_ids)

        if recommendations_df.empty:
             session.pop('current_selection_ids', None)
             # pass input_songs even if no recommendations
             return render_template('recommendations.html',
                                    recommendations=None,
                                    input_songs=input_songs_details, # pass details
                                    error="Could not generate recommendations based on your selection")

        recommendations_list = recommendations_df.to_dict('records')

        return render_template('recommendations.html',
                               recommendations=recommendations_list,
                               input_songs=input_songs_details) # pass details

    except ValueError as e:
         session.pop('current_selection_ids', None)
         print(f"Flask: Value error during recommendation: {e}")
         input_songs_details = get_song_details(selected_song_ids)
         if original_search_results:
              return render_template('search_results.html',
                                     results=original_search_results,
                                     query=original_query,
                                     input_songs=input_songs_details,
                                     error=f"Recommendation error: {e}")
         else: return redirect(url_for('index'))
    except Exception as e:
         session.pop('current_selection_ids', None)
         print(f"Flask: An unexpected error occurred during recommendation: {e}")
         traceback.print_exc(); return render_template('index.html', error="An unexpected error occurred")

@app.route('/refine', methods=['POST'])
def refine():
    if recommender is None:
        return render_template('index.html', error="Sorry, the recommendation engine is currently unavailable")

    liked_song_id = request.form.get('liked_song_id')
    original_selected_ids = session.get('current_selection_ids', [])

    if not liked_song_id: return redirect(url_for('index'))

    print(f"Flask: Refining with liked song: {liked_song_id}")
    print(f"Flask: Original selection for this list: {original_selected_ids}")
    refined_song_ids = list(set(original_selected_ids + [liked_song_id]))
    session['current_selection_ids'] = refined_song_ids
    print(f"Flask: New selection for refinement: {refined_song_ids}")

    try:
        print("Flask: Calling recommender.get_recommendations_from_ids for refinement")
        recommendations_df = recommender.get_recommendations_from_ids(refined_song_ids, top_n=10)
        print(f"Flask: Received {len(recommendations_df)} refined recommendations")

        input_songs_details = get_song_details(refined_song_ids)

        if recommendations_df.empty:
             # pass input_songs even if no recommendations
             return render_template('recommendations.html',
                                    recommendations=None,
                                    input_songs=input_songs_details, # pass details
                                    error="Could not generate further recommendations based on the refined selection")

        recommendations_list = recommendations_df.to_dict('records')

        return render_template('recommendations.html',
                               recommendations=recommendations_list,
                               input_songs=input_songs_details) # pass details

    except ValueError as e:
        print(f"Flask: Value error during refinement: {e}")
        input_songs_details = get_song_details(refined_song_ids)
        return redirect(url_for('index', error=f"Error during refinement: {e}"))
    except Exception as e:
        print(f"Flask: An unexpected error occurred during refinement: {e}")
        traceback.print_exc(); return redirect(url_for('index', error="An unexpected error occurred"))

if __name__ == '__main__':
    if recommender is None and RECOMMENDER_AVAILABLE:
         print("Recommender failed to initialize during startup")
    elif not RECOMMENDER_AVAILABLE:
         print("Could not import recommender class")

    app.run(debug=True, host='127.0.0.1', port=5000)