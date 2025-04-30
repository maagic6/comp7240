import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import h5py
import gc
import traceback

# config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# metadata
def load_metadata(filename):
    try:
        with h5py.File(filename, "r") as f:
            songs_dataset = f['metadata/songs']
            song_ids_bytes = songs_dataset['song_id'][()]
            titles_bytes = songs_dataset['title'][()]
            artists_bytes = songs_dataset['artist_name'][()]

            decode_func = lambda x: x.decode('utf-8', errors='ignore').strip()
            song_ids = list(map(decode_func, song_ids_bytes))
            titles = list(map(decode_func, titles_bytes))
            artists = list(map(decode_func, artists_bytes))

            df = pd.DataFrame({
                'song_id': song_ids,
                'title': titles,
                'artist_name': artists
            })
            return df[df['song_id'].str.len() > 0] # filter out potential empty IDs
    except FileNotFoundError:
        print(f"Metadata file not found at {filename}")
        return pd.DataFrame({'song_id': [], 'title': [], 'artist_name': []})
    except Exception as e:
        print(f"Error loading metadata from {filename}: {e}")
        return pd.DataFrame({'song_id': [], 'title': [], 'artist_name': []})

# recommender class
class AdvancedMusicRecommender:
    def __init__(self, model_type='fusion', data_dir=DATA_DIR):
        """
        Initializes the recommender
        Args:
            model_type (str): Type of model to load ('fusion', 'gmf', 'mlp')
            data_dir (str): Path to the directory containing model and data files
        """
        print(f"Initializing recommender (Type: {model_type})...")
        self.data_dir = data_dir
        self.model = None
        self.model_type = model_type

        try:
            tf.config.threading.set_inter_op_parallelism_threads(4)
            tf.config.threading.set_intra_op_parallelism_threads(4)

            if self.model_type == 'fusion':
                model_filename = 'best_fusion_model.keras'
                encoder_filename = 'song_encoder_fusion.pkl'
                max_play_filename = 'max_play_fusion.pkl'
                gmf_layer_name = 'fusion_gmf_item_embed'
                mlp_layer_name = 'fusion_mlp_item_embed'
            elif self.model_type == 'gmf':
                model_filename = 'best_gmf_model.keras'
                encoder_filename = 'song_encoder_gmf.pkl'
                max_play_filename = 'max_play_gmf.pkl'
                embedding_layer_name = 'item_embed'
            elif self.model_type == 'mlp':
                model_filename = 'best_mlp_model.keras'
                encoder_filename = 'song_encoder_mlp.pkl'
                max_play_filename = 'max_play_mlp.pkl'
                embedding_layer_name = 'item_embed'
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}. Choose 'fusion', 'gmf', or 'mlp'")

            # data filepaths
            metadata_path = os.path.join(self.data_dir, 'msd_summary_file.h5')
            model_path = os.path.join(self.data_dir, model_filename)
            encoder_path = os.path.join(self.data_dir, encoder_filename)
            max_play_path = os.path.join(self.data_dir, max_play_filename)

            # load metadata
            self.metadata = load_metadata(metadata_path)
            if self.metadata.empty:
                raise ValueError("Metadata loading failed or resulted in empty DataFrame")
            print(f"Metadata loaded: {len(self.metadata)} songs")

            with tf.device('/CPU:0'):
                self.model = tf.keras.models.load_model(model_path)
            if self.model is None:
                 raise ValueError(f"Failed to load Keras model from {model_path}")
            print(f"Model loaded: {model_filename}")

            # load encoders
            self.song_encoder = joblib.load(encoder_path)
            print(f"Song encoder loaded: {encoder_filename}")

            # handle max_play
            try:
                self.max_play = joblib.load(max_play_path)
            except FileNotFoundError:
                print(f"max_play file not found at {max_play_path}, using default max_play=1")
                self.max_play = 1
            except Exception as e:
                 print(f"Error loading {max_play_path}: {e}, using default max_play=1")
                 self.max_play = 1
            print(f"Max play value: {self.max_play}")

            # create song ID to index mapping
            self.song_id_to_idx = {
                song_id: idx
                for idx, song_id in enumerate(self.song_encoder.classes_)
            }

            # get song embeddings
            print(f"Extracting embeddings for {self.model_type} model")
            if self.model_type == 'fusion':
                gmf_emb = self.model.get_layer(gmf_layer_name).get_weights()[0]
                mlp_emb = self.model.get_layer(mlp_layer_name).get_weights()[0]
                self.song_embeddings = np.concatenate([gmf_emb, mlp_emb], axis=1)
            elif self.model_type in ['gmf', 'mlp']:
                 self.song_embeddings = self.model.get_layer(embedding_layer_name).get_weights()[0]

            print(f"Song embeddings extracted, shape: {self.song_embeddings.shape}")

            print("Recommender initialization complete")

        except ValueError as e:
             print(f"Error during initialization: {e}")
             self.model = None
        except FileNotFoundError as e:
             print(f"Required file not found during initialization: {e}")
             self.model = None
        except KeyError as e:
             print(f"Layer name {e} not found in the loaded model '{model_filename}'")
             self.model = None
        except Exception as e:
            print(f"Error during recommender initialization: {e}")
            traceback.print_exc()
            self.model = None

    def search_songs(self, query, top_k=5):
        if self.metadata.empty or query is None or not query.strip():
            return pd.DataFrame()

        query = query.strip()
        try:
            mask = (
                self.metadata['title'].str.contains(query, case=False, na=False) |
                self.metadata['artist_name'].str.contains(query, case=False, na=False)
            )
            cols_to_select = ['song_id', 'title', 'artist_name']
            return self.metadata.loc[mask, cols_to_select].head(top_k)
        except Exception as e:
             print(f"Error during song search for '{query}': {e}")
             return pd.DataFrame()


    def create_virtual_user(self, song_ids):
        """Create virtual user features from song IDs"""
        if self.model is None:
             raise RuntimeError("Recommender not properly initialized")

        valid_ids = [song_id for song_id in song_ids if song_id in self.song_id_to_idx]

        if not valid_ids:
            raise ValueError("No valid song IDs found")

        indices = [self.song_id_to_idx[song_id] for song_id in valid_ids]
        avg_embedding = np.mean(self.song_embeddings[indices], axis=0)
        return avg_embedding

    def get_recommendations_from_ids(self, selected_ids, top_n=10):
        """
        For use by Flask
        """
        if self.model is None:
             raise RuntimeError("Recommender not properly initialized")
        if not selected_ids:
             raise ValueError("No song IDs provided for recommendations")

        print(f"Generating recommendations for {len(selected_ids)} selected IDs")
        try:
            # create virtual user embedding
            virtual_user_embedding = self.create_virtual_user(selected_ids)
            print("Virtual user embedding created")

            # calculate similarity scores
            scores = np.dot(self.song_embeddings, virtual_user_embedding)
            print("Similarity scores calculated")

            # exclude selected songs
            input_indices = [
                self.song_id_to_idx[sid]
                for sid in selected_ids
                if sid in self.song_id_to_idx
            ]
            scores[input_indices] = -np.inf  # discard selected songs

            # get top N recommendations
            num_available = len(scores) - len(input_indices)
            actual_top_n = min(top_n, max(0, num_available))

            if actual_top_n == 0:
                print("No recommendations possible after filtering")
                return pd.DataFrame(columns=['song_id', 'title', 'artist_name', 'predicted_plays'])

            # partitioning top recs
            partitioned_indices = np.argpartition(scores, -actual_top_n)[-actual_top_n:]
            top_indices = partitioned_indices[np.argsort(scores[partitioned_indices])][::-1]

            top_scores = scores[top_indices]
            top_song_ids = self.song_encoder.inverse_transform(top_indices)

            # retrieve metadata for recommended songs
            recommendations = self.metadata[
                self.metadata['song_id'].isin(top_song_ids)
            ].copy()

            # add predicted plays
            if not recommendations.empty:
                try:
                    # create mapping from song_id to score
                    score_map = {song_id: score for song_id, score in zip(top_song_ids, top_scores)}
                    recommendations['score'] = recommendations['song_id'].map(score_map)

                    # calculate predicted plays
                    recommendations['predicted_plays'] = np.clip(
                        recommendations['score'].fillna(0) * self.max_play,
                        0, None
                    )
                    # sort results by predicted plays
                    recommendations = recommendations.sort_values('predicted_plays', ascending=False)
                except Exception as e:
                    print(f"Warning: Could not calculate predicted_plays accurately: {e}")
                    recommendations['predicted_plays'] = 0.0

                final_cols = ['song_id', 'title', 'artist_name', 'predicted_plays']
                recommendations = recommendations[final_cols]
            else:
                 recommendations = pd.DataFrame(columns=['song_id', 'title', 'artist_name', 'predicted_plays'])

            print(f"Generated {len(recommendations)} final recommendations")
            tf.keras.backend.clear_session()
            gc.collect()
            return recommendations

        except ValueError as e: 
             print(f"Value error during recommendation generation: {e}")
             raise # re-raise the error to be handled by app.py
        except Exception as e:
            print(f"Error during recommendation generation: {e}")
            traceback.print_exc()
            tf.keras.backend.clear_session()
            gc.collect()
            return pd.DataFrame(columns=['song_id', 'title', 'artist_name', 'predicted_plays'])