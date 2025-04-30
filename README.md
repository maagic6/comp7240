# Music recommender flask UI

Web interface for music recommender

## Setup

1.  **Create `data` folder:** Create a folder named `data` in the same directory as `app.py`
2.  **Add files to `data`:** Put these files inside the `data` folder:
    *   `msd_summary_file.h5`
    *   `best_fusion_model.keras`
    *   `song_encoder_fusion.pkl`
    *   `max_play_fusion.pkl`
3.  **Install requirements (tested on Python 3.9):**
    ```bash
    # python -m venv venv
    # source venv/bin/activate

    pip install -r requirements.txt
    ```

## Run

1.  **Run the app:**
    ```bash
    flask run
    ```
2.  **Open in browser:** Go to `http://127.0.0.1:5000/`

(Model loading may take a minute)

## Debug mode

To run with auto-reload and error pages:
```bash
# make sure debug=True in app.py's app.run() line
flask run
```

## Model type

Specify model type in app.py
```bash
...
recommender = None
if RECOMMENDER_AVAILABLE and AdvancedMusicRecommender:
    print("Initializing music recommender")
    recommender = AdvancedMusicRecommender(model_type="fusion") # fusion, gmf, mlp
    if recommender.model is None:
...
```