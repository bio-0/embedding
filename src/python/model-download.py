import os
from sentence_transformers import SentenceTransformer
from umap.umap_ import UMAP
import joblib
import numpy as np


def load_model():
    # Download model
    model_dir = "nomic-embed"  # os.environ.get('NOMIC_RESOURCES_DIR')
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1",
                                revision='02d96723811f4bb77a80857da07eda78c1549a4d',
                                trust_remote_code=True)
    # Save model
    model.save(model_dir)

    # Load data from a text file nomic_emb.csv
    x_nom = np.genfromtxt('nomic_emb.csv', delimiter=",")

    # Transform models
    # umap-display model
    disp_model = UMAP(n_components=2,
                      n_neighbors=15,
                      random_state=42,
                      min_dist=0.05,
                      metric='cosine',
                      n_jobs=1)
    u = disp_model.fit_transform(x_nom)
    filename = 'umap-models/umap-display.joblib'
    joblib.dump(disp_model, filename)
    # umap-cluster model
    clust_model = UMAP(n_components=8,
                       n_neighbors=15,
                       random_state=42,
                       min_dist=0.0,
                       metric='cosine',
                       n_jobs=1)
    u = clust_model.fit_transform(x_nom)
    filename = 'umap-models/umap-cluster.joblib'
    joblib.dump(clust_model, filename)


if __name__ == '__main__':
    load_model()
