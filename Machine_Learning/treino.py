import pickle
from pathlib import Path

import category_encoders as ce
import mlflow
import mlflow.sklearn
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------- Caminhos ----------
BASE_DIR = Path(__file__).resolve().parent.parent      # case_fraude_main
DATA_PATH = BASE_DIR / "dados" / "Dados_Case1_tratados.csv"
MODEL_PATH = Path(__file__).resolve().parent / "modelo_fraude.pkl"

# ---------- MLflow ----------
TRACKING_URI = f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}"
ARTIFACTS_DIR = BASE_DIR / "mlruns"
EXPERIMENT_NAME = "deteccao_fraude"

# ---------- Configuração do experimento ----------
COLUNAS_REMOVER = ['fraude', 'score_fraude_modelo', 'data_compra', 'produto']
COLUNAS_ENCODE = ['pais', 'categoria_produto']
TEST_SIZE = 0.2
SEED = 42


def load_csv(local):
    return pd.read_csv(local)


def build_pipeline():
    transformador_coluna = ColumnTransformer(
        transformers=[
            ('woe', ce.WOEEncoder(), COLUNAS_ENCODE),
        ],
        remainder='passthrough',
    )

    return Pipeline([
        ('oversampler', RandomOverSampler(random_state=SEED)),
        ('column_transformer', transformador_coluna),
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(
            random_state=SEED,
            n_estimators=100,
            max_depth=4,
            loss='log_loss',
            learning_rate=0.2,
        )),
    ])


def metrics(X_teste, y_teste, pipeline):
    y_pred = pipeline.predict(X_teste)
    y_proba = pipeline.predict_proba(X_teste)[:, 1]
    return {
        "recall": recall_score(y_teste, y_pred),
        "precision": precision_score(y_teste, y_pred, zero_division=0),
        "f1": f1_score(y_teste, y_pred),
        "roc_auc": roc_auc_score(y_teste, y_proba),
    }


def setup_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    # Cria o experimento já com a pasta de artefatos fixa (não depende do CWD)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME, artifact_location=ARTIFACTS_DIR.as_uri()
        )
    mlflow.set_experiment(EXPERIMENT_NAME)


def main():
    setup_mlflow()

    df = load_csv(DATA_PATH)
    X = df.drop(columns=COLUNAS_REMOVER)
    y = df['fraude']

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    pipeline = build_pipeline()

    with mlflow.start_run(run_name="gradient_boosting_woe"):
        # --- Parâmetros ---
        mlflow.log_params({
            f"model__{k}": v
            for k, v in pipeline.named_steps['model'].get_params().items()
        })
        mlflow.log_params({
            "oversampler": "RandomOverSampler",
            "colunas_woe": ",".join(COLUNAS_ENCODE),
            "colunas_removidas": ",".join(COLUNAS_REMOVER),
            "test_size": TEST_SIZE,
            "seed": SEED,
        })

        # --- Info dos dados ---
        mlflow.log_params({
            "n_treino": len(X_treino),
            "n_teste": len(X_teste),
            "taxa_fraude_treino": round(float(y_treino.mean()), 4),
        })

        # --- Treino ---
        pipeline.fit(X_treino, y_treino)

        # --- Métricas ---
        resultados = metrics(X_teste, y_teste, pipeline)
        mlflow.log_metrics(resultados)
        for nome, valor in resultados.items():
            print(f"{nome}: {valor:.4f}")

        # --- Modelo (pipeline inteira) ---
        assinatura = infer_signature(X_treino, pipeline.predict(X_treino))
        mlflow.sklearn.log_model(
        pipeline,
        name="modelo_fraude",
        signature=assinatura,
        input_example=X_treino.head(5),
        serialization_format="cloudpickle",
        )

        # --- Também salva o .pkl local, usado pelo predict.py ---
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(pipeline, f)


if __name__ == "__main__":
    main()