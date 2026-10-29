import pickle
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(__file__).resolve().parent / "modelo_fraude.pkl"
INPUT_PATH = BASE_DIR / "dados" / "novos_dados.csv"
OUTPUT_PATH = BASE_DIR / "dados" / "predicoes.csv"

# Mesmas colunas removidas no treino
COLUNAS_REMOVER = ['fraude', 'score_fraude_modelo', 'data_compra', 'produto']


def load_model(path=MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def preparar_dados(df):
    # errors="ignore": não quebra se alguma coluna (ex.: 'fraude') não existir
    return df.drop(columns=COLUNAS_REMOVER, errors="ignore")


def predict(df, pipeline, threshold=0.5):
    X = preparar_dados(df)
    proba = pipeline.predict_proba(X)[:, 1]
    resultado = df.copy()
    resultado["prob_fraude"] = proba
    resultado["pred_fraude"] = (proba >= threshold).astype(int)
    return resultado


def main():
    pipeline = load_model()
    df = pd.read_csv(INPUT_PATH)

    resultado = predict(df, pipeline)
    resultado.to_csv(OUTPUT_PATH, index=False)

    print(f"{len(resultado)} linhas processadas")
    print(f"Fraudes previstas: {resultado['pred_fraude'].sum()}")
    print(f"Salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()