import joblib
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

print('Загрузка данных')

# Загружаем модели XGBoost
model_E = joblib.load("models/model_E.pkl")
model_S = joblib.load("models/model_S.pkl")
model_G = joblib.load("models/model_G.pkl")
meta_model = joblib.load("models/meta_model.pkl")

# Загружаем RuBERT
model_name = "DeepPavlov/rubert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

# Функция получения эмбеддинга
def get_embedding(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Функция перевода ESG в рейтинг 1-150
def scale_to_rating(value, min_val=0, max_val=1):
    return int(1 + (value - min_val) / (max_val - min_val) * (150 - 1))

# Основной цикл взаимодействия с пользователем
def main():
    while True:
        print("\n Предсказане ESG")
        print("1. Ввести текст для анализа")
        print("2. Выйти")

        choice = input("Выберите действие (1/2): ").strip()

        if choice == "2":
            print("До свидания!")
            break
        elif choice == "1":
            text_input = input("\nВведите текст для ESG-анализа: ").strip()
            if not text_input:
                print("Ошибка: текст не должен быть пустым!")
                continue

            # Рассчитываем эмбеддинг
            embedding = get_embedding(text_input)

            # Делаем предсказания
            pred_E = model_E.predict(embedding)
            pred_S = model_S.predict(embedding)
            pred_G = model_G.predict(embedding)

            # Объединяем в мета-признаки и делаем финальное предсказание
            X_meta = np.column_stack([pred_E, pred_S, pred_G])
            final_prediction = meta_model.predict(X_meta)

            # Переводим в рейтинг 1-150
            rating_ESG = scale_to_rating(final_prediction[0])
            rating_E = scale_to_rating(pred_E[0])
            rating_S = scale_to_rating(pred_S[0])
            rating_G = scale_to_rating(pred_G[0])

            # Вывод результата
            print("Результат ESG-анализа")
            print(f"ESG-рейтинг: {rating_ESG}")
            print(f"Экология (E): {rating_E}")
            print(f"Социальное (S): {rating_S}")
            print(f"Управление (G): {rating_G}")
            print("-" * 30)

        else:
            print("Ошибка: введите 1 или 2.")

if __name__ == "__main__":
    main()
