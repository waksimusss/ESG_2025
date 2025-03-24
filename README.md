# Руководство пользователя
## Установка библиотек
``` bash
pip install torch transformers xgboost numpy joblib
```
## Скачивание файлов
Для корректной работы программы необходимо скачать файлы моделей и исполняемый файл:
+ [Исполняемый файл](https://github.com/waksimusss/ESG_2025/blob/master/predict_esg.py)
+ [Модель E](https://github.com/waksimusss/ESG_2025/blob/master/models/model_E.pkl)
+ [Модель S](https://github.com/waksimusss/ESG_2025/blob/master/models/model_G.pkl)
+ [Модель G](https://github.com/waksimusss/ESG_2025/blob/master/models/model_S.pkl)
+ [Модель ESG](https://github.com/waksimusss/ESG_2025/blob/master/models/meta_model.pkl)

## Запуск программы
``` bash
python predict_esg.py
```
