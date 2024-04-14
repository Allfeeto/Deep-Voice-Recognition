# Deep-Voice-Recognition

Все основные функции находятся в [DeepVoice.ipynb](https://github.com/Allfeeto/Deep-Voice-Recognition/DeepVoice.ipynb). 
Рекомендуется открывать в Google Collab.





## Реализована функциональность
- Функционал создания датасета для обучения модели (Принимает папку с аудиозаписями и параметр папки FAKE/REAL)
- Функционал обучения модели
- функционал проверки аудиофайлов (в том числе большого количества сразу)
- Интерфейс


## Особенность проекта в следующем
- Аудиофайл разбивается на сегменты по 1 секунде, из каждого сегмента извлекаются различные аудио-параметры (26 ед.)
- Модель содержит несколько сверточных слоев (Conv1D), слои LSTM (Bidirectional(LSTM)), слои Batch Normalization (BatchNormalization), слои Dropout (Dropout) и полносвязные слои (Dense). **Эта модель предназначена для обработки временных последовательностей.**
- Модель обучается сравнительно быстро (+-1ч), выдавая на тестовых значениях вероятность более 95%
- Есть Интерфейс


## Основной стек технологий 
- Python
- GitHub, Google Collab, SteamLit



# Установка и запуск

## 1 способ - StreamLit

1. Откройте [Python файл](https://github.com/Allfeeto/Deep-Voice-Recognition/main.py) с помощью IDE, которое присутствует на вашем пк 
2. В командрой строке поочередно вводим:
```
pip install streamlit
pip install librosa==0.10.1
pip install numpy==1.26.4
pip install pandas==2.2.2
pip install streamlit==1.33.0
pip install tensorflow==2.15.0
pip install scikit-learn==1.4.2

```
5. При желании вы можете загрузить свою модель, просто указав путь в нижеуказанной строке
```
loaded_model = load_model("LastModel(150.32).h5")
```
5. Для запуска сервера используем командную строку проекта и вводим:
```
streamlit run main.py
```

## 2 способ - Google Collab

1. Скопируйте [DeepVoice.ipynb](https://github.com/Allfeeto/Deep-Voice-Recognition/DeepVoice.ipynb) к себе в коллаб, запускайте нужные части кода




## РАЗРАБОТЧИКИ
- Манаков Владислав Андреевич @allfeeto
- Мельников Александр Романович @bobinka
