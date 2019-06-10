# -*- coding: utf-8 -*-
# импорт нужных функций и библиотек

import json
import numpy as np
from pandas import read_csv
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

# словарь с настройками ключ - название настройки, значение - значение настройки
config = {
    "sprashivat_configuratiu": True,
    "path_to_pos": "positive.csv",
    "path_to_neg": "negative.csv",
    "clf": "multinomial",
    "test_size": 0.2,
    "max_f": 300,
}

def change_settings(key, value):
    with open("settings.json", "r") as fp:
        json_data = json.load(fp)
    with open("settings.json", "w") as fp:
        json_data["config"][key] = value
        json.dump(json_data, fp, ensure_ascii=False)   
# функция для выборки данных


def data_select(path_to_pos, path_to_neg):
    # читаем данные и соединяем в один массив
    pos = read_csv(path_to_pos, ";")
    neg = read_csv(path_to_neg, ";")
    arr = np.concatenate((pos, neg), axis=0)

    # берём 4 и 5 столбцы - текст сообщения и тип твита
    X = arr[:, 3]
    Y = arr[:, 4]
    return X, Y


def classify(X, Y, maxf, size, cl):
    # преобразование текста в массив
    vectorizer = CountVectorizer(
        analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=maxf)
    train_data_features = vectorizer.fit_transform(X)
    train_data_features = train_data_features.toarray()

    # разделение на тестовую и обучающую выборки
    X_train, X_test, Y_train, Y_test = train_test_split(
        train_data_features, Y, test_size=size, random_state=0)
    Y_train = Y_train.astype(int)
    Y_test = Y_test.astype(int)

    # выбор классификатора
    if (cl == 'multinomial'):
        clf = MultinomialNB()
    if (cl == 'bernoulli'):
        clf = BernoulliNB()
    if (cl == 'gaussian'):
        clf = GaussianNB()

    # построение модели, прогнозирование значений, вывод оценки качества
    clf.fit(X_train, Y_train)
    res = clf.predict(X_test)
    # информация для юзверя и предсказание типа произвольного сообщения
    print("ИИ обучился, точность составила :")
    print(metrics.accuracy_score(Y_test, res))  # вывод точности ИИ
    print("введите строку, которую хотите скормить ИИ")
    user_input = str(input("-"))  # в user_input - строка
    # переводим строку в двумерный массив но с 1 элементом , строкой
    user_input = np.array([user_input])
    # переводим двумерный массив в двумерный массив с вектором внутри
    user_input = vectorizer.transform(user_input)
    user_input = user_input.toarray()  # приведение данных
    user_input = user_input.astype(int)  # опять приведение данных
    # вывод результата предсказания ИИ
    print(f"тип вашего сообщения ИИ оценил , как {clf.predict(user_input)}")
    user_input = str(
        input("1.ввести другое сообщение\n2.поменять конфигурацию\n"))  # выбор действия
    if user_input == "1":
        config['sprashivat_configuratiu'] = False
    elif user_input == "2":
        config['sprashivat_configuratiu'] = True
    else:
        print("че? сейчас совсем не понял")  # обработка опечаток


while True:  # вход в бесконечный цикл чтобы юзера не выкидывало после ввода 1 сообщенрия в ИИ
    user_input = None  # обнуление введеных пользователем данных
    print(f"""текущая конфигурация ИИ: \n
путь к файлу с позитивными сообщениями: {config['path_to_pos']}\n
путь к файлу с отрицательными сообщениями: {config['path_to_neg']}\n
классификатор:{config['clf']}\n
процент выборки для тестов : {config['test_size']}\n
максимальное количество признаков для алгоритма : {config['max_f']}
""")  # вывод конфигурации
    if config['sprashivat_configuratiu']:
        user_input = str(input("вы хотите изменить конфигурацию ИИ?\n 1.да \n 2.нет"))
        if user_input == True:
            while 1:  # тоже бесконечный цикл
                print("выберите что хотите поменять : \n")
                print("path_to_pos | path_to_neg | clf | test_size | max_f")
                user_input = str(input("-"))
                try:
                    config[user_input] = str(input("введите значение : "))
                except Exception as e:
                    print(e)
                    print("ошибка в названии конфигурации")
                    continue
                user_input = str(input("хотите еще что то поменять?(д|н)\n"))
                if user_input == "д":
                    continue
                elif user_input == "н":
                    config['sprashivat_configuratiu'] = False
                    break
                else:
                    print("что? я не понимаю")
                    continue

        elif user_input == "н":
            config['sprashivat_configuratiu'] = False
        else:
            print("че? не понял")
            continue
    else:
        try:  # вызов функций и обучение ИИ
            x, y = data_select(config['path_to_pos'], config['path_to_neg'])
            classify(x, y, int(config['max_f']), float(
                config['test_size']), config['clf'])
        except Exception as e:
            print(e)
            print("\t\tошибка в конфигурации")
