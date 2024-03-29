# Simple-experiment-data-processor

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fedor1113/simple-experiment-data-processor/master?filepath=experiment_results_processor.ipynb) [![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://opensource.org/licenses/MIT)

Небольшая утилита на Питоне для расчёта значения и погрешности по данным эксперимента. Может использовать метод Стьюдента с α равным 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998 или 0.999 и t для до 30 значений (или t предельным - как для нормального распределения) и метод Корнфельда для рассчёта погрешности. Само значение считается, как среднее арифметическое. Распределение Стьюдента точно считается только для n до 30 (а потом берётся n->inf).

Конечно, для работы необходим Python 3.

Вызов скрипта для GNU/Linux систем с Python 3:

```
python3 experiment_results_processor.py [-h] [-p PROBABILITY]
                                       [-d [DATA [DATA ...]]] [-C | -S]
```

Данные вводятся после вызова программы — или как параметры при вызове c флагом `-d`.

Опциональные параметры:
  * -h, --help            показать небольшую справку и завершить работу
  * -p PROBABILITY, --probability PROBABILITY
                        выставить вероятность для t-распределения равной PROBABILITY
  * -d [DATA [DATA ...]], --data [DATA [DATA ...]]
                        ввести данные [DATA [DATA ...]] (вместо ввода внутри программы)
  * -C, --cornfeld        использовать метод Корнфельда
  * -S, --student         использовать метод Стьюдента

Утилита выводит:

```
n = размер выборки
<x> = среднее
max = максимальное значение из выборки
min = минимальное значение из выборки
Δ(x) = абсолютная погрешность разброса
Ɛ(x) = относительная погрешность
Result: x = (<x> ± Δ(x))
α = вероятность
```


Также есть версия для Jupyter Notebook, использующая ipywidgets для ввода/вывода.
