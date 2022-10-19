# Настройка окружения
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Запуск
Для запуска необходимо настроить параметры модели в `config.ExpConfig` под желаемые, 
затем запуск через CLI:
```bash
python run.py
```
Дополнительно параметры `SWA` и других callbacks можно настроить непосредственно в `main.py`.

Чекпоинты модели будут сохранены по пути `./runs/RUN_NAME/...`, а логи Tensorboard в `./lightning_logs/RUN_NAME/...`.

# Ограничения
Дефолтно в lightning hook `on_fit_start` прописаны верхние границы числа `MACs` и количества параметров модели.

# Эксперименты
## Глубина и ширина
Модели наибольшей глубины и малой ширины показали наибольшую эффективность среди прочих. Были проведены следующие эксперементы:
1. `Deep + Narrow` -- наилучший результат
2. `Deep + Wide` -- не вмещается по ограничению на парамтеры
3. `Shallow + Narrow` -- маленький скор
4. `Shallow + Wide` -- неплохой результат, но хуже первого

## Размер сверток
`Shallow` модели с большим `kernel_size` показали себя хорошо, наравне с `Deep` и маленьким `kernel_size`. Тем не менее в итоге был выбраны глубокие модели с маленьким `kerner_size` (3), так как в области CV есть ряд небезизвестных статей о том, что фильтры малого размера показывают себя лучше.

## Нормализация WAV
Было поставлено три эксперимента:
1. Отсутствие нормализации
2. `abs()max()` Нормализация при семплировании из датасета
3. `BatchNorm` на входе в модель

эффективнее всего на LB себя показал второй тип нормализации, а третий вариант показал себя плохо, скорее всего это свзано со смещением выборок относительно громкости.

## Conv Vs. FullyConv
Было поставлено два эксперимента:
1. `conv_feature_extractor` + `fc_block` + `final_fc`
2. `conv_feature_extractor` + `final_fc`

наилучшим вариантом с точки зрения скорости обучения и регуляризации оказался второй, думаю это напрямую отсылка к практике того, что fully convolutional почти всегда лучше.

## Activations
Было поставлено три эксперимента:
1. `ReLU`
2. `LeakyReLU`
3. `SiLU`
третий вариант показал себя лучше остальных по скорости сходимости.

## Pre-Training
Был поставлен эксперимент: модель обучена на google-speech-commands датасете, а затем дообучалась на оригинальном датасете. Прироста качества это не дало. Думаю это связано с тем, что для распознавания речи не работает история с мультидоменностью.

## Batch Size
Большие batch_size показали себя лучше с точки зрения сходимости в суб-оптимальное решение. В данной работе в большинстве случаев использовался `batch_size=2048`

