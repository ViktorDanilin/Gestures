# Gestures

## Project Description
#### Это проект для людей с ограниченными возможностями и заинтересованных лиц, которые хотят выучить язык жестов

* Проект в будущем предполагает создание веб-интерфеса и приложения
* Текущая версия проекта может распознавать ASL (Американский язык жестов)
* При запуске программы появляется окно на котором все распознанные жесты помечаются, соответствующей жесту букве

## Used libraries
#### Библиотеки для создания лендинга с описанием:
* Flask
#### Библиотеки для распознавания жестов
* OpenCV
* Numpy
* CVzone
* Tensorflow
* Keras

## Project Installation (for Windows)
###1. Установка библиотек 

    pip install opencv-python

    pip install numpy

    pip install cvzone
   
###2. Установка tensorflow и keras
   
    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
    
    python -m pip install "tensorflow<2.10"
    
Проверка установки с использованием cuda

    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    
###3. Установка проекта

Открываем через терминал папку, куда планируется монтировать проект
    
    cd '/YOUR_PATH'
    
    git clone https://github.com/ViktorDanilin/Gestures.git

    cd Gestures/

    python3 SignRecognizer.py

## Demonstration video

[Demo video](https://github.com/ViktorDanilin/Gestures/assets/42595661/e3d70ff4-7e79-4e51-be9c-5771c48a6857)