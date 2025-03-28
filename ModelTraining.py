# Importing required libraries
# Importando bibliotecas necessárias
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras
models = tf.keras.models
callbacks = tf.keras.callbacks

import sys

# Print TensorFlow version
# Imprime a versão do TensorFlow
print("TensorFlow version:", tf.__version__)

# Print Python version
# Imprime a versão do Python
print("Python version:", sys.version)

# Defining the base path for the training datasets
# Definindo o caminho base para os datasets de treinamento
caminho_base = 'datasets/parts/train'

# Defining paths for compliant and non-compliant folders
# Definindo caminhos para as pastas conforme e não conforme
caminho_conforme = os.path.join(caminho_base, 'conforme')
caminho_nao_conforme = os.path.join(caminho_base, 'nao_conforme')

# Printing the paths for verification
# Imprimindo os caminhos para verificação
print(caminho_conforme)
print(caminho_nao_conforme)

# Defining preprocessing parameters and image loading
# Definindo parâmetros de pré-processamento e carregamento de imagens
batch_size = 32  # Batch size / Tamanho do lote
img_height, img_width = 299, 299  # Height and width of images / Altura e largura das imagens

# Using ImageDataGenerator to load and preprocess images with data augmentation
# Utilizando ImageDataGenerator para carregar e pré-processar as imagens com aumento de dados
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,  # Rescale pixel values to the range [0, 1] / Redimensiona os valores dos pixels para o intervalo [0, 1]
    validation_split=0.2,  # Reserve 20% of data for validation / Reserva 20% dos dados para validação
    rotation_range=20,  # Rotates images up to 20 degrees / Rotaciona as imagens em até 20 graus
    width_shift_range=0.2,  # Shifts images horizontally up to 20% / Desloca as imagens horizontalmente em até 20%
    height_shift_range=0.2,  # Shifts images vertically up to 20% / Desloca as imagens verticalmente em até 20%
    shear_range=0.2,  # Applies shear transformations / Aplica cisalhamento nas imagens
    zoom_range=0.2,  # Applies zoom transformations / Aplica zoom nas imagens
    horizontal_flip=True,  # Flips images horizontally / Realiza espelhamento horizontal
    fill_mode='nearest'  # Fills empty pixels with the nearest pixel value / Preenche pixels vazios com o valor mais próximo
)

# Loading training images
# Carregando as imagens de treinamento
train_generator = train_datagen.flow_from_directory(
    caminho_base,  # Base directory / Diretório base
    target_size=(img_height, img_width),  # Resizes images to 299x299 pixels / Redimensiona as imagens para 299x299 pixels
    batch_size=batch_size,  # Batch size / Tamanho do lote
    class_mode='binary',  # Binary classification type / Tipo de classificação binária
    subset='training'  # Uses this subset for training / Usa esta parte para treinamento
)

# Loading validation images
# Carregando as imagens de validação
validation_generator = train_datagen.flow_from_directory(
    caminho_base,  # Base directory / Diretório base
    target_size=(img_height, img_width),  # Resizes images to 299x299 pixels / Redimensiona as imagens para 299x299 pixels
    batch_size=batch_size,  # Batch size / Tamanho do lote
    class_mode='binary',  # Binary classification type / Tipo de classificação binária
    subset='validation'  # Uses this subset for validation / Usa esta parte para validação
)

# Defining the base model as InceptionV3 pre-trained on ImageNet
# Definindo o modelo base como InceptionV3 pré-treinado no ImageNet
base_model = tf.keras.applications.InceptionV3(input_shape=(img_height, img_width, 3),
                                              include_top=False,  # Excludes the final classification layer / Exclui a última camada de classificação
                                              weights='imagenet')  # Uses pre-trained weights from ImageNet / Usa pesos pré-treinados do ImageNet
base_model.trainable = False  # Freezes the base model to prevent retraining / Congela o modelo base para evitar re-treinamento

# Building the final model by adding new layers on top of the base model
# Construindo o modelo final adicionando novas camadas no topo do modelo base
model = tf.keras.Sequential([
    base_model,  # Base model / Modelo base
    tf.keras.layers.GlobalAveragePooling2D(),  # Global average pooling layer / Camada de pooling global
    tf.keras.layers.Dense(128, activation='relu'),  # Dense layer with 128 units and ReLU activation / Camada densa com 128 unidades e ativação ReLU
    tf.keras.layers.Dropout(0.5),  # Dropout layer with 50% rate to prevent overfitting / Camada de dropout com taxa de 50% para evitar overfitting
    tf.keras.layers.Dense(2, activation='softmax')  # Final layer with softmax activation for binary classification / Camada final com ativação softmax para classificação binária
])

# Defining the loss function, optimizer, and metrics
# Definindo a função de perda, otimizador e métricas
loss = tf.keras.losses.sparse_categorical_crossentropy  # Loss function for binary classification / Função de perda para classificação binária
optim = tf.keras.optimizers.Adam(learning_rate=0.001)  # Adam optimizer with learning rate 0.001 / Otimizador Adam com taxa de aprendizado 0.001
metrics = ["accuracy"]  # Accuracy metric / Métrica de acurácia

# Compiling the model
# Compilando o modelo
model.compile(optimizer=optim, loss=loss, metrics=metrics)

# Defining callbacks
# Definindo callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Stops training if validation loss does not improve after 5 epochs / Interrompe o treinamento se a perda de validação não melhorar após 5 épocas
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)  # Saves the best model based on validation loss / Salva o melhor modelo baseado na perda de validação

# Defining number of epochs
# Definindo o número de épocas
epochs = 17
steps_per_epoch = 44  # Steps per epoch / Passos por época
validation_steps = 11  # Validation steps / Passos de validação

# Training the model
# Treinamento do modelo
history = model.fit(
    train_generator,  # Training data / Dados de treinamento
    steps_per_epoch=steps_per_epoch,  # Steps per epoch / Número de passos por época
    epochs=epochs,  # Number of epochs / Número de épocas
    validation_data=validation_generator,  # Validation data / Dados de validação
    validation_steps=validation_steps,  # Validation steps / Número de passos de validação
    callbacks=[early_stopping, model_checkpoint],  # Defined callbacks / Callbacks definidos
    verbose=1
)

# Plotting results
# Plotando os resultados
plt.figure(figsize=(12, 8))

# Training and validation accuracy
# Acurácia de treinamento e validação
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Training and validation loss
# Perda de treinamento e validação
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Saving the model in TensorFlow Lite format
# Salvando o modelo no formato TensorFlow Lite
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite model saved as 'model.tflite'")



