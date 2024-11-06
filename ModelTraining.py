
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
print("TensorFlow version:", tf.__version__)

# # Print Keras version
# print("Keras version:", keras.__version__)

# Print Python version
print("Python version:", sys.version)


# Definindo o caminho base para os datasets de treinamento
caminho_base = 'datasets/parts/train'
# Definindo caminhos para as pastas conforme e não conforme
caminho_conforme = os.path.join(caminho_base, 'conforme')
caminho_nao_conforme = os.path.join(caminho_base, 'nao_conforme')

# Imprimindo os caminhos para verificação
print(caminho_conforme) # Tamanho do lote
print(caminho_nao_conforme) # Altura e largura das imagens

# Definindo parâmetros de pré-processamento e carregamento de imagens
batch_size = 32
img_height, img_width = 299, 299

# Utilizando ImageDataGenerator para carregar e pré-processar as imagens com data augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, # Redimensiona os valores dos pixels para o intervalo [0, 1]
    validation_split=0.2, # Reserva 20% dos dados para validação
    rotation_range=20, # Rotaciona as imagens em até 20 graus
    width_shift_range=0.2, # Desloca as imagens horizontalmente em até 20% da largura
    height_shift_range=0.2, # Desloca as imagens verticalmente em até 20% da altura
    shear_range=0.2, # Aplica cisalhamento nas imagens
    zoom_range=0.2, # Aplica zoom nas imagens
    horizontal_flip=True, # Realiza flip horizontal nas imagens
    fill_mode='nearest' # Preenche pixels vazios após transformações com o valor do pixel mais próximo
)

# Carregando as imagens de treinamento
train_generator = train_datagen.flow_from_directory(
    caminho_base, # Diretório base
    target_size=(img_height, img_width), # Redimensiona as imagens para 299x299 pixels
    batch_size=batch_size, # Tamanho do lote
    class_mode='binary', # Tipo de classificação binária
    subset='training' # Usa esta parte para treinamento
)

# Carregando as imagens de validação
validation_generator = train_datagen.flow_from_directory(
    caminho_base, # Diretório base
    target_size=(img_height, img_width), # Redimensiona as imagens para 299x299 pixels
    batch_size=batch_size, # Tamanho do lote
    class_mode='binary', # Tipo de classificação binária
    subset='validation' # Usa esta parte para validação
)

# Definindo o modelo base InceptionV3 pré-treinado na base ImageNet
base_model = tf.keras.applications.InceptionV3(input_shape=(img_height, img_width, 3),
                                              include_top=False,  # Exclui a última camada (classificação)
                                              weights='imagenet' ) # Usa pesos pré-treinados do ImageNet
base_model.trainable = False # Congela o modelo base para não treinar novamente

# Construindo o modelo final adicionando novas camadas no topo do modelo base
model = tf.keras.Sequential([
    base_model, # Modelo base
    tf.keras.layers.GlobalAveragePooling2D(), # Camada de pooling global
    tf.keras.layers.Dense(128, activation='relu'), # Camada densa com 128 unidades e ReLU
    tf.keras.layers.Dropout(0.5), # Dropout com taxa de 50% para evitar overfitting
    tf.keras.layers.Dense(2, activation='softmax') # Camada final com 1 unidade e ativação sigmoide para classificação binária
])

# Definindo a função de perda, otimizador e métricas
loss = tf.keras.losses.sparse_categorical_crossentropy # Função de perda para classificação binária
optim = tf.keras.optimizers.Adam(learning_rate=0.001) # Otimizador Adam com taxa de aprendizado de 0.001
metrics = ["accuracy"] # Métrica de acurácia

# Compilando o modelo
model.compile(optimizer=optim, loss=loss, metrics=metrics)

# Definindo callbacks
# EarlyStopping interrompe o treinamento se a perda de validação não melhorar após 5 épocas
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# ModelCheckpoint salva o melhor modelo baseado na perda de validação
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)

# Definindo o número de épocas
epochs = 17
steps_per_epoch = 44 # int(np.ceil(train_generator.n / batch_size))
print(steps_per_epoch)
validation_steps = 11 # int(np.ceil(validation_generator.n / batch_size))
print(validation_steps)

# Treinamento do modelo
history = model.fit(
    train_generator, # Dados de treinamento
    steps_per_epoch=steps_per_epoch, # Número de passos por época
    epochs=epochs, # Número de épocas
    validation_data=validation_generator, # Dados de validação
    validation_steps=validation_steps, # Número de passos de validação
    callbacks=[early_stopping, model_checkpoint], # Callbacks definidos
    verbose=1
)

# Plotando os resultados
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
# Extraiu histórico de acurácia e perda

# Definindo intervalo de épocas para plotagem
epochs_range = range(len(acc))

#Criando figura para plotagem
plt.figure(figsize=(12, 8))

#Plotando acurácia de treinamento e validação
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plotando perda de treinamento e validação
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Carregando o melhor modelo salvo durante o treinamento
model = tf.keras.models.load_model('best_model.h5')

# A generator that provides a representative dataset
def representative_data_gen():
  dataset_list = tf.data.Dataset.list_files(caminho_base + '/*/*')
  for i in range(100):
    image = next(iter(dataset_list))
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [299, 299])
    image = tf.cast(image / 255., tf.float32)
    image = tf.expand_dims(image, 0)
    yield [image]

# Converter o modelo para TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Otimização padrão
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # habilita os operadores TFLite incorporados
    tf.lite.OpsSet.SELECT_TF_OPS  # habilita os operadores personalizados do TensorFlow
]
tflite_model = converter.convert()

# Salvar o modelo TFLite
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Modelo TFLite salvo como 'model.tflite'")


# # Carregando o melhor modelo salvo durante o treinamento ##
# model = tf.keras.models.load_model('best_model.h5',compile=False)
# model.export('best_model.h5')

# # A generator that provides a representative dataset
# def representative_data_gen():
#   dataset_list = tf.data.Dataset.list_files(caminho_base + '/*/*')
#   for i in range(100):
#     image = next(iter(dataset_list))
#     image = tf.io.read_file(image)
#     image = tf.io.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, [299, 299])
#     image = tf.cast(image / 255., tf.float32)
#     image = tf.expand_dims(image, 0)
#     yield [image]

# # Converter o modelo para TFLite
# converter = tf.lite.TFLiteConverter.from_saved_model('best_model.h5')
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_data_gen
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
# tflite_model = converter.convert()
# with open('modelF.tflite', 'wb') as file:
#     file.write(tflite_model)






# # Salvar o modelo TFLite
# with open('model.tflite', 'wb') as f:
#     f.write(tflite_model)

# print("Modelo TFLite salvo como 'model.tflite'")




