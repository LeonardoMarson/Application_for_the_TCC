# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Writes metadata and label file to the image classifier models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from importlib import import_module

import os

from absl import app
from absl import flags
import flatbuffers
import tensorflow as tf

# pylint: disable=g-direct-tensorflow-import
from tflite_support import metadata_schema_py_generated as _metadata_fb
from tflite_support import metadata as _metadata
# pylint: enable=g-direct-tensorflow-import

FLAGS = flags.FLAGS


def define_flags():
  flags.DEFINE_string("model_file", "model.tflite", "./model.tflite")
  flags.DEFINE_string("label_file", "label.txt", "label.txt")
  flags.DEFINE_string("export_directory", "saida", "model_last.tflite")
  
  flags.mark_flag_as_required("model_file")
  flags.mark_flag_as_required("label_file")
  flags.mark_flag_as_required("export_directory")


class ModelSpecificInfo(object):
  """Holds information that is specificly tied to an image classifier."""

  def __init__(self, name, version, image_width, image_height, image_min,
               image_max, mean, std, num_classes, author):
    self.name = name
    self.version = version
    self.image_width = image_width
    self.image_height = image_height
    self.image_min = image_min
    self.image_max = image_max
    self.mean = mean
    self.std = std
    self.num_classes = num_classes
    self.author = author


_MODEL_INFO = {
    "model.tflite":
        ModelSpecificInfo(
            name="Detecção de Falhas em Peças na Indústria Moveleira Após Processo de Pintura Utilizando Técnicas de Processamento de Imagens com Redes Neurais Artificiais",
            version="v1",
            image_width=299,
            image_height=299,
            image_min=0,
            image_max=255,
            mean=[127.5],
            std=[127.5],
            num_classes=2,
            author="Leonardo Marson")
}


class MetadataPopulatorForImageClassifier(object):
  """Populates the metadata for an image classifier."""

  def __init__(self, model_file, model_info, label_file_path):
    self.model_file = model_file
    self.model_info = model_info
    self.label_file_path = label_file_path
    self.metadata_buf = None

  def populate(self):
    """Creates metadata and then populates it for an image classifier."""
    self._create_metadata()
    self._populate_metadata()

  def _create_metadata(self):
    """Creates the metadata for an image classifier."""

    # Creates model info.
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = self.model_info.name
    model_meta.description = ("Identifica se um objeto está conforme ou nao_conforme entre %d categorias." %
                              self.model_info.num_classes)
    model_meta.version = self.model_info.version
    model_meta.author = self.model_info.author
    model_meta.license = ("Licança Apache. Versão 2.0 "
                          "http://www.apache.org/licenses/LICENSE-2.0.")

    # Creates input info.
    input_meta = _metadata_fb.TensorMetadataT()
    input_meta.name = "image"
    input_meta.description = (
        "Imagem de entrada a ser classificada. A imagem esperada é de {0} x {1}, com "
        "três canais (red, blue, e green) por pixel. Cada valor no tensor "
        "é um único byte entre {2} e {3}.".format(
            self.model_info.image_width, self.model_info.image_height,
            self.model_info.image_min, self.model_info.image_max))
    input_meta.content = _metadata_fb.ContentT()
    input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
    input_meta.content.contentProperties.colorSpace = (
        _metadata_fb.ColorSpaceType.RGB)
    input_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.ImageProperties)
    input_normalization = _metadata_fb.ProcessUnitT()
    input_normalization.optionsType = (
        _metadata_fb.ProcessUnitOptions.NormalizationOptions)
    input_normalization.options = _metadata_fb.NormalizationOptionsT()
    input_normalization.options.mean = self.model_info.mean
    input_normalization.options.std = self.model_info.std
    input_meta.processUnits = [input_normalization]
    input_stats = _metadata_fb.StatsT()
    input_stats.max = [self.model_info.image_max]
    input_stats.min = [self.model_info.image_min]
    input_meta.stats = input_stats

    # Creates output info.
    output_meta = _metadata_fb.TensorMetadataT()
    output_meta.name = "Probabilidade"
    output_meta.description = "Probabilidade de %d labels respectivamente." % self.model_info.num_classes
    output_meta.content = _metadata_fb.ContentT()
    output_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
    output_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    output_stats = _metadata_fb.StatsT()
    output_stats.max = [1.0]
    output_stats.min = [0.0]
    output_meta.stats = output_stats
    label_file = _metadata_fb.AssociatedFileT()
    label_file.name = os.path.basename(self.label_file_path)
    label_file.description = "Labels para os objetos que o modelo pode reconhecer."
    label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
    output_meta.associatedFiles = [label_file]

    # Creates subgraph info.
    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = [input_meta]
    subgraph.outputTensorMetadata = [output_meta]
    model_meta.subgraphMetadata = [subgraph]

    b = flatbuffers.Builder(0)
    b.Finish(
        model_meta.Pack(b),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    self.metadata_buf = b.Output()

  def _populate_metadata(self):
    """Populates metadata and label file to the model file."""
    populator = _metadata.MetadataPopulator.with_model_file(self.model_file)
    populator.load_metadata_buffer(self.metadata_buf)
    populator.load_associated_files([self.label_file_path])
    populator.populate()


def main(_):
  model_file = FLAGS.model_file
  model_basename = os.path.basename(model_file)
  if model_basename not in _MODEL_INFO:
    raise ValueError(
        "A informação sobre o modelo {0} ainda não foi definida.".format(model_basename))

  export_model_path = os.path.join(FLAGS.export_directory, model_basename)

  # Copies model_file to export_path.
  tf.io.gfile.copy(model_file, export_model_path, overwrite=False)

  # Generate the metadata objects and put them in the model file
  populator = MetadataPopulatorForImageClassifier(
      export_model_path, _MODEL_INFO.get(model_basename), FLAGS.label_file)
  populator.populate()

  # Validate the output model file by reading the metadata and produce
  # a json file with the metadata under the export path
  displayer = _metadata.MetadataDisplayer.with_model_file(export_model_path)
  export_json_file = os.path.join(FLAGS.export_directory,
                                  os.path.splitext(model_basename)[0] + ".json")
  json_file = displayer.get_metadata_json()
  with open(export_json_file, "w") as f:
    f.write(json_file)

  print("Finalizado popular metadados e e arquivo associado ao modelo: ")
  print(model_file)
  print("Os metadados json forma salvos em: ")
  print(export_json_file)
  print("O arquivo associado que foi enpacotado junto ao modelo é o: ")
  print(displayer.get_packed_associated_file_list())


if __name__ == "__main__":
  define_flags()
  app.run(main)
