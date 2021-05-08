# deepfashion_to_tfrecords
Convert deepfashion to tfrecords to learn multimodal models

## Usage

First you need to get https://github.com/switchablenorms/DeepFashion2 then:

`pip install tensorflow fire`

`python deepfashion_to_tfrecords --dataset_path=deepfashion/train --num_shards=100 --output_folder=outputs`

Check the results at [deepfashion to tfrecords](read_with_tfdata.ipynb)
