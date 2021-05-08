# deepfashion_to_tfrecords
Convert deepfashion to tfrecords to learn multimodal models

![image](https://user-images.githubusercontent.com/2346494/117541312-996f2f80-b013-11eb-973c-7d60ee68fb5d.png)


## Usage

First you need to get https://github.com/switchablenorms/DeepFashion2 then:

`pip install tensorflow fire`

`python deepfashion_to_tfrecords --dataset_path=deepfashion/train --num_shards=100 --output_folder=outputs`
It runs in 5min when using 20 cores

Check the results at [deepfashion to tfrecords](read_with_tfdata.ipynb)
