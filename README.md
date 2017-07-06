# train-cnn

A GBDX task that trains a CNN classifier on labeled image chips using the GPU.


## Run

Here we run a sample execution of the train-cnn task. Sample inputs are provided on S3 in the locations specified below.

1. In a Python terminal create a GBDX interface and specify the task input location:

    ```python
    from gbdxtools import Interface
    from os.path import join
    import uuid

    gbdx = Interface()

    input_location = 's3://gbd-customer-data/32cbab7a-4307-40c8-bb31-e2de32f940c2/platform-stories/train-cnn/'
    ```

2. Create a task instance and set the required inputs:

    ```python
    cnn_task = gbdx.Task('train-cnn')
    cnn_task.inputs.train_data = join(input_location, 'train_data')
    cnn_task.inputs.bit_depth = '8'
    cnn_task.inputs.nb_epoch = '15'
    ```

3. Create a single-task workflow object and define where the output data should be saved.

    ```python
    workflow = gbdx.Workflow([cnn_task])
    random_str = str(uuid.uuid4())
    output_location = join('platform-stories/trial-runs', random_str)

    workflow.savedata(cnn_task.outputs.trained_model, output_location)
    ```

4. Execute the workflow and monitor its status as follows:

    ```python
    workflow.execute()
    workflow.status
    ```

## Input Ports

GBDX input ports can only be of "Directory" or "String" type. Booleans, integers and floats are passed to the task as strings, e.g., "True", "10", "0.001".

| Name  | Type | Description | Required |
|---|---|---|---|
| train_data | directory | Contains training images X.npz and corresponding labels y.npz. | True |
| nb_epoch | string | Number of training epochs to perform during training. Defaults to 10. | False |
| bit_depth | string | Bit depth of the input images. This parameter is necessary for proper normalization. Defaults to 8. | False |

## Output Ports

| Name  | Type | Description |
|---|---|---|
| trained_model | directory | Contains the fully trained model with the architecture stored as model_arch.json and the weights as model_weights.h5. |


## Development

### Build the Docker Image

You need to install [Docker](https://docs.docker.com/engine/installation).

Clone the repository:

```bash
git clone https://github.com/platformstories/train-cnn
```

Then

```bash
cd train-cnn
docker build -t train-cnn .
```

### Try out locally

Create a container in interactive mode and mount the sample input under `/mnt/work/input/`:

```bash
docker run --rm -v full/path/to/sample-input:/mnt/work/input -it train-cnn
```

Then, within the container:

```bash
python /train-cnn.py
```

### Docker Hub

Login to Docker Hub:

```bash
docker login
```

Tag your image using your username and push it to DockerHub:

```bash
docker tag train-cnn yourusername/train-cnn
docker push yourusername/train-cnn
```

The image name should be the same as the image name under containerDescriptors in train-cnn.json.

Alternatively, you can link this repository to a [Docker automated build](https://docs.docker.com/docker-hub/builds/). Every time you push a change to the repository, the Docker image gets automatically updated.
### Register on GBDX

In a Python terminal:
```python
from gbdxtools import Interface
gbdx=Interface()
gbdx.task_registry.register(json_filename="train-cnn.json")
```

Note: If you change the task image, you need to reregister the task with a higher version number in order for the new image to take effect. Keep this in mind especially if you use Docker automated build.
