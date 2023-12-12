# Retinal Blood-Vessel Segmentation

![sample](https://github.com/rumbleFTW/retinal-segmentation/assets/85807431/0cb61689-8441-4d5a-a21b-0ae88ce7d251)

## **Running on local machine**

To set up the project, you will need to have Python 3.7 or above and Poetry installed on your machine.

Clone the repository to your local machine:

```bash
git clone https://github.com/rumbleFTW/retinal-segmentation.git
cd retinal-segmentation
```

Navigate to the project directory:

```bash
cd retinal-segmentation
```

Install the dependencies:

```
pip install -r requirements.txt
```

## **Training**

To train models, run the following command in the terminal:

```bash
python train.py --data /path/to/data --network att_unet --device cuda
```

**Arguments**

- `--data`: (required) The path to the data folder.
- `--network`: (required) The network type to use. Available options are `att_unet`, `unet` &, `seg_net`.
- `--device`: (optional) The device to use. Default is `cpu`.

**Example**

Here's an example of training an `AttentionUNet` Network on `cuda`:

```bash
python train.py --data ./data/DRIVE --network att_unet --device cuda
```

## **Inference**

To inference on images, run the following command in the terminal:

```bash
python infer.py --img /path/to/image.jpg --network unet --device cpu --checkpt /path/to/checkpoint.pth
```

**Arguments**

- `--img`: (required) the path to the image to test.
- `--network`: (required) the type of neural network to use for testing. Available options include att_unet, unet, and seg_net
- `--device`: (optional) the device to use for testing, default is "cpu".
- `--checkpt`: (required) the path to the checkpoint .pth file.

## **Testing**

To test models on a dataset, run the following command in the terminal:

```bash
python test.py --data /path/to/data --network att_unet --checkpt /path/to/checkpoint.pth
```

**Arguments**

- `--data`: (required) The path to the data folder.
- `--network`: (required) The network type to use. Available options are `att_unet`, `unet` &, `seg_net`.
- `--checkpt`: (required) the path to the checkpoint .pth file.

## **Gradio demo**

Place checkpoints in the directory `checkpts` under the name: `att_unet-weights.pth`, `seg_net-weights.pth`, `unet-weights.pth`

```python
  python gradio_demo.py
```
