# Retinal Blood-Vessel Segmentation

## **Running on local machine**

To use the models, clone the repository and navigate to the root directory.

```bash
git clone https://github.com/rumbleFTW/retinal-segmentation.git
cd retinal-segmentation
```

Install necessary modules

```bash
pip install -r requirements.txt
```

## **Training**

To train models, run the following command in the terminal:

```bash
python training.py --data /path/to/data --network att_unet --device cuda
```

**Arguments**

- `--data`: (required) The path to the data folder.
- `--network`: (required) The network type to use. Available options are `att_unet`, `unet`, and `seg_net`.
- `--device`: (optional) The device to use. Default is `cpu`.

**Example**

Here's an example of training an `AttentionUNet` Network on `cuda`:

```bash
python training.py --data ./DRIVE --network att_unet --device cuda
```
