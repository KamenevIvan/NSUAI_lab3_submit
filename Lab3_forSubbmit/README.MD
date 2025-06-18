# Detecting people in drone images using YOLOv5s
The project is aimed at detecting people in drone images. A distinctive feature is the small size of the person in the picture, as the photo is taken from a great height.

## Requirements
Firstly you want to download all dependencies:

```
pip install -r requirements.txt
```

## Detection
You can pass the path to a specific file or folder to the script, and the result will be one or more images in the output folder.

```
python detect.py path/to/imageOrFolder --output_path path/to/OutputFolder --model_weights path/to/castom.pt --save_txt
```

## Arguments
You can pass the next arguments
| Argument           | Description                                                 |
| ------------------ | ----------------------------------------------------------- |
| `input_path`       | Path to input file or folder                                |
| `--output_path`    | Path to output files. Has default                           |
| `--model_weights`  | Path to model weights. Has default                          |
| `--confidence`     | Confidence threshold. Has default                           |
| `--save_txt`       | If you want to save text results                            |

## Output
Each input image generates:
    .jpg file in your output dir with boundig boxes
    .txt file in your output dir with boundig boxes coordinates and confidence score (if --save_txt on)