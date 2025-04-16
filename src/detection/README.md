# Detection Module Instructions

## Setup Instructions

1.**Download YOLO PT File**

    Download the YOLO`.pt` file from this [LINK](https://docs.ultralytics.com/tasks/pose/#models "Ultralytics") and place it in the `src/models/yolo` folder.

2.**Prepare `sftl54.yaml` File**

    Ensure the`sftl54.yaml` file is prepared. Refer to the `README.md` file located in the `facial_dataset` folder inside the `data/external` directory for more details.

## Training Instructions to train the model, use the following command:

```bash
python train.py --weights yolo11m-pose.pt --epochs 240 --optimizer SGD --lrf 1e-5 --weight-decay 5e-3 2>&1 | tee-a ../../reports/yolo_training_results.txt
```

> **Note:**

> - Running the script above requires executing it from this directory.

> - If running from the root directory, use the following command:

> ```bash
>> python src/detection/train.py --weights yolo11m-pose.pt --epochs 240 --optimizer SGD --lrf 1e-5 --weight-decay 5e-3 2>&1 | tee-areports/yolo_training_results.txt
> ```
