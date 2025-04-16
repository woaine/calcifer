import argparse
import os

from ultralytics import YOLO

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='yolo11m-pose.pt', help='Initial weights path. File must be in the config folder.')
    parser.add_argument('--data', default='sftl54.yaml', help='Path to data file. File must be in the config folder.')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=160, help='Image size')
    parser.add_argument('--device', default='0', help='CUDA device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--resume', type=bool, nargs='?', const=True, default=False, help='Resume most recent training')
    parser.add_argument('--optimizer', default=None, help='SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto')
    parser.add_argument('--lrf', type=float, default=None, help='Final learning rate (lr0 * lrf)')
    parser.add_argument('--weight-decay', type=float, default=None, help='Optimizer weight decay')
    opt = parser.parse_args()

    current_dir = os.path.dirname(__file__)
    config_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'config'))

    params = {
        'data': os.path.join(config_dir, opt.data),
        'epochs': opt.epochs,
        'batch': opt.batch,
        'imgsz': opt.imgsz,
        'device': opt.device,
        'resume': opt.resume,
    }
    if opt.optimizer:
        params['optimizer'] = opt.optimizer
    if opt.lrf:
        params['lrf'] = opt.lrf
    if opt.weight_decay:
        params['weight_decay'] = opt.weight_decay

    model = YOLO(os.path.join(config_dir, opt.weights))
    model.train(**params)