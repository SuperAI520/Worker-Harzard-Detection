import sys
sys.path.insert(0, './yolor')
import os
import time
import copy
import argparse
import torch.utils.data
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import DataParallel
from datetime import datetime
from yolor.models.models import *
from yolor.utils.datasets import create_dataloader
from yolor.utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, fitness_p, fitness_r, fitness_ap50, fitness_ap, fitness_f, strip_optimizer, get_latest_run,\
    check_dataset, check_file, check_git_status, check_img_size, print_mutation, set_logging
import constants
from tqdm import tqdm

class QuantizedDarknet(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedDarknet, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x

def calibrate_model(model, loader, device=torch.device("cpu")):

    model.to(device)
    model.eval()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')

    s_time = time.time()
    with torch.no_grad():
        for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(loader, desc=s)):
            img = img.to(device, non_blocking=True)
            img = img.float()
            # img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)
            _, _ = model(img)
    e_time = time.time()
    print(f'>>>>>>>>>>>>>    calibration time   {(e_time-s_time)}')
    # for inputs, labels in loader:
    #     inputs = inputs.to(device)
    #     labels = labels.to(device)
    #     _ = model(inputs)


def model_equivalence(model_1, model_2, device, rtol=1e-05, atol=1e-08, num_tests=100, input_size=(1,3,32,32)):

    model_1.to(device)
    model_2.to(device)

    for _ in range(num_tests):
        x = torch.rand(size=input_size).to(device)
        y1 = model_1(x).detach().cpu().numpy()
        y2 = model_2(x).detach().cpu().numpy()
        if np.allclose(a=y1, b=y2, rtol=rtol, atol=atol, equal_nan=False) == False:
            print("Model equivalence test sample failed: ")
            print(y1)
            print(y2)
            return False

    return True


def load_torchscript_model(model_filepath, device):

    model = torch.jit.load(model_filepath, map_location=device)

    return model


def measure_inference_latency(model,
                              device,
                              input_size=(1, 3, 112, 112),
                              num_samples=100,
                              num_warmups=10):

    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave


def save_torchscript_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    # torch.jit.save(torch.jit.script(model), model_filepath)
    torch.save(model.state_dict(), model_filepath)
    # torch.save(model, model_filepath)


def main(args):
    yaml_path = './dataset/custom.yaml'
    with open(yaml_path) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
    train_path = data_dict['train']
    nc, names = int(data_dict['nc']), data_dict['names']  # number classes, names
    # Image sizes
    gs = 64 #int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in [1280, 1280]]  # verify imgsz are gs-multiples

    hyp_path = './dataset/hyp.scratch.1280.yaml'
    with open(hyp_path) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
        if 'box' not in hyp:
            warn('Compatibility: %s missing "box" which was renamed from "giou" in %s' %
                 (opt.hyp, 'https://github.com/ultralytics/yolov5/pull/1120'))
            hyp['box'] = hyp.pop('giou')

    batch_size = 8
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, args,
                                                hyp=hyp, augment=True, cache=False, rect=False,
                                                rank=-1, world_size=1, workers=8)

    # torch support only cpu option for the quantization
    device = torch.device("cpu")
    # device = torch.device("cuda:0")
    cuda_device = torch.device("cuda:0")
    cfg = constants.YOLOR_CONFIG
    img_size = constants.YOLOR_IMG_SIZE
    model = Darknet(cfg, img_size)

    model_dir = "models"
    quantized_model_filename = "yolor_quantized.pt"
    quantized_model_filepath = os.path.join(model_dir, quantized_model_filename)

    # load pretrained model weight
    model.load_state_dict(torch.load(constants.DETECTION_MODEL_PATH, map_location=device)['model'])
    model.to(device)

    calibrate_model(model=model, loader=dataloader)
    fused_model = copy.deepcopy(model)

    model.eval()
    fused_model.eval()  
    for module_name, module in fused_model.named_children():
        if "output" in module_name:
            continue
        for basic_block_name, basic_block in module.named_children():
            for sub_block_name, sub_block in basic_block.named_children():
                if sub_block_name in ['BatchNorm2d']:
                    torch.quantization.fuse_modules(basic_block, [["Conv2d", "BatchNorm2d"]], inplace=True)
                    break
            
    fused_model.eval()  
                

    # Print FP32 model.
    # print(model)
    # Print fused model.
    # print(fused_model)

    # Prepare the model for static quantization. This inserts observers in
    # the model that will observe activation tensors during calibration.
    
    quantized_model = QuantizedDarknet(model_fp32=fused_model)

    # Select quantization schemes from
    # https://pytorch.org/docs/stable/quantization-support.html
    quantization_config = torch.quantization.get_default_qconfig("fbgemm")

    quantized_model.qconfig = quantization_config

    # Print quantization configurations
    # print(quantized_model.qconfig)

    torch.quantization.prepare(quantized_model)

    # Use training data for calibration.
    # transform = transforms.Compose([
    #     transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    # ])

    # trainset = CASIAWebFace(args.train_root, args.train_file_list, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)

    calibrate_model(model=quantized_model, loader=dataloader)

    quantized_model = torch.quantization.convert(quantized_model)
    quantized_model.eval()

    # Print quantized model.
    # print(quantized_model)
    
    # Save quantized model.
    save_torchscript_model(model=quantized_model, model_dir=model_dir, model_filename=quantized_model_filename)

    # Load quantized model.
    # quantized_jit_model = load_torchscript_model(model_filepath=quantized_model_filepath, device=device)

    # _, fp32_eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=cpu_device, criterion=None)
    # _, int8_eval_accuracy = evaluate_model(model=quantized_jit_model, test_loader=test_loader, device=cpu_device,
    #                                        criterion=None)

    # Skip this assertion since the values might deviate a lot.
    # assert model_equivalence(model_1=model, model_2=quantized_jit_model, device=cpu_device, rtol=1e-01, atol=1e-02, num_tests=100, input_size=(1,3,32,32)), "Quantized model deviates from the original model too much!"

    # print("FP32 evaluation accuracy: {:.3f}".format(fp32_eval_accuracy))
    # print("INT8 evaluation accuracy: {:.3f}".format(int8_eval_accuracy))

    """
    fp32_cpu_inference_latency = measure_inference_latency(model=model, device=device, input_size=(1, 3, 112, 112),
                                                           num_samples=100)
    int8_cpu_inference_latency = measure_inference_latency(model=quantized_model, device=device,
                                                           input_size=(1, 3, 112, 112), num_samples=100)
    int8_jit_cpu_inference_latency = measure_inference_latency(model=quantized_jit_model, device=device,
                                                               input_size=(1, 3, 112, 112), num_samples=100)
    # fp32_gpu_inference_latency = measure_inference_latency(model=model, device=cuda_device, input_size=(1, 3, 112, 112),
    #                                                        num_samples=100)

    print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(fp32_cpu_inference_latency * 1000))
    # print("FP32 CUDA Inference Latency: {:.2f} ms / sample".format(fp32_gpu_inference_latency * 1000))
    print("INT8 CPU Inference Latency: {:.2f} ms / sample".format(int8_cpu_inference_latency * 1000))
    print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(int8_jit_cpu_inference_latency * 1000))
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch - Quantization Part')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    args = parser.parse_args()

    main(args)
