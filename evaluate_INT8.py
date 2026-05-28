import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import utilsint as utils
from networks.soaint import get_model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, jaccard_score, classification_report
from torchinfo import summary
import onnxruntime as ort
import time
import psutil

process = psutil.Process(os.getpid())

# autocast compatibility
try:
    from torch.amp import autocast
    def AUTocast():
        return autocast("cuda")
except Exception:
    from torch.cuda.amp import autocast
    def AUTocast():
        return autocast()


# ---------------------------------------------------------
# Tile Dataset (Sliding Window)
# ---------------------------------------------------------
#class TileInferenceDataset(Dataset):
#    def __init__(self, fn, chip_size, stride, transform=None):
#        self.fn = fn
#        self.chip_size = chip_size
#        self.stride = stride
#        self.transform = transform
#
#        with rasterio.open(self.fn) as f:
#            self.height, self.width = f.height, f.width
#            self.num_channels = f.count
#
#        self.chip_coordinates = [
#            (y, x)
#            for y in list(range(0, self.height - chip_size, stride)) + [self.height - chip_size]
#            for x in list(range(0, self.width - chip_size, stride)) + [self.width - chip_size]
#        ]
#
#    def __getitem__(self, idx):
#        y, x = self.chip_coordinates[idx]
#        with rasterio.open(self.fn) as f:
#            img = np.moveaxis(
#                f.read(window=Window(x, y, self.chip_size, self.chip_size)), 0, -1
#            )
#        if self.transform:
#            img = self.transform(img)
#        return img, np.array((y, x))
#
#    def __len__(self):
#        return len(self.chip_coordinates)

class TileInferenceDataset(Dataset):
    def __init__(self, s1_fn, s2_fn, chip_size, stride, mode, transform=None):
        self.s1_fn = s1_fn
        self.s2_fn = s2_fn
        self.chip_size = chip_size
        self.stride = stride
        self.mode = mode
        self.transform = transform

        # Open S2 to get spatial size
        with rasterio.open(self.s2_fn) as f:
            self.height, self.width = f.height, f.width

        # Preload S1 and S2 full images into memory
        with rasterio.open(self.s1_fn) as f:
            self.s1_full = np.moveaxis(f.read(), 0, -1).astype(np.float32)

        with rasterio.open(self.s2_fn) as f:
            self.s2_full = np.moveaxis(f.read(), 0, -1).astype(np.float32)

        # Sliding window coordinates
        self.chip_coordinates = [
            (y, x)
            for y in list(range(0, self.height - chip_size, stride)) + [self.height - chip_size]
            for x in list(range(0, self.width - chip_size, stride)) + [self.width - chip_size]
        ]

    def __getitem__(self, idx):
        y, x = self.chip_coordinates[idx]

        # Extract chips
        if self.mode == "s1only":
            img = self.s1_full[y:y+self.chip_size, x:x+self.chip_size, :]  # (H,W,2)

        elif self.mode == "s2only":
            img = self.s2_full[y:y+self.chip_size, x:x+self.chip_size, :]  # (H,W,13)

        elif self.mode == "s1s2":
            s1_chip = self.s1_full[y:y+self.chip_size, x:x+self.chip_size, :]
            s2_chip = self.s2_full[y:y+self.chip_size, x:x+self.chip_size, :]
            img = np.concatenate([s1_chip, s2_chip], axis=-1)  # (H,W,15)

        if self.transform:
            img = self.transform(img)

        return img, np.array((y, x))

    def __len__(self):
        return len(self.chip_coordinates)


# ---------------------------------------------------------
# Normalization transform (mode-aware)
# ---------------------------------------------------------
def make_image_transform(mean, std):
    def _transform(img):
        img = (img - mean) / std
        img = np.moveaxis(img, -1, 0).astype(np.float32)
        return torch.from_numpy(img)
    return _transform



# ---------------------------------------------------------
# Inference + Evaluation
# ---------------------------------------------------------
def inference_and_eval(args, model, device):

#    # Select correct MEAN/STDs based on mode
#    if args.s_mode == "s1only":
#        IMAGE_MEANS = utils.IMAGE_MEANS[:2]
#        IMAGE_STDS  = utils.IMAGE_STDS[:2]
#
#    elif args.s_mode == "s2only":
#        IMAGE_MEANS = utils.IMAGE_MEANS[2:]   # all 13 S2 bands
#        IMAGE_STDS  = utils.IMAGE_STDS[2:]
#
#    elif args.s_mode == "s1s2":
#        IMAGE_MEANS = utils.IMAGE_MEANS
#        IMAGE_STDS  = utils.IMAGE_STDS
    if args.s_mode == "s1only":
        IMAGE_MEANS = mean[:2]
        IMAGE_STDS  = std[:2]
    
    elif args.s_mode == "s2only":
        IMAGE_MEANS = mean[2:]
        IMAGE_STDS  = std[2:]
    
    elif args.s_mode == "s1s2":
        IMAGE_MEANS = mean
        IMAGE_STDS  = std


    image_transform = make_image_transform(IMAGE_MEANS, IMAGE_STDS)


    # Only PyTorch models have .eval()
    if not isinstance(model, tuple):
        model.eval()

    df = pd.read_csv(args.test_list)

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.comparisons_dir, exist_ok=True)
    os.makedirs(args.metrics_log, exist_ok=True)

    y_true_all, y_pred_all = [], []

    # Performance accumulators
    total_tiles = 0
    latency_list = []
    memory_list = []
    memory_listGPU = []
    power_list = []
    max_gpu_mem = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        s1_fn, s2_fn = row["S1image_fn"], row["S2image_fn"]
        gt_fn = row.get("hr_label_fn", None)

        base_name = os.path.basename(s2_fn).replace(".tif", "")
        print(f"[{idx+1}/{len(df)}] Processing {base_name}...")

        # Read S1 + S2
        s1 = np.moveaxis(rasterio.open(s1_fn).read(), 0, 2).astype(np.float32)
        s2 = np.moveaxis(rasterio.open(s2_fn).read(), 0, 2).astype(np.float32)

        # Build input depending on mode
        if args.s_mode == "s1only":
            img_full = s1
        elif args.s_mode == "s2only":
            img_full = s2
        elif args.s_mode == "s1s2":
            img_full = np.concatenate([s1, s2], axis=-1)

        H, W, _ = img_full.shape

        if args.s_mode == "s1only":
            raster_for_sliding = s1_fn
        elif args.s_mode == "s2only":
            raster_for_sliding = s2_fn
        else:
            raster_for_sliding = s2_fn  # S1+S2 handled inside transform
        
        dataset = TileInferenceDataset(
            s1_fn=s1_fn,
            s2_fn=s2_fn,
            chip_size=args.chip_size,
            stride=args.chip_stride,
            mode=args.s_mode,
            transform=image_transform
        )



        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                num_workers=4, pin_memory=True)

        # Output accumulation
        output_tensor = torch.zeros((args.num_classes, H, W),
                                    dtype=torch.float32, device=device)
        count_tensor = torch.zeros((H, W),
                                   dtype=torch.float32, device=device)
        kernel = torch.ones((args.chip_size, args.chip_size),
                            dtype=torch.float32, device=device)

        # Inference loop
        for data, coords in dataloader:
            data = data.to(device, non_blocking=True)
            coords = coords.numpy()

            # Reset GPU memory stats
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            # Measure latency + memory
            start_time = time.time()
            mem_before = process.memory_info().rss / 1024
            mem_beforeGPU = torch.cuda.memory_allocated() / 1024 if device.type == "cuda" else 0
            cpu_before = psutil.cpu_percent(interval=None)

            # Run model
            with torch.no_grad():
                if isinstance(model, tuple):
                    backend = model[0]
                    if backend == "onnx":
                        _, sess, input_name = model
                        out = sess.run(None, {input_name: data.cpu().numpy()})[0]
                        out = torch.from_numpy(out).float().to(device)
                    elif backend == "openvino":
                        _, compiled, input_name = model
                        out = compiled([data])[compiled.outputs[0]]
                        out = torch.from_numpy(out).float().to(device)
                else:
                    with AUTocast():
                        out = model(data)

            end_time = time.time()

            # Latency
            batch_latency_ms = (end_time - start_time) * 1000
            latency_list.append(batch_latency_ms / data.shape[0])
            total_tiles += data.shape[0]

            # Memory
            mem_after = process.memory_info().rss / 1024
            memory_list.append(mem_after - mem_before)

            if device.type == "cuda":
                mem_afterGPU = torch.cuda.memory_allocated() / 1024
                memory_listGPU.append(mem_afterGPU - mem_beforeGPU)
                peak = torch.cuda.max_memory_allocated(device) / (1024)
                max_gpu_mem = max(max_gpu_mem, peak)

            cpu_after = psutil.cpu_percent(interval=None)
            cpu_usage = cpu_after - cpu_before
            estimated_power = (cpu_usage / 100.0) * 200
            power_list.append(estimated_power)
# -----------------------------------------------------------------------------------------------------------
            # Post-processing
            if isinstance(out, (tuple, list)):
                seg_logits = out[0]
            elif isinstance(out, dict):
                seg_logits = out.get("logits", list(out.values())[0])
            else:
                seg_logits = out

            preds_seg = F.softmax(seg_logits, dim=1)
            preds_seg *= kernel.unsqueeze(0).unsqueeze(0)

            for j in range(preds_seg.shape[0]):
                y, x = coords[j]
                output_tensor[:, y:y+args.chip_size, x:x+args.chip_size] += preds_seg[j]
                count_tensor[y:y+args.chip_size, x:x+args.chip_size] += kernel

        # Normalize accumulated predictions
        output_tensor = output_tensor / count_tensor.clamp(min=1e-6)
        output_hard = output_tensor.argmax(dim=0).byte().cpu().numpy()

        # Save GeoTIFF
        profile = rasterio.open(s2_fn).profile
        profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0)
        pred_fn = os.path.join(args.save_path, f"{base_name}_pred.tif")
        with rasterio.open(pred_fn, "w", **profile) as dst:
            dst.write(output_hard, 1)
            try:
                dst.write_colormap(1, utils.LABEL_IDX_COLORMAP)
            except Exception:
                pass
        print(f"Saved prediction: {pred_fn}")

        # Collect GT
        gt = None
        if gt_fn and os.path.exists(gt_fn):
            gt = rasterio.open(gt_fn).read(1)
            gt = np.take(utils.LABEL_CLASS_TO_IDX_MAP_GT, gt, mode="clip")
            valid_mask = (gt != 0)
            y_true_all.append(gt[valid_mask].flatten())
            y_pred_all.append(output_hard[valid_mask].flatten())

        # Visualization
        # S1 RGB
        if args.s_mode in ["s1only", "s1s2"]:
            s1_rgb = np.zeros((H, W, 3), dtype=np.float32)
            s1_rgb[...,0] = (s1[...,0] - s1[...,0].min()) / (s1[...,0].ptp() + 1e-6)
            s1_rgb[...,1] = (s1[...,1] - s1[...,1].min()) / (s1[...,1].ptp() + 1e-6)
        else:
            s1_rgb = np.zeros((H, W, 3), dtype=np.float32)

        # S2 RGB
        if args.s_mode in ["s2only", "s1s2"]:
            s2_rgb = np.stack([s2[...,3], s2[...,2], s2[...,1]], axis=-1)
            s2_rgb = np.clip(s2_rgb / 3000.0, 0, 1)
        else:
            s2_rgb = np.zeros((H, W, 3), dtype=np.float32)

        pred_rgb = utils.class_map_to_rgb(output_hard)
#        gt_rgb = utils.class_map_to_rgb(gt) if gt is not None else np.zeros_like(pred_rgb)
        gt_rgb = utils.class_map_to_rgb(gt)

    
    
        fig, axs = plt.subplots(1, 4, figsize=(18, 6))
        axs[0].imshow(s1_rgb); axs[0].set_title("S1 (VV/VH)")
        axs[1].imshow(s2_rgb); axs[1].set_title("S2 (RGB)")
        axs[2].imshow(pred_rgb); axs[2].set_title("Prediction")
        axs[3].imshow(gt_rgb); axs[3].set_title("Ground Truth")

        for ax in axs:
            ax.axis("off")
            
        patches = [mpatches.Patch(color=np.array(color)/255.0, label=utils.LABEL_NAMES[idx])
        for idx, color in utils.LABEL_IDX_COLORMAP.items() if idx != 0]
        fig.legend(handles=patches, loc="lower center", ncol=len(patches))
        
        vis_fn = os.path.join(args.comparisons_dir, f"{base_name}_comparison.png")
        plt.savefig(vis_fn, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved visualization: {vis_fn}")
# -----------------------------------------------------------------------------------------------------
    # === Performance Summary ===
    avg_latency = sum(latency_list) / len(latency_list)
    throughput = 1000.0 / avg_latency
    
    avg_memory = sum(memory_list) / len(memory_list)
    peak_memory = max(memory_list)
    
#    avg_memoryGPU = sum(memory_listGPU) / len(memory_listGPU)
#    peak_memoryGPU = max(memory_listGPU)
    
    # Default values (CPU inference or no CUDA)
    avg_memoryGPU = 0
    peak_memoryGPU = 0
    
    # GPU memory reporting only if CUDA is used
    if device.type == "cuda" and len(memory_listGPU) > 0:
        avg_memoryGPU = sum(memory_listGPU) / len(memory_listGPU)
        peak_memoryGPU = max(memory_listGPU)
        print(f"Average GPU Memory per Tile: {avg_memoryGPU:.2f} KB")
        print(f"Peak GPU Memory per Tile: {peak_memoryGPU:.2f} KB")
    else:
        print("GPU memory tracking skipped (CPU inference or no CUDA).")

    
    
    avg_power = sum(power_list) / len(power_list)
    peak_power = max(power_list)
    
    # Save everything in a dictionary
    metrics = {
        "avg_latency(ms)": avg_latency,
        "throughput_tiles(sec)": throughput,
        
        "avg_memory(KB)": avg_memory,
        "peak_memory(KB)": peak_memory,
        
        "avg_memory_gpu(KB)": avg_memoryGPU,
        "peak_memory_gpu(KB)": peak_memoryGPU,
        "Peak_GPU_Memory(KB)": max_gpu_mem,
        
        "avg_power(Watt)": avg_power,
        "peak_power(Watt)": peak_power,
    }

    print("\n=== Performance Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")
    with open(os.path.join(args.metrics_log, "performance.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.3f}\n")

                    
    # Final metrics
    if len(y_true_all) > 0:
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        mask = (y_true_all != 0)
        y_true_eval = y_true_all[mask]
        y_pred_eval = y_pred_all[mask]

        oa = accuracy_score(y_true_eval, y_pred_eval)
        kappa = cohen_kappa_score(y_true_eval, y_pred_eval)
        miou = jaccard_score(y_true_eval, y_pred_eval, average="macro")

        cm = confusion_matrix(y_true_eval, y_pred_eval, labels=list(range(1, args.num_classes)))
        report = classification_report(
            y_true_eval, y_pred_eval,
            labels=list(range(1, args.num_classes)),
            target_names=[utils.LABEL_NAMES[i] for i in range(1, args.num_classes)],
            digits=4
        )

        print("\n=== Final Metrics ===")
        print(f"OA: {oa:.4f}, Kappa: {kappa:.4f}, mIoU: {miou:.4f}")
        print("Confusion Matrix:\n", cm)
        print(report)

        with open(os.path.join(args.metrics_log, "metrics.txt"), "w") as f:
            f.write(f"OA: {oa:.4f}\nKappa: {kappa:.4f}\nmIoU: {miou:.4f}\n")
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(cm, separator=', '))
            f.write("\n\nClassification Report:\n")
            f.write(report)
    else:
        print("[Warning] No GT pixels collected; metrics not computed.")


# ---------------------------------------------------------
# ONNX + OpenVINO loaders
# ---------------------------------------------------------
def load_onnx_model(onnx_path):
    sess = ort.InferenceSession(
        onnx_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    return sess, input_name


from openvino.runtime import Core
def load_openvino_model(path):
    ie = Core()
    model = ie.read_model(path)
    compiled = ie.compile_model(model, "CPU")
    input_name = compiled.inputs[0].get_any_name()
    return compiled, input_name

import glob

def load_tif(path):
    with rasterio.open(path) as f:
        arr = f.read().astype(np.float32)
        return torch.from_numpy(arr)

def compute_mean_std(S1_DIR, S2_DIR, DEVICE="cpu"):
    s1_files = sorted(glob.glob(os.path.join(S1_DIR, "*.tif")))
    s2_files = sorted(glob.glob(os.path.join(S2_DIR, "*.tif")))

    assert len(s1_files) == len(s2_files) > 0, "Dataset mismatch"

    n_samples = 0
    channel_sum = None
    channel_sq_sum = None

    for i, (s1_path, s2_path) in enumerate(zip(s1_files, s2_files)):
        s1 = load_tif(s1_path).to(DEVICE)   # (2,H,W)
        s2 = load_tif(s2_path).to(DEVICE)   # (13,H,W)
        #print("S1 shape:", s1.shape, "S2 shape:", s2.shape, s2_path)
        
        img = torch.cat([s1, s2], dim=0)    # (15,H,W)
        C, H, W = img.shape
        pixels = H * W

        if channel_sum is None:
            channel_sum = torch.zeros(C, dtype=torch.float64, device=DEVICE)
            channel_sq_sum = torch.zeros(C, dtype=torch.float64, device=DEVICE)

        channel_sum += img.sum(dim=[1, 2])
        channel_sq_sum += (img ** 2).sum(dim=[1, 2])
        n_samples += pixels

    mean = channel_sum / n_samples
    std = torch.sqrt(channel_sq_sum / n_samples - mean ** 2)

    return mean.cpu().numpy(), std.cpu().numpy()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_list", type=str, default="./dataset/CSV_list/C_NYC-test.csv")
    parser.add_argument("--model_path", type=str, default="./log_INT8/unettS1/unet_epoch_30.pth")
    parser.add_argument("--model", type=str, default="unet", choices=["qnn", "unet"])
    parser.add_argument("--save_path", type=str, default="./log_INT8/unettS1/test/")
    parser.add_argument("--comparisons_dir", type=str, default="./log_INT8/unettS1/test/comparisons/")
    parser.add_argument("--metrics_log", type=str, default="./log_INT8/unettS1/test/")
    parser.add_argument("--chip_size", type=int, default=256)
    parser.add_argument("--chip_stride", type=int, default=256) # no overlap if chip_size = chip_stride, stride < size (e.g. 128 < 256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--gpu", type=str, default="0")

    # NEW: S1/S2 mode
    parser.add_argument("--s_mode", type=str, default="s1only",
                        choices=["s1s2", "s2only", "s1only"])

    args = parser.parse_args()
    
# ----------------------------------------------------------
    print("Computing dataset mean/std ...")
    mean, std = compute_mean_std(
        S1_DIR="./dataset/s1_validation/s1_validation",
        S2_DIR="./dataset/s2_validation/s2_validation",
        DEVICE="cpu"
    )
    print("Mean:", mean)
    print("Std:", std)


#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    print("Using device:", device)
    device = torch.device("cpu")
    print("Using CPU")
    
    # Set correct in_channels
    if args.s_mode == "s1only":
        args.in_channels = 2
    elif args.s_mode == "s2only":
        args.in_channels = 13
    elif args.s_mode == "s1s2":
        args.in_channels = 15

    # Load model
    if args.model_path.endswith(".xml"):
        print("Using OpenVINO INT8 model")
        compiled, input_name = load_openvino_model(args.model_path)
        model = ("openvino", compiled, input_name)

    elif args.model_path.endswith(".onnx"):
        print("Using ONNX model")
        sess, input_name = load_onnx_model(args.model_path)
        model = ("onnx", sess, input_name)

    else:
        print("Using PyTorch model")
        model = get_model(args.model, num_classes=args.num_classes,
                          in_channels=args.in_channels, img_size=args.chip_size)
        state_dict = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.to(device)

    inference_and_eval(args, model, device)
