from utils.parser import parse_yaml
from models.centernet import CenterNet
import argparse
import torch
import thop

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./config/model/base_model.yaml")
    parser.add_argument('--device', default='cuda', help='device to use for testing')
    parser.add_argument('--profiler', action='store_true', help='display details of model performance using pytorch profiler')
    return parser.parse_args()

def count_model_params(model):
    return sum(p.numel() for p in model.parameters())

def get_model_memory(model):
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    return mem_params + mem_bufs # in bytes

def get_model_flops(model, batch_size=1, input_size=[512, 512], device='cuda'):

    model.eval()
    model = model.to(device)
    input = torch.rand(batch_size, 3, input_size[0], input_size[1]).to(device)

    macs, params = thop.profile(model, inputs=(input,), verbose=False)
    return macs

@torch.inference_mode()
def benchmark(model, batch_size=1, input_size=[512, 512], times=100, device='cuda', profiler=False):
    model.eval()
    model = model.to(device)
    
    input = torch.rand(batch_size, 3, input_size[0], input_size[1]).to(device)
    for _ in range(10): model(input) #gpu warmup

    if profiler:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_flops=True) as prof:
            model(input)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    avg_time = 0
    for _ in range(0, times):
        input = torch.rand(batch_size,3,input_size[0],input_size[1]).to(device)
        start.record()
        model(input)
        end.record()
        torch.cuda.synchronize()
        avg_time += start.elapsed_time(end)

    avg_time /= times
    return avg_time
    

def profile(args, option):
    # Load model
    model = CenterNet(option).to(args.device)
    model.eval()
    
    img_w = option["MODEL"]["INPUT_SIZE"]["WIDTH"]
    img_h = option["MODEL"]["INPUT_SIZE"]["HEIGHT"]
    
    print(f'gpu: {torch.cuda.get_device_name(0)}, torch: {torch.__version__}, cuda: {torch.version.cuda}, cudnn: {torch.backends.cudnn.version()}')
    print(f"#params : {count_model_params(model)/1e6:.1f} (M), mem: {get_model_memory(model)/1e6:.1f} (Mb), macs: {get_model_flops(model, input_size=[img_h, img_w])/1e9:.1f} (G)" )
    print(f"Latency (ms): {benchmark(model, input_size=[img_h, img_w], times=100, profiler=args.profiler):.3f}")
    
if __name__ == "__main__":
    args = parse_args()
    
    # Parse yaml files
    model_option = parse_yaml(args.model)
    profile(args, model_option)
