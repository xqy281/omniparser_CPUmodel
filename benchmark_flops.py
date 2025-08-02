# benchmark_flops.py

import time
import torch

def benchmark_flops(device_name: str, matrix_size: int = 4096, iterations: int = 100):
    """
    在指定的设备上运行矩阵乘法基准测试，并报告有效GFLOPS。

    Args:
        device_name (str): 设备名称，如 'cpu' 或 'cuda'。
        matrix_size (int): 用于测试的方阵大小。
        iterations (int): 测试迭代次数。
    """
    print("-" * 40)
    print(f"开始在设备 '{device_name}' 上进行基准测试...")
    print(f"矩阵大小: {matrix_size}x{matrix_size}, 迭代次数: {iterations}")

    try:
        # 设置设备
        device = torch.device(device_name)

        # 创建测试用的随机矩阵
        a = torch.randn(matrix_size, matrix_size, device=device)
        b = torch.randn(matrix_size, matrix_size, device=device)

        # 预热运行 (Warm-up)
        # 第一次运算通常较慢，需要预热以获得准确的计时
        print("正在预热...")
        for _ in range(5):
            c = torch.matmul(a, b)
        
        # 如果是GPU，等待所有CUDA核心完成工作
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # 开始计时
        print("开始正式测试...")
        start_time = time.time()

        # 执行核心计算
        for _ in range(iterations):
            c = torch.matmul(a, b)

        # 再次同步，确保所有计算都已完成
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()

        # 计算性能
        duration = end_time - start_time
        
        # 矩阵乘法 (N*N) x (N*N) 的计算量大约是 2 * N^3
        # (N^3次乘法和N^3次加法)
        total_flops = 2 * (matrix_size ** 3) * iterations
        
        # GFLOPS = 十亿次浮点运算 / 秒
        gflops = (total_flops / 1e9) / duration

        print("\n--- 测试结果 ---")
        print(f"总耗时: {duration:.4f} 秒")
        print(f"有效性能: {gflops:.2f} GFLOPS (每秒十亿次浮点运算)")
        print("-" * 40)

    except Exception as e:
        print(f"\n在设备 '{device_name}' 上测试失败: {e}")
        print("请确保PyTorch和相关驱动已正确安装。")
        print("-" * 40)


if __name__ == "__main__":
    # --- 在CPU上测试 ---
    benchmark_flops('cpu')

    # --- 检查是否有可用的GPU并测试 ---
    if torch.cuda.is_available():
        benchmark_flops('cuda')
    else:
        print("\n未检测到可用的NVIDIA GPU (CUDA)。跳过GPU测试。")