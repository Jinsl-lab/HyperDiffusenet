"""
HyperDiffuseNet 安装测试脚本
该脚本用于测试HyperDiffuseNet所需的所有依赖是否正确安装，并验证基本功能是否正常。
"""

import sys
import platform
import importlib
import numpy as np
import torch


def check_module(module_name):
    """
    检查指定模块是否已安装并获取其版本

    参数:
    module_name: str, 要检查的模块名称

    返回:
    (bool, str): 安装状态和版本信息
    """
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, '__version__'):
            return True, module.__version__
        else:
            return True, "未知版本"
    except ImportError:
        return False, None


def main():
    """主函数，执行安装测试流程"""
    print("HyperDiffuseNet 安装测试")
    print("=" * 40)

    # 检查Python版本
    python_version = platform.python_version()
    python_ok = int(python_version.split('.')[0]) >= 3 and int(python_version.split('.')[1]) >= 7
    print(f"Python版本: {python_version} {'✓' if python_ok else '✗'}")

    # 定义必需的包
    required_packages = [
        "torch",
        "numpy",
        "scipy",
        "scanpy",
        "numba",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "networkx",
        "python-louvain"
    ]

    # 检查每个包
    all_packages_installed = True
    print("\n检查必需的包:")
    for package in required_packages:
        installed, version = check_module(package)
        status = "✓" if installed else "✗"
        version_str = f"v{version}" if installed else "未安装"
        print(f"  {package:15s}: {version_str:15s} {status}")
        if not installed:
            all_packages_installed = False

    # 检查PyTorch GPU支持
    if check_module("torch")[0]:
        print("\nPyTorch配置:")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  GPU设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    设备 {i}: {torch.cuda.get_device_name(i)}")

    # 尝试导入HyperDiffuseNet模块
    print("\n检查HyperDiffuseNet模块:")
    modules_to_check = [
        "lorentzian_helper",
        "wrapped_normal",
        "layers",
        "preprocess",
        "tsne_helper",
        "single_cell_tools",
        "HyperDiffuseNet_HyperSpatial_Attention"
    ]

    all_modules_ok = True
    for module in modules_to_check:
        try:
            importlib.import_module(module)
            print(f"  {module:40s} ✓")
        except ImportError as e:
            all_modules_ok = False
            print(f"  {module:40s} ✗ ({str(e)})")

    # 基本功能测试
    if all_packages_installed and all_modules_ok:
        print("\n运行简单功能测试:")
        try:
            # 创建小型随机数据集
            data = np.random.rand(100, 20)
            spatial = np.random.rand(100, 2)

            # 测试双曲空间函数
            from lorentzian_helper import lorentz2poincare, poincare2lorentz
            x = torch.rand(10, 3)
            y = lorentz2poincare(x)
            z = poincare2lorentz(y)

            print("  基本功能测试通过 ✓")

        except Exception as e:
            print(f"  基本功能测试失败: {str(e)}")

    # 总结
    print("\n安装测试总结:")
    if python_ok and all_packages_installed and all_modules_ok:
        print("✅ HyperDiffuseNet安装成功!")
        print("您可以开始使用该库。")
    else:
        print("❌ HyperDiffuseNet安装不完整。")
        print("请修复上述问题后再使用该库。")


if __name__ == "__main__":
    main()