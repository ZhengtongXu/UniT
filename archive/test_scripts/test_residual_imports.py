"""
快速验证脚本: 测试所有新文件是否可正常导入

运行: python test_residual_imports.py
"""

import sys
import traceback

def test_import(module_path, description):
    """测试单个模块导入"""
    try:
        exec(f"import {module_path}")
        print(f"✅ {description}: {module_path}")
        return True
    except Exception as e:
        print(f"❌ {description}: {module_path}")
        print(f"   错误: {str(e)}")
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("Residual Tactile Learning - 导入测试")
    print("="*60 + "\n")

    tests = [
        ("UniT.dataset.residual_tactile_dataset", "数据集类"),
        ("UniT.taming.models.residual_vqmodel", "模型定义"),
        ("UniT.workspace.train_residual_vqvae_workspace", "训练Workspace"),
        ("UniT.model.residual_feature_extractor", "特征提取器"),
    ]

    results = []
    for module, desc in tests:
        result = test_import(module, desc)
        results.append((desc, result))

    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)

    success_count = sum(1 for _, result in results if result)
    total_count = len(results)

    for desc, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {desc}")

    print(f"\n总计: {success_count}/{total_count} 通过")

    if success_count == total_count:
        print("\n🎉 所有模块导入成功!")
        print("\n下一步:")
        print("  1. 运行数据转换:")
        print("     bash UniT/scripts/train_residual_quickstart.sh")
        print("\n  2. 或手动运行:")
        print("     python UniT/dataset/convert_residual_data.py --help")
        return 0
    else:
        print("\n⚠️  部分模块导入失败,请检查依赖和代码")
        return 1

if __name__ == "__main__":
    sys.exit(main())
