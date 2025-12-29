"""
主验证脚本 - 顺序运行所有验证
=====================================

运行顺序:
1. debug_data_pipeline.py
2. debug_model_forward.py
3. debug_loss_and_gradients.py
4. debug_training_dynamics.py

总运行时间: 约30-42分钟
"""

import sys
import os
import pathlib
import subprocess
import json
import time
from datetime import datetime

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
os.chdir(ROOT_DIR)


def print_section(title):
    """打印分隔线"""
    print(f"\n{'#'*70}")
    print(f"# {title}")
    print(f"{'#'*70}\n")


def run_script(script_name, description):
    """运行单个验证脚本"""
    print_section(f"运行: {script_name}")
    print(f"描述: {description}\n")

    script_path = os.path.join('debug_scripts', script_name)

    # 检查脚本存在
    if not os.path.exists(script_path):
        print(f"❌ 错误: 脚本不存在: {script_path}")
        return False, None

    # 运行脚本
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=1200  # 20分钟超时
        )

        elapsed = time.time() - start_time

        # 打印输出
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        # 检查返回码
        success = result.returncode == 0

        print(f"\n{'='*70}")
        if success:
            print(f"✓ {script_name} 完成 (耗时: {elapsed:.1f}秒)")
        else:
            print(f"❌ {script_name} 失败 (返回码: {result.returncode})")
        print(f"{'='*70}\n")

        return success, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"\n❌ 超时: {script_name} (>{elapsed:.1f}秒)")
        return False, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ 异常: {e}")
        import traceback
        traceback.print_exc()
        return False, elapsed


def load_report(report_path):
    """加载验证报告"""
    if not os.path.exists(report_path):
        return None

    try:
        with open(report_path, 'r') as f:
            return json.load(f)
    except:
        return None


def generate_summary_report(results):
    """生成总结报告"""
    print_section("验证总结报告")

    # 统计
    total_scripts = len(results)
    passed = sum(1 for r in results if r['success'])
    failed = total_scripts - passed
    total_time = sum(r['elapsed'] for r in results if r['elapsed'] is not None)

    print(f"总脚本数: {total_scripts}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)\n")

    # 详细结果
    print(f"{'='*70}")
    print(f"{'脚本':<40} {'状态':<10} {'耗时':<15}")
    print(f"{'='*70}")

    for r in results:
        status = "✓ PASS" if r['success'] else "❌ FAIL"
        elapsed = f"{r['elapsed']:.1f}s" if r['elapsed'] is not None else "N/A"
        print(f"{r['name']:<40} {status:<10} {elapsed:<15}")

    print(f"{'='*70}\n")

    # 加载各报告的详细信息
    print("详细问题:")
    print(f"{'='*70}")

    all_issues_count = 0

    for r in results:
        report = load_report(r['report_path'])
        if report:
            issues = report.get('issues', [])
            if issues:
                print(f"\n{r['name']}:")
                for i, issue in enumerate(issues, 1):
                    print(f"  {i}. {issue}")
                all_issues_count += len(issues)

    if all_issues_count == 0:
        print("\n✓ 无问题!")
    else:
        print(f"\n总计问题数: {all_issues_count}")

    print(f"{'='*70}\n")

    # 生成综合报告JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_scripts': total_scripts,
        'passed': passed,
        'failed': failed,
        'total_time_seconds': total_time,
        'results': results,
        'all_issues_count': all_issues_count
    }

    summary_path = 'debug_outputs/validation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"综合报告已保存: {summary_path}")

    # 最终判断
    print(f"\n{'='*70}")
    if failed == 0 and all_issues_count == 0:
        print("✓✓✓ 所有验证通过! 可以启动完整训练! ✓✓✓")
        print(f"{'='*70}\n")
        return True
    else:
        print("❌ 存在失败或问题, 请检查后再启动训练")
        print(f"{'='*70}\n")
        return False


def main():
    """主函数"""
    print_section("Residual VQ-VAE 验证套件")

    print("本脚本将顺序运行以下验证:")
    print("  1. debug_data_pipeline.py (数据流验证)")
    print("  2. debug_model_forward.py (模型前向验证)")
    print("  3. debug_loss_and_gradients.py (Loss和梯度验证)")
    print("  4. debug_training_dynamics.py (训练动态验证)")
    print("\n预计总耗时: 30-42分钟\n")

    # 确认 (检查命令行参数)
    import sys
    if '--yes' in sys.argv or '-y' in sys.argv:
        print("是否继续? [y/N]: y (自动确认)")
    else:
        response = input("是否继续? [y/N]: ")
        if response.lower() != 'y':
            print("取消验证")
            return

    # 验证脚本列表
    scripts = [
        {
            'name': 'debug_data_pipeline.py',
            'description': '数据流验证 (Dataset, DataLoader, 归一化)',
            'report_path': 'debug_outputs/data_pipeline_report.json'
        },
        {
            'name': 'debug_model_forward.py',
            'description': '模型前向传播验证 (Prophet, Obs, Residual, Decoder)',
            'report_path': 'debug_outputs/model_forward_report.json'
        },
        {
            'name': 'debug_loss_and_gradients.py',
            'description': 'Loss和梯度验证 (Recon, Sparse, 梯度流)',
            'report_path': 'debug_outputs/loss_and_gradients_report.json'
        },
        {
            'name': 'debug_training_dynamics.py',
            'description': '训练动态验证 (参数更新, Loss趋势, Checkpoint)',
            'report_path': 'debug_outputs/training_dynamics_report.json'
        }
    ]

    # 运行所有脚本
    results = []
    overall_start = time.time()

    for script in scripts:
        success, elapsed = run_script(script['name'], script['description'])

        results.append({
            'name': script['name'],
            'description': script['description'],
            'report_path': script['report_path'],
            'success': success,
            'elapsed': elapsed
        })

        # 如果失败, 询问是否继续
        if not success:
            print(f"\n⚠ {script['name']} 失败!")
            if '--yes' in sys.argv or '-y' in sys.argv:
                print("是否继续运行下一个脚本? [y/N]: y (自动确认)")
            else:
                response = input("是否继续运行下一个脚本? [y/N]: ")
                if response.lower() != 'y':
                    print("中止验证")
                    break

    overall_elapsed = time.time() - overall_start

    # 生成总结报告
    all_passed = generate_summary_report(results)

    print(f"\n总耗时: {overall_elapsed:.1f}秒 ({overall_elapsed/60:.1f}分钟)")

    # 下一步建议
    if all_passed:
        print("\n" + "="*70)
        print("下一步:")
        print("  1. 查看可视化结果: debug_outputs/")
        print("  2. 启动完整训练:")
        print("     cd /root/code/zpw/tac/UniT-Align")
        print("     bash launch_residual_training.sh")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("下一步:")
        print("  1. 查看详细报告: debug_outputs/*_report.json")
        print("  2. 修复问题后重新运行验证")
        print("  3. 或单独运行失败的脚本:")
        for r in results:
            if not r['success']:
                print(f"     python debug_scripts/{r['name']}")
        print("="*70)

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
