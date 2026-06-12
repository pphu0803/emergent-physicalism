"""
可视化模块

生成模拟结果的图表，用于论文和报告。
"""

import numpy as np
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def plot_results(
    results: Dict[str, Any],
    save_path: Optional[str] = None,
    title: str = "Political Economy Simulation",
) -> None:
    """绘制单个模拟结果的完整面板图"""
    ts = results.get('time_series', {})
    if not ts:
        print("  No time series data to plot")
        return

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    ticks = ts['tick']

    # 1. 基尼系数（不平等趋势）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ticks, ts['gini'], color='#e74c3c', linewidth=1.5)
    ax1.set_title('Gini Coefficient (Inequality)', fontsize=11)
    ax1.set_xlabel('Tick')
    ax1.set_ylabel('Gini')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # 2. 人口变化
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ticks, ts['population'], color='#2ecc71', linewidth=1.5)
    ax2.set_title('Population', fontsize=11)
    ax2.set_xlabel('Tick')
    ax2.set_ylabel('Alive Agents')
    ax2.grid(True, alpha=0.3)

    # 3. 财富份额分布
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.stackplot(
        ticks,
        ts['bottom50_share'],
        [t - b for t, b in zip(ts['top10_share'], ts['bottom50_share'])],
        [t for t in ts['top10_share']],
        labels=['Bottom 50%', '50-90%', 'Top 10%'],
        colors=['#3498db', '#f39c12', '#e74c3c'],
        alpha=0.7,
    )
    ax3.set_title('Wealth Distribution', fontsize=11)
    ax3.set_xlabel('Tick')
    ax3.set_ylabel('Share')
    ax3.legend(loc='center right', fontsize=8)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    # 4. 平均与中位财富
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(ticks, ts['mean_wealth'], color='#9b59b6', linewidth=1.5, label='Mean')
    ax4.plot(ticks, ts['median_wealth'], color='#1abc9c', linewidth=1.5, label='Median')
    ax4.set_title('Mean vs Median Wealth', fontsize=11)
    ax4.set_xlabel('Tick')
    ax4.set_ylabel('Wealth')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. 雇主数量
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(ticks, ts['num_employers'], color='#e67e22', linewidth=1.5)
    ax5.set_title('Number of Employers (Capital Concentration)', fontsize=11)
    ax5.set_xlabel('Tick')
    ax5.set_ylabel('Employers')
    ax5.grid(True, alpha=0.3)

    # 6. 生产总量
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(ticks, ts['total_production'], color='#27ae60', linewidth=1.5)
    ax6.set_title('Total Production', fontsize=11)
    ax6.set_xlabel('Tick')
    ax6.set_ylabel('Output')
    ax6.grid(True, alpha=0.3)

    # 7. HHI指数
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(ticks, ts['hhi'], color='#c0392b', linewidth=1.5)
    ax7.set_title('HHI Index (Market Concentration)', fontsize=11)
    ax7.set_xlabel('Tick')
    ax7.set_ylabel('HHI')
    ax7.set_ylim(0, 1)
    ax7.grid(True, alpha=0.3)

    # 8. Top-1% 份额
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(ticks, ts['top1_share'], color='#8e44ad', linewidth=1.5)
    ax8.set_title('Top 1% Wealth Share', fontsize=11)
    ax8.set_xlabel('Tick')
    ax8.set_ylabel('Share')
    ax8.grid(True, alpha=0.3)

    # 9. 平均工具水平（资本化程度）
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(ticks, ts['avg_tools'], color='#2c3e50', linewidth=1.5)
    ax9.set_title('Average Tool Level (Capitalization)', fontsize=11)
    ax9.set_xlabel('Tick')
    ax9.set_ylabel('Tools')
    ax9.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved to {save_path}")

    plt.close(fig)


def plot_comparison(
    results_list: List[Dict[str, Any]],
    labels: List[str],
    metrics: List[str] = None,
    save_path: Optional[str] = None,
    title: str = "Institutional Comparison",
) -> None:
    """对比多个模拟结果（不同制度规则）"""
    if metrics is None:
        metrics = ['gini', 'population', 'hhi', 'avg_tools', 'bottom50_share']

    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 3 * n_metrics), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, len(results_list)))

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for j, (results, label) in enumerate(zip(results_list, labels)):
            ts = results.get('time_series', {})
            if metric in ts:
                ax.plot(ts['tick'], ts[metric], color=colors[j], label=label, linewidth=1.2)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)

    axes[-1].set_xlabel('Tick')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Comparison plot saved to {save_path}")

    plt.close(fig)
