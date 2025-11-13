"""
================================================================
可视化工具函数
提供训练结果、回放结果和鲁棒性测试的可视化
================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Union, List

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10


def visualize_training_results(network, 
                              xi_replayed: np.ndarray,
                              eval_results: Dict,
                              save_path: Optional[str] = None,
                              title_suffix: str = "",
                              show_images: bool = False,
                              sequence_index: int = 0):
    """
    可视化训练和回放结果（9宫格布局）
    
    参数:
        network: 训练好的网络对象
        xi_replayed: 回放序列
        eval_results: 评估结果
        save_path: 保存路径
        title_suffix: 标题后缀
        show_images: 是否显示图片
        sequence_index: 显示哪个序列（多序列模式）
    """
    fig = plt.figure(figsize=(14, 9))
    
    num_epochs = len(network.mu_history)
    max_steps = xi_replayed.shape[0]
    
    # 确定使用哪个训练序列
    if hasattr(network, 'training_sequences') and len(network.training_sequences) > 0:
        training_seq = network.training_sequences[sequence_index]
        num_seq = len(network.training_sequences)
    else:
        training_seq = network.training_sequence
        num_seq = 1
    
    # 提取评估结果
    if 'evaluation_mode' in eval_results:
        if eval_results['evaluation_mode'] == 'full_sequence_matching':
            recall_acc = eval_results['recall_accuracy']
            match_indices = eval_results.get('match_indices', np.zeros(max_steps, dtype=int))
        elif eval_results['evaluation_mode'] == 'multiple_trials':
            recall_acc = eval_results['recall_accuracy']
            match_indices = np.zeros(max_steps, dtype=int)
    else:
        match_indices = eval_results.get('match_indices', np.zeros(max_steps, dtype=int))
        recall_acc = eval_results.get('recall_accuracy', 0.0)
    
    # 子图1: 隐藏层训练误差
    ax1 = plt.subplot(3, 3, 1)
    plt.plot(range(1, num_epochs + 1), network.mu_history, 'b-', linewidth=2)
    plt.xlabel('训练轮数')
    plt.ylabel('平均误差 μ')
    plt.title('隐藏层训练误差')
    plt.grid(True, alpha=0.3)
    
    # 子图2: 可见层训练误差
    ax2 = plt.subplot(3, 3, 2)
    plt.plot(range(1, num_epochs + 1), network.nu_history, 'r-', linewidth=2)
    plt.xlabel('训练轮数')
    plt.ylabel('平均误差 ν')
    plt.title('可见层训练误差')
    plt.grid(True, alpha=0.3)
    
    # 子图3: 双误差对比
    ax3 = plt.subplot(3, 3, 3)
    plt.plot(range(1, num_epochs + 1), network.mu_history, 'b-', 
             linewidth=1.5, label='μ (隐藏层)')
    plt.plot(range(1, num_epochs + 1), network.nu_history, 'r-', 
             linewidth=1.5, label='ν (可见层)')
    plt.xlabel('训练轮数')
    plt.ylabel('平均误差')
    if num_seq > 1:
        plt.title(f'误差收敛曲线 ({num_seq}个序列)')
    else:
        plt.title('误差收敛曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图4: 训练序列
    ax4 = plt.subplot(3, 3, 4)
    plt.imshow(training_seq.T, cmap='gray', 
               aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('时间步')
    plt.ylabel('神经元索引')
    if num_seq > 1:
        plt.title(f'训练序列 #{sequence_index+1}')
    else:
        plt.title('训练序列（可见层）')
    
    # 子图5: 回放序列
    ax5 = plt.subplot(3, 3, 5)
    plt.imshow(xi_replayed.T, cmap='gray', aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('时间步')
    plt.ylabel('神经元索引')
    plt.title('回放序列（可见层）')
    
    # 子图6: U权重矩阵
    ax6 = plt.subplot(3, 3, 6)
    im6 = plt.imshow(network.U, cmap='jet', aspect='auto', interpolation='nearest')
    plt.colorbar(im6)
    plt.xlabel('可见神经元')
    plt.ylabel('隐藏神经元')
    plt.title('权重矩阵 U')
    
    # 子图7: V权重矩阵
    ax7 = plt.subplot(3, 3, 7)
    im7 = plt.imshow(network.V, cmap='jet', aspect='auto', interpolation='nearest')
    plt.colorbar(im7)
    plt.xlabel('隐藏神经元')
    plt.ylabel('可见神经元')
    plt.title('权重矩阵 V')
    
    # 子图8: P固定投影矩阵
    ax8 = plt.subplot(3, 3, 8)
    im8 = plt.imshow(network.P, cmap='jet', aspect='auto', interpolation='nearest')
    plt.colorbar(im8)
    plt.xlabel('可见神经元')
    plt.ylabel('隐藏神经元')
    plt.title('固定投影矩阵 P')
    
    # 子图9: 序列匹配追踪
    ax9 = plt.subplot(3, 3, 9)
    
    if 'evaluation_mode' in eval_results and \
       eval_results['evaluation_mode'] == 'full_sequence_matching':
        if 'match_indices' in eval_results:
            match_indices = eval_results['match_indices']
            plt.plot(range(1, len(match_indices) + 1), match_indices, 'o', 
                     markersize=4, alpha=0.5, color='gray', label='逐帧匹配')
            
            if eval_results.get('found_sequence', False):
                match_start = eval_results.get('match_start_idx', -1)
                T = len(training_seq)
                if match_start >= 0:
                    complete_match_x = range(match_start + 1, match_start + T + 1)
                    complete_match_y = range(1, T + 1)
                    plt.plot(complete_match_x, complete_match_y, 'o-', 
                             linewidth=2, markersize=6, color='green', 
                             label='完整序列匹配')
                    plt.axvspan(match_start + 1, match_start + T, 
                               alpha=0.2, color='green')
                
                title_text = f'序列匹配追踪\n完整匹配: ✓\n逐帧准确率: {eval_results.get("frame_recall_accuracy", 0)*100:.1f}%'
            else:
                title_text = f'序列匹配追踪\n完整匹配: ✗\n逐帧准确率: {eval_results.get("frame_recall_accuracy", 0)*100:.1f}%'
            
            plt.xlabel('回放时间步')
            plt.ylabel('匹配的训练序列位置')
            plt.title(title_text, fontsize=10)
            plt.ylim([0, len(training_seq) + 1])
            plt.legend(loc='best', fontsize=8)
            plt.grid(True, alpha=0.3)
    elif 'evaluation_mode' in eval_results and eval_results['evaluation_mode'] == 'multiple_trials':
        success_rate = eval_results['recall_accuracy'] * 100
        ax9.axis('off')
        info_text = f"多次试验评估结果\n\n"
        info_text += f"成功率: {success_rate:.1f}%\n"
        info_text += f"试验次数: {eval_results.get('num_trials', 'N/A')}\n"
        info_text += f"成功次数: {eval_results.get('success_count', 'N/A')}\n"
        if eval_results.get('avg_convergence_steps') is not None:
            info_text += f"平均收敛步数: {eval_results['avg_convergence_steps']:.1f}\n"
        plt.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    else:
        plt.plot(range(1, max_steps + 1), match_indices, 'o-', 
                 linewidth=1.5, markersize=6)
        plt.xlabel('回放时间步')
        plt.ylabel('匹配的训练序列索引')
        plt.title(f'序列匹配追踪 (准确率: {recall_acc*100:.1f}%)')
        plt.ylim([0, len(training_seq) + 1])
        plt.grid(True, alpha=0.3)
    
    # 主标题
    main_title = f'序列吸引子网络训练与回放{title_suffix}'
    if 'evaluation_mode' in eval_results:
        eval_mode_text = {
            'full_sequence_matching': '完整序列匹配',
            'multiple_trials': '多次试验统计'
        }.get(eval_results['evaluation_mode'], '')
        if eval_mode_text:
            main_title += f'\n评估方式: {eval_mode_text}'
    
    plt.suptitle(main_title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
    if show_images:
        plt.show()
    else:
        plt.close()


def visualize_robustness(noise_levels: np.ndarray, 
                        robustness_scores: Union[np.ndarray, Dict],
                        save_path: Optional[str] = None,
                        title_suffix: str = "",
                        show_images: bool = False,
                        labels: Optional[List[str]] = None):
    """
    可视化噪声鲁棒性测试结果
    
    参数:
        noise_levels: 噪声水平数组
        robustness_scores: 成功率数组或字典（多序列模式）
        save_path: 保存路径
        title_suffix: 标题后缀
        show_images: 是否显示图片
        labels: 自定义标签（多序列模式）
    """
    plt.figure(figsize=(10, 6))
    
    if isinstance(robustness_scores, np.ndarray):
        # 单序列模式
        plt.plot(noise_levels * 100, robustness_scores * 100, '-o',
                 linewidth=2.5, markersize=8, color='#A23B72',
                 label='单序列')
        title = f'序列吸引子的噪声鲁棒性{title_suffix}'
    
    elif isinstance(robustness_scores, dict):
        # 多序列模式
        colors = plt.cm.tab10(np.linspace(0, 1, len(robustness_scores)))
        
        for i, (seq_name, scores) in enumerate(robustness_scores.items()):
            if labels is not None and i < len(labels):
                label = labels[i]
            else:
                seq_idx = int(seq_name.split('_')[1])
                label = f'序列 #{seq_idx+1}'
            
            plt.plot(noise_levels * 100, scores * 100, '-o',
                    linewidth=2, markersize=6, 
                    color=colors[i],
                    label=label)
        
        title = f'多序列噪声鲁棒性对比{title_suffix}'
    
    else:
        raise ValueError("robustness_scores 必须是 np.ndarray 或 Dict")
    
    plt.xlabel('噪声水平 (%)', fontsize=12)
    plt.ylabel('恢复到原序列的成功率 (%)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
    if show_images:
        plt.show()
    else:
        plt.close()


def visualize_multi_sequence_overview(network,
                                     save_path: Optional[str] = None,
                                     title_suffix: str = "",
                                     show_images: bool = False):
    """
    可视化多序列学习概览
    
    参数:
        network: 网络对象
        save_path: 保存路径
        title_suffix: 标题后缀
        show_images: 是否显示图片
    """
    if not hasattr(network, 'training_sequences') or len(network.training_sequences) == 0:
        print("警告：没有训练序列可以可视化")
        return
    
    K = len(network.training_sequences)
    
    # 动态计算子图布局
    n_cols = min(K, 3)
    n_rows = (K + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows + 3))
    
    # 顶部：训练误差曲线
    ax_top = plt.subplot(n_rows + 1, 1, 1)
    num_epochs = len(network.mu_history)
    plt.plot(range(1, num_epochs + 1), network.mu_history, 'b-', 
             linewidth=1.5, label='μ (隐藏层)')
    plt.plot(range(1, num_epochs + 1), network.nu_history, 'r-', 
             linewidth=1.5, label='ν (可见层)')
    plt.xlabel('训练轮数')
    plt.ylabel('平均误差')
    plt.title(f'多序列训练误差收敛 ({K}个序列)', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 为每个序列创建回放测试
    for k in range(K):
        xi_replayed = network.replay(sequence_index=k, 
                                     max_steps=network.training_sequences[k].shape[0] * 2)
        eval_result = network.evaluate_replay(xi_replayed, sequence_index=k)
        
        # 训练序列
        ax_train = plt.subplot(n_rows + 1, n_cols * 2, n_cols * 2 + k * 2 + 1)
        plt.imshow(network.training_sequences[k].T, cmap='gray', 
                   aspect='auto', interpolation='nearest')
        plt.title(f'序列 #{k+1} (训练)', fontsize=10)
        plt.xlabel('时间步')
        plt.ylabel('神经元')
        
        # 回放序列
        ax_replay = plt.subplot(n_rows + 1, n_cols * 2, n_cols * 2 + k * 2 + 2)
        plt.imshow(xi_replayed.T, cmap='gray', 
                   aspect='auto', interpolation='nearest')
        plt.title(f'序列 #{k+1} (回放, 准确率={eval_result["recall_accuracy"]*100:.0f}%)', 
                  fontsize=10)
        plt.xlabel('时间步')
        plt.ylabel('神经元')
    
    plt.suptitle(f'多序列学习结果概览{title_suffix}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"多序列概览图已保存: {save_path}")
    if show_images:
        plt.show()
    else:
        plt.close()

