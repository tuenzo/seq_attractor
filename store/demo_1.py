# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
# ---
"""
================================================================
Python实现：Learning Sequence Attractors in RNN with Hidden Neurons
基于 Lu & Wu (2024) 论文
Jupyter Notebook 版本
================================================================
"""

# %% [markdown]
# # 序列吸引子RNN网络实现
# 
# 基于 Lu & Wu (2024) 论文的Python实现
# 
# ## 依赖库安装
# ```bash
# pip install numpy matplotlib
# ```

# %% 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体支持
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 设置图表样式
plt.style.use('seaborn-v0_8-darkgrid')
print("✓ 库导入成功")

# %% [markdown]
# ## 1. 参数设置

# %% 参数配置
N_v = 100              # 可见神经元数量
T = 50                # 序列长度
N_h = round((T - 1) * 5)  # 隐藏神经元数量 (论文建议至少 T-1)
eta = 0.001           # 学习率
kappa = 5             # 鲁棒性参数（margin）
num_epochs = 500      # 训练轮数

print(f"参数配置:")
print(f"  可见神经元数量 (N_v): {N_v}")
print(f"  序列长度 (T): {T}")
print(f"  隐藏神经元数量 (N_h): {N_h}")
print(f"  学习率 (eta): {eta}")
print(f"  Margin参数 (kappa): {kappa}")
print(f"  训练轮数: {num_epochs}")

# %% [markdown]
# ## 2. 生成训练序列
# 
# 生成随机二值序列 {-1, +1}，确保：
# - 序列具有周期性：x(1) = x(T)
# - 中间状态无重复

# %% 生成训练序列
np.random.seed(42)  # 设定随机种子以确保可重复性
x = np.sign(np.random.randn(T, N_v))  # T x N_v
x[x == 0] = 1

# 确保序列中没有重复（除了首尾）
for t in range(1, T - 1):
    while np.any(np.all(x[t, :] == x[:t, :], axis=1)):
        x[t, :] = np.sign(np.random.randn(N_v))
        x[t, x[t, :] == 0] = 1

x[T - 1, :] = x[0, :]  # 周期性：最后一个等于第一个

print(f'✓ 训练序列生成完成')
print(f'  序列形状: {x.shape}')
print(f'  周期性检查: x[0] == x[{T-1}] ? {np.all(x[0] == x[T-1])}')

# 可视化训练序列
plt.figure(figsize=(12, 4))
plt.imshow(x.T, cmap='RdBu_r', aspect='auto', interpolation='nearest')
plt.colorbar(label='神经元状态')
plt.xlabel('时间步', fontsize=12)
plt.ylabel('神经元索引', fontsize=12)
plt.title('训练序列可视化', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. 初始化权重矩阵
# 
# - **U**: 可见层 → 隐藏层 (N_h × N_v) - 可训练
# - **V**: 隐藏层 → 可见层 (N_v × N_h) - 可训练
# - **P**: 随机投影矩阵 (N_h × N_v) - 固定不变

# %% 初始化权重
U = np.random.randn(N_h, N_v) * 1e-6
V = np.random.randn(N_v, N_h) * 1e-6
P = np.random.randn(N_h, N_v) / np.sqrt(N_v)  # 固定，不参与学习

print(f"✓ 权重矩阵初始化完成")
print(f"  U形状: {U.shape}")
print(f"  V形状: {V.shape}")
print(f"  P形状: {P.shape}")

# %% [markdown]
# ## 4. 三因子学习规则训练
# 
# ### 训练步骤：
# 1. **更新U**: 使用margin perceptron规则训练隐藏层权重
# 2. **更新V**: 训练输出层权重以重构序列

# %% 训练网络
print('开始训练...\n')

mu_history = np.zeros(num_epochs)  # 隐藏层误差
nu_history = np.zeros(num_epochs)  # 可见层误差

for epoch in range(num_epochs):
    total_mu = 0
    total_nu = 0
    
    for t in range(T - 1):  # 遍历序列对 (x(t), x(t+1))
        x_current = x[t, :].reshape(-1, 1)      # N_v x 1
        x_next = x[(t + 1) % T, :].reshape(-1, 1)
        
        # ===== 步骤1：更新U =====
        # 计算目标隐藏层状态 z(t+1)
        z_target = np.sign(P @ x_next)  # N_h x 1
        z_target[z_target == 0] = 1
        
        # 计算当前隐藏层输入
        h_input = U @ x_current  # N_h x 1
        
        # 计算误差项 mu(t) - margin perceptron规则
        mu = (z_target * h_input < kappa).astype(float)  # N_h x 1
        
        # 更新U：U_ij <- U_ij + eta * mu_i(t) * z_i(t+1) * x_j(t)
        for i in range(N_h):
            if mu[i, 0] > 0:
                U[i, :] += eta * mu[i, 0] * z_target[i, 0] * x_current.flatten()
        
        # ===== 步骤2：更新V =====
        # 计算实际隐藏层输出 y(t)
        y_actual = np.sign(U @ x_current)  # N_h x 1
        y_actual[y_actual == 0] = 1
        
        # 计算可见层输入
        v_input = V @ y_actual  # N_v x 1
        
        # 计算误差项 nu(t)
        nu = (x_next * v_input < kappa).astype(float)  # N_v x 1
        
        # 更新V：V_ji <- V_ji + eta * nu_j(t) * x_j(t+1) * y_i(t)
        for j in range(N_v):
            if nu[j, 0] > 0:
                V[j, :] += eta * nu[j, 0] * x_next[j, 0] * y_actual.flatten()
        
        # 累积误差
        total_mu += np.sum(mu)
        total_nu += np.sum(nu)
    
    # 记录平均误差
    mu_history[epoch] = total_mu / (N_h * (T - 1))
    nu_history[epoch] = total_nu / (N_v * (T - 1))
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, μ={mu_history[epoch]:.4f}, ν={nu_history[epoch]:.4f}')

print('\n✓ 训练完成')

# %% [markdown]
# ## 5. 训练过程可视化

# %% 可视化训练误差
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 子图1: 隐藏层误差
axes[0].plot(range(1, num_epochs + 1), mu_history, 'b-', linewidth=2)
axes[0].set_xlabel('训练轮数', fontsize=11)
axes[0].set_ylabel('平均误差 μ', fontsize=11)
axes[0].set_title('隐藏层训练误差', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# 子图2: 可见层误差
axes[1].plot(range(1, num_epochs + 1), nu_history, 'r-', linewidth=2)
axes[1].set_xlabel('训练轮数', fontsize=11)
axes[1].set_ylabel('平均误差 ν', fontsize=11)
axes[1].set_title('可见层训练误差', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# 子图3: 双误差对比
axes[2].plot(range(1, num_epochs + 1), mu_history, 'b-', linewidth=1.5, label='μ (隐藏层)')
axes[2].plot(range(1, num_epochs + 1), nu_history, 'r-', linewidth=1.5, label='ν (可见层)')
axes[2].set_xlabel('训练轮数', fontsize=11)
axes[2].set_ylabel('平均误差', fontsize=11)
axes[2].set_title('误差收敛曲线', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. 序列回放测试

# %% 测试序列回放
print('测试序列回放能力...\n')

# 从第一个状态开始（可添加噪声）
xi_test = x[0, :].copy().reshape(-1, 1)
noise_level = 0.0  # 可调整噪声比例（0.0 - 0.5）
noise_mask = np.random.rand(N_v, 1) < noise_level
xi_test[noise_mask] = -xi_test[noise_mask]

max_steps = T * 2
xi_replayed = np.zeros((max_steps, N_v))

for step in range(max_steps):
    # 隐藏层激活
    zeta = np.sign(U @ xi_test)
    zeta[zeta == 0] = 1
    
    # 可见层更新
    xi_test = np.sign(V @ zeta)
    xi_test[xi_test == 0] = 1
    
    xi_replayed[step, :] = xi_test.flatten()

# 评估回放质量
match_count = 0
for step in range(max_steps):
    for t in range(T):
        if np.all(xi_replayed[step, :] == x[t, :]):
            match_count += 1
            break

recall_accuracy = match_count / max_steps
print(f'✓ 回放序列匹配率: {recall_accuracy * 100:.2f}%')

# %% 可视化回放结果
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# 训练序列
im1 = axes[0].imshow(x.T, cmap='RdBu_r', aspect='auto', interpolation='nearest')
axes[0].set_xlabel('时间步', fontsize=11)
axes[0].set_ylabel('神经元索引', fontsize=11)
axes[0].set_title('训练序列（可见层）', fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=axes[0])

# 回放序列
im2 = axes[1].imshow(xi_replayed.T, cmap='RdBu_r', aspect='auto', interpolation='nearest')
axes[1].set_xlabel('时间步', fontsize=11)
axes[1].set_ylabel('神经元索引', fontsize=11)
axes[1].set_title('回放序列（可见层）', fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()

# %% 序列匹配追踪
match_matrix = np.zeros(max_steps)
for step in range(max_steps):
    for t in range(T):
        if np.all(xi_replayed[step, :] == x[t, :]):
            match_matrix[step] = t + 1  # +1 for 1-indexed display
            break

plt.figure(figsize=(12, 4))
plt.plot(range(1, max_steps + 1), match_matrix, 'o-', linewidth=1.5, markersize=6, color='#2E86AB')
plt.xlabel('回放时间步', fontsize=11)
plt.ylabel('匹配的训练序列索引', fontsize=11)
plt.title('序列匹配追踪', fontsize=12, fontweight='bold')
plt.ylim([0, T + 1])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. 权重矩阵可视化

# %% 可视化权重矩阵
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# U权重矩阵
im1 = axes[0].imshow(U, cmap='coolwarm', aspect='auto', interpolation='nearest')
axes[0].set_xlabel('可见神经元', fontsize=11)
axes[0].set_ylabel('隐藏神经元', fontsize=11)
axes[0].set_title('权重矩阵 U (可见→隐藏)', fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=axes[0])

# V权重矩阵
im2 = axes[1].imshow(V, cmap='coolwarm', aspect='auto', interpolation='nearest')
axes[1].set_xlabel('隐藏神经元', fontsize=11)
axes[1].set_ylabel('可见神经元', fontsize=11)
axes[1].set_title('权重矩阵 V (隐藏→可见)', fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=axes[1])

# P固定投影矩阵
im3 = axes[2].imshow(P, cmap='viridis', aspect='auto', interpolation='nearest')
axes[2].set_xlabel('可见神经元', fontsize=11)
axes[2].set_ylabel('隐藏神经元', fontsize=11)
axes[2].set_title('固定投影矩阵 P', fontsize=12, fontweight='bold')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. 噪声鲁棒性测试
# 
# 测试网络在不同噪声水平下恢复原始序列的能力

# %% 噪声鲁棒性测试
print('进行噪声鲁棒性测试...\n')
noise_levels = np.arange(0, 0.8, 0.05)
robustness_scores = np.zeros(len(noise_levels))
num_trials = 50
stability_threshold = 3  # 连续不变化的步数阈值视为收敛

for i, noise_level in enumerate(noise_levels):
    success_count = 0

    for trial in range(num_trials):
        # 添加噪声
        xi_noisy = x[0, :].copy().reshape(-1, 1)
        noise_mask = np.random.rand(N_v, 1) < noise_level
        xi_noisy[noise_mask] = -xi_noisy[noise_mask]

        stable_steps = 0
        converged = False

        for step in range(T * 5):  # 给足够长时间演化
            xi_prev = xi_noisy.copy()

            # 隐藏层更新
            zeta = np.sign(U @ xi_noisy)
            zeta[zeta == 0] = 1

            # 可见层更新
            xi_noisy = np.sign(V @ zeta)
            xi_noisy[xi_noisy == 0] = 1

            # 检查是否恢复原始状态 (x[0])
            if np.all(xi_noisy.flatten() == x[0, :]):
                converged = True
                break

            # 检查是否卡在固定点
            if np.array_equal(xi_noisy, xi_prev):
                stable_steps += 1
            else:
                stable_steps = 0

            if stable_steps >= stability_threshold:
                break

        if converged:
            success_count += 1

    robustness_scores[i] = success_count / num_trials
    print(f'噪声水平 {noise_level:.2f}: 成功率 {robustness_scores[i]*100:.1f}%')


# %% 可视化鲁棒性结果
plt.figure(figsize=(10, 6))
plt.plot(noise_levels * 100, robustness_scores * 100, '-o', 
         linewidth=2.5, markersize=8, color='#A23B72')
plt.xlabel('噪声水平 (%)', fontsize=12)
plt.ylabel('序列恢复成功率 (%)', fontsize=12)
plt.title('序列吸引子的噪声鲁棒性', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim([0, 105])
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. 总结与调优建议
# 
# ### 如果误差未收敛或回放失败，请尝试：
# 1. **增加训练轮数** `num_epochs` (如 1000-2000)
# 2. **调整学习率** `eta` (建议范围 1e-4 到 1e-2)
# 3. **增加隐藏神经元数** `N_h` (尝试 2*T 或更多)
# 4. **减小序列长度** `T` 或可见神经元数 `N_v`
# 5. **调整margin参数** `kappa` (尝试 0.5-2.0)

# %% 打印最终统计
print('\n' + '='*50)
print('程序运行完成！')
print('='*50)
print(f'\n最终统计:')
print(f'  训练误差 (μ): {mu_history[-1]:.4f}')
print(f'  训练误差 (ν): {nu_history[-1]:.4f}')
print(f'  回放准确率: {recall_accuracy * 100:.2f}%')
print(f'  最大噪声容忍度: {noise_levels[robustness_scores > 0.5][-1]*100:.0f}% (成功率>50%)')
print('='*50)

# %%
