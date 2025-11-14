from  SAN_multi_seq_1 import SequenceAttractorNetwork, parameter_sweep, visualize_results, visualize_robustness
import numpy as np
# 快速训练
network = SequenceAttractorNetwork(N_v=100, T=70, N_h=1000, eta=0.001, kappa=1)
train_results = network.train(num_epochs=500,V_only=True)
xi_replayed = network.replay()
eval_results = network.evaluate_replay(xi_replayed)
robustness_scores = network.test_robustness(noise_levels=np.array([0.1, 0.01, 0.0]), num_trials=50)

print(f"\n训练完成:")
print(f"  最终 μ 误差: {train_results['final_mu']:.4f}")
print(f"  最终 ν 误差: {train_results['final_nu']:.4f}")
print(f"  回放准确率: {eval_results['recall_accuracy']*100:.2f}%")
    
visualize_results(network, xi_replayed, eval_results, save_path='quick_train_test.png')

# # 参数扫描
# base_params = {
#     'N_v': 100,
#     'T': 70,
#     'N_h': 100,  # 自动计算
#     'kappa': 1,
#     'num_epochs': 300
# }
# parameter_sweep('kappa', [0.001, 0.01, 0.1, 1, 10, 100], base_params)
