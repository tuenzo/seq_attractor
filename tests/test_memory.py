"""
统一记忆网络测试
"""

import numpy as np

from src.models.memory import MemorySequenceAttractorNetwork


class TestMemorySequenceAttractorNetwork:
    """MemorySequenceAttractorNetwork 综合测试。"""

    def test_multi_sequence_training(self, memory_network, multiple_sequences):
        """验证多序列训练路径。"""
        result = memory_network.train(
            x=multiple_sequences,
            num_epochs=50,
            verbose=False,
            interleaved=True,
        )

        assert isinstance(result, dict)
        assert result["training_mode"] == "multi_sequence"
        assert memory_network.num_sequences == len(multiple_sequences)
        assert len(memory_network.training_sequences) == len(multiple_sequences)
        assert np.array_equal(
            memory_network.training_sequences[0], multiple_sequences[0]
        )
        assert memory_network.sequence_training_info[-1]["training_mode"] == "multi_sequence"

    def test_incremental_training_flow(self, memory_network):
        """验证增量学习路径能够保留历史记忆。"""
        seq1 = memory_network.generate_random_sequence(seed=123)
        seq2 = memory_network.generate_random_sequence(seed=456)

        result_first = memory_network.train(
            x=seq1,
            num_epochs=20,
            verbose=False,
            incremental=False,
        )
        assert result_first["training_mode"] == "学习新序列"
        assert memory_network.num_sequences == 1
        assert memory_network._total_epochs_trained == 20

        result_second = memory_network.train(
            x=seq2,
            num_epochs=30,
            verbose=False,
            incremental=True,
        )
        assert result_second["training_mode"] == "增量学习新序列"
        assert memory_network.num_sequences == 2
        assert memory_network._total_epochs_trained == 50
        assert memory_network.sequence_training_info[-1]["incremental"] is True

        # 增量训练后可回放第一个序列
        replayed = memory_network.replay(sequence_index=0, max_steps=memory_network.T * 2)
        evaluation = memory_network.evaluate_replay(replayed, sequence_index=0)
        assert "recall_accuracy" in evaluation

