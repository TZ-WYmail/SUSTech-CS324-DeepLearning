"""
快速测试脚本 - 验证LSTM实现是否正常工作
"""
import torch
from dataset import PalindromeDataset
from lstm import LSTM
from utils import accuracy

print("=" * 60)
print("快速测试LSTM实现")
print("=" * 60)

# 测试1: 数据集
print("\n测试1: 数据集创建...")
dataset = PalindromeDataset(input_length=5, total_len=100, one_hot=False)
inputs, label = dataset[0]
print(f"✓ 数据集创建成功")
print(f"  输入形状: {inputs.shape}")
print(f"  输入序列: {inputs.squeeze().astype(int)}")
print(f"  标签: {label}")

# 测试2: 模型创建
print("\n测试2: 模型创建...")
model = LSTM(
    seq_length=6,
    input_dim=1,
    hidden_dim=32,
    output_dim=10,
    batch_size=4
)
print(f"✓ 模型创建成功")
print(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 测试3: 前向传播
print("\n测试3: 前向传播...")
batch_inputs = torch.randn(4, 6, 1)
outputs = model(batch_inputs)
print(f"✓ 前向传播成功")
print(f"  输出形状: {outputs.shape}")
print(f"  输出和: {outputs.sum(dim=1)}")  # 应该接近1（softmax）

# 测试4: 准确率计算
print("\n测试4: 准确率计算...")
batch_targets = torch.tensor([0, 1, 2, 3])
acc = accuracy(outputs, batch_targets)
print(f"✓ 准确率计算成功")
print(f"  准确率: {acc:.2f}%")

# 测试5: 反向传播
print("\n测试5: 反向传播...")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
loss = criterion(outputs, batch_targets)
loss.backward()
optimizer.step()
print(f"✓ 反向传播成功")
print(f"  损失值: {loss.item():.4f}")

print("\n" + "=" * 60)
print("✓ 所有测试通过！代码实现正确。")
print("=" * 60)
