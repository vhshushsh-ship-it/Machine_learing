import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# ============================================
# 1. 生成模拟数据
# ============================================
np.random.seed(42)  # 固定随机种子，让结果可复现c

# ---------- 回归数据：y = 2*x + 噪声 ----------
X_reg = np.linspace(0, 10, 100).reshape(-1, 1)          # 特征，100个点
y_reg = 2 * X_reg.ravel() + np.random.normal(0, 1, 100) # 真实关系+噪声

# ---------- 分类数据：根据 x 值分成两类 ----------
# 构造一个简单二分类：x < 5 为类别0，x >= 5 为类别1，并加一点混淆
X_cls = np.linspace(0, 10, 200).reshape(-1, 1)
y_cls = (X_cls.ravel() >= 5).astype(int)
# 在边界附近随机翻转一些点，增加难度
flip_idx = np.random.choice(len(y_cls), size=20, replace=False)
y_cls[flip_idx] = 1 - y_cls[flip_idx]

# ============================================
# 2. 训练回归模型（线性回归）
# ============================================
reg_model = LinearRegression()
reg_model.fit(X_reg, y_reg)
y_reg_pred = reg_model.predict(X_reg)

# ============================================
# 3. 训练分类模型（逻辑回归）
# ============================================
cls_model = LogisticRegression()
cls_model.fit(X_cls, y_cls)
# 得到预测类别和概率
y_cls_pred = cls_model.predict(X_cls)
y_cls_prob = cls_model.predict_proba(X_cls)[:, 1]  # 属于类别1的概率

# ============================================
# 4. 可视化对比
# ============================================
plt.figure(figsize=(14, 5))

# 子图1：回归任务
plt.subplot(1, 2, 1)
plt.scatter(X_reg, y_reg, alpha=0.7, label='真实数据')
plt.plot(X_reg, y_reg_pred, color='red', linewidth=2, label='回归拟合线')
plt.title(f'回归任务：预测连续值\n均方误差(MSE): {mean_squared_error(y_reg, y_reg_pred):.2f}')
plt.xlabel('特征 X')
plt.ylabel('目标值 y (连续)')
plt.legend()
plt.grid(True)

# 子图2：分类任务
plt.subplot(1, 2, 2)
# 画真实类别 (0/1)
plt.scatter(X_cls[y_cls==0], y_cls[y_cls==0], label='类别0', c='blue', alpha=0.6)
plt.scatter(X_cls[y_cls==1], y_cls[y_cls==1], label='类别1', c='orange', alpha=0.6)
# 画模型决策边界 (概率=0.5的位置)
# 计算所有点的分类概率，找出概率接近0.5的点作为边界线
prob_grid = cls_model.predict_proba(X_cls)[:, 1]
boundary_idx = np.where(np.abs(prob_grid - 0.5) < 0.05)[0]
if len(boundary_idx) > 0:
    plt.axvline(x=X_cls[boundary_idx[0]], color='green', linestyle='--', label='决策边界 (概率=0.5)')
plt.title(f'分类任务：预测离散类别\n准确率: {accuracy_score(y_cls, y_cls_pred)*100:.1f}%')
plt.xlabel('特征 X')
plt.ylabel('类别 (0 或 1)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ============================================
# 5. 打印模型参数，帮助理解
# ============================================
print("\n========== 回归模型 ==========")
print(f"系数 (斜率): {reg_model.coef_[0]:.2f}")
print(f"截距: {reg_model.intercept_:.2f}")
print(f"方程: y = {reg_model.coef_[0]:.2f} * x + {reg_model.intercept_:.2f}")

print("\n========== 分类模型 ==========")
print(f"逻辑回归系数: {cls_model.coef_[0][0]:.2f}")
print(f"逻辑回归截距: {cls_model.intercept_[0]:.2f}")
print("决策边界位置: x = {:.2f}".format(-cls_model.intercept_[0] / cls_model.coef_[0][0]))