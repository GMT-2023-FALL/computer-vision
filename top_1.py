import pandas as pd

# 假设这些是你的数据
data = {
    'Model': ['Base', 'Var1', 'Var2', 'Var3', 'Var4'],
    'Training Top-1 Accuracy':   [93.71, 93.03, 86.68, 88.11, 86.31],  # 这里填入你的准确率
    'Validation Top-1 Accuracy': [88.66, 89.05, 88.06, 89.61, 87.91],  # 这里填入你的准确率
}
if __name__ == '__main__':

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 显示DataFrame
    print(df)