# import csv
#
# # 要读取的 CSV 文件名
# input_filename = 'new_data/styles.csv'
# # 要保存的新 CSV 文件名
# output_filename = '9.csv'
# # 想要选择的列的索引，这里我们选择第一列（索引为 0）和第十列（索引为 9）
# columns_indices = [0, 4]
# # 只有当第十列的值出现在这个列表中时，才会被写入新文件
# filter_values = ['Formal Shoes']  # 示例值，根据需要更改
# if __name__ == '__main__':
#
#     # 读取 CSV 文件，并只选择指定的列，且第十列的值必须在 filter_values 中
#     with open(input_filename, mode='r', newline='', encoding='utf-8') as infile, \
#             open(output_filename, mode='w', newline='', encoding='utf-8') as outfile:
#         reader = csv.reader(infile)
#         writer = csv.writer(outfile)
#
#         for row in reader:
#             # 检查该行的第十列的值是否在 filter_values 列表中
#             if row[columns_indices[1]] in (filter_values[0]):
#                 # 只选择并写入指定索引的列，且值满足过滤条件
#                 # 这里我们保存第一列和第十列
#                 writer.writerow([row[columns_indices[0]], row[columns_indices[1]]])

import csv

if __name__ == '__main__':
    # for i in range(0, 10):
    #
    #     # 要读取的原始 CSV 文件名
    #     input_filename = '{}.csv'.format(i)
    #     # 要保存的修改后的 CSV 文件名
    #     output_filename = '{}_modified.csv'.format(i)
    #     # 要修改的列的索引
    #     column_index = 1
    #     # 要将列的所有值更改为的新值
    #     new_value = i
    #
    #     # 读取原始 CSV 文件，并将指定列的值更改为新值
    #     with open(input_filename, mode='r', newline='', encoding='utf-8') as infile, \
    #             open(output_filename, mode='w', newline='', encoding='utf-8') as outfile:
    #         reader = csv.reader(infile)
    #         writer = csv.writer(outfile)
    #
    #         for row in reader:
    #             # 更改指定列的值为新值
    #             row[column_index] = new_value
    #             # 写入修改后的行到新文件
    #             writer.writerow(row)
    #
    #     print(f'文件已修改并保存为 {output_filename}。')

    # import csv
    #
    # # 要合并的 CSV 文件名列表
    # input_filenames = ['{}_modified.csv'.format(i) for i in range(0, 10)]
    # # 要保存的合并后的 CSV 文件名
    # output_filename = 'merged.csv'
    #
    # # 使用 'w' 模式打开输出文件，以便写入
    # with open(output_filename, mode='w', newline='', encoding='utf-8') as outfile:
    #     writer = csv.writer(outfile)
    #
    #     # 对每个输入文件进行遍历
    #     for filename in input_filenames:
    #         with open(filename, mode='r', newline='', encoding='utf-8') as infile:
    #             reader = csv.reader(infile)
    #             # 写入当前文件的所有行到输出文件
    #             for row in reader:
    #                 writer.writerow(row)
    #
    # print(f'所有文件已合并到 {output_filename} 中。')

    import csv
    import shutil
    import os

    # CSV 文件路径
    csv_file_path = 'merged.csv'
    # 原始图片所在的文件夹路径
    source_folder_path = 'new_data/images/'
    # 图片要被复制到的目标文件夹路径
    destination_folder_path = 'fashion_data/'

    # 确保目标文件夹存在，如果不存在，则创建它
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)

    # 读取 CSV 文件，并获取第一列中的每个图片文件名
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            image_file_name = row[0] + ".jpg"  # 假设图片文件名在第一列
            source_file_path = os.path.join(source_folder_path, image_file_name)
            destination_file_path = os.path.join(destination_folder_path, image_file_name)

            # 复制图片文件到目标文件夹
            try:
                shutil.copy(source_file_path, destination_file_path)
            except FileNotFoundError as e:
                print(f'文件 {source_file_path} 不存在。')

    print(f'所有图片已被复制到 {destination_folder_path}。')