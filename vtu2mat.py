import os
import meshio
import scipy.io

def vtu_to_mat_single(input_file, output_dir, dim=2):
    # 读取VTU文件
    mesh = meshio.read(input_file)

    # 提取所有点数据
    data = {key: mesh.point_data[key] for key in mesh.point_data.keys()}

    # 提取坐标数据
    data['x_coor'] = mesh.points[:, 0]
    data['y_coor'] = mesh.points[:, 1]
    if dim == 3:
        data['z_coor'] = mesh.points[:, 2]

    # 获取输入文件的名称，不包括扩展名
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # 将数据保存为MAT文件，文件名与输入文件相同，保存在指定的输出目录下
    output_file = os.path.join(output_dir, base_name + '.mat')
    scipy.io.savemat(output_file, data)
    # 打印消息以显示已经生成的.mat文件
    print(f'MAT file generated: {output_file}')

def vtu_to_mat_batch(input_dir,dim=2):
    output_dir = os.path.join('matdata', input_dir)
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    # 遍历输入目录中的所有.vtu文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.vtu'):
            input_file = os.path.join(input_dir, filename)
            vtu_to_mat_single(input_file, output_dir, dim)

input_dir = '2D-NS'

# 批量处理目录中的所有文件
vtu_to_mat_batch(input_dir)