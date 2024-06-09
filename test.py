import vtk
from vtk.util.numpy_support import vtk_to_numpy
import scipy.io

def convert_vtu_to_mat(vtu_filename, mat_filename):
    # 读取 VTU 文件
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_filename)
    reader.Update()
    data = reader.GetOutput()

    # 提取点坐标
    points_vtk_array = data.GetPoints().GetData()
    points = vtk_to_numpy(points_vtk_array)

    # 提取单元信息
    cells_vtk_array = data.GetCells().GetData()
    cells = vtk_to_numpy(cells_vtk_array)

    # 保存为 MAT 文件
    scipy.io.savemat(mat_filename, {'points': points, 'cells': cells})

# 示例用法
vtu_filename = '2D-NS/parameter_1.000000/realtime_0.000000_timestep_0000.0.vtu'
mat_filename = 'output_file.mat'
convert_vtu_to_mat(vtu_filename, mat_filename)
