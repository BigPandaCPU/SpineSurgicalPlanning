import numpy as np

from tools.vtk_tools import *
import os
from tools.template_coordinate import *


def registration_method(points_file, origin_stl_file, stl_file, coordinate_file=''):
    all_actors = []
    cur_spine_actor = createActorFromSTL(stl_file, opacity=0.75)
    all_actors.append(cur_spine_actor)
    spine_polydata = cur_spine_actor.GetMapper().GetInput()
    origin_spine_actor = createActorFromSTL(origin_stl_file)
    origin_spine_polydata = origin_spine_actor.GetMapper().GetInput()
   # all_actors.append(origin_spine_actor)

    points = parsePointsFile(points_file)
    points_vtk = vtkPoints()
    # pointset_vtk = vtk.vtkPointSet()
    points_polydata = vtkPolyData()

    for p in points.values():
        # print(p)
        p_actor = createSphereActor(p ,3)
        # all_actors.append(p_actor)
        points_vtk.InsertNextPoint(p)
    points_polydata.SetPoints(points_vtk)
    # pointset_vtk.SetPoints(points_vtk)

    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(origin_spine_polydata)
    icp.SetTarget(spine_polydata)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(100)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()
    # transform_matrix = icp.GetMatrix()

    icpTransformFilter = vtkTransformPolyDataFilter()
    icpTransformFilter.SetInputData(origin_spine_polydata)
    icpTransformFilter.SetTransform(icp)
    icpTransformFilter.Update()

    # new_points = np.array(icpTransformFilter.GetOutput().GetPoints().GetData())
    # for p in new_points:
    #     p_actor = createSphereActor(p, 3, color='Green')
    #     # all_actors.append(p_actor)

    points_mapper = vtkPolyDataMapper()
    points_mapper.SetInputData(icpTransformFilter.GetOutput())
    points_actor = vtkActor()
    points_actor.SetMapper(points_mapper)
    spine_mapper = vtkPolyDataMapper()
    spine_mapper.SetInputData(spine_polydata)
    spine_actor = vtkActor()
    spine_actor.SetMapper(spine_mapper)
    # spine_actor.GetProperty().SetColor(0.75, 0.75, 0.75)
   # spine_actor.GetProperty().SetOpacity(0.5)
    all_actors.append(spine_actor)
    # all_actors.append(points_actor)
    showActors(all_actors)
    pass

def init_coordinate(label_num):
    coordinate_template = Coordinate_Template()
    if label_num.lower() == "c1":
        target_init_function = coordinate_template.init_c1
    elif label_num.lower() == "c2":
        target_init_function = coordinate_template.init_c2
    elif label_num.lower() == "c3":
        target_init_function = coordinate_template.init_c3
    elif label_num.lower() == "c4":
        target_init_function = coordinate_template.init_c4
    elif label_num.lower() == "c5":
        target_init_function = coordinate_template.init_c5
    elif label_num.lower() == "c6":
        target_init_function = coordinate_template.init_c6
    elif label_num.lower() == "c7":
        target_init_function = coordinate_template.init_c7
    elif label_num.lower() == "t1":
        target_init_function = coordinate_template.init_t1
    elif label_num.lower() == "t2":
        target_init_function = coordinate_template.init_t2
    elif label_num.lower() == "t3":
        target_init_function = coordinate_template.init_t3
    elif label_num.lower() == "t4":
        target_init_function = coordinate_template.init_t4
    elif label_num.lower() == "t5":
        target_init_function = coordinate_template.init_t5
    elif label_num.lower() == "t6":
        target_init_function = coordinate_template.init_t6
    elif label_num.lower() == "t7":
        target_init_function = coordinate_template.init_t7
    elif label_num.lower() == "t8":
        target_init_function = coordinate_template.init_t8
    elif label_num.lower() == "t9":
        target_init_function = coordinate_template.init_t9
    elif label_num.lower() == "t10":
        target_init_function = coordinate_template.init_t10
    elif label_num.lower() == "t11":
        target_init_function = coordinate_template.init_t11
    elif label_num.lower() == "t12":
        target_init_function = coordinate_template.init_t12
    elif label_num.lower() == "l1":
        target_init_function = coordinate_template.init_l1
    elif label_num.lower() == "l2":
        target_init_function = coordinate_template.init_l2
    elif label_num.lower() == "l3":
        target_init_function = coordinate_template.init_l3
    elif label_num.lower() == "l4":
        target_init_function = coordinate_template.init_l4
    elif label_num.lower() == "l5":
        target_init_function = coordinate_template.init_l5
    else:
        target_init_function = coordinate_template.init_t10

    coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R, \
        pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid = target_init_function()
    return coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R,\
            pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid


def transfer_point(point, transfer_matrix):
    if len(point) == 3:
        point.append(1)
    return transfer_matrix.MultiplyPoint(point)[:-1]


def registration_polydata(label_num, spine_polydata):
    all_actors = []
    new_coordinate = []
    colors = vtkNamedColors()
    target_dir = './registration_test_data'

    stl_file_path = os.path.join(target_dir, label_num+".stl")

    # coordinate_file_path = os.path.join(target_dir, label_num+".txt")
    # 从txt读取初始模板
    # origin, coordinates,plane_center_point_L,plane_center_point_R,pedicle_pipeline_L_normal,pedicle_pipeline_R_normal\
    #     = load_coordinate(coordinate_file_path)

    origin, coordinates, plane_center_point_L, plane_center_point_R, pedicle_pipeline_L_normal,\
        pedicle_pipeline_R_normal, pedicle_lower_reference_point_mid = init_coordinate(label_num)

    cur_spine_actor = createActorFromSTL(stl_file_path, opacity=0.75, color='Green')
   # all_actors.append(cur_spine_actor)

    origin_spine_polydata = cur_spine_actor.GetMapper().GetInput()
    original_axis_actor = createSpineAxisActor(origin, coordinates)
  # all_actors.extend(original_axis_actor)

    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(origin_spine_polydata)
    icp.SetTarget(spine_polydata)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(100)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()
    transform_matrix = icp.GetMatrix()

    icpTransformFilter = vtkTransformPolyDataFilter()
    icpTransformFilter.SetInputData(origin_spine_polydata)
    icpTransformFilter.SetTransform(icp)
    icpTransformFilter.Update()

    aligned_spine_mapper = vtkPolyDataMapper()
    aligned_spine_mapper.SetInputData(icpTransformFilter.GetOutput())
    aligned_spine_actor = vtkActor()
    aligned_spine_actor.SetMapper(aligned_spine_mapper)
    aligned_spine_actor.GetProperty().SetOpacity(0.5)
    aligned_spine_actor.GetProperty().SetDiffuseColor(colors.GetColor3d('Red'))
    all_actors.append(aligned_spine_actor)

    new_origin = transfer_point(origin, transform_matrix)
    plane_center_point_L = transfer_point(plane_center_point_L, transform_matrix)
    plane_center_point_R = transfer_point(plane_center_point_R, transform_matrix)
    pedicle_lower_reference_point_mid = transfer_point(pedicle_lower_reference_point_mid, transform_matrix)
    # 坐标转换将translate置0
    transform_matrix.SetElement(0, 3, 0)
    transform_matrix.SetElement(1, 3, 0)
    transform_matrix.SetElement(2, 3, 0)

    for axis in coordinates:
        axis.append(1)
        new_axis = transform_matrix.MultiplyPoint(axis)[:-1]
        new_axis = new_axis / np.linalg.norm(new_axis)
        new_coordinate.append(new_axis)
    new_axis = np.asarray(new_coordinate)

    # code used for rendering
    plane_center_point_L_actor = createSphereActor(plane_center_point_L, 3)
    all_actors.append(plane_center_point_L_actor)
    plane_center_point_R_actor = createSphereActor(plane_center_point_R, 3)
    all_actors.append(plane_center_point_R_actor)

    new_origin_actor = createSphereActor(new_origin, 3)
    all_actors.append(new_origin_actor)

    axis_actor = createSpineAxisActor(new_origin, new_axis)
    all_actors.extend(axis_actor)

    origin_spine_mapper = vtkPolyDataMapper()
    origin_spine_mapper.SetInputData(spine_polydata)
    origin_spine_actor = vtkActor()
    origin_spine_actor.SetMapper(origin_spine_mapper)
    origin_spine_actor.GetProperty().SetOpacity(0.5)
    all_actors.append(origin_spine_actor)

    # showActors(all_actors)
    return np.asarray(new_axis), np.asarray(new_origin), np.asarray(plane_center_point_L), np.asarray(plane_center_point_R),\
        np.asarray(pedicle_pipeline_L_normal), np.asarray(pedicle_pipeline_R_normal), np.asarray(pedicle_lower_reference_point_mid)

if __name__ == '__main__':
    points_file_path = '../DeepSpineData/registration_test_data/1.3.6.1.4.1.9328.50.4.0069_seg.nii_label_22_picked_points.pp'
    coordinate_file_path = ''
    stl_file_path = '../DeepSpineData/registration_test_data/1.3.6.1.4.1.9328.50.4.0069_seg.nii_label_21.stl'
    stl_file_path2 = '../DeepSpineData/all_stl/conlon_stl/1.3.6.1.4.1.9328.50.4.0001_seg.nii_label_22.stl'
    registration_method(points_file_path, stl_file_path, stl_file_path2, coordinate_file_path)