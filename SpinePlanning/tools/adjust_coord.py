import copy

import numpy as np

from tools.coordinate_registration import registration_polydata
from tools.vtk_tools import *


def coordinate(axis_x_normal, axis_y_normal, axis_z_normal,\
                           spine_polydata, implant_origin, rotation=[0,0,0],  color='LightSteelBlue', opacity=0.5):
    matrix = vtk.vtkMatrix4x4()
    colors = vtkNamedColors()

    trans = vtkTransform()
    trans.Translate(implant_origin[0], implant_origin[1], implant_origin[2])
    for idx, ele in enumerate(rotation):
        if ele !=0:
            if idx == 0:
                trans.RotateWXYZ(ele, axis_x_normal)
            if idx == 1:
                trans.RotateWXYZ(ele, axis_y_normal)
            if idx == 2:
                trans.RotateWXYZ(ele, axis_z_normal)
    trans.Translate(-implant_origin[0], -implant_origin[1], -implant_origin[2])

    transformPD = vtkTransformPolyDataFilter()
    transformPD.SetTransform(trans)
    transformPD.SetInputData(spine_polydata)
    transformPD.Update()

    matrix.DeepCopy(trans.GetMatrix())

    return matrix


def translate_along_axis(origin, axis, translate_length=[0,0,0]):
    print("translate along axis")
    origin = np.asarray(origin)
    axis= np.asarray(axis)
    for idx,ele in enumerate(translate_length):
        origin = origin + ele*axis[idx]
    return origin


def adjust_polydata(label_num, spine_polydata):
    all_actors = []
    new_coordinate = []
    colors = vtkNamedColors()
    target_dir = '../registration_test_data'

    coordinate_file_path = os.path.join(target_dir, label_num+".txt")
    stl_file_path = os.path.join(target_dir, label_num+".stl")
    stl_file_path = "../test_data/bone_test.stl"
    coordinate_file_path = "../test_data/bone_test.txt"
    origin, coordinates, plane_center_point_L, plane_center_point_R, pedicle_pipeline_L_normal,pedicle_pipeline_R_normal, \
            pedicle_reference_point0, pedicle_reference_point1 = load_coordinate_reference_point(coordinate_file_path)
    print(pedicle_reference_point0)
    print(pedicle_reference_point1)
    coordinates = np.asarray(coordinates)
    pedicle_pipeline_L_normal= np.asarray(pedicle_pipeline_L_normal)
    pedicle_pipeline_R_normal = np.asarray(pedicle_pipeline_R_normal)
    cur_spine_actor = createActorFromSTL(stl_file_path, opacity=0.5, color='Green')
    all_actors.append(cur_spine_actor)
   # origin_actor = createSphereActor(origin, 20)
    #all_actors.append(origin_actor)

    origin_spine_polydata = cur_spine_actor.GetMapper().GetInput()

    # trans_matrix = coordinate(coordinates[0], coordinates[1], coordinates[2], \
    #                spine_polydata, origin, rotation=[-33, -4, -3], color='LightSteelBlue', opacity=0.5)
    #
    # trans_matrix.SetElement(0, 3, 0)
    # trans_matrix.SetElement(1, 3, 0)
    # trans_matrix.SetElement(2, 3, 0)
    #
    # for axis in coordinates:
    #     axis = copy.deepcopy(axis)
    #     axis.append(1)
    #     new_axis = trans_matrix.MultiplyPoint(axis)[:-1]
    #     new_axis = new_axis / np.linalg.norm(new_axis)
    #     new_coordinate.append(new_axis)
    # print(new_coordinate)
    # new_axis = np.asarray(new_coordinate)
    translate_length_p0 = np.array([0, 0, 0])
    translate_length_p1 = np.array([0, 0, 0])
    # plane_center_point_L = plane_center_point_L + 15 * coordinates[2] - 28 * coordinates[1]- 8 * new_axis[0]
    # plane_center_point_R = plane_center_point_R + 15 * coordinates[2] - 28 * coordinates[1]+ 3 * new_axis[0]

    normal_l = 0.2*coordinates[0] + 0.8*coordinates[1]
    normal_r = -0.2*coordinates[0] + 0.8*coordinates[1]
    # pedicle_reference_point0 = translate_along_axis(plane_center_point_L,  coordinates, translate_length_p0)
    # pedicle_reference_point1 = translate_along_axis(plane_center_point_R,  coordinates, translate_length_p1)
    # print("p0 :", pedicle_reference_point0)
    # print("p1 :", pedicle_reference_point1)

    pedicle_reference_point0 = createSphereActor(pedicle_reference_point0, 1, color="Red")
    pedicle_reference_point1 = createSphereActor(pedicle_reference_point1, 1, color="Blue")
    all_actors.append(pedicle_reference_point0)
    all_actors.append(pedicle_reference_point1)
    # cutplane = vtkPlane()
    # cutplane.SetOrigin([origin[0], origin[1], origin[2]])
    # cutplane.SetNormal(new_axis[0])
    #
    # cutter = vtkCutter()
    # cutter.SetCutFunction(cutplane)
    # cutter.SetInputData(origin_spine_polydata)
    #
    # stripper = vtk.vtkStripper()
    # stripper.SetInputConnection(cutter.GetOutputPort())
    # stripper.Update()
    # slicing_boundary = stripper.GetOutput()
    # all_actors = []
    # actor = vtkActor()
    # mapper = vtkPolyDataMapper()
    # mapper.SetInputData(cutter.GetOutput())
    # actor.SetMapper(mapper)
    # all_actors.append(actor)

   #  plane_center_point_L_actor = createSphereActor(plane_center_point_L, 5)
   #  plane_center_point_R_actor = createSphereActor(plane_center_point_R,5)
   #  all_actors.append(plane_center_point_R_actor)
   #  all_actors.append(plane_center_point_L_actor)

    original_axis_actor = createSpineAxisActor(origin, coordinates)
    all_actors.extend(original_axis_actor)

    p0 = 30*coordinates[0] + 30*coordinates[1] + origin
    p0_normal = p0/np.linalg.norm(p0)
    print(normal_l)
    print(normal_r)
    left_normal = createAxisActor(
        [plane_center_point_L - normal_l * 40, plane_center_point_L + pedicle_pipeline_L_normal * 40])
    right_normal = createAxisActor(
        [plane_center_point_R - normal_r * 40, plane_center_point_R + pedicle_pipeline_R_normal * 40])
    all_actors.append(right_normal)
    all_actors.append(left_normal)
   #  original_axis_actor_new = createSpineAxisActor(origin, new_axis)
   #  all_actors.extend(original_axis_actor_new)
    showActors(all_actors)


    # save_coordinate(label_num, origin, new_axis, plane_center_point_L, plane_center_point_R,\
    #                 pedicle_pipeline_L_normal, pedicle_pipeline_R_normal)
#    return np.asarray(new_axis), np.asarray(new_origin)


if __name__ == "__main__":
    stl_file = "../test_data\\bone_test.stl"
    label = stl_file.split('\\')[-1][:-4]
    label_num = label[-8:]
    spine_polydata = load_stl(stl_file)
    label_num = "L2"
    new_axis, new_origin = adjust_polydata(label_num, spine_polydata)