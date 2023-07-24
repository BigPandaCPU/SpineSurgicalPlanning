import os
import time
import numpy as np
from vtkmodules.all import vtkActor
from vtkmodules.vtkCommonColor import vtkNamedColors
from tools.vtk_tools import *
from tools.calculate_center_point import calculate_center_point
from tools.coordinate_registration import registration_polydata


def PedicleSurgeryPlanning(spine_polydata, label_num, save_result=False, render=False):
    all_actors = []
    ### step1: create the spine axis ###
    spine_coordinate_template = registration_polydata(label_num, spine_polydata)
    new_axis = spine_coordinate_template[0]
    new_origin = spine_coordinate_template[1]
    plane_center_point_L_template = spine_coordinate_template[2]
    plane_center_point_R_template = spine_coordinate_template[3]
    pedicle_pipeline_L_normal = spine_coordinate_template[4]
    pedicle_pipeline_R_normal = spine_coordinate_template[5]
    pedicle_lower_reference_point_mid = spine_coordinate_template[6]
    # save_coordinate(label_num, new_origin,new_axis)

    cur_spine_axis_normal = new_axis
    cur_spine_axis_origin = new_origin
    cur_spine_points_center = new_origin
    cur_spine_points_main_vector_points = [new_origin + 40 * new_axis[1], new_origin - 40 * new_axis[1]]
    cur_spine_axis_z_normal = new_axis[2]

    ### step2:get the middle point of the epedicle pipeline of the left and right ###
    plane_center_point_L, plane_center_point_R = calculate_center_point(cur_spine_axis_normal, spine_polydata,
                                                                        cur_spine_points_center, spine_coordinate_template, test_mode=False)
    # if plane_center_point_L.size == 0 or plane_center_point_R.size == 0:
    #     plane_center_point_L = plane_center_point_L_template
    #     plane_center_point_R = plane_center_point_R_template
        
    ### step3: surgical planning  ###
    project_angle = getAngleBetweenLineAndPlane(cur_spine_points_main_vector_points[0],
                                                cur_spine_points_main_vector_points[1], cur_spine_axis_z_normal,
                                                cur_spine_axis_origin)
    #project_point = np.array([0.0, np.cos(project_angle), np.sin(project_angle)]) ### pedicle pipeline parallel to spine points main vector
    project_point = np.array([0.0, 1.0, 0.0]) ### pedicle pipeline parallel to spine axis z=0 plane

    pedicle_pipeline_R_normal = createPediclePipelineNormal(-12, cur_spine_axis_normal, project_point)
    pedicle_pipeline_L_normal = createPediclePipelineNormal(12, cur_spine_axis_normal, project_point)

    if save_result:
        save_coordinate(label_num, cur_spine_points_center, cur_spine_axis_normal, plane_center_point_L, plane_center_point_R,
                    pedicle_pipeline_L_normal, pedicle_pipeline_R_normal)

    if render:  # create actor and visualize the result
        # create spine actor
        colors = vtkNamedColors()
        spine_mapper = vtkPolyDataMapper()
        spine_mapper.SetInputData(spine_polydata)
        spine_actor = vtkActor()
        spine_actor.SetMapper(spine_mapper)
        color = "Cornsilk"
        spine_actor.GetProperty().SetColor(colors.GetColor3d(color))
        spine_actor.GetProperty().SetOpacity(0.75)
        all_actors.append(spine_actor)

        #     actor = vtkActor()
        #     actor.SetMapper(mapper)
        #     actor.GetProperty().SetColor(colors.GetColor3d(color))
        #     actor.GetProperty().SetOpacity(opacity)


        # left and right center point of PediclePipeline
        L_point_actor = createSphereActor(plane_center_point_L, 3, color="magenta" )
        R_point_actor = createSphereActor(plane_center_point_R, 3, color = "yellow")
        all_actors.append(L_point_actor)
        all_actors.append(R_point_actor)

        # in and out point auxiliary line
        pedicle_pipeline_points_R = np.array([plane_center_point_R - 50.0 * pedicle_pipeline_R_normal,
                                              plane_center_point_R + 50.0 * pedicle_pipeline_R_normal])
        pedicle_pipeline_actor_R = createLineActor(pedicle_pipeline_points_R, color="yellow", line_width=5.0)
        pedicle_pipeline_points_L = np.array([plane_center_point_L - 50 * pedicle_pipeline_L_normal,
                                              plane_center_point_L + 50.0 * pedicle_pipeline_L_normal])
        pedicle_pipeline_actor_L = createLineActor(pedicle_pipeline_points_L, color="magenta", line_width=5.0)
        all_actors.append(pedicle_pipeline_actor_R)
        all_actors.append(pedicle_pipeline_actor_L)

        # actors of the coordinate and pac axis
        cur_spine_points_main_vector_actor = createLineActor(cur_spine_points_main_vector_points, color="blue",
                                                             line_width=5.0)

        cur_spine_points_center_actor = createSphereActor(cur_spine_points_center, radius=2.0, opacity=1.0, color='Red')
        cur_spine_axis_actor = createSpineAxisActor(cur_spine_axis_origin, cur_spine_axis_normal)

        # all_actors.append(cur_spine_points_main_vector_actor)
        all_actors.append(cur_spine_points_center_actor)
        all_actors.extend(cur_spine_axis_actor)

        # intersection point of screw and pedicle
        # intersect_points_R = getIntersectPointsFromLineAndPolyData(
        #     plane_center_point_R - 100.0 * pedicle_pipeline_R_normal,
        #     plane_center_point_R + 100.0 * pedicle_pipeline_R_normal,
        #     spine_polydata)
        # intersect_points_R_actor = createIntersectPointsActor(intersect_points_R, 1.0, opacity=1.0, color="yellow")
        # all_actors.extend(intersect_points_R_actor)

        # intersect_points_L = getIntersectPointsFromLineAndPolyData(
        #     plane_center_point_L - 100.0 * pedicle_pipeline_L_normal,
        #     plane_center_point_L + 100.0 * pedicle_pipeline_L_normal,
        #     spine_polydata)
        # intersect_points_L_actor = createIntersectPointsActor(intersect_points_L, 1.0, opacity=1.0, color="magenta")
        # all_actors.extend(intersect_points_L_actor)

        # simulated screw
        # pedicle_pipeline_cylinder_R_actor = createPediclePipelineCylinderActor(intersect_points_R[0],
        #                                                                        intersect_points_R[1],
        #                                                                        color="yellow", )
        # pedicle_pipeline_cylinder_L_actor = createPediclePipelineCylinderActor(intersect_points_L[0],
        #                                                                        intersect_points_L[1],
        #                                                                        color="magenta", )
        # all_actors.append(pedicle_pipeline_cylinder_R_actor)
        # all_actors.append(pedicle_pipeline_cylinder_L_actor)

        showActors(all_actors, label_num)

label_dict={1:"C1", 2:"C2", 3:"C3", 4:"C4", 5:"C5", 6:"C6", 7:"C7", 8:"T1", 9:"T2",
                 10:"T3", 11:"T4", 12:"T5", 13:"T6", 14:"T7", 15:"T8", 16:"T9", 17:"T10",
                 18:"T11", 19:"T12", 20:"L1", 21:"L2", 22:"L3", 23:"L4", 24:"L5", 25:"L5"}



if __name__ == "__main__":
    # stl_dir = "D:\pointnet2\PointNet2_deepspine\DeepSpineData\\all_stl\conlon_stl"
    # stl_names = os.listdir(stl_dir)
    # for stl_name in stl_names:
    #     if "1.3.6.1.4.1.9328.50.4.0002_seg.nii_label_19" not in stl_name:
    #         continue
    # stl_file = "D:\pointnet2\PointNet2_deepspine\DeepSpineData\\all_stl\conlon_stl\\1.3.6.1.4.1.9328.50.4.0747_seg.nii_label_23.stl"
    # label_num = "L4"
    # spine_polydata = load_stl(stl_file)
    # PedicleSurgeryPlanning(spine_polydata, label_num, render=True, save_result=False)

    stl_dir = "/media/alg/data3/DeepSpineData/CTSpine1k_new/image/stl"
    stl_names = os.listdir(stl_dir)
    for stl_name in stl_names:
        if "Test09" not in stl_name:
            continue
        stl_file = os.path.join(stl_dir, stl_name)
        cur_label = int(stl_name.split("_")[3][:2])
        label_num = label_dict[cur_label]
        spine_polydata = load_stl(stl_file)
        PedicleSurgeryPlanning(spine_polydata, label_num, render=True, save_result=False)



    # nii_file = "./test_data/label/Test10_seg.nii.gz"
    # img_array, origin, spacing, direction = getImageFromNII(nii_file)
    # unique_values = np.unique(img_array)
    #
    # label_dict={1:"C1", 2:"C2", 3:"C3", 4:"C4", 5:"C5", 6:"C6", 7:"C7", 8:"T1", 9:"T2",
    #             10:"T3", 11:"T4", 12:"T5", 13:"T6", 14:"T7", 15:"T8", 16:"T9", 17:"T10",
    #             18:"T11", 19:"T12", 20:"L1", 21:"L2", 22:"L3", 23:"L4", 24:"L5"}
    #
    # tmp_array = np.zeros_like(img_array)
    # for i in range(len(unique_values)):
    #     cur_value = unique_values[i]
    #     print("label :", cur_value)
    #     tmp_array[:] = 0
    #     if cur_value > 0 and cur_value < 25:
    #         label_num = label_dict[i]
    #         if cur_value != 6:
    #             continue
    #         cur_idx = np.where(img_array == cur_value)
    #         tmp_array[cur_idx] = 1
    #         #tmp_array = tmp_array.astype(int)
    #
    #         cur_polydata_normal, cur_polydata = createPolyDataNormalsFromArray(tmp_array, spacing, origin)
    #
    #         PedicleSurgeryPlanning(cur_polydata, label_num, render=True, save_result=False)
    #
