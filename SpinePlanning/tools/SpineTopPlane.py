import os
import open3d as o3d
from vtkmodules.util import numpy_support
from tools.vtk_tools import *

if __name__ == "__main__":
    #point_file_dir = "E:/data/DeepSpineData/picked_points"
    stl_dir = "E:/data/DeepSpineData/stl"
    stl_file_names = os.listdir(stl_dir)
    for stl_file_name in stl_file_names:
        if "1.3.6.1.4.1.9328.50.4.0002_seg.nii_label_" not in stl_file_name:
            continue
        print("\n"+stl_file_name)
        all_actors = []
        # stl_file_name = point_file_name.replace("_picked_points.pp", ".stl")
        stl_file_path = os.path.join(stl_dir, stl_file_name)
        spine_actor = createActorFromSTL(stl_file_path, opacity=0.75)
        #print(cur_spine_actor.GetMapper().GetInput())

        spine_points = getPointsFromSTL(stl_file_path, num_points=5000)
        spine_points_center, spine_points_main_vector_points, spine_points_main_vector_normal = getMainVectorByPCA(spine_points, delta=70.0)
        spine_points_main_vector_normal_actor = createLineActor(spine_points_main_vector_points)

        intersect_points_with_spine = getIntersectPointsFromLineAndPolyData(spine_points_main_vector_points[0], spine_points_main_vector_points[1], spine_actor.GetMapper().GetInput())
        spine_points_main_vector_normal = checkSpineMainVectorNormal(intersect_points_with_spine, spine_points_center, spine_points_main_vector_normal)
        spine_points_main_vector_points = [spine_points_center + 60.0 * spine_points_main_vector_normal, spine_points_center - 60.0 * spine_points_main_vector_normal]

        print(intersect_points_with_spine)

        spine_points_center_plane_actor = createCirclePlaneActor(spine_points_center, spine_points_main_vector_normal, color='cyan', radius=30.0, opacity=1.0)



        cut_center = spine_points_center
        cliped_spine_polydata = createClipedPolydata(cut_center, spine_points_main_vector_normal, spine_actor.GetMapper().GetInput())
        cliped_spine_actor = createActorFromPolydata(cliped_spine_polydata, opacity=0.75)


        cliped_spine_points = np.asarray(cliped_spine_polydata.GetPoints().GetData())
        #print(cliped_spine_points.shape)

        aim_points = cliped_spine_points[0:-1:10,:]
        #print(aim_points.shape)

        cliped_spine_points_fit_plane_actor,\
        cliped_spine_points_fit_plane_center, \
        cliped_spine_points_fit_plane_normal = fitPlaneActorFromPoints(cliped_spine_points)

        #cliped_spine_points_fit_plane_normal = -cliped_spine_points_fit_plane_normal


        cliped_spine_points_fit_plane_normal_points = np.array([cliped_spine_points_fit_plane_center, cliped_spine_points_fit_plane_center + 20.0 * cliped_spine_points_fit_plane_normal])
        cliped_spine_points_main_vector_actor = createLineActor(cliped_spine_points_fit_plane_normal_points , color="green", line_width=5.0)



        project_point0 = calProjectedPointCoordOnPlane(spine_points_main_vector_normal, spine_points_center, cliped_spine_points_fit_plane_normal_points[0])
        project_point1 = calProjectedPointCoordOnPlane(spine_points_main_vector_normal, spine_points_center, cliped_spine_points_fit_plane_normal_points[1])

        project_normal = normalizeVector(project_point1-project_point0)
        project_normal_points = np.array([cliped_spine_points_fit_plane_center, cliped_spine_points_fit_plane_center + 20.0 * project_normal])
        project_normal_actor = createLineActor(project_normal_points, color="yellow", line_width=5.0)

        intersect_points = getIntersectPointsFromLineAndPolyData(project_normal_points[0], project_normal_points[1], cliped_spine_polydata)
        intersect_point0_actor = createSphereActor(intersect_points[0], radius=5.0, opacity=1.0, color='red')
        #all_actors.append(intersect_point0_actor)

        dis = np.sqrt(np.sum(np.square(cliped_spine_points - intersect_points[0]), axis=1))
        radius = 6.0
        idx = np.where(dis<radius)
        aim_points = cliped_spine_points[idx]
        aim_points_actor = createPointsActor(aim_points, radius=0.2, color='red')

        spine_axis_z_plane_actor, spine_axis_origin, spine_axis_z_normal = fitPlaneActorFromPoints(aim_points,)
        spine_axis_z_normal_points = np.array([spine_axis_origin+30.0*spine_axis_z_normal, spine_axis_origin-30.0*spine_axis_z_normal])
        spine_axis_z_actor = createLineActor(spine_axis_z_normal_points)

        spine_axis_y_point0 = calProjectedPointCoordOnPlane(spine_axis_z_normal, spine_axis_origin, spine_points_main_vector_points[0])
        spine_axis_y_point1 = calProjectedPointCoordOnPlane(spine_axis_z_normal, spine_axis_origin, spine_points_main_vector_points[1])
        spine_axis_y_normal = normalizeVector(spine_axis_y_point1 - spine_axis_y_point0)

        spine_axis_y_points = np.array([spine_axis_origin, spine_axis_origin+20*spine_axis_y_normal])
        spine_axis_y_actor = createLineActor(spine_axis_y_points, color='green')

        spine_axis_x_normal = vectorCross(spine_axis_z_normal, spine_axis_y_normal)
        spine_axis_x_points = np.array([spine_axis_origin, spine_axis_origin-20*spine_axis_x_normal])
        spine_axis_x_actor = createLineActor(spine_axis_x_points, color='red')

        spine_axis_y_cut_plane_actor = createCirclePlaneActor(spine_points_center, spine_axis_y_normal,
                                                                 color='yellow', radius=30.0, opacity= 0.9)

        num_points = intersect_points_with_spine.shape[0]
        colors = ['red', 'green', 'blue', 'yellow', 'pink']
        for i in range(num_points):
            cur_point = intersect_points_with_spine[i]
            cur_point_actor = createSphereActor(cur_point, radius=2.0, opacity=1.0, color=colors[i])
            all_actors.append(cur_point_actor)


        all_actors.append(spine_actor)
        all_actors.append(spine_points_main_vector_normal_actor)
        all_actors.append(cliped_spine_points_fit_plane_actor)
        all_actors.append(cliped_spine_points_main_vector_actor)

        all_actors.append(spine_points_center_plane_actor)
        all_actors.append(spine_axis_x_actor)
        all_actors.append(spine_axis_y_actor)
        all_actors.append(spine_axis_z_actor)

        all_actors.extend(aim_points_actor)

        all_actors.append(project_normal_actor)
        all_actors.append(spine_axis_z_plane_actor)
        all_actors.append(spine_axis_y_cut_plane_actor)
       # showActors(all_actors)




