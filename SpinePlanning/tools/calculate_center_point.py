import numpy as np
from PCA_coordinate_creation.create_coordinate import create_coordinate_from_polydata
from tools.vtk_tools import *
import cv2
import copy
import matplotlib.pyplot as plt
colors = vtkNamedColors()


def draw_contour(contour_list, size=[300,300]):
    background = np.zeros(size)
    for contour in contour_list:
        background[contour[:, 0], contour[:, 1]] = 1
    return background.astype(np.uint8)


def get_left_right(sliced_contour_array, mode):
    target_contours = []
    contour_array = np.array([])
    contours, hierarchy = cv2.findContours(np.uint8(sliced_contour_array.T), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if mode == 'outer':
        right_contour = np.array([])
        left_contour = np.array([])

        for contour in contours:
            contour_array = np.append(contour_array, np.squeeze(contour)[:, 0])
        mid_y_point = (contour_array.max()+contour_array.min())//2

        for idx, contour in enumerate(contours):
            if np.squeeze(contour)[:, 0].mean() > mid_y_point:
                if right_contour.shape[0] == 0:
                    right_contour = np.squeeze(contour)
                else:
                    right_contour = np.append(right_contour, np.squeeze(contour), axis=0)
            else:
                if left_contour.shape[0] == 0:
                    left_contour = np.squeeze(contour)
                else:
                    left_contour = np.append(left_contour, np.squeeze(contour),  axis=0)
        return left_contour, right_contour

    else:
        contours = sorted(contours, key=lambda c: c.shape[0], reverse=True)

        target_contours.append(np.squeeze(contours[0]))
        target_contours.append(np.squeeze(contours[1]))
        if target_contours[0][0][0] < target_contours[1][0][0]:
            left = target_contours[0]
            right = target_contours[1]
        else:
            left = target_contours[1]
            right = target_contours[0]
        return left, right



def scanning(sliced_contour_list, alternative=False):
    unique_inner_x = np.unique(sliced_contour_list[1][:, 1])
    # initial_array = draw_contour(sliced_contour_list)
    # initial_array[50, 0] = 10
    # plt.imshow(initial_array)
    # plt.show()

    if alternative:
        inner_lefted = np.intersect1d(np.where(sliced_contour_list[1][:, 1] > unique_inner_x.min() + 3)[0], \
                                      np.where(sliced_contour_list[1][:, 1] < unique_inner_x.max() - 3)[0])
        outer_lefted = np.intersect1d(np.where(sliced_contour_list[0][:, 1] > unique_inner_x.min() - 5)[0], \
                                      np.where(sliced_contour_list[0][:, 1] < unique_inner_x.max() - 3)[0])
    else:
        inner_lefted = np.intersect1d(np.where(sliced_contour_list[1][:, 1] > unique_inner_x.min()+3)[0],\
                                       np.where(sliced_contour_list[1][:, 1] < unique_inner_x.max()-3)[0])
        outer_lefted = np.intersect1d(np.where(sliced_contour_list[0][:, 1] > unique_inner_x.min()-10)[0],\
                                       np.where(sliced_contour_list[0][:, 1] < unique_inner_x.max()+5)[0])

    sliced_contour_list[0] = sliced_contour_list[0][outer_lefted, :]
    sliced_contour_list[1] = sliced_contour_list[1][inner_lefted, :]
    bg_outer = draw_contour([sliced_contour_list[0]])
    bg_inner = draw_contour([sliced_contour_list[1]])
    # plt.imshow(bg_outer)
    # plt.show()
    # plt.imshow(bg_inner)
    # plt.show()

    left_outer, right_outer = get_left_right(bg_outer, 'outer')
    left_inner, right_inner = get_left_right(bg_inner, 'inner')
    return [left_outer, right_outer], [left_inner, right_inner]


# def cal_nearest_point_fatmethod2_singlecontours(contour, offset_min, z_coordinate, test_mode=False, alternative=False):


def cal_nearest_point_fatmethod2(outer, inner, offset_min, z_coordinate, test_mode=False, alternative=False):

    # select the fattest points pair, outer: [left, right], inner:[left, right]
    outer_left = outer[0]
    outer_right = outer[1]
    inner_right = inner[1]
  # [left_outer, right_outer], [left_inner, right_inner]
    dist = lambda p, inner_p: np.linalg.norm(p-inner_p)
    inner_dist_max = 0

    outer_left_dist_min = 100
    outer_right_dist_min = 100

    for ilp_idx, inner_left_point in enumerate(inner[0]):
        x_coordinate = inner_left_point[1]
        corres_right_points = inner_right[np.where(inner_right[:, 1] == x_coordinate)]
        for cores_point in corres_right_points:
            dist_cores_inner_points = dist(inner_left_point, cores_point)
            if dist_cores_inner_points > inner_dist_max:
                inner_dist_max = dist_cores_inner_points
                max_inner_left_point = inner_left_point
                max_inner_right_point = cores_point

    if alternative:
        for olp_idx, outer_left_point in enumerate(outer_left):
            outer_left_dist = dist(max_inner_left_point, outer_left_point)
            if outer_left_dist < outer_left_dist_min and abs(max_inner_left_point[1] - outer_left_point[1]) > 3:
                outer_left_dist_min = outer_left_dist
                outer_min_left_point_idx = olp_idx
                outer_left_min_point = outer_left_point

        for orp_idx, outer_right_point in enumerate(outer_right):
            outer_right_dist = dist(max_inner_right_point, outer_right_point)
            if outer_right_dist < outer_right_dist_min and abs(max_inner_left_point[1] - outer_right_point[1]) > 3:
                outer_right_dist_min = outer_right_dist
                outer_min_right_point_idx = orp_idx
                outer_right_min_point = outer_right_point
    else:
        for olp_idx, outer_left_point in enumerate(outer_left):
            outer_left_dist = dist(max_inner_left_point, outer_left_point)
            if outer_left_dist < outer_left_dist_min and abs(max_inner_left_point[1]-outer_left_point[1]) < 3:
                outer_left_dist_min = outer_left_dist
                outer_min_left_point_idx = olp_idx
                outer_left_min_point = outer_left_point

        for orp_idx, outer_right_point in enumerate(outer_right):
            outer_right_dist = dist(max_inner_right_point, outer_right_point)
            if outer_right_dist < outer_right_dist_min and abs(max_inner_left_point[1]-outer_right_point[1])< 3:
                outer_right_dist_min = outer_right_dist
                outer_min_right_point_idx = orp_idx
                outer_right_min_point = outer_right_point

    # if (max(outer_right_dist_min, outer_left_dist_min)-min(outer_right_dist_min, outer_left_dist_min)) > 5:

    if test_mode:
        left_outer_point = np.append(np.append(outer_left_min_point, z_coordinate), 1)
        right_outer_point = np.append(np.append(outer_right_min_point, z_coordinate), 1)
        max_inner_left_point = np.append(np.append(max_inner_left_point, z_coordinate), 1)
        max_inner_right_point = np.append(np.append(max_inner_right_point, z_coordinate), 1)
    else:
        left_outer_point = np.append(np.append(outer_left_min_point+offset_min, z_coordinate), 1)
        right_outer_point = np.append(np.append(outer_right_min_point+offset_min, z_coordinate), 1)
        max_inner_left_point = np.append(np.append(max_inner_left_point+offset_min, z_coordinate), 1)
        max_inner_right_point = np.append(np.append(max_inner_right_point+offset_min, z_coordinate), 1)

    return [left_outer_point, max_inner_left_point], [right_outer_point, max_inner_right_point]


def system_alignment_in_vtk(coordinate, spine_polydata, origin):
   #  print('system coordinate aligning')
    all_actors = []

    world_x_actor = createAxisActor([np.array([0, 0, 0]), np.array([0, 0, 0])+40*np.array([1, 0, 0])], 'DarkRed')
    world_y_actor = createAxisActor([np.array([0, 0, 0]), np.array([0, 0, 0])+40*np.array([0, 1, 0])], 'DarkGreen')
    world_z_actor = createAxisActor([np.array([0, 0, 0]), np.array([0, 0, 0])+40*np.array([0, 0, 1])], 'DarkBlue')
    all_actors.append(world_x_actor)
    all_actors.append(world_y_actor)
    all_actors.append(world_z_actor)
    #
    # spine_x_actor = createAxisActor([origin, origin + 40 * coordinate[0]], 'DarkRed')
    # spine_y_actor = createAxisActor([origin, origin + 40 * coordinate[1]], 'DarkGreen')
    # spine_z_actor = createAxisActor([origin, origin + 40 * coordinate[2]], 'DarkBlue')
    # all_actors.append(spine_x_actor)
    # all_actors.append(spine_y_actor)
    # all_actors.append(spine_z_actor)

    # red:x, green:y, blue:z
    world_origin_actor = createSphereActor([0, 0, 0], 3)
    all_actors.append(world_origin_actor)
    # spine_origin_actor = createSphereActor(origin, 3)
    # all_actors.append(spine_origin_actor)

    spine2world_TF, world2spine_TF = spine2world_vtk(origin, coordinate, True)

    # spine_pca.transform(spine2world_TF)
    trans = vtkTransform()
    transformPD = vtkTransformPolyDataFilter()
    trans.SetMatrix(spine2world_TF)
    transformPD.SetTransform(trans)
    transformPD.SetInputData(spine_polydata)
    transformPD.Update()
    aligned_spine_polydata = transformPD.GetOutput()
    # COM_actor = create_center_of_mass_actor(aligned_spine_polydata)
    # COM_actor.GetProperty().SetColor(colors.GetColor3d("Red"))
    # all_actors.append(COM_actor)

    spine_mapper = vtkPolyDataMapper()
    spine_mapper.SetInputData(aligned_spine_polydata)
    spine_actor = vtkActor()
    spine_actor.SetMapper(spine_mapper)
    spine_actor.GetProperty().SetOpacity(0.5)
    all_actors.append(spine_actor)
   # showActors(all_actors)
    return aligned_spine_polydata, world2spine_TF


def verify_contour_correctness(contours, default_z_slicing, i):
    # if contours[1].shape[0] < 50 or contours[1].shape[0] > 110 or contours[0].shape[0] < 150:
    #     print('z_slicing :', default_z_slicing + i, ',gets out of upper or lower boundary of spine body')
    #     return False
    outer_array = draw_contour([contours[0]])
    inner_array = draw_contour([contours[1]])
    outer_bbox = cv2.boundingRect(outer_array)
    inner_bbox = cv2.boundingRect(inner_array)
    if inner_bbox[0] > outer_bbox[0]\
        and inner_bbox[1] > outer_bbox[1]\
        and inner_bbox[0]+inner_bbox[2] < outer_bbox[0]+outer_bbox[2]\
        and inner_bbox[1]+inner_bbox[3] < outer_bbox[1]+outer_bbox[3]:
        return True

    else:
        print('z_slicing :', default_z_slicing + i, ',two contours are not enclosed')
        return False


def points_variation_filtering(points, n_delete_points=1):
    # delete the y_direction outliers
    # print(n_delete_points, 'need to be deleted')
    points_array = np.array(points)
    points_array_y = points_array[:, 1]
    outlier_y = np.where(np.abs(points_array_y - np.mean(points_array_y, axis=0)) > 5)
    if outlier_y[0].shape[0] < points_array.shape[0]:
        points_array = np.delete(points_array, outlier_y, axis=0)
    # delete points too far from mid point
    if (n_delete_points - len(outlier_y)) > 0:
        n_delete_points = n_delete_points - len(outlier_y)
        dist2center = np.linalg.norm(points_array - np.mean(points_array, axis=0), axis=1)
        for i in range(n_delete_points):
            max_dist_idx = np.argmax(dist2center)
            points_array = np.delete(points_array, max_dist_idx, axis=0)
            dist2center = np.delete(dist2center, max_dist_idx)
    return points_array


def create_fitted_plane(points):
    points = np.asarray(points)
    my_points = vtkPoints()
    for i in range(points.shape[0]):
        # print(points[i])
        my_points.InsertNextPoint(points[i])

    fitted_plane = vtk.vtkPlane()
    plane_origin = [0.0]*3
    plane_normal = [0.0]*3

    fitted_plane.ComputeBestFittingPlane(my_points, plane_origin, plane_normal)
    plane = vtk.vtkPlaneSource()
    plane.SetCenter(plane_origin)
    plane.SetNormal(plane_normal)
    plane.Update()

    scaler = vtkTransform()
    scaler.Translate(plane_origin[0], plane_origin[1], plane_origin[2])
    scaler.Scale(25, 25, 25)
    scaler.Translate(-plane_origin[0], -plane_origin[1], -plane_origin[2])

    transformPD = vtkTransformPolyDataFilter()
    transformPD.SetTransform(scaler)
    transformPD.SetInputData(plane.GetOutput())
    transformPD.Update()

    plane_mapper = vtkPolyDataMapper()
    plane_mapper.SetInputData(transformPD.GetOutput())
    plane_actor = vtkActor()
    plane_actor.SetMapper(plane_mapper)
    plane_actor.GetProperty().SetOpacity(0.8)
    return plane_actor, [plane_origin, plane_normal]


def adjust_points(mean_points, coordinates, adjust_coef):
    print('The point need to be adjusted')
    z_move_dist = 4.5 - adjust_coef
    mean_points[0] = mean_points[0] + coordinates[2]*z_move_dist
    mean_points[1] = mean_points[1] + coordinates[2]*z_move_dist
    return mean_points


def points2vtk(left_points, right_points, spine_polydata, origin, coordinate,\
               world2spine_TF, find_closest_point=True):
    find_closest_point = False
    if world2spine_TF.__class__.__name__ == 'vtkMatrix4x4':
        narray = np.eye(4)
        world2spine_TF.DeepCopy(narray.ravel(), world2spine_TF)
    world2spine_TF = narray
    all_actors = []

    right_points_in_vtk = []
    left_points_in_vtk = []
    point_locator = vtk.vtkPointLocator()
    point_locator.SetDataSet(spine_polydata)
    point_locator.BuildLocator()

    spine_mapper = vtkPolyDataMapper()
    spine_mapper.ScalarVisibilityOff()
    spine_mapper.SetInputData(spine_polydata)
    spine_actor = vtkActor()
    spine_actor.SetMapper(spine_mapper)
    spine_actor.GetProperty().SetColor(colors.GetColor3d("Cornsilk"))
    if find_closest_point:
        spine_actor.GetProperty().SetOpacity(0.5)
    else:
        spine_actor.GetProperty().SetOpacity(0.5)
    all_actors.append(spine_actor)

    # origin_in_vtk = [origin[0]*img_parameters[0][0], origin[1]*img_parameters[0][1], origin[2]*img_parameters[0][2]]
    origin_in_vtk = origin
    spine_origin_actor = createSphereActor(origin_in_vtk, 3)
    all_actors.append(spine_origin_actor)

    x_actor = createAxisActor([origin_in_vtk, origin_in_vtk + 40 * coordinate[0]], 'DarkRed')
    y_actor = createAxisActor([origin_in_vtk, origin_in_vtk + 40 * coordinate[1]], 'DarkGreen')
    z_actor = createAxisActor([origin_in_vtk, origin_in_vtk + 40 * coordinate[2]], 'DarkBlue')
    all_actors.append(x_actor)
    all_actors.append(y_actor)
    all_actors.append(z_actor)

    right_vtk_points = vtkPoints()
    left_vtk_points = vtkPoints()

    for rp in right_points:
        rp_real = np.dot(world2spine_TF, rp)
        # rp_in_vtk = [rp_real[0] * img_parameters[0][0], rp_real[1] * img_parameters[0][1], rp_real[2] * img_parameters[0][2]]
        rp_in_vtk = rp_real[:3]
        if find_closest_point:
            closest_point = spine_polydata.GetPoint(point_locator.FindClosestPoint(rp_in_vtk))
            right_vtk_points.InsertNextPoint(closest_point)
            right_points_in_vtk.append(closest_point)
        else:
            right_points_in_vtk.append(rp_in_vtk)
            right_vtk_points.InsertNextPoint(rp_in_vtk)
    right_points_in_vtk = points_variation_filtering(right_points_in_vtk)
    right_mean_point = np.mean(right_points_in_vtk, axis=0)
    right_plane_actor, right_plane_parameter = create_fitted_plane(right_points_in_vtk)
    # right_plane_actor.GetProperty().SetColor(colors.GetColor3d("Green"))
    # all_actors.append(right_plane_actor)
    right_mean_point_actor = createSphereActor(right_mean_point, 3, color = 'Red')
    all_actors.append(right_mean_point_actor)

    for lp in left_points:
        lp_real = np.dot(world2spine_TF, lp)
        # lp_in_vtk = [lp_real[0] * img_parameters[0][0], lp_real[1] * img_parameters[0][1], lp_real[2] * img_parameters[0][2]]
        lp_in_vtk = lp_real[:3]
        if find_closest_point:
            closest_point = spine_polydata.GetPoint(point_locator.FindClosestPoint(lp_in_vtk))
            left_vtk_points.InsertNextPoint(closest_point)
            left_points_in_vtk.append(closest_point)
        else:
            left_points_in_vtk.append(lp_in_vtk)
            left_vtk_points.InsertNextPoint(lp_in_vtk)
    left_points_in_vtk = points_variation_filtering(left_points_in_vtk)
    left_mean_point = np.mean(left_points_in_vtk, axis=0)
    left_mean_point_actor = createSphereActor(left_mean_point, 3, color = 'Green')
    all_actors.append(left_mean_point_actor)
    # left_plane_actor, left_plane_parameter = create_fitted_plane(left_points_in_vtk)
    # left_plane_actor.GetProperty().SetColor(colors.GetColor3d("Green"))
    # all_actors.append(left_plane_actor)

    for point in right_points_in_vtk:
        p_actor = createSphereActor(point, 3)
        all_actors.append(p_actor)
    for point in left_points_in_vtk:
        p_actor = createSphereActor(point, 3)
        all_actors.append(p_actor)

    # showActors(all_actors)
    return [left_mean_point, right_mean_point]


def calculate_center_point(coordinates, spine_polydata, spine_origin_in_vtk, spine_coordinate_template, test_mode=False, alternative_mode=False):
    default_z_slicing = -3
    n_slices = 12
    slice_idx_sum = 0
    aligned_spine_polydata, world2spine_TF = system_alignment_in_vtk(coordinates, spine_polydata, spine_origin_in_vtk)
    l_points = []
    r_points = []
    for i in range(n_slices):
        # if i != 4:
        #     continue
        contours, offset_min, z_coordinate = slicing_xy_vtk(aligned_spine_polydata, default_z_slicing + i)
        # print(contours[0].shape[0], contours[1].shape[0])
        # if i==11:
        # contour_array = draw_contour(contours)
        # plt.imshow(contour_array)
        # plt.show()

        is_correct = verify_contour_correctness(contours, default_z_slicing, i)
        if not is_correct:
            continue
        slice_idx_sum += (i+default_z_slicing)
        # print(i+default_z_slicing)
        try:
            outer, inner = scanning(contours)
            # outer_array = draw_contour(outer)
            # inner_array = draw_contour(inner)
            # plt.imshow(outer_array)
            # plt.show()
            # plt.imshow(inner_array)
            # plt.show()
            # if i == 6:

            left_points, right_points = cal_nearest_point_fatmethod2(outer, inner, offset_min, z_coordinate, test_mode)
        except Exception as e:
            print(Exception, ":", e)
            continue

        # MAX = contours[0].max(), contours[1].max()
        if test_mode:
            contour_array = draw_contour(contours)
            contour_array[left_points[0][0], left_points[0][1]] = 2
            contour_array[left_points[1][0], left_points[1][1]] = 2
            contour_array[right_points[0][0], right_points[0][1]] = 4
            contour_array[right_points[1][0], right_points[1][1]] = 4

            plt.imshow(np.uint8(contour_array))
            plt.show()
            left_points[0][0] += offset_min
            left_points[0][1] += offset_min
            left_points[1][0] += offset_min
            left_points[1][1] += offset_min
            right_points[0][0] += offset_min
            right_points[0][1] += offset_min
            right_points[1][0] += offset_min
            right_points[1][1] += offset_min

        l_points = l_points + left_points
        r_points = r_points + right_points
    # if True:
    if len(l_points) < 1 or len(r_points) < 1:  # at lease 2 points are needed to create the mid point
        print("get points directly")
        return spine_coordinate_template[3], spine_coordinate_template[2]
    else:
        # plane parameters: [origin, normal]
        mean_points = points2vtk(l_points, r_points, spine_polydata, spine_origin_in_vtk,\
                                                                   coordinates, world2spine_TF, find_closest_point=True)

        # if slice_idx_sum/n_slices <= 2:
        mean_points = adjust_points(mean_points, coordinates, slice_idx_sum/n_slices)
        # create_spine_cutplane_center(left_plane_parameters, right_plane_parameters, spine_polydata, coordinates,
        #                              spine_origin_in_vtk)

        # verify left and right, right point always in the positive direction of x-axis
        plane_position = coordinates[0][0]*(spine_origin_in_vtk[0]-mean_points[0][0]) + \
            coordinates[0][1]*(spine_origin_in_vtk[1]-mean_points[0][1]) + \
            coordinates[0][2]*(spine_origin_in_vtk[2]-mean_points[0][2])
    # plane_position2 = coordinates[0][0]*(spine_origin_in_vtk[0]-mean_points[1][0]) + \
    #     coordinates[0][1]*(spine_origin_in_vtk[1]-mean_points[1][1]) + \
    #     coordinates[0][2]*(spine_origin_in_vtk[2]-mean_points[1][2])
        if plane_position > 0:  # right point
            return mean_points[0], mean_points[1]
        else:
            return mean_points[1], mean_points[0]


def calculate_center_point_alternative(coordinates, spine_polydata, spine_origin_in_vtk):
    alternative = True
    default_z_slicing = -3
    n_slices = 12
    slice_idx_sum = 0
    aligned_spine_polydata, world2spine_TF = system_alignment_in_vtk(coordinates, spine_polydata, spine_origin_in_vtk,)
    l_points = []
    r_points = []
    for i in range(n_slices):
        contours, offset_min, z_coordinate = slicing_xy_vtk(aligned_spine_polydata, default_z_slicing + i)
        # print(contours[0].shape[0], contours[1].shape[0])
        # contour_array = draw_contour(contours)
        # plt.imshow(contour_array)
        # plt.show()

        is_correct = verify_contour_correctness(contours, default_z_slicing, i)
        if not is_correct:
            continue
        slice_idx_sum += (i+default_z_slicing)
        try:
            outer, inner = scanning(contours, alternative)
            left_points, right_points = cal_nearest_point_fatmethod2(outer, inner, offset_min, z_coordinate, test_mode=False, alternative=alternative)
        except Exception as e:
            print(Exception, ":", e)
            continue
        # MAX = contours[0].max(), contours[1].max()
        # contour_array = draw_contour(contours)
        #
        # contour_array[left_points[0][0], left_points[0][1]] = 2
        # contour_array[left_points[1][0], left_points[1][1]] = 2
        # contour_array[right_points[0][0], right_points[0][1]] = 4
        # contour_array[right_points[1][0], right_points[1][1]] = 4
        #
        # plt.imshow(np.uint8(contour_array))
        # plt.show()
        # left_points[0][0] += offset_min
        # left_points[0][1] += offset_min
        # left_points[1][0] += offset_min
        # left_points[1][1] += offset_min
        # right_points[0][0] += offset_min
        # right_points[0][1] += offset_min
        # right_points[1][0] += offset_min
        # right_points[1][1] += offset_min

        l_points = l_points + left_points
        r_points = r_points + right_points
    if len(l_points) == 0 or len(r_points) == 0:
        print('Still no points is found with alternative method')
        return spine_origin_in_vtk, spine_origin_in_vtk

    # plane parameters: [origin, normal]
    mean_points = points2vtk(l_points, r_points, spine_polydata, spine_origin_in_vtk,\
                                                               coordinates, world2spine_TF, find_closest_point=True)

    if slice_idx_sum/n_slices <= 2:
        mean_points = adjust_points(mean_points, coordinates, slice_idx_sum/n_slices)
    # create_spine_cutplane_center(left_plane_parameters, right_plane_parameters, spine_polydata, coordinates,
    #                              spine_origin_in_vtk)

    # verify left and right, right point always in the positive direction of x-axis
    plane_position = coordinates[0][0]*(spine_origin_in_vtk[0]-mean_points[0][0]) + \
        coordinates[0][1]*(spine_origin_in_vtk[1]-mean_points[0][1]) + \
        coordinates[0][2]*(spine_origin_in_vtk[2]-mean_points[0][2])
    # plane_position2 = coordinates[0][0]*(spine_origin_in_vtk[0]-mean_points[1][0]) + \
    #     coordinates[0][1]*(spine_origin_in_vtk[1]-mean_points[1][1]) + \
    #     coordinates[0][2]*(spine_origin_in_vtk[2]-mean_points[1][2])
    if plane_position > 0:  # right point, axis direction of the alternative method is different with
        return mean_points[1], mean_points[0]
    else:
        return mean_points[0], mean_points[1]