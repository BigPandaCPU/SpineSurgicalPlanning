import os
import numpy as np
import vtk
from tools.vtk_tools import *


def create_coordinate_from_polydata(spine_polydata):
    # print('using coordinates created by main axis and second axis of PCA calculated with vtkPCAStatistics')
    # using coordinates created by main axis and second axis
    all_actors = []
    coordinate = []
    colors = vtkNamedColors()

    ConnectivityFilter = vtk.vtkPolyDataConnectivityFilter()
    ConnectivityFilter.SetInputData(spine_polydata)
    ConnectivityFilter.SetExtractionModeToLargestRegion()
    ConnectivityFilter.Update()
    biggest_spine_polydata = ConnectivityFilter.GetOutput()

    center_mass = vtk.vtkCenterOfMass()
    center_mass.SetInputData(biggest_spine_polydata)
    center_mass.SetUseScalarsAsWeights(False)
    center_mass.Update()
    origin_in_vtk = center_mass.GetCenter()
    spine_origin_actor = createSphereActor(origin_in_vtk, 3)
    all_actors.append(spine_origin_actor)

    points = vtkPoints()
    points.DeepCopy(biggest_spine_polydata.GetPoints())
    # bounds = points.GetBounds()
    # points_origin = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2]
    ev, _ = get_eigen_vectors_values(points)
    coordinate.append(np.array(ev[1]))
    coordinate.append(np.array(ev[0]))

    # coordinate = varify_main_axis(coordinate)
    # coordinate = varify_main_axis_points_intersection(coordinate, origin_in_vtk, biggest_spine_polydata)
    coordinate.append(np.array(np.cross(coordinate[0], coordinate[1])))

    coordinate[0] = -coordinate[0]
    coordinate[1] = -coordinate[1]
   # coordinate[2] = -coordinate[2]

    spine_mapper = vtkPolyDataMapper()
    spine_mapper.SetInputData(biggest_spine_polydata)
    spine_mapper.ScalarVisibilityOff()
    spine_actor = vtkActor()
    spine_actor.SetMapper(spine_mapper)
    spine_actor.GetProperty().SetOpacity(0.5)
    spine_actor.GetProperty().SetColor(colors.GetColor3d('Cornsilk'))
    all_actors.append(spine_actor)
    # create_origin_actor = createSphereActor([0,0,0], 3)
    # all_actors.append(create_origin_actor)
    # print(coordinate, '\n', origin_in_vtk)

    # main_axis_actor = createAxisActor([origin_in_vtk - 40 * np.array(ev[0]), origin_in_vtk + 40 * np.array(ev[0])], 'yellow')
    # second_axis_actor = createAxisActor([origin_in_vtk - 40 * np.array(ev[1]), origin_in_vtk + 40 * np.array(ev[1])],
    #                                   'yellow')

    # showActors(all_actors)

   #  new_coordinate = []
   #  trans = vtkTransform()
   # # trans.Translate(-origin_in_vtk[0], -origin_in_vtk[1], -origin_in_vtk[2])
   #  # trans.RotateX(10)
   #  # trans.Translate(origin_in_vtk[0], origin_in_vtk[1], origin_in_vtk[2])
   #  transform_matrix = trans.GetMatrix()
   #
   #  for axis in coordinate:
   #      axis = np.append(axis, 1)
   #      new_axis = transform_matrix.MultiplyPoint(axis)[:-1]
   #      new_axis = new_axis / np.linalg.norm(new_axis)
   #      new_coordinate.append(new_axis)
   #
   #  x_actor = createAxisActor([origin_in_vtk, origin_in_vtk + 40 * new_coordinate[0]], 'DarkRed')
   #  y_actor = createAxisActor([origin_in_vtk, origin_in_vtk + 40 * new_coordinate[1]], 'DarkGreen')
   #  z_actor = createAxisActor([origin_in_vtk, origin_in_vtk + 40 * new_coordinate[2]], 'DarkBlue')
   #  all_actors.append(x_actor)
   #  all_actors.append(y_actor)
   #  all_actors.append(z_actor)
    # save_coordinate('1.3.6.1.4.1.9328.50.4.0003_seg.nii_label_22', origin_in_vtk, new_coordinate)

    world_x_actor = createAxisActor([origin_in_vtk, origin_in_vtk + 40 * np.array([1,0,0])], 'Red')
    world_y_actor = createAxisActor([origin_in_vtk, origin_in_vtk + 40 * np.array([0,1,0])], 'White')
    world_z_actor = createAxisActor([origin_in_vtk, origin_in_vtk + 40 * np.array([0,0,1])], 'Blue')
    # all_actors.append(world_x_actor)
    # all_actors.append(world_y_actor)
    # all_actors.append(world_z_actor)
    # showActors(all_actors)

    return coordinate, biggest_spine_polydata, origin_in_vtk