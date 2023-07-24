import math
import time

import vtk
import os
#import open3d as o3d
import numpy as np
#from PyQt5.QtWidgets import QMessageBox
import SimpleITK as sitk
#from skspatial.objects import Plane, Points
import copy
from vtkmodules.all import vtkClipClosedSurface, vtkCutter
from vtkmodules.all import vtkClipPolyData
from vtkmodules.all import vtkDataSetMapper
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkPolyData,
    vtkPolyLine
)

from vtkmodules.all import vtkOBBTree
from vtkmodules.all import vtkCylinderSource

from vtkmodules.vtkCommonCore import (
    VTK_VERSION_NUMBER,
    vtkVersion
)
from vtkmodules.vtkRenderingCore import (
    vtkCamera,
    vtkRenderer
)

from vtkmodules.all import vtkPolyDataConnectivityFilter
from vtkmodules.vtkFiltersSources import vtkRegularPolygonSource

from vtkmodules.vtkFiltersCore import (
    vtkFlyingEdges3D,
    vtkMarchingCubes,
    vtkWindowedSincPolyDataFilter,
    vtkPolyDataNormals,
)

from vtkmodules.vtkRenderingCore import (
    vtkPolyDataMapper,
    vtkActor,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
)

from vtkmodules.all import vtkInteractorStyleTrackballCamera
#from vtk.util import numpy_support

from vtkmodules.util.vtkConstants import VTK_UNSIGNED_CHAR

from vtkmodules.all import vtkPlane, vtkPlaneCollection
from vtkmodules.all import vtkTransform, vtkTransformPolyDataFilter
from vtkmodules.all import vtkPlaneSource, vtkDiskSource
from vtkmodules.util import numpy_support
from vtkmodules.all import vtkDataSetSurfaceFilter

from vtkmodules.vtkIOGeometry import vtkSTLWriter



def saveSTLFile(save_stl_file_name, poly_data_normals):
    """

    :param save_stl_file_name:
    :param poly_data_normals:
    :return:
    """
    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName(save_stl_file_name)
    stl_writer.SetInputConnection(poly_data_normals.GetOutputPort())
    stl_writer.SetFileTypeToBinary()
    stl_writer.Write()
    stl_writer.Update()

def getImageFromNII(nii_file):
    """
    func:get mask array,origin, spacing from nii
    :param nii_file:
    :return:
    """
    if not os.path.exists(nii_file):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("the nrrd file %s is not exist!"%nii_file)
            msg.exec_()
            return

    reader = sitk.ReadImage(nii_file)
    origin = reader.GetOrigin()
    spacing = reader.GetSpacing()
    direction = reader.GetDirection()
    img = sitk.GetArrayFromImage(reader)
    return img, origin, spacing, direction


def numpy2VTK(img, spacing=[1.0, 1.0, 1.0], origin=[0, 0, 0]):
    # evolved from code from Stou S.,
    # on http://www.siafoo.net/snippet/314
    importer = vtk.vtkImageImport()

    img_data = img.astype('uint8')
    img_string = img_data.tobytes()  # type short
    dim = img.shape

    importer.CopyImportVoidPointer(img_string, len(img_string))
    importer.SetDataScalarType(VTK_UNSIGNED_CHAR)
    importer.SetNumberOfScalarComponents(1)

    extent = importer.GetDataExtent()
    importer.SetDataExtent(extent[0], extent[0] + dim[2] - 1,
                           extent[2], extent[2] + dim[1] - 1,
                           extent[4], extent[4] + dim[0] - 1)
    importer.SetWholeExtent(extent[0], extent[0] + dim[2] - 1,
                            extent[2], extent[2] + dim[1] - 1,
                            extent[4], extent[4] + dim[0] - 1)

    importer.SetDataSpacing(spacing[0], spacing[1], spacing[2])
    importer.SetDataOrigin(origin[0], origin[1], origin[2])

    return importer


def saveSTLFileFromPolyData(save_stl_file_name, poly_data):
    """

    :param save_stl_file_name:
    :param poly_data_normals:
    :return:
    """
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(poly_data)
    surface_filter.Update()

    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputConnection(surface_filter.GetOutputPort())


    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName(save_stl_file_name)
    #stl_writer.SetInputData(poly_data)
    stl_writer.SetInputConnection(triangle_filter.GetOutputPort())
    stl_writer.Write()
    stl_writer.Update()

def savePointFile(points, save_point_file):
    """
    :param points: numpy array, m*4, [x,y,z, label]
    :param save_point_file:
    :return:
    """
    fp = open(save_point_file, "w")
    for i in range(points.shape[0]):
        out_str = "%.3f %.3f %.3f %d\n"%(points[i][0], points[i][1], points[i][2], points[i][3])
        fp.write(out_str)
    fp.close()



def saveSTLFileFromPolyDataNormals(save_stl_file_name, poly_data_normals):
    """

    :param save_stl_file_name:
    :param poly_data_normals:
    :return:
    """
    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName(save_stl_file_name)
    stl_writer.SetInputConnection(poly_data_normals.GetOutputPort())
    stl_writer.SetFileTypeToBinary()
    stl_writer.Write()
    stl_writer.Update()

def saveActorAsSTL(save_stl_file, actor):
    """
    func:将actor保存为STL文件
    :param save_stl_file:
    :param actor:
    :return:
    """
    mapper = actor.GetMapper()
    poly_data = mapper.GetInput()
    #points = poly_data.GetPoints()
    #src_points = numpy_support.vtk_to_numpy(poly_data.GetPoints().GetData())
    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName(save_stl_file)
    stl_writer.SetInputData(poly_data)
    stl_writer.Write()
    stl_writer.Update()

def createPolyDataNormalsFromArray(img_array, spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0], use_flying_edges=True, get_largest_connect_region=True):
    """

    :param img_array:
    :param spacing:
    :param origin:
    :param use_flying_edges:
    :return:
    """

    importer = numpy2VTK(img_array, spacing=spacing, origin=origin)

    if not use_flying_edges:
        try:
            skin_extractor = vtkFlyingEdges3D()

        except AttributeError:
            skin_extractor = vtkMarchingCubes()
    else:
        skin_extractor = vtkMarchingCubes()

        # femur process #
    skin_extractor.ComputeGradientsOff()
    skin_extractor.ComputeNormalsOff()
    skin_extractor.SetInputConnection(importer.GetOutputPort())
    skin_extractor.SetValue(0, 1)
    skin_extractor.Update()

    smooth = vtkWindowedSincPolyDataFilter()
    smooth.SetInputData(skin_extractor.GetOutput())
    smooth.SetNumberOfIterations(20)
    pass_band = 0.01
    smooth.SetPassBand(pass_band)
    smooth.BoundarySmoothingOff()
    smooth.FeatureEdgeSmoothingOff()
    smooth.NonManifoldSmoothingOn()
    smooth.NormalizeCoordinatesOn()
    smooth.Update()

    normal_gen = vtkPolyDataNormals()
    normal_gen.ConsistencyOn()  # discreate marching cubes may generate inconsistent surface
    # we almost always perform smoothing, so aplitting would not be able to preserve any sharp features
    # (and sharp edges would look like artifacts in the smooth surface).
    normal_gen.SplittingOff()

    if get_largest_connect_region:
        ConnectivityFilter = vtk.vtkPolyDataConnectivityFilter()
        ConnectivityFilter.SetInputData(smooth.GetOutput())
        ConnectivityFilter.SetExtractionModeToLargestRegion()
        ConnectivityFilter.Update()
        normal_gen.SetInputData(ConnectivityFilter.GetOutput())
    else:
        normal_gen.SetInputData(smooth.GetOutput())
    normal_gen.Update()

    return normal_gen, smooth.GetOutput()

# def createPlaneActor(plane_center, plane_normal, opacity=0.5, color="Cornsilk"):
#     """
#     func:
#     :param plane_center:
#     :param plane_normal:
#     :param color:
#     :return:
#     """
#     colors = vtkNamedColors()
#     plane_source = vtkDiskSource()
#     plane_source.SetInnerRadius(0.0)
#     plane_source.SetOuterRadius(10.0)
#
#     plane_source.Update()
#     plane = plane_source.GetOutput()
#
#     # Create a mapper and actor
#     mapper = vtkPolyDataMapper()
#     mapper.SetInputData(plane)
#
#     actor = vtkActor()
#     actor.SetMapper(mapper)
#     actor.GetProperty().SetColor(colors.GetColor3d(color))
#     actor.GetProperty().SetOpacity(opacity)
#     return actor

def createCirclePlaneActor(plane_center, plane_normal, color="cyan", radius=8.0, opacity=0.5):
    """

    :param plane_center:
    :param plane_normal:
    :return:
    """
    colors = vtkNamedColors()
    polygonSource = vtkRegularPolygonSource()
    # Comment this line to generate a disk instead of a circle.
    polygonSource.GeneratePolygonOff()
    polygonSource.SetNumberOfSides(500)
    polygonSource.SetRadius(radius)
    polygonSource.SetGeneratePolygon(True)
    # polygonSource.SetCenter()
    # polygonSource.SetNormal()
    polygonSource.SetCenter(plane_center[0], plane_center[1], plane_center[2])
    polygonSource.SetNormal(plane_normal[0], plane_normal[1], plane_normal[2])

    #  Visualize
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(polygonSource.GetOutputPort())

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d(color))
    actor.GetProperty().SetOpacity(opacity)
    return actor


def create_center_of_mass_actor(polydata, opacity=0.5):
    centerOfMassFilter = vtk.vtkCenterOfMass()
    centerOfMassFilter.SetInputData(polydata)
    centerOfMassFilter.SetUseScalarsAsWeights(False)
    centerOfMassFilter.Update()

    COM_actor = createSphereActor(centerOfMassFilter.GetCenter(), 3, opacity)
    return COM_actor, centerOfMassFilter.GetCenter()


def createActorFromPolydata(polydata, opacity=1.0, color='Cornsilk'):
    """

    :param polydata:
    :param opacity:
    :param color:
    :return:
    """
    colors = vtkNamedColors()
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d(color))
    actor.GetProperty().SetOpacity(opacity)
    return actor


def createPointsActor(points, radius=1.0 ,opacity=0.5, color='Cornsilk'):
    """

    :param points:
    :param radius:
    :param color:
    :return:
    """
    points_actor = []
    for point in points:
        point_actor = createSphereActor(point, radius, opacity, color)
        points_actor.append(point_actor)
    return points_actor

def fitPlaneActorFromPoints(pts, color="cyan", radius=8.0):
    """
    func:最小二乘算法拟合算法拟合出点云的平面

    :return:
    """
    points = Points(pts)
    plane = Plane.best_fit(points)
    plane_normal = np.array([plane.normal[0], plane.normal[1], plane.normal[2]])
    plane_point = np.array([plane.point[0], plane.point[1], plane.point[2]])
    plane_actor = createCirclePlaneActor(plane_point, plane_normal, color, radius)
    return plane_actor, plane_point, plane_normal


def createSphereActor(center, radius, opacity=0.5, color='Cornsilk'):
    """

    :param center:
    :param radius:
    :param opacity:
    :param color:
    :return:
    """
    colors = vtkNamedColors()
    sphere = vtkSphereSource()
    sphere.SetCenter(center[0], center[1], center[2])
    sphere.SetRadius(radius)
    sphere.SetPhiResolution(100)
    sphere.SetThetaResolution(100)
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d(color))
    actor.GetProperty().SetOpacity(opacity)
    return actor


def load_stl(stl_file_path):
    cur_spine_actor = createActorFromSTL(stl_file_path, opacity=0.75)
    cur_polydata = cur_spine_actor.GetMapper().GetInput()
    return cur_polydata


def getPointsFromSTL(stl_file, num_points=1000):
    """

    :param stl_file:
    :param num_points:
    :return:
    """
    stl = o3d.io.read_triangle_mesh(stl_file)
    pcd = stl.sample_points_poisson_disk(
        number_of_points=num_points)  # sample_points_poisson_disk  sample_points_uniformly
    points_xyz = np.asarray(pcd.points)
    return points_xyz

def getNearestNPointsFromSrcPoints(seed_point, src_points, around_points_num=10):
    """
    func:
    :param seed_point:
    :param src_points:
    :return:
    """
    src_points_shape = src_points.shape
    src_points = src_points.reshape(src_points_shape[0], 1, src_points_shape[1])
    seed_point = np.expand_dims(seed_point, axis=0)

    dis_square = np.sum(np.power((src_points - seed_point), 2), axis=2)

    idx = np.argsort(dis_square, axis=0)

    aim_points_idx = idx[0:around_points_num, :]
    aim_points_idx = np.reshape(aim_points_idx, -1)
    src_points = np.squeeze(src_points, axis=1)
    aim_points = src_points[aim_points_idx,:]
    return aim_points

def createPolyDataFromSTL(stl_file):
    """
    :param stl_file:
    :return:
    """
    reader = vtkSTLReader()
    reader.SetFileName(stl_file)
    reader.Update()
    poly_data = reader.GetOutput()
    return poly_data

def createActorFromSTL(stl_file, color='LightSteelBlue', opacity=1.0):
    """

    :param stl_file:
    :return:
    """

    colors = vtkNamedColors()
    reader = vtkSTLReader()
    reader.SetFileName(stl_file)
    reader.Update()
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetDiffuse(0.8)
    actor.GetProperty().SetDiffuseColor(colors.GetColor3d(color))
    actor.GetProperty().SetSpecular(0.3)
    actor.GetProperty().SetSpecularPower(60.0)
    actor.GetProperty().SetOpacity(opacity)
    return actor

def createLineActor(points_array, color="blue", line_width=4.0):
    """

    :param points_array:
    :param colors:
    :return:
    """
    colors = vtkNamedColors()
    points = vtkPoints()
    points.InsertNextPoint(points_array[0])
    points.InsertNextPoint(points_array[1])

    poly_line = vtkPolyLine()
    poly_line.GetPointIds().SetNumberOfIds(2)
    for i in range(0, 2):
        poly_line.GetPointIds().SetId(i, i)

    cells = vtkCellArray()
    cells.InsertNextCell(poly_line)

    poly_data = vtkPolyData()
    poly_data.SetPoints(points)

    poly_data.SetLines(cells)

    mapper = vtkPolyDataMapper()
    mapper.SetInputData(poly_data)

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d(color))
    actor.GetProperty().SetLineWidth(line_width)
    return actor


def spine2world_vtk(spine_origin, spine_coordinate, inverse=False):
    matrix = vtk.vtkMatrix4x4()
    matrix.DeepCopy((spine_coordinate[0][0], spine_coordinate[1][0], spine_coordinate[2][0], spine_origin[0],
                     spine_coordinate[0][1], spine_coordinate[1][1], spine_coordinate[2][1], spine_origin[1],
                     spine_coordinate[0][2], spine_coordinate[1][2], spine_coordinate[2][2], spine_origin[2],
                     0, 0, 0, 1,
                     ))
    T = np.array([[spine_coordinate[0][0], spine_coordinate[1][0], spine_coordinate[2][0], spine_origin[0]],
     [spine_coordinate[0][1], spine_coordinate[1][1], spine_coordinate[2][1], spine_origin[1]],
     [spine_coordinate[0][2], spine_coordinate[1][2], spine_coordinate[2][2], spine_origin[2]],
     [0, 0, 0, 1]])
    if not inverse:
        return matrix

    else:
        R_w2s = T[:3,:3]
        R_s2w = R_w2s.T
        P_w2s = T[:3,3]
        P_s2w = -np.dot(R_s2w, P_w2s)
        inverse_matrix = vtk.vtkMatrix4x4()

        inverse_matrix.DeepCopy((R_s2w[0][0], R_s2w[0][1], R_s2w[0][2], P_s2w[0],
                                 R_s2w[1][0], R_s2w[1][1], R_s2w[1][2], P_s2w[1],
                                 R_s2w[2][0], R_s2w[2][1], R_s2w[2][2], P_s2w[2],
                                 0, 0, 0, 1
                                ))

    return inverse_matrix, matrix


def screen_shot_generate(all_actors, coordinates, origin):
    # label_out_dir = '../CTSpine1K-main/data/label/split_label'
    #path_list = single_spine_path.split('/')
    # case_name = path_list[-2][:-7]
    # dataset_name = path_list[-3]
    # label_name = path_list[-1][:-7]
    # label_out_file = os.path.join(label_out_dir, dataset_name , label_name)
    # if not os.path.isdir(label_out_file):
    #     os.makedirs(label_out_file)

    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetOffScreenRendering(1)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)
    # renderer.RemoveAllLights()
    # world coordinate compare with spine coordinate
    x_actor = createAxisActor([origin, origin + 40 * np.array([1,0,0])], 'Red')
    y_actor = createAxisActor([origin, origin + 40 * np.array([0,1,0])], 'Green')
    z_actor = createAxisActor([origin, origin + 40 * np.array([0,0,1])], 'Blue')
    x_actor.GetProperty().SetOpacity(0.4)
    y_actor.GetProperty().SetOpacity(0.4)
    z_actor.GetProperty().SetOpacity(0.4)
    all_actors.append(x_actor)
    all_actors.append(y_actor)
    all_actors.append(z_actor)

    for actor in all_actors:
        renderer.AddActor(actor)

    camera = vtk.vtkCamera()
    camera.SetFocalPoint(origin[0], origin[1], origin[2])
    camera.SetPosition(origin[0]+coordinates[2][0]*160, origin[1]+coordinates[2][1]*160, origin[2] + coordinates[2][2]*160)
    camera.SetViewUp(-coordinates[1])
    renderer.SetActiveCamera(camera)
    renderWindow.Render()

    # save image as numpy
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(renderWindow)
    window_to_image_filter.SetInputBufferTypeToRGB()
    window_to_image_filter.SetScale(1)
    window_to_image_filter.Update()
    renderWindow.Finalize()

    # label_name_top = case_name +'_'+ label_name+'_top.jpeg'
    writer = vtk.vtkJPEGWriter()
    writer.SetQuality(100)
   #  writer.SetFileName(os.path.join(label_out_file, label_name_top))
    writer.SetFileName(os.path.join('top.jpeg'))
    writer.SetInputConnection(window_to_image_filter.GetOutputPort())
    # writer.SetQuality(100)
    writer.Write()

    camera2 = vtk.vtkCamera()
    camera2.SetFocalPoint(origin[0], origin[1], origin[2])
    camera2.SetPosition(origin[0]+coordinates[0][0]*150, origin[1]+coordinates[0][1]*150, origin[2]+coordinates[0][2]*150)
    camera2.SetViewUp(coordinates[2])
    renderer.SetActiveCamera(camera2)
    renderWindow.Render()

    # save image as numpy
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(renderWindow)
    window_to_image_filter.SetInputBufferTypeToRGB()
    window_to_image_filter.SetScale(1)
    window_to_image_filter.Update()
    renderWindow.Finalize()

    # label_name_left = case_name +'_'+label_name + '_left.jpeg'
    writer = vtk.vtkJPEGWriter()
    writer.SetQuality(100)
    writer.SetFileName(os.path.join('left.jpeg'))
    writer.SetInputConnection(window_to_image_filter.GetOutputPort())
    writer.SetQuality(100)
    writer.Write()

    camera3 = vtk.vtkCamera()
    camera3.SetFocalPoint(origin[0], origin[1], origin[2])
    camera3.SetPosition(origin[0] + coordinates[1][0] * 150, origin[1] + coordinates[1][1] * 150,
                        origin[2] + coordinates[1][2] * 150)
    camera3.SetViewUp(coordinates[2])
    renderer.SetActiveCamera(camera3)
    renderWindow.Render()

    # save image as numpy
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(renderWindow)
    window_to_image_filter.SetInputBufferTypeToRGB()
    window_to_image_filter.SetScale(1)
    window_to_image_filter.Update()
    renderWindow.Finalize()

    # label_name_back = case_name +'_'+ label_name + '_back.jpeg'
    writer = vtk.vtkJPEGWriter()
    writer.SetQuality(100)
    writer.SetFileName(os.path.join('back.jpeg'))
    writer.SetInputConnection(window_to_image_filter.GetOutputPort())
    writer.SetQuality(100)
    writer.Write()


def top_two(n_points_list):
    top_one_idx = 0
    top_one = 0
    top_two_idx = 0
    top_two = 0

    for idx, n_points in enumerate(n_points_list):
        if n_points > top_one:
            top_two = top_one
            top_two_idx = top_one_idx
            top_one = n_points
            top_one_idx = idx
        elif n_points > top_two:
            top_two = n_points
            top_two_idx = idx
    return top_one_idx, top_two_idx


def top_one(n_points_list):
    top_one_idx = 0
    top_one = 0

    for idx, n_points in enumerate(n_points_list):
        if n_points > top_one:
            top_one = n_points
            top_one_idx = idx
    return top_one_idx

def slicing_xy_vtk(aligned_spine_polydata, z_coordinate):
    all_actors = []
    cutplane = vtkPlane()
    cutplane.SetOrigin([0, 0, z_coordinate])
    cutplane.SetNormal([0, 0, 1])

    cutter = vtkCutter()
    cutter.SetCutFunction(cutplane)
    cutter.SetInputData(aligned_spine_polydata)

    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(cutter.GetOutputPort())
    stripper.Update()
    slicing_boundary = stripper.GetOutput()
    all_actors = []
    actor = vtkActor()
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(slicing_boundary)
    actor.SetMapper(mapper)
    all_actors.append(actor)
   # showActors(all_actors)
    n_cells = slicing_boundary.GetNumberOfCells()
    if n_cells > 2:  # biggest two stripper
        n_points = []
        for c in range(n_cells):
            n_points.append(slicing_boundary.GetCell(c).GetNumberOfPoints())

        c_0_idx, c_1_idx = top_two(n_points)
        points0 = np.ceil(np.asarray(slicing_boundary.GetCell(c_0_idx).GetPoints().GetData()))[:, :2]
        points1 = np.ceil(np.asarray(slicing_boundary.GetCell(c_1_idx ).GetPoints().GetData()))[:, :2]
    else:
        points0 = np.ceil(np.asarray(slicing_boundary.GetCell(0).GetPoints().GetData()))[:, :2]
        points1 = np.ceil(np.asarray(slicing_boundary.GetCell(1).GetPoints().GetData()))[:, :2]

    points0 = np.unique(points0, axis=0)
    points1 = np.unique(points1, axis=0)
    if points1.shape[0] < 45 or points0.shape[0] < 45:  # num rows
        offset_min = int(points0.min())
    else:
        offset_min = int(min([points1.min(), points0.min()]))
    points0 -= offset_min
    points1 -= offset_min
    contours = [points0.astype(np.uint8), points1.astype(np.uint8)]
    contours = sorted(contours, key=lambda c: c.shape[0], reverse=True)

    # array = draw_contour(contours)
    # plt.imshow(array)
    # plt.title(f'{points0.shape[0]}, {points1.shape[1]}')
    # plt.show()
    return contours, offset_min, z_coordinate


def createAxisActor(points_array, color="blue", line_width=4.0):
    """
    :param points_array:
    :param colors:
    :return:
    """
    colors = vtkNamedColors()
    points = vtkPoints()
    points.InsertNextPoint(points_array[0])
    points.InsertNextPoint(points_array[1])

    poly_line = vtkPolyLine()
    poly_line.GetPointIds().SetNumberOfIds(2)
    for i in range(0, 2):
        poly_line.GetPointIds().SetId(i, i)

    cells = vtkCellArray()
    cells.InsertNextCell(poly_line)

    poly_data = vtkPolyData()
    poly_data.SetPoints(points)

    poly_data.SetLines(cells)

    mapper = vtkPolyDataMapper()
    mapper.SetInputData(poly_data)

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d(color))
    actor.GetProperty().SetLineWidth(line_width)
    return actor

def showActors(actors, window_name=""):
    ren = vtkRenderer()
    for cur_actor in actors:
        ren.AddActor(cur_actor)

    win = vtkRenderWindow()
    win.AddRenderer(ren)
    win.SetWindowName("show spine "+window_name)

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(win)
    style = vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)

    ren.ResetCamera()
    win.Render()
    iren.Initialize()
    iren.Start()

def getIntersectPointsFromLineAndPolyData(line_point0, line_point1, poly_data):
    """

    :param line_point0:
    :param line_point1:
    :param polydata:
    :return:
    """
    tree = vtkOBBTree()
    tree.SetDataSet(poly_data)
    tree.BuildLocator()

    intersect_points = vtkPoints()

    tree.IntersectWithLine(line_point0, line_point1, intersect_points, None)

    num_intersect_points = intersect_points.GetNumberOfPoints()
    #print("NumPoints", num_intersect_points)

    intersect_points_list = []
    for i in range(num_intersect_points):
        cur_point = intersect_points.GetPoint(i)
        intersect_points_list.append(cur_point)
    return np.array(intersect_points_list)



def PCA(data, correlation=False, sort=True):
    X = np.asarray(data).T
    X_mean = np.mean(X,axis=1).reshape(3,1)
    X_head = X - X_mean
    H = X_head.dot(X_head.T)
    eigenvalues,eigenvectors = np.linalg.eig(H)

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]
    return eigenvalues, eigenvectors

def getMainVectorByPCA(points_xyz, delta=10.0):
    """
    :param points:
    :return:
    """
    points_center = np.mean(points_xyz, axis=0)
    points_w, points_v = PCA(points_xyz)
    points_cloud_vector = points_v[:, 0] #点云主方向对应的向量
    points_main_vector_points = [
        points_center+delta*points_cloud_vector,
        points_center-delta*points_cloud_vector,
    ]
    points_main_vector_normal = points_cloud_vector
    return points_center, points_main_vector_points, points_main_vector_normal

def calProjectedPointCoordOnPlane(plane_normal_vector, plane_point, point):
    """
    :param plane_normal_vector: 平面法向量, xyz， numpy array, (3,)
    :param plane_point: 平面法向量与平面的交点,xyz, numpy array,(3,)
    :param point: 平面外的一点,numpy array,xyz,(3,)
    :return: projected_point:平面外一点在该平面上的投影点,xyz, numpy array,(3,)
    method reference:https://blog.csdn.net/fsac213330/article/details/53219949
    https://blog.csdn.net/soaryy/article/details/82884691
    """
    eps = 1.0e-8
    A = plane_normal_vector[0]
    B = plane_normal_vector[1]
    C = plane_normal_vector[2]
    r = A*A + B*B + C*C
    assert (r > eps), "plane normal vector should not be (0, 0, 0)"
    D = -(A*plane_point[0]+B*plane_point[1]+C*plane_point[2])

    x0 = point[0]
    y0 = point[1]
    z0 = point[2]

    x_projected = ( (B*B + C*C)*x0 - A*(B*y0+C*z0+D) )/r
    y_projected = ( (A*A + C*C)*y0 - B*(A*x0+C*z0+D) )/r
    z_projected = ( (A*A + B*B)*z0 - C*(A*x0+B*y0+D) )/r

    return np.array([x_projected, y_projected, z_projected])

def getAngleBetweenLineAndPlane(line_point0, line_point1, plane_normal, plane_point):
    """
    func:get the angle between line and plane
    :param line_point0:
    :param line_point1:
    :param plane_normal:
    :param plane_point:
    :return:
    """
    projected_point0 = calProjectedPointCoordOnPlane(plane_normal, plane_point, line_point0)
    projected_point1 = calProjectedPointCoordOnPlane(plane_normal, plane_point, line_point1)

    projected_dis = np.sqrt(np.sum(np.square(projected_point0-projected_point1)))
    line_dis = np.sqrt(np.sum(np.square(line_point0-line_point1)))
    res = projected_dis / line_dis
    if res > 1.0:
        res = 1
    projected_angle = np.arccos(res)
    return projected_angle

def createRotateMatrixAroundAxis(rotate_angle, axis="X"):
    """
    func:
    :param rotate_angle: degree
    :param axis:
    :return:
    """
    rotate_rad = rotate_angle/180.0*np.pi
    rotate_matrix = np.zeros([4,4], dtype=np.float64)
    rotate_matrix[0,0] = 1.0
    rotate_matrix[1,1] = 1.0
    rotate_matrix[2,2] = 1.0
    rotate_matrix[3,3] = 1.0
    if axis == "X":
        rotate_matrix[1, 1] = np.cos(rotate_rad)
        rotate_matrix[1, 2] = -np.sin(rotate_rad)
        rotate_matrix[2, 1] = np.sin(rotate_rad)
        rotate_matrix[2, 2] = np.cos(rotate_rad)
    elif axis == "Y":
        rotate_matrix[0, 0] = np.cos(rotate_rad)
        rotate_matrix[0, 2] = np.sin(rotate_rad)
        rotate_matrix[2, 0] = -np.sin(rotate_rad)
        rotate_matrix[2, 2] = np.cos(rotate_rad)
    elif axis == "Z":
        rotate_matrix[0, 0] = np.cos(rotate_rad)
        rotate_matrix[0, 1] = -np.sin(rotate_rad)
        rotate_matrix[1, 0] = np.sin(rotate_rad)
        rotate_matrix[1, 1] = np.cos(rotate_rad)
    else:
        print("input axis must be X, Y or Z")
    return rotate_matrix

def normalizeVector(v):
    """
    v: vector,np.array, (3,)
    :return:
    v_normal:
    """
    r = np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    v_normal = v/r
    return v_normal


def vectorCross(v1, v2):
    """
    func:计算两个向量的叉乘
    :param v1: numpy array,xyz
    :param v2: numpy array,xyz
    :return:
    """
    a1 = v1[0]
    b1 = v1[1]
    c1 = v1[2]

    a2 = v2[0]
    b2 = v2[1]
    c2 = v2[2]

    return np.array([b1*c2-b2*c1, c1*a2-a1*c2, a1*b2-a2*b1])


def createPediclePipelineNormal(rotate_angle, cur_spine_axis_normal, project_point):
    """
    func:创建椎弓根通道的法向量
    :param rotate_angle:
    :param cur_spine_axis_normal:
    :param project_point:
    :return:
    """
    rotate_matrix = createRotateMatrixAroundAxis(rotate_angle, "Z")
    coor_trans_matrix = np.zeros([4, 4], dtype=np.float64)
    coor_trans_matrix[0:3, 0] = cur_spine_axis_normal[0]
    coor_trans_matrix[0:3, 1] = cur_spine_axis_normal[1]
    coor_trans_matrix[0:3, 2] = cur_spine_axis_normal[2]
    coor_trans_matrix[0:3, 3] = 0.0
    coor_trans_matrix[3, 3] = 1.0
    tmp0 = np.dot(rotate_matrix, np.array([project_point[0], project_point[1], project_point[2], 1.0]).T)
    transed_point = np.dot(coor_trans_matrix, tmp0)[0:3]
    pedicle_pipeline_normal = normalizeVector(transed_point)
    return pedicle_pipeline_normal

def createcylinderActor2(point0, point1, color='Green', opacity=1.0, radius = 1.5):
    #print("radius :", radius)
    colors = vtkNamedColors()
    line = vtk.vtkLineSource()
    line.SetPoint1(point0[0], point0[1], point0[2])
    line.SetPoint2(point1[0], point1[1], point1[2])

    tubeFilter = vtk.vtkTubeFilter()
    tubeFilter.SetInputConnection(line.GetOutputPort())
    # tubeFilter.SetVaryRadiusToVaryRadiusOff()
    tubeFilter.SetRadius(radius)
    tubeFilter.SetNumberOfSides(100)
    tubeFilter.CappingOn()
    tubeFilter.Update()

    # out = tubeFilter.GetOutput()
    tube_mapper = vtkPolyDataMapper()
    tube_mapper.SetInputData(tubeFilter.GetOutput())
    tube_actor = vtkActor()
    tube_actor.SetMapper(tube_mapper)
    tube_actor.GetProperty().SetColor(colors.GetColor3d(color))
    tube_actor.GetProperty().SetOpacity(opacity)
    # out = tubeFilter.GetOutput()
    # showActors([tube_actor])
    return tube_actor


def createcylinderActor(center, hegiht, radius, direction, color="green", opacity=1.0, cur_side="left"):
    """
    func:创建指定方向的圆柱体
    :param center:
    :param hegiht:
    :param radius:
    :param direction:
    :param color:
    :param opacity:
    :return:
    """
    colors = vtkNamedColors()

    # Create a sphere
    cylinderSource = vtkCylinderSource()
    cylinderSource.SetCenter(0.0, 0.0, 0.0)
    cylinderSource.SetRadius(radius)
    cylinderSource.SetHeight(hegiht)
    cylinderSource.SetResolution(100)
    cylinderSource.Update()
    # print("origin cylinder center:")
    # print(cylinderSource.GetCenter())

    plane_normal = (0.0, 0.0, 1.0)
    plane_point = (0.0, 0.0, 0.0)
    vertical_phi_rad = getAngleBetweenLineAndPlane((0.0, 0.0, 0.0), direction, plane_normal, plane_point)

    if np.dot(plane_normal, direction) < 1.0e-8:
        vertical_phi_rad = -vertical_phi_rad


    projected_point = calProjectedPointCoordOnPlane(plane_normal, plane_point, direction)
    projected_point_x = projected_point[0]
    projected_point_y = projected_point[1]
    projected_len = np.sqrt(np.square(projected_point_x) + np.square(projected_point_y))
    horizontal_theta_rad = np.arccos(np.abs(projected_point_x)/projected_len)

    #step1:rotate X axis, round (vertical_phi)
    transform1 = vtkTransform()
    transform1.RotateX(-180.0/np.pi*vertical_phi_rad)

    transF1 = vtkTransformPolyDataFilter()
    transF1.SetInputData(cylinderSource.GetOutput())
    transF1.SetTransform(transform1)
    transF1.Update()

    #step2:rotate Z axis, round (np.pi-horizontal_theta)

    transform2 = vtkTransform()
    if cur_side == 'left':
        transform2.RotateZ(180.0 / np.pi * (-np.pi/2.0 + horizontal_theta_rad))
    else:
        transform2.RotateZ(180.0 / np.pi * (np.pi / 2.0 - horizontal_theta_rad))
    #print(transform2)
    transF2 = vtkTransformPolyDataFilter()
    transF2.SetInputData(transF1.GetOutput())
    transF2.SetTransform(transform2)
    transF2.Update()

    transform3 = vtkTransform()
    transform3.Translate(center[0], center[1], center[2])
    #print(transform3)
    transF3 = vtkTransformPolyDataFilter()
    transF3.SetInputData(transF2.GetOutput())
    transF3.SetTransform(transform3)
    transF3.Update()

    # Create a mapper and actor
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(transF3.GetOutput())
    actor = vtkActor()
    actor.GetProperty().SetColor(colors.GetColor3d(color))
    actor.GetProperty().SetOpacity(opacity)
    actor.SetMapper(mapper)
    return actor


def varify_main_axis_points_intersection(coordinate, origin, spine_polydata):
    # adjust main axis, in case of horizontal main axis
    all_actors = []
    origin_array = np.array(origin)
    x_axis = coordinate[0]  # x轴是横向的!!!!! 次轴为x!
    x_max_idx = np.argmax(x_axis)
    if x_max_idx == 1:  # x轴是横向的!!!!!
        coordinate[0], coordinate[1] = coordinate[1], coordinate[0]

    # calculate the intersection points of main axis with spine polydata
    tree = vtk.vtkOBBTree()
    tree.SetDataSet(spine_polydata)
    tree.BuildLocator()

    intersection_points = vtkPoints()
    tree.IntersectWithLine(origin_array+coordinate[1]*100, origin_array-coordinate[1]*100, intersection_points, None)
    intersection_points = np.asarray(intersection_points.GetData())
    for point in intersection_points:
        point_actor = createSphereActor(point, 3)
        all_actors.append(point_actor)

    dis = np.sqrt(np.sum(np.square(intersection_points - origin), axis=1))
    min_idx = np.argmin(dis)
    main_axis_direction = (intersection_points[min_idx] - origin_array) / np.linalg.norm(intersection_points[min_idx] - origin_array)

    # if np.argmax(main_axis_direction) != np.argmax(coordinate[1]):
    #     coordinate[0], coordinate[1] = coordinate[1], coordinate[0]

    main_direction_axis = main_axis_direction[np.argmax(main_axis_direction)]
    main_direction_coordinate0 = coordinate[1][np.argmax(coordinate[1])]

    if main_direction_coordinate0*main_direction_axis < 0:
        coordinate[0] = -coordinate[0]
        coordinate[1] = -coordinate[1]
    z_axis = np.cross(coordinate[0], coordinate[1])
    coordinate.append(z_axis)
    return coordinate


def get_eigen_vectors_values(points):
    '''
    returns eigen values, vector
    @param points: list of points (x, y, z)
    @return: eigen values, vector
    '''
    # vtk double array
    xArr = vtk.vtkDoubleArray()
    xArr.SetName("x")
    xArr.SetNumberOfComponents(1)

    yArr = vtk.vtkDoubleArray()
    yArr.SetName("y")
    yArr.SetNumberOfComponents(1)

    zArr = vtk.vtkDoubleArray()
    zArr.SetName("z")
    zArr.SetNumberOfComponents(1)

    for i in range(points.GetNumberOfPoints()):
        xArr.InsertNextValue(points.GetPoint(i)[0])
        yArr.InsertNextValue(points.GetPoint(i)[1])
        zArr.InsertNextValue(points.GetPoint(i)[2])

    # vtk table
    table = vtk.vtkTable()
    table.AddColumn(xArr)
    table.AddColumn(yArr)
    table.AddColumn(zArr)

    # vtk pca
    pca = vtk.vtkPCAStatistics()
    pca.SetInputData(table)
    pca.SetColumnStatus("x", 1)
    pca.SetColumnStatus("y", 1)
    pca.SetColumnStatus("z", 1)
    pca.RequestSelectedColumns()
    pca.SetDeriveOption(True)
    pca.Update()

    # eigenvalues
    eigenvalues = vtk.vtkDoubleArray()
    pca.GetEigenvalues(eigenvalues)
    ev = []
    # print("Eigenvalues: ")
    for i in range(eigenvalues.GetNumberOfTuples()):
        # print(eigenvalues.GetValue(i))
        ev.append(eigenvalues.GetValue(i))

    # eigenvectors
    eigenvectors = vtk.vtkDoubleArray()
    pca.GetEigenvectors(eigenvectors)

    eig_vec = [vtk.vtkDoubleArray() for i in range(eigenvectors.GetNumberOfTuples())]
    for i in range(eigenvectors.GetNumberOfTuples()):
        pca.GetEigenvector(i, eig_vec[i])

    eig_vec_2 = []
    for i in range(len(eig_vec)):
        eig_vec_2.append((eig_vec[i].GetValue(0), eig_vec[i].GetValue(1), eig_vec[i].GetValue(2)))
    return eig_vec_2, ev


def createSpineAxisActor(axis_origin, axis_normal,line_width=4.0, x_color="DarkRed", y_color="DarkGreen", z_color="DarkBlue", len=30):
    """
    :param origin:
    :param axis_x_normal:
    :param axis_y_normal:
    :param axis_z_normal:
    :param x_color:
    :param y_color:
    :param z_color:
    :param len:
    :return:
    """
    axis_origin = np.asarray(axis_origin)
    axis_normal = np.asarray(axis_normal)
    axis_x_points = [axis_origin, axis_origin + axis_normal[0] * len]
    axis_x_actor = createAxisActor(axis_x_points, x_color, line_width)

    axis_y_points = [axis_origin, axis_origin + axis_normal[1] * len]
    axis_y_actor = createAxisActor(axis_y_points, y_color, line_width)

    axis_z_points = [axis_origin, axis_origin + axis_normal[2] * len]
    axis_z_actor = createAxisActor(axis_z_points, z_color, line_width)

    return [axis_x_actor, axis_y_actor, axis_z_actor]

def createSpineAxisActor_new(axis_origin, axis_normal,line_width=4.0, x_color="DarkRed", y_color="DarkGreen", z_color="DarkBlue", len=50, opacity=0.3):
    """
    :param origin:
    :param axis_x_normal:
    :param axis_y_normal:
    :param axis_z_normal:
    :param x_color:
    :param y_color:
    :param z_color:
    :param len:
    :return:
    """
    axis_origin = np.asarray(axis_origin)
    axis_normal = np.asarray(axis_normal)
    axis_x_points = [axis_origin - axis_normal[0] * len, axis_origin + axis_normal[0] * len]
    axis_x_actor = createAxisActor(axis_x_points, x_color, line_width)
    axis_x_actor.GetProperty().SetOpacity( opacity)

    axis_y_points = [axis_origin - axis_normal[1] * len, axis_origin + axis_normal[1] * len]
    axis_y_actor = createAxisActor(axis_y_points, y_color, line_width)
    axis_y_actor.GetProperty().SetOpacity( opacity)

    axis_z_points = [axis_origin - axis_normal[2] * len, axis_origin + axis_normal[2] * len]
    axis_z_actor = createAxisActor(axis_z_points, z_color, line_width)
    axis_z_actor.GetProperty().SetOpacity( opacity)
    return [axis_x_actor, axis_y_actor, axis_z_actor]

def createOriginAxisActor(axis_len=100.0, line_width=4.0, x_color="red", y_color='green',z_color="blue"):
    """
    :return:
    """
    origin_axis_x_actor = createAxisActor([np.array([0.0, 0.0, 0.0]), np.array([axis_len, 0.0, 0.0])], x_color, line_width)
    origin_axis_y_actor = createAxisActor([np.array([0.0, 0.0, 0.0]), np.array([0.0, axis_len, 0.0])], y_color, line_width)
    origin_axis_z_actor = createAxisActor([np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, axis_len])], z_color, line_width)
    return [origin_axis_x_actor, origin_axis_y_actor, origin_axis_z_actor]

def createPediclePipelineCylinderActor(point0, point1, radius = 3.5/2.0, color="magenta"):
    pedicle_pipeline_cylinder_point0 = point0
    pedicle_pipeline_cylinder_point1 = point0 + 0.8 * (point1 - point0)
    pedicle_pipeline_cylinder_actor = createcylinderActor2(pedicle_pipeline_cylinder_point0, pedicle_pipeline_cylinder_point1, color, radius=radius)
    return pedicle_pipeline_cylinder_actor

def save_coordinate(case_name, origin, coordinates, plane_center_point_L=None, plane_center_point_R=None,\
                    pedicle_pipeline_L_normal=None, pedicle_pipeline_R_normal=None):
    print('save coordinate for ', case_name)
    save_path = os.path.join('./', case_name+'TEST.txt')

    fp_w = open(save_path, "w")
    fp_w.write("origin\n")
    fp_w.write("%s\n" % str(origin[0]))
    fp_w.write("%s\n" % str(origin[1]))
    fp_w.write("%s\n" % str(origin[2]))

    for idx, coordinate in enumerate(coordinates):
        if idx==0:
            writepoint2file(fp_w, "x_axis", coordinate)
        elif idx==1:
            writepoint2file(fp_w, "y_axis", coordinate)
        else:
            writepoint2file(fp_w, "z_axis", coordinate)
    # if plane_center_point_L.all() != None:
    #     writepoint2file(fp_w, "plane_center_point_L", plane_center_point_L)
    # if plane_center_point_R.all() != None:
    #     writepoint2file(fp_w, "plane_center_point_R", plane_center_point_R)
    # if pedicle_pipeline_L_normal.all() != None:
    #     writepoint2file(fp_w, "pedicle_pipeline_L_normal", pedicle_pipeline_L_normal)
    # if pedicle_pipeline_R_normal.all() != None:
    #     writepoint2file(fp_w, "pedicle_pipeline_R_normal", pedicle_pipeline_R_normal)

    writepoint2file(fp_w, "plane_center_point_L", plane_center_point_L)

    writepoint2file(fp_w, "plane_center_point_R", plane_center_point_R)

    writepoint2file(fp_w, "pedicle_pipeline_L_normal", pedicle_pipeline_L_normal)

    writepoint2file(fp_w, "pedicle_pipeline_R_normal", pedicle_pipeline_R_normal)
    fp_w.close()

def writepoint2file(file, name, point):
    file.write(f'{name}\n')
    for coord in point:
        file.write("%s\n" % coord)


def load_coordinate(case_path):
    origin = []
    x_axis = []
    y_axis = []
    z_axis = []
    plane_center_point_L = []
    plane_center_point_R = []
    pedicle_pipeline_L_normal =[]
    pedicle_pipeline_R_normal = []
    file_r = open(case_path, 'r')
    lines = file_r.readlines()
    file_r.close()

    for idx, line in enumerate(lines):
        cur_line = line.strip()
        if cur_line == 'origin':
            origin.append(float(lines[idx + 1]))
            origin.append(float(lines[idx + 2]))
            origin.append(float(lines[idx + 3]))
        if cur_line == 'x_axis':
            x_axis.append(float(lines[idx + 1]))
            x_axis.append(float(lines[idx + 2]))
            x_axis.append(float(lines[idx + 3]))
        if cur_line == 'y_axis':
            y_axis.append(float(lines[idx + 1]))
            y_axis.append(float(lines[idx + 2]))
            y_axis.append(float(lines[idx + 3]))
        if cur_line == 'z_axis':
            z_axis.append(float(lines[idx + 1]))
            z_axis.append(float(lines[idx + 2]))
            z_axis.append(float(lines[idx + 3]))

        if cur_line == 'plane_center_point_L':
            plane_center_point_L.append(float(lines[idx + 1]))
            plane_center_point_L.append(float(lines[idx + 2]))
            plane_center_point_L.append(float(lines[idx + 3]))

        if cur_line == 'plane_center_point_R':
            plane_center_point_R.append(float(lines[idx + 1]))
            plane_center_point_R.append(float(lines[idx + 2]))
            plane_center_point_R.append(float(lines[idx + 3]))
        if cur_line == 'pedicle_pipeline_L_normal':
            pedicle_pipeline_L_normal.append(float(lines[idx + 1]))
            pedicle_pipeline_L_normal.append(float(lines[idx + 2]))
            pedicle_pipeline_L_normal.append(float(lines[idx + 3]))
        if cur_line == 'pedicle_pipeline_R_normal':
            pedicle_pipeline_R_normal.append(float(lines[idx + 1]))
            pedicle_pipeline_R_normal.append(float(lines[idx + 2]))
            pedicle_pipeline_R_normal.append(float(lines[idx + 3]))
    return origin, [x_axis, y_axis, z_axis],plane_center_point_L,plane_center_point_R,pedicle_pipeline_L_normal,pedicle_pipeline_R_normal


def load_coordinate_reference_point(case_path):
    origin = []
    x_axis = []
    y_axis = []
    z_axis = []
    plane_center_point_L = []
    plane_center_point_R = []
    pedicle_pipeline_L_normal =[]
    pedicle_pipeline_R_normal = []
    pedicle_reference_point0 = []
    pedicle_reference_point1 = []
    file_r = open(case_path, 'r')
    lines = file_r.readlines()
    file_r.close()

    for idx, line in enumerate(lines):
        cur_line = line.strip()
        if cur_line == 'origin':
            origin.append(float(lines[idx + 1]))
            origin.append(float(lines[idx + 2]))
            origin.append(float(lines[idx + 3]))
        if cur_line == 'x_axis':
            x_axis.append(float(lines[idx + 1]))
            x_axis.append(float(lines[idx + 2]))
            x_axis.append(float(lines[idx + 3]))
        if cur_line == 'y_axis':
            y_axis.append(float(lines[idx + 1]))
            y_axis.append(float(lines[idx + 2]))
            y_axis.append(float(lines[idx + 3]))
        if cur_line == 'z_axis':
            z_axis.append(float(lines[idx + 1]))
            z_axis.append(float(lines[idx + 2]))
            z_axis.append(float(lines[idx + 3]))

        if cur_line == 'plane_center_point_L':
            plane_center_point_L.append(float(lines[idx + 1]))
            plane_center_point_L.append(float(lines[idx + 2]))
            plane_center_point_L.append(float(lines[idx + 3]))

        if cur_line == 'plane_center_point_R':
            plane_center_point_R.append(float(lines[idx + 1]))
            plane_center_point_R.append(float(lines[idx + 2]))
            plane_center_point_R.append(float(lines[idx + 3]))
        if cur_line == 'pedicle_pipeline_L_normal':
            pedicle_pipeline_L_normal.append(float(lines[idx + 1]))
            pedicle_pipeline_L_normal.append(float(lines[idx + 2]))
            pedicle_pipeline_L_normal.append(float(lines[idx + 3]))
        if cur_line == 'pedicle_pipeline_R_normal':
            pedicle_pipeline_R_normal.append(float(lines[idx + 1]))
            pedicle_pipeline_R_normal.append(float(lines[idx + 2]))
            pedicle_pipeline_R_normal.append(float(lines[idx + 3]))
        if cur_line == 'pedicle_reference_point0':
            pedicle_reference_point0.append(float(lines[idx + 1]))
            pedicle_reference_point0.append(float(lines[idx + 2]))
            pedicle_reference_point0.append(float(lines[idx + 3]))
        if cur_line == 'pedicle_reference_point1':
            pedicle_reference_point1.append(float(lines[idx + 1]))
            pedicle_reference_point1.append(float(lines[idx + 2]))
            pedicle_reference_point1.append(float(lines[idx + 3]))
    return origin, [x_axis, y_axis, z_axis],plane_center_point_L,plane_center_point_R,\
        pedicle_pipeline_L_normal,pedicle_pipeline_R_normal,\
        pedicle_reference_point0, pedicle_reference_point1


def createIntersectPointsActor(intersect_points, radius=1.0, opacity=1.0, color="megenta"):
    """
    func:
    :param intersect_points:
    :param radius:
    :param opacity:
    :param color:
    :return:
    """
    intersect_point0_actor = createSphereActor(intersect_points[0], radius, opacity, color)
    intersect_point1_actor = createSphereActor(intersect_points[1], radius, opacity, color)
    return [intersect_point0_actor, intersect_point1_actor]


def parsePointsFile(points_file):
    """
    func:get points from point files
    :param points_file:
    :return:
    """
    fp = open(points_file, "r")
    lines = fp.readlines()
    fp.close()
    cur_points_dict = {}
    for cur_line in lines:
        cur_line = cur_line.strip()
        if "<point" in cur_line:
            cur_line = cur_line.replace("/>", "")
            cur_line_lists = cur_line.split(" ")
            cur_point_x = 0
            cur_point_y = 0
            cur_point_z = 0
            cur_point_name = ""
            for cur_line_list in cur_line_lists:
                if "x=" in cur_line_list:
                    cur_point_x = float(cur_line_list[3:-1])
                if "y=" in cur_line_list:
                    cur_point_y = float(cur_line_list[3:-1])
                if "z=" in cur_line_list:
                    cur_point_z = float(cur_line_list[3:-1])
                if "name=" in cur_line_list:
                    cur_point_name = cur_line_list[6:-1]
            #print(("%s:x=%.2f y=%.2f z=%.2f")%(cur_point_name, cur_point_x, cur_point_y, cur_point_z))
            cur_points_dict[cur_point_name] = np.array([cur_point_x, cur_point_y, cur_point_z])
    return cur_points_dict


def createClipedActor(plane_center, plane_normal, poly_data):
    """

    :return:
    """
    colors = vtkNamedColors()
    plane1 = vtkPlane()
    plane1.SetOrigin(plane_center[0], plane_center[1], plane_center[2])
    plane1.SetNormal(plane_normal[0], plane_normal[1], plane_normal[2])

    planes = vtkPlaneCollection()
    planes.AddItem(plane1)
    # planes.AddItem(plane2)

    clipper = vtkClipPolyData()
    clipper.SetInputData(poly_data)
    clipper.SetClipFunction(plane1)

    clip_mapper = vtkDataSetMapper()
    clip_mapper.SetInputConnection(clipper.GetOutputPort())

    clip_actor = vtkActor()
    clip_actor.SetMapper(clip_mapper)
    # clip_actor.GetProperty().SetColor(0.0, 0.0, 0.0)
    # clip_actor.GetProperty().SetInterpolationToFlat()
    clip_actor.GetProperty().SetDiffuse(0.8)
    clip_actor.GetProperty().SetDiffuseColor(colors.GetColor3d("LightSteelBlue"))
    clip_actor.GetProperty().SetSpecular(0.3)
    clip_actor.GetProperty().SetSpecularPower(60.0)
    clip_actor.GetProperty().SetOpacity(1.5)
    return clip_actor


def createClipedPolydata(plane_center, plane_normal, poly_data):
    """

    :return:
    """
    colors = vtkNamedColors()
    plane1 = vtkPlane()
    plane1.SetOrigin(plane_center[0], plane_center[1], plane_center[2])
    plane1.SetNormal(plane_normal[0], plane_normal[1], plane_normal[2])

    planes = vtkPlaneCollection()
    planes.AddItem(plane1)

    clipper = vtkClipPolyData()
    clipper.SetInputData(poly_data)
    clipper.SetClipFunction(plane1)
    clipper.Update()
    return clipper.GetOutput()

def checkSpineMainVectorNormal(intersect_points, spine_points_center, spine_points_main_vector_normal):
    """
    func:
    :param intersect_points:
    :param spine_points_center:
    :return:
    """
    assert intersect_points.shape[0] > 3, print("the interset points is less 4")
    dis = np.sqrt(np.sum(np.square(intersect_points - spine_points_center), axis=1))
    order_idxs = np.argsort(dis)
    assert np.abs((order_idxs[0] - order_idxs[1])) == 1, print("the intersect pints error")
    if order_idxs[0] > order_idxs[1]:
        spine_points_main_vector_normal = normalizeVector(intersect_points[1] - intersect_points[0])
    else:
        spine_points_main_vector_normal = normalizeVector(intersect_points[0] - intersect_points[1])

    return spine_points_main_vector_normal

def createSpineAxis(cliped_spine_polydata,cliped_spine_points,cliped_spine_points_fit_plane_center,cliped_spine_points_fit_plane_normal,spine_points_info):
    """

    :param cliped_spine_polydata:
    :param cliped_spine_points:
    :param cliped_spine_points_fit_plane_center:
    :param cliped_spine_points_fit_plane_normal:
    :param spine_points_info:
    :return:
    """
    all_actors = []
    spine_points_center = spine_points_info[0]
    spine_points_main_vector_normal = spine_points_info[1]
    spine_points_main_vector_points = spine_points_info[2]

    cliped_spine_points_fit_plane_normal_points = np.array([cliped_spine_points_fit_plane_center,
                                                            cliped_spine_points_fit_plane_center + 20.0 * cliped_spine_points_fit_plane_normal])

    project_point0 = calProjectedPointCoordOnPlane(spine_points_main_vector_normal, spine_points_center,
                                                   cliped_spine_points_fit_plane_normal_points[0])
    project_point1 = calProjectedPointCoordOnPlane(spine_points_main_vector_normal, spine_points_center,
                                                   cliped_spine_points_fit_plane_normal_points[1])

    project_normal = normalizeVector(project_point1 - project_point0)
    project_normal_points = np.array(
        [cliped_spine_points_fit_plane_center, cliped_spine_points_fit_plane_center + 20.0 * project_normal])
    # project_normal_actor = createLineActor(project_normal_points, color="yellow", line_width=5.0)

    intersect_points = getIntersectPointsFromLineAndPolyData(project_normal_points[0], project_normal_points[1],
                                                             cliped_spine_polydata)
    intersect_point0_actor = createSphereActor(intersect_points[0], radius=5.0, opacity=1.0, color='red')
    all_actors.append(intersect_point0_actor)

    dis = np.sqrt(np.sum(np.square(cliped_spine_points - intersect_points[0]), axis=1))
    radius = 6.0
    idx = np.where(dis < radius)
    aim_points = cliped_spine_points[idx]

    aim_points_actors = createPointsActor(aim_points, radius=0.2, color='red')

    spine_axis_z_plane_actor, spine_axis_origin, spine_axis_z_normal = fitPlaneActorFromPoints(aim_points)

    if np.dot(spine_axis_z_normal, cliped_spine_points_fit_plane_normal) < 1.0e-8:
        spine_axis_z_normal = -spine_axis_z_normal

    spine_axis_z_normal_points = np.array(
        [spine_axis_origin + 30.0 * spine_axis_z_normal, spine_axis_origin - 0.0 * spine_axis_z_normal])

    spine_axis_z_actor = createLineActor(spine_axis_z_normal_points)

    spine_axis_y_point0 = calProjectedPointCoordOnPlane(spine_axis_z_normal, spine_axis_origin,
                                                        spine_points_main_vector_points[0])
    spine_axis_y_point1 = calProjectedPointCoordOnPlane(spine_axis_z_normal, spine_axis_origin,
                                                        spine_points_main_vector_points[1])
    spine_axis_y_normal = normalizeVector(spine_axis_y_point1 - spine_axis_y_point0)

    spine_axis_y_points = np.array([spine_axis_origin, spine_axis_origin + 20 * spine_axis_y_normal])

    spine_axis_y_actor = createLineActor(spine_axis_y_points, color='green')

    spine_axis_x_normal = vectorCross(spine_axis_z_normal, spine_axis_y_normal)
    spine_axis_x_points = np.array([spine_axis_origin, spine_axis_origin - 20 * spine_axis_x_normal])

    spine_axis_x_actor = createLineActor(spine_axis_x_points, color='red')

    spine_axis_y_cut_plane_actor = createCirclePlaneActor(spine_points_center, spine_axis_y_normal,
                                                          color='yellow', radius=30.0, opacity=0.9)
    all_actors.extend(aim_points_actors)
    all_actors.append(spine_axis_x_actor)
    all_actors.append(spine_axis_y_actor)
    all_actors.append(spine_axis_z_actor)
    all_actors.append(spine_axis_z_plane_actor)
    all_actors.append(spine_axis_y_cut_plane_actor)

    spine_axis_info = [spine_axis_origin, spine_axis_x_normal, spine_axis_y_normal, spine_axis_z_normal]
    return spine_axis_info, all_actors


def getReferenceSpineAxisZNormal(cur_spine_stl_file):
    """
    get the reference spine axis z normal from spine stl file
    :param spine_stl_file:
    :return:
    """
    short_name = cur_spine_stl_file[:-6]
    cur_label_index = int(cur_spine_stl_file[-6:-4])
    pre_spine_stl_file = short_name+str(cur_label_index-1)+".stl"
    next_spine_stl_file = short_name + str(cur_label_index + 1) + ".stl"

    cur_spine_points = getPointsFromSTL(cur_spine_stl_file)
    cur_spine_points_center = np.mean(cur_spine_points, axis=0)
    reference_spine_axis_z_normal = None

    if os.path.exists(pre_spine_stl_file):
        pre_spine_points = getPointsFromSTL(pre_spine_stl_file)
        pre_spine_points_center = np.mean(pre_spine_points, axis=0)
        reference_spine_axis_z_normal = normalizeVector(pre_spine_points_center - cur_spine_points_center)
    elif os.path.exists(next_spine_stl_file):
        next_spine_points = getPointsFromSTL(next_spine_stl_file)
        next_spine_points_center = np.mean(next_spine_points, axis=0)
        reference_spine_axis_z_normal = normalizeVector(cur_spine_points_center - next_spine_points_center)
    else:
        pass
    return reference_spine_axis_z_normal

def createSpineInfoFromStl(spine_stl_file):
    """
    func:get the spine axis info, and the spine points info
    :param spine_stl_file:
    :return: spine_axis_info:[spine_axis_origin, spine_axis_x_normal, spine_axis_y_normal, spine_axis_z_normal]
             spine_points_info:[spine_points_center, spine_points_main_vector_normal]
    """
    all_actors = []
    reference_spine_axis_z_normal = getReferenceSpineAxisZNormal(spine_stl_file)

    spine_polydata = createPolyDataFromSTL(spine_stl_file)
    spine_actor = createActorFromPolydata(spine_polydata, opacity=0.75)

    spine_points = getPointsFromSTL(spine_stl_file, num_points=5000)
    spine_points_center, \
    spine_points_main_vector_points, \
    spine_points_main_vector_normal = getMainVectorByPCA(spine_points, delta=70.0)

    spine_points_main_vector_normal_actor = createLineActor(spine_points_main_vector_points)

    intersect_points_with_spine = getIntersectPointsFromLineAndPolyData(spine_points_main_vector_points[0],
                                                                        spine_points_main_vector_points[1],
                                                                        spine_polydata)
    spine_points_main_vector_normal = checkSpineMainVectorNormal(intersect_points_with_spine, spine_points_center,
                                                                 spine_points_main_vector_normal)
    spine_points_main_vector_points = [spine_points_center + 60.0 * spine_points_main_vector_normal,
                                       spine_points_center - 60.0 * spine_points_main_vector_normal]

    spine_points_info = [spine_points_center, spine_points_main_vector_normal, spine_points_main_vector_points]

    cut_center = spine_points_center
    cliped_spine_polydata = createClipedPolydata(cut_center, spine_points_main_vector_normal, spine_polydata)
    cliped_spine_points = np.asarray(cliped_spine_polydata.GetPoints().GetData())

    cliped_spine_points_fit_plane_actor, \
    cliped_spine_points_fit_plane_center, \
    cliped_spine_points_fit_plane_normal = fitPlaneActorFromPoints(cliped_spine_points)

    cliped_spine_points_fit_plane_normal_actor = createLineActor([cliped_spine_points_fit_plane_center, cliped_spine_points_fit_plane_center+20.0*cliped_spine_points_fit_plane_normal], 'red')

    all_actors.append(cliped_spine_points_fit_plane_normal_actor)
    all_actors.append(spine_actor)
    all_actors.append(cliped_spine_points_fit_plane_actor)
    all_actors.append(spine_points_main_vector_normal_actor)
   # showActors(all_actors)

    if reference_spine_axis_z_normal is not None:
        two_vectors_product = np.dot(cliped_spine_points_fit_plane_normal, reference_spine_axis_z_normal)
        print("two vectors product:", two_vectors_product)
        if two_vectors_product < 1.0e-8:
            cliped_spine_points_fit_plane_normal = -cliped_spine_points_fit_plane_normal

    cliped_spine_points_fit_plane_center_actor = createSphereActor(cliped_spine_points_fit_plane_center, radius=1.0, opacity=1.0, color='red')

    spine_axis_info, cur_all_actors = createSpineAxis(cliped_spine_polydata,cliped_spine_points,cliped_spine_points_fit_plane_center,cliped_spine_points_fit_plane_normal,spine_points_info)
    #check the cliped_spine_points_normal
    all_actors.append(spine_actor)
    all_actors.append(spine_points_main_vector_normal_actor)
    all_actors.extend(cur_all_actors)
    all_actors.append(cliped_spine_points_fit_plane_actor)
  #  showActors(all_actors)
    return spine_axis_info, spine_points_info, all_actors


def cutpolydata(polydata, cutplane):
    cutter = vtkCutter()
    cutter.SetCutFunction(cutplane)
    cutter.SetInputData(polydata)
    cutter.Update()

    cutter_mapper = vtkPolyDataMapper()
    cutter_mapper.SetInputData(cutter.GetOutput())
    cutter_actor = vtkActor()
    cutter_actor.SetMapper(cutter_mapper)
    return cutter_actor, cutter.GetOutput()


def create_spine_cutplane_center(left_plane_parameters, right_plane_parameters, spine_polydata, coordinates, origin_in_vtk):
    all_actors = []
    sym_cutplane = vtkPlane()
    sym_cutplane2 = vtkPlane()
    sym_cutplane.SetOrigin(origin_in_vtk)
    sym_cutplane2.SetOrigin(origin_in_vtk)
    sym_cutplane.SetNormal(coordinates[0])
    sym_cutplane2.SetNormal(-coordinates[0])

    clip_planes = vtkPlaneCollection()
    clip_planes2 = vtkPlaneCollection()
    clip_planes.AddItem(sym_cutplane)
    clip_planes2.AddItem(sym_cutplane2)

    clipper = vtkClipClosedSurface()
    clipper.SetInputData(spine_polydata)
    clipper.SetClippingPlanes(clip_planes)
    clipper.Update()
    clipped_spine = vtkPolyData()
    clipped_spine.DeepCopy(clipper.GetOutput())

    clipper2 = vtkClipClosedSurface()
    clipper2.SetInputData(spine_polydata)
    clipper2.SetClippingPlanes(clip_planes2)
    clipper2.Update()
    clipped_spine2 = vtkPolyData()
    clipped_spine2.DeepCopy(clipper2.GetOutput())

    clipMapper = vtkDataSetMapper()
    clipMapper.SetInputData(clipped_spine2)
    clipMapper2 = vtkPolyDataMapper()
    clipMapper2.SetInputData(clipped_spine)

    clipActor = vtkActor()
    clipActor.SetMapper(clipMapper)
    clipActor.GetProperty().SetOpacity(0.5)
    clipActor2 = vtkActor()
    clipActor2.SetMapper(clipMapper2)
    clipActor2.GetProperty().SetOpacity(0.5)
    all_actors.append(clipActor2)
    all_actors.append(clipActor)

    left_center_actor = createSphereActor(left_plane_parameters[0], 3)
    right_center_actor = createSphereActor(right_plane_parameters[0], 3)
    all_actors.append(right_center_actor)
    all_actors.append(left_center_actor)
    # clipActor.GetProperty().SetColor(1.0000, 0.3882, 0.2784)
    # clipActor.GetProperty().SetInterpolationToFlat()
    # showActors(all_actors)

    com_actor, com_point = create_center_of_mass_actor(clipped_spine)
    com2_actor, com_point2 = create_center_of_mass_actor(clipped_spine2)
    dist_com2left = np.linalg.norm(np.asarray(com_point) - np.asarray(left_plane_parameters[0]))
    dist_com2right = np.linalg.norm(np.asarray(com_point2) - np.asarray(right_plane_parameters[0]))

    if dist_com2left < dist_com2right:
        left_clipper = left_plane_parameters
        right_clipper = right_plane_parameters
    else:
        left_clipper = right_plane_parameters
        right_clipper = left_plane_parameters

    left_cutplane = vtkPlane()
    left_cutplane.SetOrigin(left_clipper[0])
    left_cutplane.SetNormal(left_clipper[1])

    right_cutplane = vtkPlane()
    right_cutplane.SetOrigin(right_clipper[0])
    right_cutplane.SetNormal(right_clipper[1])

    left_cutplane_actor, left_cutplane = cutpolydata(clipped_spine, left_cutplane)
    all_actors.append(left_cutplane_actor)

    showActors(all_actors)


def cal_max_radius(spine_polydata, direction, plane_center_point):
    points_list = []
    dist = lambda p, inner_p: np.linalg.norm(p - inner_p)

    cutplane = vtkPlane()
    cutplane.SetOrigin(plane_center_point)
    cutplane.SetNormal(direction)

    cutter = vtkCutter()
    cutter.SetCutFunction(cutplane)
    cutter.SetInputData(spine_polydata)

    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(cutter.GetOutputPort())
    stripper.Update()
    slicing_boundary = stripper.GetOutput()

    min_dist_idx = 0
    min_dist = 1000
    n_cells = slicing_boundary.GetNumberOfCells()

    for c in range(n_cells):
        points_list.append(slicing_boundary.GetCell(c).GetPoints().GetPoint(0))

    for idx, point in enumerate(points_list):
        distboundary2center = dist(point, plane_center_point)
        if distboundary2center < min_dist:
            min_dist = distboundary2center
            min_dist_idx = idx

    target_points = np.asarray(slicing_boundary.GetCell(min_dist_idx).GetPoints().GetData())
    min_dist = 1000
    for p in target_points:
        dist2center = dist(p, plane_center_point)
        if dist2center < min_dist:
            min_dist = dist2center

    return (2*min_dist*0.7) // 0.5 * 0.5
