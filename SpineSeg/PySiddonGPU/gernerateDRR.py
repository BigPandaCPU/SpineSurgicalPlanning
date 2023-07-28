import numpy as np
import itk
import SimpleITK as sitk
from PIL import Image
from PyQt5.QtCore import QUuid, QDateTime
from SpineSeg.PySiddonGPU.libPySiddonGpu import PySiddonGpu as pySiddonGpu

import os
import copy


class DicomInfo:
    def __init__(self):
        self.projectVersion = "v1.1"
        self.imageData = None
        self.patientName = ""
        self.patientID = ""
        self.patientSex = ""
        self.patientAge = ""
        self.seriesDate = ""  #
        self.createTime = ""  #手术创建时间

        self.dimensions = []
        self.origin = []
        self.spacing = []
        self.direction = []
        self.extent = []



def ImageReader(image_file_path):
    """
    :param image_file_path:
    :return:
    """
    dicom_info = DicomInfo()
    if "nii.gz" in image_file_path:
        image = sitk.ReadImage(image_file_path)
    else:
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(image_file_path)
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)
        series_reader.MetaDataDictionaryArrayUpdateOn()
        image = series_reader.Execute()

        dicom_info.patientName = series_reader.GetMetaData(0, "0010|0010")  # patient name
        dicom_info.patientID = series_reader.GetMetaData(0, "0010|0020")  # patient id
        dicom_info.patientSex = series_reader.GetMetaData(0, "0010|0040")  # patient sex
        dicom_info.patientAge = series_reader.GetMetaData(0, "0010|1010")  # patient age
        dicom_info.seriesDate = series_reader.GetMetaData(0, "0008|0021")  # series data

        dicom_info.dimensions = image.GetSize()
        dicom_info.origin = image.GetOrigin()
        dicom_info.spacing = image.GetSpacing()
        dicom_info.extent = [0, dicom_info.dimensions[0] - 1, 0, dicom_info.dimensions[1] - 1, 0,
                             dicom_info.dimensions[2] - 1]
        dicom_info.direction = image.GetDirection()


    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    size = image.GetSize()
    direction = image.GetDirection()


    dicom_info.imageData = copy.deepcopy(image)
    dicom_info.createTime = QDateTime().currentMSecsSinceEpoch()


    volume_center = np.asarray(origin) + np.multiply(spacing, np.divide(size, 2.))
    image_info = {'Spacing': spacing, 'Origin': origin, 'Size': size,  'Volume_center': volume_center}
    image_np = sitk.GetArrayFromImage(image).astype(np.float32)

    properties = {}
    properties["original_size_of_raw_data"] = np.array(image.GetSize())[[2, 1, 0]]
    properties["size_after_cropping"] = properties["original_size_of_raw_data"]
    properties["original_spacing"] = np.array(image.GetSpacing())[[2, 1, 0]]
    properties["list_of_data_files"] = [image_file_path]
    properties["seg_file"] = None

    properties["itk_origin"] = origin
    properties["itk_spacing"] = spacing
    properties["itk_direction"] = direction
    properties["crop_bbox"] =[[0, size[2]],[0,size[1]], [0, size[0]]]

    return image_np, image_info, properties, dicom_info

def createDRRView(mov_img_info, projector_info, drr_type="front"):
    """
    :return:
    """
    Dimension = 3
    PixelType = itk.F
    ImageType = itk.Image[PixelType, Dimension]
    RegionType = itk.ImageRegion[Dimension]
    PhyImageType = itk.Image[itk.Vector[itk.F, Dimension], Dimension]

    source_point = [0.0] * Dimension
    source_point[2] = mov_img_info['Volume_center'][2]

    if drr_type == "front":
        source_point[0] = mov_img_info['Volume_center'][0]
        source_point[1] = mov_img_info['Volume_center'][1] - projector_info['VCS']

    else:
        source_point[0] = mov_img_info['Volume_center'][0] - projector_info['VCS']
        source_point[1] = mov_img_info['Volume_center'][1]

    source_point = np.array(source_point, dtype=np.float32)

    DRR = ImageType.New()
    DRRregion = RegionType()

    DRRstart = itk.Index[Dimension]()
    DRRstart.Fill(0)

    DRRsize = [0] * Dimension

    if drr_type == "front":
        DRRsize[0] = projector_info['DRRsize_x']
        DRRsize[1] = 1
    else:
        DRRsize[0] = 1
        DRRsize[1] = projector_info['DRRsize_x']

    DRRsize[2] = projector_info['DRRsize_y']

    DRRregion.SetSize(DRRsize)
    DRRregion.SetIndex(DRRstart)

    DRRspacing = itk.Point[itk.F, Dimension]()
    DRRspacing[0] = 1.0  # x spacing
    DRRspacing[1] = 1.0  # y spacing
    DRRspacing[2] = 1.0  # z spacing

    DRRorigin = itk.Point[itk.F, Dimension]()
    if drr_type == "front":
        DRRorigin[0] = mov_img_info['Volume_center'][0] - DRRspacing[0] * (DRRsize[0] - 1.0) / 2.
        DRRorigin[1] = mov_img_info['Volume_center'][1] + projector_info['VCD']
        DRRorigin[2] = mov_img_info['Volume_center'][2] - DRRspacing[2] * (DRRsize[2] - 1.0) / 2.
    else:
        DRRorigin[0] = mov_img_info['Volume_center'][0] + projector_info['VCD']
        DRRorigin[1] = mov_img_info['Volume_center'][1] - DRRspacing[1] * (DRRsize[1] - 1.0) / 2.
        DRRorigin[2] = mov_img_info['Volume_center'][2] - DRRspacing[2] * (DRRsize[2] - 1.0) / 2.


    DRR.SetRegions(DRRregion)
    DRR.Allocate()
    DRR.SetSpacing(DRRspacing)
    DRR.SetOrigin(DRRorigin)

    # direction = DRR.GetDirection()
    # direction_array = itk.array_from_matrix(direction)
    # print(direction_array)

    # Get array of physical coordinates for the DRR at the initial position
    PhysicalPointImagefilter = itk.PhysicalPointImageSource[PhyImageType].New()
    PhysicalPointImagefilter.SetReferenceImage(DRR)
    PhysicalPointImagefilter.SetUseReferenceImage(True)
    PhysicalPointImagefilter.Update()
    sourceDRR = PhysicalPointImagefilter.GetOutput()

    tmp = itk.GetArrayFromImage(sourceDRR)
    sourceDRR_array_to_reshape = np.squeeze(tmp)
    if drr_type == "front":
        sourceDRR_array_reshaped = sourceDRR_array_to_reshape.reshape(DRRsize[0] * DRRsize[2], Dimension)
    else:
        sourceDRR_array_reshaped = sourceDRR_array_to_reshape.reshape(DRRsize[1] * DRRsize[2], Dimension)
    DRRPhy_array = np.ravel(sourceDRR_array_reshaped, order='C').astype(np.float32)

    if drr_type == "front":
        drr_size = (DRRsize[2], DRRsize[0])
    else:
        drr_size = (DRRsize[2], DRRsize[1])
    return source_point, DRRPhy_array, drr_size


def saveDrrImg(data, save_png_file):
    """
    :param data:
    :param save_png_file:
    :return:
    """
    data_min = np.min(data)
    data_max = np.max(data)

    # print("data_min:", data_min)
    # print("data_max:", data_max)
    data = (data - data_min) / (data_max - data_min) * 255
    data = data[::-1]
    image_out = Image.fromarray(data.astype('uint8'))
    img_as_img = image_out.convert("RGB")
    img_as_img.save(save_png_file)

def getRealSizeDRRData(drr_data, movImageInfo, projector_info, drr_type="side"):
    """
    """
    movImgSize = movImageInfo['Size']
    movSpacing = movImageInfo['Spacing']
    Xdim = movImgSize[0]
    Xspacing = movSpacing[0]

    Ydim = movImgSize[1]
    Yspacing = movSpacing[1]

    Zdim = movImgSize[2]
    Zspacing = movSpacing[2]

    VCD = projector_info['VCD']
    VCS = projector_info['VCS']

    DRRsize_x = projector_info["DRRsize_x"]
    DRRsize_y = projector_info["DRRsize_y"]

    if drr_type == "front":
        startX = int((DRRsize_x - (VCD+VCS)/VCS*Xdim*Xspacing)/2.0)
        if startX < 0:
            startX = 0
        endX =  DRRsize_x - startX

    else:
        startX = int((DRRsize_x - (VCD + VCS) / VCS * Ydim * Yspacing) / 2.0)
        if startX < 0:
            startX = 0
        endX = DRRsize_x - startX

    startY = int((DRRsize_y - (VCD+VCS)/VCS*Zdim*Zspacing)/2.0)
    if startY < 0:
        startY = 0
    endY = DRRsize_y - startY

    return drr_data[startY:endY, startX:endX]



def generateDRR(input_img_file, save_png_dir):
    """
    """
    myu_water = 0.2683

    Projector_info = {}
    Projector_info['VCD'] = 500
    Projector_info['VCS'] = 1000
    Projector_info['DRRspacing_x'] = 1.0
    Projector_info['DRRspacing_y'] = 1.0
    Projector_info['DRR_ppx'] = 0
    Projector_info['DRR_ppy'] = 0
    Projector_info['DRRsize_x'] = 1000
    Projector_info['DRRsize_y'] = 2000
    Projector_info['threadsPerBlock_x'] = 16
    Projector_info['threadsPerBlock_y'] = 16
    Projector_info['threadsPerBlock_z'] = 1

    save_front_png_file = os.path.join(save_png_dir, "DRR_front.png")
    save_side_png_file = os.path.join(save_png_dir, "DRR_side.png")

    movingImageFileName = input_img_file
    print("\nread image.")
    movImage, movImageInfo, properties, dicomInfo = ImageReader(movingImageFileName)

    movImgSize = movImageInfo['Size']
    save_img_info_file = os.path.join(save_png_dir, "img_info.txt")
    fp = open(save_img_info_file, "w")
    fp.write("%d %d %d"%(movImgSize[0], movImgSize[1], movImgSize[2]))
    fp.close()

    origin_img = copy.deepcopy(movImage)
    #origin_img = np.expand_dims(origin_img,axis=0)


    print("read image done.")
    movImgArray_1d = np.ravel(movImage, order='C')  # ravel does not generate a copy of the array (it is faster than flatten)
    idx0 = np.where(movImgArray_1d < 0.0)
    movImgArray_1d = movImgArray_1d / 1000.0 * myu_water
    movImgArray_1d[idx0] = 0.0

    NumThreadsPerBlock = np.array([16, 16, 1]).astype(np.int32)
    MovSize_forGpu = np.array([movImageInfo['Size'][0], movImageInfo['Size'][1], movImageInfo['Size'][2]]).astype(
        np.int32)
    MovSpacing_forGpu = np.array(
        [movImageInfo['Spacing'][0], movImageInfo['Spacing'][1], movImageInfo['Spacing'][2]]).astype(np.float32)

    # Calculate side planes
    X0 = movImageInfo['Volume_center'][0] - movImageInfo['Spacing'][0] * movImageInfo['Size'][0] / 2.0
    Y0 = movImageInfo['Volume_center'][1] - movImageInfo['Spacing'][1] * movImageInfo['Size'][1] / 2.0
    Z0 = movImageInfo['Volume_center'][2] - movImageInfo['Spacing'][2] * movImageInfo['Size'][2] / 2.0
    DRRsize_forGpu = np.array([Projector_info['DRRsize_x'], Projector_info['DRRsize_y'], 1]).astype(np.int32)

    print("pySiddonGpu init:")
    projector = pySiddonGpu(NumThreadsPerBlock,
                            movImgArray_1d,
                            MovSize_forGpu,
                            MovSpacing_forGpu,
                            X0.astype(np.float32), Y0.astype(np.float32), Z0.astype(np.float32),
                            DRRsize_forGpu)

    print("pySiddonGpu init done.")

    source_point_front, drr_phy_array_front, drr_size_front = createDRRView(movImageInfo, Projector_info,
                                                                            drr_type="front")
    drrImgArrayFront = np.zeros(DRRsize_forGpu[0] * DRRsize_forGpu[1] * DRRsize_forGpu[2], dtype=np.float32, order='C')
    output_front = np.ravel(drrImgArrayFront, order='C').astype(np.float32)

    projector.generateDRR(source_point_front, drr_phy_array_front, output_front)
    output_reshaped_front = np.reshape(output_front, drr_size_front, order='C')
    output_reshaped_front = getRealSizeDRRData(output_reshaped_front, movImageInfo, Projector_info, drr_type="front")
    saveDrrImg(output_reshaped_front, save_front_png_file)

    source_point_side, drr_phy_array_side, drr_size_side = createDRRView(movImageInfo, Projector_info,
                                                                         drr_type="side")

    drrImgArraySide = np.zeros(DRRsize_forGpu[0] * DRRsize_forGpu[1] * DRRsize_forGpu[2], dtype=np.float32, order='C')
    output_side = np.ravel(drrImgArraySide, order='C').astype(np.float32)
    projector.generateDRR(source_point_side, drr_phy_array_side, output_side)
    output_reshaped_side = np.reshape(output_side, drr_size_side, order='C')
    output_reshaped_side = getRealSizeDRRData(output_reshaped_side, movImageInfo, Projector_info, drr_type="side")

    saveDrrImg(output_reshaped_side, save_side_png_file)
    projector.releaseMem()
    # del projector
    #print("done\n")
    return origin_img, properties, dicomInfo



if __name__=="__main__":

    import time
    import tqdm
    input_img_dir = "/media/alg/data3/DeepSpineData/spine_test/Test09/nii"
    save_png_dir = "/media/alg/data3/DeepSpineData/spine_test/Test09/drr"
    img_names = os.listdir(input_img_dir)
    for img_name in tqdm.tqdm(img_names):
        # if "0068" not in img_name:
        #     continue
        shot_name = img_name.replace(".nii.gz","")
        print(shot_name)
        save_png_path = os.path.join(save_png_dir)
        os.makedirs(save_png_path, exist_ok=True)
        img_file = os.path.join(input_img_dir, img_name)
        generateDRR(img_file, save_png_path)
