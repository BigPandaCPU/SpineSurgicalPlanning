import os
import time
import zipfile
import numpy as np
from PyQt5.QtWidgets import *
import SimpleITK as sitk
from PyQt5.QtCore import QUuid, QDateTime
from SpinePlanning.tools.vtk_tools import createPolyDataNormalsFromArray, saveSTLFile

SpineLabelDict_int2str={1:"C1", 2:"C2", 3:"C3", 4:"C4", 5:"C5", 6:"C6", 7:"C7", 8:"T1", 9:"T2",
                 10:"T3", 11:"T4", 12:"T5", 13:"T6", 14:"T7", 15:"T8", 16:"T9", 17:"T10",
                 18:"T11", 19:"T12", 20:"L1", 21:"L2", 22:"L3", 23:"L4", 24:"L5", 25:"L6"}
SpineLabelDict_str2int = {}
for k,v in SpineLabelDict_int2str.items():
    SpineLabelDict_str2int[v] = k

Index2Color = {
        1:"Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff\\xff\\xff\\0\\0\\0\\0\\0\\0)",
        2:"Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff\\0\\0\\xff\\xff\\0\\0\\0\\0)",
        3:"Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff\\0\\0\\0\\0\\xff\\xff\\0\\0)",
        4:"Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff\\xff\\xff\\xff\\xff\\0\\0\\0\\0)",
        5:"Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff\\0\\0\\xff\\xff\\xff\\xff\\0\\0)",
        6:"Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff\\xff\\xff\\0\\0\\xff\\xff\\0\\0)",
        7:"Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff\\xff\\xff\\xef\\xef\\xd5\\xd5\\0\\0)",
        8:"Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff\\0\\0\\0\\0\\xcd\\xcd\\0\\0)",
        9:"Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff\\xcd\\xcd\\x85\\x85??\\0\\0)",
        10:"Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff\\xd2\\xd2\\xb4\\xb4\\x8c\\x8c\\0\\0)",
        11:"Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff\\x66\\x66\\xcd\\xcd\\xaa\\xaa\\0\\0)",
        12:"Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff\\0\\0\\0\\0\\x80\\x80\\0\\0)",
        13:"Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff\\0\\0\\x8b\\x8b\\x8b\\x8b\\0\\0)",
        14:"Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff..\\x8b\\x8bWW\\0\\0)",
        15:"Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff\\xff\\xff\\xe4\\xe4\\xe1\\xe1\\0\\0)",
        16:"Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xffjjZZ\\xcd\\xcd\\0\\0)",
        17:"Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff\\xdd\\xdd\\xa0\\xa0\\xdd\\xdd\\0\\0)",
        18:"Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff\\xe9\\xe9\\x96\\x96zz\\0\\0)",
        19:"Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff\\xa5\\xa5****\\0\\0)",
        20:"Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff\\xff\\xff\\xfa\\xfa\\xfa\\xfa\\0\\0)",
        21:"Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff\\x93\\x93pp\\xdb\\xdb\\0\\0)",
        22:"Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff\\xda\\xdapp\\xd6\\xd6\\0\\0)",
        23:"Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xffKK\\0\\0\\x82\\x82\\0\\0)",
        24:"Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff\\xff\\xff\\xb6\\xb6\\xc1\\xc1\\0\\0)",
        25:"Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff<<\\xb3\\xb3\\x17\\x17\\0\\0)",
}


def write2File(fp,line_id, id, point):
    """
    func:
    :param fp: file pointer
    :param point:
    :return:
    """
    fp.write("%d\\ID=%d\n" % (line_id, id))
    fp.write("%d\\x=%.4f\n" % (line_id, point[0]))
    fp.write("%d\\y=%.4f\n" % (line_id, point[1]))
    fp.write("%d\\z=%.4f\n" % (line_id, point[2]))


def writeVirtualWallProp(save_virtual_wall_file, femurProsthesisInfo, tibiaProsthesisInfo):
    tibia_wall = ', '.join(str(i) for i in tibiaProsthesisInfo.virtualwall)
    femur_wall = ', '.join(str(i) for i in femurProsthesisInfo.virtualwall)
    with open(save_virtual_wall_file, 'w') as f:
        f.write("[FemurWall]\n")
        f.write("Femur=")
        f.write(f'{femur_wall}\n')

        f.write("[TibiaWall]\n")
        f.write("Tibia=")
        f.write(f'{tibia_wall}\n')


def writeKeyPointsProp(save_key_point_file, femurProsthesisInfo, tibiaProsthesisInfo):

    femur_key_points = femurProsthesisInfo.keyPoints
    tibia_key_points = tibiaProsthesisInfo.keyPoints
    fp_w = open(save_key_point_file, "w")

    femur_head_center = femur_key_points['femur_head_center']

    tibia_ankle_center_label = 12
    tibia_ankle_center_point = tibia_key_points[tibia_ankle_center_label]

    tibia_platform_center_label = 13
    tibia_platform_center_point = tibia_key_points[tibia_platform_center_label]



    key_points_ID = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 25, 26, 27, 28]
    fp_w.write("[PointsProperty]\n")
    line_id = 1
    size_count = 21
    for id in key_points_ID:
        if id in femur_key_points.keys():
            cur_point = femur_key_points[id]
            write2File(fp_w, line_id, id, cur_point)


        if id in tibia_key_points.keys():
            cur_point = tibia_key_points[id]
            write2File(fp_w, line_id, id, cur_point)

        if id == 28:
            cur_point = femur_head_center
            write2File(fp_w, line_id, id, cur_point)
        line_id += 1
    fp_w.write("size=%d\n\n" % size_count)
    fp_w.write("[ForceLine]\n")
    fp_w.write("Method=2\n")
    fp_w.write("Line=%f, %.4f, %.4f, %.4f, %.4f, %.4f\n" % (tibia_ankle_center_point[0],
                                                            tibia_ankle_center_point[1],
                                                            tibia_ankle_center_point[2],
                                                            tibia_platform_center_point[0],
                                                            tibia_platform_center_point[1],
                                                            tibia_platform_center_point[2]))
    fp_w.close()

def convertChinese2Unicode(string):
    """
    func:将带有中文的字符串转换为unicode编码
    :param string:
    :return:
    """
    string_out = ""
    for i in range(len(string)):
        ch = string[i]
        if '\u4e00' <= ch <= '\u9fff':
            tmp = ch.encode('unicode-escape').decode()
            ch_new = tmp.replace("u", "x")
            string_out += ch_new
        else:
            string_out += ch
    return string_out

def getImageFromNrrd(nrrd_file):
    """
    func:get mask array,origin, spacing from nrrd
    :param nrrd_file:
    :return:
    """
    if not os.path.exists(nrrd_file):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("the nrrd file %s is not exist!"%nrrd_file)
            msg.exec_()
            return

    reader = sitk.ReadImage(nrrd_file)
    origin = reader.GetOrigin()
    spacing = reader.GetSpacing()
    direction = reader.GetDirection()
    img = sitk.GetArrayFromImage(reader)
    return img, origin, spacing, direction


def compressBinaryArray(mask_data):
    """
    func:compress mask data
    :param mask_data:numpy, shape(x, y, z)
    :return:
    """

    size = mask_data.size
    mask_data = mask_data.reshape(-1, size)[0]
    mask_data.flat[0] = 0
    diff = mask_data[1:] - mask_data[:-1]
    idx_neq = np.where(diff != 0)[0]
    binary_statistics = [idx_neq[0]+1, ] + list(idx_neq[1:]-idx_neq[:-1]) + [size - idx_neq[-1]-1]
    return np.array(binary_statistics, np.int32)

def compressBinaryArray2(mask_data):
    size = mask_data.size
    mask_data.flat[0] = 0
    binary_statistics=[]
    count = 1
    for i in range(1, size):
        if mask_data.flat[i] == mask_data.flat[i-1]:
            count += 1
        else:
            binary_statistics.append(count)
            count = 1
    binary_statistics.append(count)
    # print(len(binary_statistics))
    # print(binary_statistics[:20])
    return np.array(binary_statistics, np.int32)


def writeDcmImage(save_dcm_file, dicom_info):
    """
    func:
    :param save_dcm_file:
    :param dicom_info:
    :return:
    """
    with open(save_dcm_file, "wb") as fp:
        project_version = dicom_info.projectVersion
        dims = np.array(dicom_info.dimensions, dtype=np.int32)
        origin = np.array(dicom_info.origin, dtype=np.float64)
        spacing = np.array(dicom_info.spacing, dtype=np.float64)
        extend = np.array(dicom_info.extent, dtype=np.int32)
        white_space = np.array([0, ] * 256, dtype=np.uint8)
        fp.write(project_version.encode())
        fp.write(dims.tobytes())
        fp.write(origin.tobytes())
        fp.write(spacing.tobytes())
        fp.write(extend.tobytes())
        fp.write(white_space.tobytes())
        image3D_array = sitk.GetArrayFromImage(dicom_info.imageData)
        image3D_array = image3D_array.astype(np.int16)
        fp.write(image3D_array.tobytes())
        fp.close()

def writeMaskImage(save_mask_file, dicom_info, mask_array):
    """
    func:
    :param mask_file:
    :param dicom_info:
    :param mask_array:
    :return:
    """
    with open(save_mask_file, "wb") as fp:
        project_version = dicom_info.projectVersion
        dims = np.array(dicom_info.dimensions, dtype=np.int32)
        origin = np.array(dicom_info.origin, dtype=np.float64)
        spacing = np.array(dicom_info.spacing, dtype=np.float64)
        extend = np.array(dicom_info.extent, dtype=np.int32)
        white_space = np.array([0,]*256, dtype=np.uint8)
        compress_mask_data = compressBinaryArray(mask_array)
        size = np.array([compress_mask_data.size * compress_mask_data.itemsize], dtype=np.int32)
        fp.write(project_version.encode())
        fp.write(dims.tobytes())
        fp.write(origin.tobytes())
        fp.write(spacing.tobytes())
        fp.write(extend.tobytes())
        fp.write(white_space.tobytes())
        fp.write(size.tobytes())
        fp.write(compress_mask_data.tobytes())
        fp.close()

def writeMaskProp(save_mask_prop_file):
    """
    :param save_mask_prop:
    :return:
    """
    fp = open(save_mask_prop_file, "w")
    fp.write("[MaskProperty]\n")
    fp.write("mask1\Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xffUU\\xa8\\xa8\\x5\\x5\\0\\0)\n")
    fp.write("mask1\ID=1\n")
    fp.write("mask2\Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff\\0\\0\\x8b\\x8b\\xea\\xea\\0\\0)\n")
    fp.write("mask2\ID=2\n")
    fp.close()

def writeSpineMaskProp(spine_mask_info, save_mask_prop_file):
    """
    :param save_mask_prop:
    :return:
    """
    fp = open(save_mask_prop_file, "w")
    fp.write("[MaskProperty]\n")

    for k in spine_mask_info.keys():
        fp.write("%s\%s\n"%(k, spine_mask_info[k]["Color"]))
        fp.write("%s\ID=%d\n"%(k, spine_mask_info[k]["ID"]))
        fp.write("%s\DisplayPostfix=%d\n"%(k, spine_mask_info[k]["DisplayPostfix"]))
    fp.close()

def writePatientProp(save_patient_prop_file, dicom_info, cur_part):
    """
    :param patient_info:
    :return:
    """
    fp = open(save_patient_prop_file, "w")
    fp.write("[Version]\n")
    fp.write("VName=%s\n\n" % dicom_info.projectVersion)
    fp.write("[PatientProperty]\n")
    fp.write("Name=%s\n" % dicom_info.patientName)
    fp.write("ID=%s\n" % dicom_info.patientID)
    fp.write("Sex=%s\n" % dicom_info.patientSex)
    fp.write("Date=%s\n" % dicom_info.seriesDate)
    fp.write("Age=%s\n" % dicom_info.patientAge)
    fp.write("Part=%s\n" % str(cur_part))
    fp.write("CreateTime=%s\n" % dicom_info.createTime)
    fp.close()
def writeSpinePatientProp(save_patient_prop_file, dicom_info):
    """
    :param patient_info:
    :return:
    """
    fp = open(save_patient_prop_file, "w")
    fp.write("[Version]\n")
    fp.write("VName=%s\n\n" % dicom_info.projectVersion)
    fp.write("[PatientProperty]\n")
    fp.write("Name=%s\n" % dicom_info.patientName)
    fp.write("ID=%s\n" % dicom_info.patientID)
    fp.write("Sex=%s\n" % dicom_info.patientSex)
    fp.write("Date=%s\n" % dicom_info.seriesDate)
    fp.write("Age=%s\n" % dicom_info.patientAge)
    fp.write("CreateTime=%s\n" % dicom_info.createTime)
    fp.close()

def writeStlProp(save_stl_prop_file):
    """
    :param save_stl_prop_file:
    :return:
    """
    fp = open(save_stl_prop_file, "w")
    fp.write("[StlProperty]\n")
    fp.write("bone1\ID=1\n")
    fp.write("bone1\BoneType=1\n")
    fp.write("bone1\Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff\\xf7\\xf7\\xda\\xda\\b\\b\\0\\0)\n")

    fp.write("bone2\ID=2\n")
    fp.write("bone2\BoneType=2\n")
    fp.write("bone2\Color=@Variant(\\0\\0\\0\\x43\\x1\\xff\\xff\\xd3\\xd3\\x8e\\x8evv\\0\\0)\n")
    fp.close()
def writeSpineStlProp(spine_stl_info, save_stl_prop_file):
    """
    :param save_mask_prop:
    :return:
    """
    fp = open(save_stl_prop_file, "w")
    fp.write("[StlProperty]\n")
    for k in spine_stl_info.keys():
        fp.write("%s\%s\n"%(k, spine_stl_info[k]["Color"]))
        fp.write("%s\ID=%d\n"%(k, spine_stl_info[k]["ID"]))
        fp.write("%s\BoneType=%d\n"%(k, spine_stl_info[k]["BoneType"]))
    fp.close()

def writeOsteotomyProp(save_planning_file, femurProsthesisInfo, tibiaProsthesisInfo):
    fp = open(save_planning_file, "w")
    fp.write("[OsteotomyFemurProperty]\n")
    femurProsthesisType = convertChinese2Unicode(femurProsthesisInfo.type)
    tibiaProsthesisType = convertChinese2Unicode(tibiaProsthesisInfo.type)

    fp.write("Types=%s, %s\n" % (femurProsthesisType, femurProsthesisInfo.name))
    fp.write("Data=")
    femur_matrix_Array = femurProsthesisInfo.matrix
    for i in range(4):
        for j in range(4):
            fp.write("%g, " % femur_matrix_Array[i, j])

    fp.write("%g, " % femurProsthesisInfo.varusAngle)
    fp.write("%g, " % femurProsthesisInfo.valgusAngle)
    fp.write("%g, " % femurProsthesisInfo.anteversionAngle)
    fp.write("%g, " % femurProsthesisInfo.casterAngle)
    fp.write("%g, " % femurProsthesisInfo.internalRotationAngle)
    fp.write("%g, " % femurProsthesisInfo.externalRotationAngle)
    fp.write("%g, " % femurProsthesisInfo.distalMedialOsteotomy)    #远端内侧
    fp.write("%g, " % femurProsthesisInfo.distalLateralOsteotomy)   #远端外侧
    fp.write("%g, " % femurProsthesisInfo.medialCondyleOsteotomy)   #后髁内侧
    fp.write("%g\n" % femurProsthesisInfo.lateralCondyleOsteotomy)  #后髁外侧

    fp.write("\n")
    fp.write("[OsteotomyTibiaProperty]\n")
    fp.write("Types=%s, %s\n" % (tibiaProsthesisType, tibiaProsthesisInfo.name))
    fp.write("Data=")
    tibia_matrix_Array = tibiaProsthesisInfo.matrix
    for i in range(4):
        for j in range(4):
            fp.write("%g, " % tibia_matrix_Array[i, j])

    fp.write("%g, " % tibiaProsthesisInfo.varusAngle)
    fp.write("%g, " % tibiaProsthesisInfo.valgusAngle)
    fp.write("%g, " % tibiaProsthesisInfo.anteversionAngle)
    fp.write("%g, " % tibiaProsthesisInfo.casterAngle)
    fp.write("%g, " % tibiaProsthesisInfo.internalRotationAngle)
    fp.write("%g, " % tibiaProsthesisInfo.externalRotationAngle)
    fp.write("%g, " % tibiaProsthesisInfo.proximalMedialOsteotomy)   #近端内侧
    fp.write("%g\n" % tibiaProsthesisInfo.proximalLateralOsteotomy)  #近端外侧

    fp.write("\n")
    fp.write("[OsteotomyFemurAdditionAngle]\n")
    fp.write("Data=%g,%g,%g\n" % (0, 0, 0))
    fp.write("\n")

    fp.write("[OsteotomyTibiaAdditionAngle]\n")
    fp.write("Data=%g,%g,%g\n" % (0, 0, 0))
    fp.write("\n")
    fp.close()


def createHurFile(src_dir, save_dir, patient_id, surgical_side):
    """

    :param save_hur_file:
    :param src_dir:
    :return:
    """
    SurgicalSideDict = {"left":"L", "right":"R"}
    uid = QUuid.createUuid().toString()
    uid = uid[2:-1]
    cur_date_time = QDateTime.currentDateTime().toString("yyyy.MM.dd.hh.mm")
    hur_file_name = cur_date_time + "_" + uid + ".hur"
    save_hur_dir = os.path.join(save_dir, patient_id+"_"+SurgicalSideDict[surgical_side])
    os.makedirs(save_hur_dir, exist_ok=True)

    save_hur_file = os.path.join(save_hur_dir, hur_file_name)
    z = zipfile.ZipFile(save_hur_file, "w")

    file_names = ["Stl.prop", "Points.prop", "Patient.prop", "Osteotomy.prop", "Mask.prop", "mask1.image",
                  "mask2.image", "bone1.stl", "bone2.stl", "dcm.image", "VirtualWall.prop"]
    for file_name in file_names:
        file_path = os.path.join(src_dir, file_name)
        z.write(file_path, file_path.replace(src_dir, ''))
    z.close()

def createSpineHurFile(src_dir, save_dir, patient_id):
    """

    :param save_hur_file:
    :param src_dir:
    :return:
    """
    uid = QUuid.createUuid().toString()
    uid = uid[2:-1]
    cur_date_time = QDateTime.currentDateTime().toString("yyyy.MM.dd.hh.mm")
    hur_file_name = cur_date_time + "_" + uid + "_spine.hur"
    #save_hur_dir = os.path.join(save_dir, patient_id+"_spine")
    os.makedirs(save_dir, exist_ok=True)

    save_hur_file = os.path.join(save_dir, hur_file_name)
    z = zipfile.ZipFile(save_hur_file, "w")

    file_names = os.listdir(src_dir)
    for file_name in file_names:
        file_path = os.path.join(src_dir, file_name)
        z.write(file_path, file_path.replace(src_dir, ''))
    z.close()

def writeNrrdFile(save_nrrd_file, img_array, origin, spacing, direction):
    """

    :param save_nrrd_file:
    :param img_array:
    :param origin:
    :param spacing:
    :return:
    """
    img = sitk.GetImageFromArray(img_array)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)
    img.SetDirection(direction)
    sitk.WriteImage(img, save_nrrd_file, True)

def getStlFileNameFromProp(stl_prop):
    """
    :param stl_prop:
    :return:
    """
    fp = open(stl_prop, "r")
    lines = fp.readlines()
    fp.close()
    new_lines = []
    for i in range(len(lines)):
        cur_line = lines[i].strip()
        if "Color" in cur_line or "ID" in cur_line:
            continue
        new_lines.append(cur_line.replace("%U5B9E%U4F53", "实体"))

    femur_stl_name = None
    tibia_stl_name = None

    for i in range(len(new_lines)):
        cur_line = new_lines[i]
        if "\\" not in cur_line:
            continue

        cur_line_list = cur_line.split("\\")
        cur_file_name = cur_line_list[0]+".stl"
        cur_bone_type = cur_line_list[1].split("=")[1]
        if cur_bone_type == "1":
            femur_stl_name = cur_file_name
        if cur_bone_type == "2":
            tibia_stl_name = cur_file_name
    return femur_stl_name, tibia_stl_name

def createSpinePlanningPropFile(mask_data, dicom_info, save_dir, start_label=None, end_label=None):
    from collections import OrderedDict
    origin, spacing = dicom_info.origin, dicom_info.spacing
    unique_values = np.unique(mask_data)
    spine_stl_info = OrderedDict()
    spine_mask_info = OrderedDict()

    tmp_array = np.zeros_like(mask_data)
    bone_count = 0
    if start_label and end_label:
        stat_index = SpineLabelDict_str2int[start_label]
        end_index = SpineLabelDict_str2int[end_label]
        for i in range(stat_index, end_index+1):
            bone_count += 1
            cur_value = i
            tmp_array[:] = 0
            cur_idx = np.where(mask_data == cur_value)
            tmp_array[cur_idx] = cur_value

            save_mask_file = os.path.join(save_dir, "mask%d.image" % bone_count)
            writeMaskImage(save_mask_file, dicom_info, tmp_array)
            spine_mask_info['mask%d' % bone_count] = {'Color': Index2Color[cur_value], "ID": bone_count,
                                                      "DisplayPostfix": bone_count}
            save_stl_file_path = os.path.join(save_dir, "bone%d.stl" % bone_count)
            cur_polydata_normal, cur_polydata = createPolyDataNormalsFromArray(tmp_array, spacing, origin)
            saveSTLFile(save_stl_file_path, cur_polydata_normal)
            spine_stl_info['bone%d' % bone_count] = {'Color': Index2Color[cur_value], "ID": bone_count, "BoneType": cur_value}
            print(save_stl_file_path, " saved done!")
    else:
        for i in range(len(unique_values)):
            cur_value = unique_values[i]
            tmp_array[:] = 0
            if cur_value > 0:
                bone_count += 1
                cur_idx = np.where(mask_data == cur_value)
                tmp_array[cur_idx] = cur_value

                save_mask_file = os.path.join(save_dir, "mask%d.image" % bone_count)
                writeMaskImage(save_mask_file, dicom_info, tmp_array)
                spine_mask_info['mask%d' % bone_count] = {'Color': Index2Color[cur_value], "ID": bone_count,
                                                          "DisplayPostfix": bone_count}

                save_stl_file_path = os.path.join(save_dir, "bone%d.stl" % bone_count)
                cur_polydata_normal, cur_polydata = createPolyDataNormalsFromArray(tmp_array, spacing, origin)
                saveSTLFile(save_stl_file_path, cur_polydata_normal)
                spine_stl_info['bone%d' % bone_count] = {'Color': Index2Color[cur_value], "ID": bone_count,
                                                         "BoneType": cur_value}
                print(save_stl_file_path, " saved done!")
    save_stl_prop_file = os.path.join(save_dir, "Stl.prop")
    writeSpineStlProp(spine_stl_info, save_stl_prop_file)

    save_mask_prop_file = os.path.join(save_dir, "Mask.prop")
    writeSpineMaskProp(spine_mask_info, save_mask_prop_file)

    save_patient_prop_file = os.path.join(save_dir, "Patient.prop")
    writeSpinePatientProp(save_patient_prop_file, dicom_info)

    save_dcm_file = os.path.join(save_dir, "dcm.image")
    writeDcmImage(save_dcm_file, dicom_info)

    print("done")





if __name__ == "__main__":
    mask_nrrd_file = "E:/data/DeepBoneData/third_patch/10010/10010knee10136601/predict/Femur.nrrd"

    #img, origin, spacing = getImageFromNrrd(mask_nrrd_file)
    img = np.array([[0, 0, 0, 1, 1, 0, 0, 1, 1,1], [0,1,1,1,1,1, 0, 0, 1, 0]])
    tmp = img[1:]
    print(tmp)
    #compressBinaryArray(img)

    uid = QUuid.createUuid().toString()
    uid = uid[2:-1]
    cur_date_time = QDateTime.currentDateTime().toString("yyyy.MM.dd.hh.mm")
    print(uid)
    print(cur_date_time)
    hur_file_name = cur_date_time+"_"+uid+".hur"
    print(hur_file_name)



# 47089
# [638, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

# 126839
# [167338317, 2, 510, 3, 508, 5, 507, 4, 509, 2, 1529, 1, 257539, 3, 508, 5, 506, 7, 505, 8]