import os
import shutil
import SimpleITK as sitk

def getAllDcmFilesFromSrcDCMDir(src_dir):
    count = 0
    src_path = os.path.join(src_dir, "DICOM")
    if os.path.exists(src_path):
        dst_path = os.path.join(src_dir, "AimDICOM")
        os.makedirs(dst_path, exist_ok=True)

        for rt, sub_dirs, file_names in os.walk(src_path):
            for file_name in file_names:
                src_file_path = os.path.join(rt, file_name)
                dst_file_path = os.path.join(dst_path, "%04d.dcm " % count)
                count += 1
                shutil.move(src_file_path, dst_file_path)
        print(src_dir, " sum count ", count)
        shutil.rmtree(src_path)
    else:
        print(src_path, " doesn't exist!")

def getAllDcmFilesFromSrcDir(src_dir):
    count = 0
    all_src_files = []
    for rt, sub_dirs, file_names in os.walk(src_dir):
        for file_name in file_names:
            src_file = os.path.join(rt, file_name)
            count += 1
            all_src_files.append(src_file)
    print(src_dir, " sum count ", count)

    dst_path = os.path.join(src_dir, "AimDICOM")
    os.makedirs(dst_path, exist_ok=True)
    for i, src_file in enumerate(all_src_files):
        dst_file = os.path.join(dst_path,"%04d.dcm " % i )
        shutil.move(src_file, dst_file)

    sub_dirs = os.listdir(src_dir)
    for sub_dir in sub_dirs:
        sub_path = os.path.join(src_dir, sub_dir)
        if os.path.isfile(sub_path):
            os.remove(sub_path)
        else:
            if "AimDICOM" in sub_path:
                continue
            else:
                shutil.rmtree(sub_path)

def get_max_dcm_seires(src_dcm_dir):
    #for sub_dir in tqdm(sub_dirs):
    src_dir_path = os.path.join(src_dcm_dir, "AimDICOM/")
    dst_dir_path = os.path.join(src_dcm_dir, "DICOM")
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(src_dir_path)
    file_reader = sitk.ImageFileReader()
    aim_series = ''
    aim_series_files =[]
    max_count = -1
    assert len(series_ids)>0,'seires error, less than 1'

    if len(series_ids) >1:
        os.makedirs(dst_dir_path, exist_ok=True)
        for series in series_ids:
            series_file_names = reader.GetGDCMSeriesFileNames(src_dir_path, series)
            if max_count < len(series_file_names):
                max_count = len(series_file_names)
                aim_series_files = series_file_names
                aim_series = series
        for src_series_file in aim_series_files:
            file_dir, file_name = os.path.split(src_series_file)
            dst_series_file = os.path.join(dst_dir_path, file_name)
            shutil.copy(src_series_file, dst_series_file)
        print("aim file:%s, sum files:%d"%(src_dcm_dir, len(aim_series_files)))
        shutil.rmtree(src_dir_path)
    else:
        os.rename(src_dir_path, dst_dir_path)


if __name__=="__main__":
    from tqdm import tqdm
    src_dir = "/media/alg/data3/DeepSpineData/spine_test/"
    sub_dirs = os.listdir(src_dir)
    #sub_dirs = ["1.3.6.1.4.1.9328.50.4.0001"]
    sub_dirs = ['Test12','Test13']
    for sub_dir in tqdm(sub_dirs):
        #if "dt" in sub_dir:
        print(sub_dir)
        sub_path = os.path.join(src_dir, sub_dir)
        getAllDcmFilesFromSrcDir(sub_path)
        get_max_dcm_seires(sub_path)
        print("\n")
        #break


