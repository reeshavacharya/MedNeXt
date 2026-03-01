#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from nnunet_mednext.paths import nnUNet_raw_data
from batchgenerators.utilities.file_and_folder_operations import *
import os
import shutil


if __name__ == "__main__":
    # BTCV data (all organs) is assumed to be available at
    #   /data/reeshav/MedNeXt_dataset/Abdomen/RawData
    # with the following structure (standard BTCV release):
    #   RawData/Training/img/   -> training images  (imgXXXX.nii.gz)
    #   RawData/Training/label/ -> training labels  (labelXXXX.nii.gz)
    #   RawData/Testing/img/    -> test images      (imgXXXX.nii.gz)
    #
    # We point directly to this external data location so that the
    # conversion populates nnUNet_raw_data for Task017 using the
    # official multi-organ BTCV annotations.

    base = "/data/reeshav/MedNeXt_dataset/Abdomen/RawData"

    task_id = 17
    task_name = "AbdominalOrganSegmentation"
    prefix = 'ABD'

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    # Map to the BTCV folder names present under the new RawData location
    train_img_folder = join(base, "Training", "img")
    train_label_folder = join(base, "Training", "label")
    test_img_folder = join(base, "Testing", "img")
    train_patient_names = []
    test_patient_names = []

    # The official BTCV files are named like
    #   imgXXXX.nii.gz   (image)
    #   labelXXXX.nii.gz (label)
    # We iterate over all training labels, derive the corresponding
    # image filename, and generate sequential patient IDs
    # (ABD_001, ABD_002, ...).
    train_labels = subfiles(train_label_folder, join=False, suffix='nii.gz')
    for i, p in enumerate(sorted(train_labels)):
        serial_number = i + 1
        train_patient_name = f'{prefix}_{serial_number:03d}.nii.gz'
        # p should be like 'labelXXXX.nii.gz'
        stem = p[:-7]  # 'labelXXXX'
        if not stem.startswith('label'):
            raise RuntimeError(f"Unexpected training label file name: {p}")
        case_id = stem[len('label'):]
        image_file = join(train_img_folder, f"img{case_id}.nii.gz")
        label_file = join(train_label_folder, p)
        shutil.copy(image_file, join(imagestr, f'{train_patient_name[:7]}_0000.nii.gz'))
        shutil.copy(label_file, join(labelstr, train_patient_name))
        train_patient_names.append(train_patient_name)

    # Test images are in Testing/img/imgXXXX.nii.gz. We preserve
    # the numeric order derived from XXXX when constructing
    # patient IDs (ABD_XXXX).
    test_imgs = subfiles(test_img_folder, join=False, suffix=".nii.gz")
    for p in sorted(test_imgs):
        stem = p[:-7]  # 'imgXXXX'
        if not stem.startswith('img'):
            raise RuntimeError(f"Unexpected test image file name: {p}")
        case_id = stem[len('img'):]
        image_file = join(test_img_folder, p)
        serial_number = int(case_id)
        test_patient_name = f'{prefix}_{serial_number:03d}.nii.gz'
        shutil.copy(image_file, join(imagests, f'{test_patient_name[:7]}_0000.nii.gz'))
        test_patient_names.append(test_patient_name)

    json_dict = OrderedDict()
    json_dict['name'] = "AbdominalOrganSegmentation"
    json_dict['description'] = "Multi-Atlas Labeling Beyond the Cranial Vault Abdominal Organ Segmentation"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "https://www.synapse.org/#!Synapse:syn3193805/wiki/217789"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = OrderedDict({
        "00": "background",
        "01": "spleen",
        "02": "right kidney",
        "03": "left kidney",
        "04": "gallbladder",
        "05": "esophagus",
        "06": "liver",
        "07": "stomach",
        "08": "aorta",
        "09": "inferior vena cava",
        "10": "portal vein and splenic vein",
        "11": "pancreas",
        "12": "right adrenal gland",
        "13": "left adrenal gland"}
    )
    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s" % train_patient_name, "label": "./labelsTr/%s" % train_patient_name} for i, train_patient_name in enumerate(train_patient_names)]
    json_dict['test'] = ["./imagesTs/%s" % test_patient_name for test_patient_name in test_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))
