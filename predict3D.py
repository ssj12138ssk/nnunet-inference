import sys
import os
import SimpleITK as sitk
import numpy as np
import shutil
import glob
import argparse
import vtk
import torch
import multiprocessing
import re
import psutil

current_dir = './data'
nnunet_raw = os.path.join(current_dir, "raw")
nnunet_preprocessed = os.path.join(current_dir, "preprocessed")
nnunet_results = os.path.join(current_dir, "results")

os.makedirs(nnunet_raw, exist_ok=True)
os.makedirs(nnunet_preprocessed, exist_ok=True)
os.makedirs(nnunet_results, exist_ok=True)

os.environ['nnUNet_raw'] = nnunet_raw
os.environ['nnUNet_preprocessed'] = nnunet_preprocessed
os.environ['nnUNet_results'] = nnunet_results

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import join

nnUNet_results = nnunet_results
nnUNet_raw = nnunet_raw
nnUNet_preprocessed = nnunet_preprocessed


'''def check_resources(min_gpu_mem_gb=2.5):
    """Check if system has sufficient resources for inference"""
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        free_mem = torch.cuda.mem_get_info(0)[0] / (1024 ** 3)
        print(f"GPU memory - Total: {total_mem:.2f}GB, Free: {free_mem:.2f}GB")

        if free_mem < min_gpu_mem_gb:
            raise RuntimeError(f"Insufficient GPU memory. Required: {min_gpu_mem_gb}GB, Available: {free_mem:.2f}GB")
    else:
        print("No GPU available - using CPU")

    cpu_cores = multiprocessing.cpu_count()
    print(f"CPU cores available: {cpu_cores}")'''


def check_resources(min_gpu_mem_gb=2.5, min_ram_gb=6):
    """检查系统是否满足推理所需的资源"""
    # 检查GPU资源
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        free_mem = torch.cuda.mem_get_info(0)[0] / (1024 ** 3)
        print(f"GPU memory - Total: {total_mem:.2f}GB, Free: {free_mem:.2f}GB")

        if free_mem < min_gpu_mem_gb:
            raise RuntimeError(f"Insufficient GPU memory. Required: {min_gpu_mem_gb}GB, Available: {free_mem:.2f}GB")
    else:
        print("No GPU available - using CPU")

    ram = psutil.virtual_memory()
    free_ram_gb = ram.available / (1024 ** 3)
    print(f"System Memory - Total: {ram.total / (1024 ** 3):.2f}GB, Available: {free_ram_gb:.2f}GB")

    if free_ram_gb < min_ram_gb:
        raise RuntimeError(
            f"Insufficient system memory. Required: {min_ram_gb}GB, Available: {free_ram_gb:.2f}GB"
        )

    cpu_cores = multiprocessing.cpu_count()
    print(f"CPU cores available: {cpu_cores}")

def verify_pelvis_file(input_folder):
    """Check that Pelvis.mhd exists in input folder"""
    pelvis_path = os.path.join(input_folder, "Pelvis.mhd")
    if not os.path.exists(pelvis_path):
        raise FileNotFoundError(f"Required file not found: Pelvis.mhd in {input_folder}")
    return pelvis_path


def convert_to_nii_gz(input_folder: str, temp_folder: str):
    """Convert Pelvis.mhd to nnUNet-compatible .nii.gz format"""
    os.makedirs(temp_folder, exist_ok=True)
    conversion_map = {}

    pelvis_mhd = os.path.join(input_folder, "Pelvis.mhd")
    pelvis_raw = os.path.join(input_folder, "Pelvis.raw")

    if not os.path.exists(pelvis_mhd):
        raise FileNotFoundError("Pelvis.mhd not found in input folder")

    try:
        img = sitk.ReadImage(pelvis_mhd)
        output_basename = "Pelvis_0000"
        output_path = os.path.join(temp_folder, f"{output_basename}.nii.gz")
        sitk.WriteImage(img, output_path)

        conversion_map[output_basename] = {
            'original': [pelvis_mhd, pelvis_raw] if os.path.exists(pelvis_raw) else [pelvis_mhd],
            'temp': output_path,
            'type': 'mhd_raw',
            'original_basename': "Pelvis"
        }

    except Exception as e:
        print(f"Conversion failed: {str(e)}")
        raise

    return conversion_map


def process_segmentation_image(img: sitk.Image) -> sitk.Image:
    """Process segmentation image - keep only label 1 and convert to uint8"""
    np_img = sitk.GetArrayFromImage(img)
    np_img = np.where(np_img == 1, 255, 0).astype(np.uint8)
    new_img = sitk.GetImageFromArray(np_img)
    new_img.CopyInformation(img)
    return new_img


def create_smooth_surface(mask_image: sitk.Image, output_filename: str):
    """Create smooth STL surface from binary segmentation"""
    try:
        np_mask = sitk.GetArrayFromImage(mask_image)

        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(mask_image.GetSize())
        vtk_image.SetSpacing(mask_image.GetSpacing())
        vtk_image.SetOrigin(mask_image.GetOrigin())
        vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        for z in range(np_mask.shape[0]):
            for y in range(np_mask.shape[1]):
                for x in range(np_mask.shape[2]):
                    vtk_image.SetScalarComponentFromFloat(x, y, z, 0, np_mask[z, y, x])

        discrete_cubes = vtk.vtkDiscreteMarchingCubes()
        discrete_cubes.SetInputData(vtk_image)
        discrete_cubes.GenerateValues(1, 255, 255)

        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(discrete_cubes.GetOutputPort())
        smoother.SetNumberOfIterations(50)
        smoother.SetPassBand(0.01)
        smoother.Update()

        selector = vtk.vtkThreshold()
        selector.SetInputConnection(smoother.GetOutputPort())
        selector.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,
                                        vtk.vtkDataSetAttributes.SCALARS)
        selector.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
        selector.SetLowerThreshold(255.0)
        selector.SetUpperThreshold(255.0)

        geometry = vtk.vtkGeometryFilter()
        geometry.SetInputConnection(selector.GetOutputPort())
        geometry.Update()

        mesh = geometry.GetOutput()
        if mesh.GetNumberOfPoints() == 0:
            print("Warning: No surface mesh generated (empty segmentation)")
            return False

        stl_writer = vtk.vtkSTLWriter()
        stl_writer.SetFileName(f"{output_filename}")
        stl_writer.SetInputData(mesh)
        stl_writer.Write()

        return True

    except Exception as e:
        print(f"STL creation failed: {str(e)}")
        return False


def update_mask_offset(original_mhd_path, mask_mhd_path):
    """Update Offset in mask MHD file to match original"""
    try:
        with open(original_mhd_path, 'r') as f:
            content = f.read()
            offset_match = re.search(r'Offset\s*=\s*(.*)\n', content)
            if not offset_match:
                print("Warning: Offset not found in original MHD")
                return

            original_offset = offset_match.group(1)

        with open(mask_mhd_path, 'r') as f:
            mask_content = f.read()

        updated_content = re.sub(
            r'Offset\s*=\s*(.*)\n',
            f'Offset = {original_offset}\n',
            mask_content
        )

        with open(mask_mhd_path, 'w') as f:
            f.write(updated_content)


    except Exception as e:
        print(f"Error updating mask offset: {str(e)}")


def convert_from_nii_gz(output_nii: str, conversion_info: dict, final_output_folder: str):
    """Convert nnUNet output back to original format and create STL"""
    os.makedirs(final_output_folder, exist_ok=True)
    original_basename = conversion_info['original_basename']

    try:
        img = sitk.ReadImage(output_nii)
        processed_img = process_segmentation_image(img)

        mask_mhd_path = os.path.join(final_output_folder, f"{original_basename}Mask.mhd")
        sitk.WriteImage(processed_img, mask_mhd_path)

        raw_output_path = os.path.join(final_output_folder, f"{original_basename}Mask.raw")
        if not os.path.exists(raw_output_path):
            open(raw_output_path, 'wb').close()

        original_mhd = conversion_info['original'][0]
        update_mask_offset(original_mhd, mask_mhd_path)

        stl_filename = os.path.join(final_output_folder, original_basename)
        create_smooth_surface(processed_img, stl_filename)

    except Exception as e:
        print(f"Post-processing failed: {str(e)}")
        raise


def run_nnunet_inference(input_folder: str, output_folder: str, dataset_id: int, cuda_device: int = 0):
    """Run nnUNet inference using Python API"""
    device = torch.device(f'cuda:{cuda_device}') if torch.cuda.is_available() and cuda_device >= 0 else torch.device(
        'cpu')

    configuration = "3d_fullres"
    trainer = "nnUNetTrainerNoMirroring"
    folds = 'all'
    checkpoint_name = "checkpoint_final.pth"

    print(f"Starting inference on device: {device}")

    try:
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_gpu=True,
            device=device,
            verbose=False
        )

        model_folder = join(nnUNet_results, f"Dataset{dataset_id:03d}_Hip", "models")
        if not os.path.exists(model_folder):
            raise FileNotFoundError(f"Model path not found: {model_folder}")

        predictor.initialize_from_trained_model_folder(
            model_folder,
            folds,
            checkpoint_name=checkpoint_name
        )

        predictor.predict_from_files(
            input_folder,
            output_folder,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0
        )

        print("Inference completed successfully")
        return True

    except Exception as e:
        print(f"Inference failed: {str(e)}")
        return False


def predict_with_3d_model(current_dir, input_folder, output_folder, dataset_id=701, cuda_device=0):
    """Run full inference pipeline"""
    temp_input = os.path.join(current_dir, "./data/temp_input")
    temp_output = os.path.join(current_dir, "./data/temp_output")

    shutil.rmtree(temp_input, ignore_errors=True)
    shutil.rmtree(temp_output, ignore_errors=True)
    os.makedirs(temp_input, exist_ok=True)

    try:
        check_resources()
        pelvis_path = verify_pelvis_file(input_folder)
        conversion_map = convert_to_nii_gz(input_folder, temp_input)

        os.makedirs(temp_output, exist_ok=True)
        if not run_nnunet_inference(temp_input, temp_output, dataset_id, cuda_device):
            return False

        for temp_basename, conv_info in conversion_map.items():
            nii_path = os.path.join(temp_output, f"Pelvis.nii.gz")
            if os.path.exists(nii_path):
                convert_from_nii_gz(nii_path, conv_info, output_folder)
            else:
                print(f"Output file not found: Pelvis.nii.gz")
                return False

        return True

    finally:
        shutil.rmtree(temp_input, ignore_errors=True)
        shutil.rmtree(temp_output, ignore_errors=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="nnUNet 3D Hip Segmentation")
    parser.add_argument("--input", required=True, help="Input directory containing Pelvis.mhd")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--dataset_id", type=int, default=701, help="Dataset ID")
    parser.add_argument("--cuda_device", type=int, default=0, help="CUDA device ID")

    try:
        args = parser.parse_args()
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
    except SystemExit:
        print("Argument parsing failed")
        sys.exit(1)

    current_dir = os.getcwd()

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(args.cuda_device)}")
    else:
        print("Using CPU")

    try:
        success = predict_with_3d_model(
            current_dir,
            input_folder=args.input,
            output_folder=args.output,
            dataset_id=args.dataset_id,
            cuda_device=args.cuda_device
        )

        if success:
            print("\nProcessing completed successfully")
            print(f"Results saved to: {args.output}")
        else:
            print("\nProcessing failed")
            sys.exit(1)

    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        sys.exit(1)



        #   python predict3D.py --input C:\Users\MedbotAI\Desktop\nnunet-infer\test\input --output C:\Users\MedbotAI\Desktop\nnunet-infer\test\testoutput
        #   predict3D.exe --input C:\Users\MedbotAI\Desktop\nnunet-infer\test\input --output C:\Users\MedbotAI\Desktop\nnunet-infer\test\testoutput