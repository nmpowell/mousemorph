#!/usr/bin/python

"""MouseMorph, an automatic mouse MR image processor

This is a Python program to run niftkN4BiasFieldCorrection from NiftK, with appropriate parameters for mouse brains. A mask is optional.

Your NIfTI file's header information needs to be present and correct: check in particular the pixel dimensions, which should be in mm and (for the defaults here) of the order 0.01--0.1mm

Example usage:
    python mm_nuc_n4.py -i "input_dir" -m "mask_dir" -n 100 -o "output_dir"

NUC Parameters:
    -n      number of iterations (default 200)
    
MouseMorph parameters:
    -i      input directory or single NIfTI image
    -o      output directory (default input dir)
    -m      mask_directory (default None)
    -fn     input file name filter: if specifying an input directory with more files than you want to process, only process files containing this string.
    -mn     mask name filter: as above (masks' file names must always also contain their corresponding image's file name, however. E.g. 'mask_image_name.nii')
    
NB: Masks work in the following way: given input file "root_name.nii.gz", mask_directory is searched for files containing "*root_name*.nii*". It is assumed that name-matches imply the mask corresponds to the input. mask_directory should be different from input directory.

"""

import os
import sys
import csv
import copy
import glob
import argparse
import subprocess
import multiprocessing
import numpy as np
import nibabel as nib
from scipy import ndimage, linalg
from numpy import array, ndarray, sum

import mm_functions as mmfn
import mm_niftk
import mm_parse_inputs

__author__ = 'Nick Powell (PhD student, CMIC & CABI, UCL, UK), nicholas.powell.11@ucl.ac.uk'
__version__ = '0.2.20150213'
__created__ = '2014-08-23, Saturday'

def nuc_n4_debias(nk, sargs, nf_path, iterations=200, fwhm=0.15, subsampleFactor=4, nlevels=4, conv=0.001, nhistbins=256):
    """Runs niftkN4BiasFieldCorrection on NIfTI files.
    
    Returns the path of the corrected output.
    """
    
    # Get the input file and name, removing path and 1+ extensions
    nf_path = os.path.normpath(os.path.join(nf_path))
    nf_name = (os.path.basename(nf_path)).split(os.extsep)[0]
    
    temp_dir = mmfn.create_temp_directories(sargs.output_directory)
    
    outputBiasImage = os.path.normpath(os.path.join(temp_dir, nf_name + '_biasfield' + sargs.ext))
    outputSubSampledMask = os.path.normpath(os.path.join(temp_dir, nf_name + '_ss_mask' + sargs.ext))
    outputMask = os.path.normpath(os.path.join(temp_dir, nf_name + '_outmask' + sargs.ext))
    outputSubSampledImage = os.path.normpath(os.path.join(temp_dir, nf_name + '_ss_img' + sargs.ext))
    
    output_path = os.path.normpath(os.path.join(sargs.output_directory, sargs.out_name_prepend + nf_name + sargs.out_name_append + sargs.ext))
    
    # Run non-uniformity correction ...
    
    n4_calling_list = [nk.niftkN4BiasFieldCorrection,
                                '--echo',
                                '-i', nf_path, 
                                '-o', output_path,
                                '--outBiasField', outputBiasImage,
                                '--nlevels', str(nlevels),
                                '--sub', str(subsampleFactor),
                                '--niters', str(iterations),
                                '--FWHM', str(fwhm),
                                '--nbins', str(nhistbins),
                                '--convergence', str(conv),
                                '--outSubsampledMask', outputSubSampledMask,
                                '--outMask', outputMask,
                                '--outSubsampledImage', outputSubSampledImage]
    if sargs.mask_directory is not None:
        mask_path = mmfn.get_corresponding_file(sargs.mask_directory, nf_name, sargs.mask_name_filter, path_only=True)
        n4_calling_list.extend(['--inMask', mask_path])
        print("  Using mask: {0} ...".format(mask_path))
        
        if mask_path is None:
            return mask_path
    
    # Check for output
    if not os.path.isfile(output_path):
        # subprocess.call(n4_calling_list)
        mmfn.subprocess_call_and_log(n4_calling_list, temp_dir)
    else:
        print("  Output file, {0} already exists!".format(output_path))
    
    
    return output_path
    
def nuc_n4_debias_star(args):
    return nuc_n4_debias(*args)
    
def main():
    
    # Parse input arguments and get sanitised version
    sargs = mm_parse_inputs.SanitisedArgs(mm_parse_inputs.parse_input_arguments())
    
    # Enable calling NiftyReg and NiftySeg
    nk = mm_niftk.MM_Niftk()
    
    mmfn.check_create_directories([sargs.output_directory])
    
    # Set default parameters
    if sargs.number is not None:
        iterations = sargs.number
    else:
        iterations = 200
        
    fwhm = 0.15			    # default 0.15
    subsampleFactor = 4	    # default 4; recommend 2 for in vivo mouse brains with resolution >100 micrometres.
    nlevels = 4             # default 4
    conv = 0.001			# default 0.001
    nhistbins = 256		    # default 200

    # Get list of files (use file_name_filter)
    nifti_files_list = sorted(glob.glob(os.path.join(sargs.input_directory, sargs.file_name_filter + '.nii*')))
    print("  Processing {0} files: \n   - - -\n   \t{1}\n   - - -\n  ...".format(len(nifti_files_list), '\n\t'.join([ str(item) for item in nifti_files_list ])))
    
    TASKS = [(nk, sargs, nf_path, iterations, fwhm, subsampleFactor, nlevels, conv, nhistbins) for nf_path in nifti_files_list]
    # Set the number of parallel processes to use
    pool = multiprocessing.Pool(np.int(multiprocessing.cpu_count() / 2))
    # The _star function will unpack TASKS to use in the actual function
    # Using pool.map because we do care about the order of the results.
    all_output_paths = pool.map(nuc_n4_debias_star, TASKS)
    pool.close()
    pool.join()
    
    all_output_paths = []
    
    for counter, nf_path in enumerate(nifti_files_list):
        outpath = nuc_n4_debias(nk, sargs, nf_path, iterations, fwhm, subsampleFactor, nlevels, conv, nhistbins)
        all_output_paths.append(outpath)
    
    print(" all_output_paths: {0}".format(all_output_paths))
    
    print("  Copying headers from original images to bias-corrected images ...")
    
    for counter, nifti_file in enumerate(nifti_files_list):
        print "  Processing {0} / {1}: {2} ...".format((counter + 1), len(nifti_files_list), nifti_file)
        
        original_nf_name = os.path.basename(nifti_file).split(os.extsep)[0]
        original_nifti = nib.load(nifti_file)
        bias_corrected_path = mmfn.get_corresponding_file(sargs.output_directory, original_nf_name, path_only=True)
        
        print("Bias corrected result path: '{0}'".format(bias_corrected_path))
        
        updated_nii = mmfn.copy_header(nib.load(bias_corrected_path), original_nifti)
        # This will overwrite the bias corrected files
        nib.save(updated_nii, bias_corrected_path)
    
    print("  Bias field correction completed; files saved to: \n{0}".format('\n\t'.join([ str(item) for item in all_output_paths ])))
    
def go(mm):
    
    args = mm.args
    
    # Enable calling NiftyReg and NiftySeg
    nk = mm_niftk.MM_Niftk()
    
    # Get list of files
    input_files_list = sorted(glob.glob(os.path.join(args.input_directory, args.input_name_filter + '.nii*')))
    print("  Processing {0} files: \n   - - -\n   \t{1}\n   - - -\n  ...".format(len(input_files_list), '\n\t'.join([ str(item) for item in input_files_list ])))
    
    TASKS = [(nk, args, f_path, args.iterations, args.fwhm, args.subsample, args.nlevels, args.convergence, args.nhistbins) for f_path in args.input_files_list]
    
    if args.parallel:
    
        # all_output_paths = []
        # print("[nuc_n4_debias] + TASKS: {0}".format([nuc_n4_debias] + TASKS))
        # for task in TASKS:
            # all_output_paths.append(nuc_n4_debias(*task))
            # mmfn.function_star([nuc_n4_debias] + task)
        
        import multiprocessing
        
        # Set the number of parallel processes to use
        pool = multiprocessing.Pool(np.int(multiprocessing.cpu_count()))
        # Prepend the function name to the list, so function_star() works
        all_output_paths = pool.map(mmfn.function_star, [nuc_n4_debias] + TASKS)
        pool.close()
        pool.join()
        
    else:
    
        all_output_paths = []
        for task in TASKS:
            all_output_paths.append(nuc_n4_debias(*task))
        
        print(" All output images: \n{0}".format(all_output_paths))
        
        print("  Copying headers from original images to bias-corrected images ...")
    
    for counter, nifti_file in enumerate(input_files_list):
        print "  Processing {0} / {1}: {2} ...".format((counter + 1), len(input_files_list), nifti_file)
        
        original_nf_name = os.path.basename(nifti_file).split(os.extsep)[0]
        original_nifti = nib.load(nifti_file)
        bias_corrected_path = mmfn.get_corresponding_file(args.output_directory, original_nf_name, path_only=True)
        
        updated_nii = mmfn.copy_header(nib.load(bias_corrected_path), original_nifti)
        # This will overwrite the bias corrected files
        nib.save(updated_nii, bias_corrected_path)
    
    print("  Bias field correction completed; files saved to: \n{0}".format('\n\t'.join([ str(item) for item in all_output_paths ])))
        
        
if __name__ == '__main__':
    time_start = mmfn.notify_script_start(__file__)
    main()
    mmfn.notify_script_complete(__file__, time_start=time_start)
    
# End