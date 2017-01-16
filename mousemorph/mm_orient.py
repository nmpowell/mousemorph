#!/usr/bin/python

"""MouseMorph, an automatic mouse MR image processor

This is a Python program to orient 3D objects (MR images of mouse brains) to a standard space.

Standard space, for example Right Anterior Superior (RAS), can be either the same as a set of external atlas images (user provides the directory) or based upon the data itself (user manually chooses N correctly-oriented candidates midway through the script). The correct orientation is chosen for each data image via comparison with the atlas images/correctly-oriented subset (currently just one is fine).

This script outputs a "niftifilename_orienting_affine.txt" file, which may be used to orient files which correspond to the data (such as masks or phase images), with mm_orient_corresponding_files.py. This may also be used to create resampled, correctly-oriented files for, e.g., video flythroughs or other presentations.

The final images can be aligned in relation to the image origin. The origin can be placed at the image corner, at the geometric centre, at the object centroid, or an arbitrary position. To adjust this, see "target_vox =" around line 995.

Example usage
-------------
python mm_orient.py -i "input_directory_or_file" [-o "output_directory" -at "atlas_directory"]

Notes
-----
- Resampling option is available, but the solution here is hacky. We lack knowledge of the shape of the final result, so a larger-than-necessary 'reference' image is created for reg_resample, then cropped. The origin of the new image is set to its centre (instead of, e.g., the centre of mass (CoM)). Resampling will inevitably change the centre of mass anyway, so this shouldn't be a problem.

# Optionally orient any files which correspond to the data, using mm_orient_corresponding_files.py

"""

import os
import sys
import csv
import copy
import glob
import time
import argparse
import subprocess
import numpy as np
import nibabel as nib
from scipy import ndimage, linalg, stats
from numpy import array, ndarray, sum
from itertools import chain
import multiprocessing

import mm_niftk
import mm_functions as mmfn
import mm_parse_inputs

__author__ = 'Nick Powell (PhD student, CMIC & CABI, UCL, UK), nicholas.powell.11@ucl.ac.uk'
__version__ = '0.3.20150214'
__created__ = '2014-06-19, Thursday'

def testing_ods():
    # Incomplete.
    
    # If the header already contains an orientation matrix, you can either completely replace it with the output from mm_orient (default behaviour), take it into account so only a "fine adjustment" is made by mm_orient, or proceed as if it isn't there (which could produce unexpected, disastrous results).
    
    if mmfn.check_zero_off_diagonals(matrix, ignore_final_column=False) is False:
        print "do something"
        # off-diagonal elements of the matrix (ignoring translation elements) are non-zero - this means an orientation already exists in the header.
        
def centre_of_mass( data, weighted=True ):
    """Returns the voxel centre of mass of an ND Numpy array.
    
    If weighted is True (default), each voxel is weighted by its value. Otherwise, each voxel is weighted equally (as if binary).
    """
    
    # If we have a 4D image, only use the first timepoint.
    if np.size(data.shape) > 3:
        data = data[:,:,:,0]
    
    # indices of non-zero elements
    rows, cols, slices = np.ndarray.nonzero(data)
    rows, cols, slices = rows + 1, cols + 1, slices + 1
    
    # Unweighted CoM
    com = np.array([np.mean(rows), np.mean(cols), np.mean(slices)])
    
    if weighted:
        total_sum = np.ndarray.sum(data)
        
        # linear list of actual elements (must be same length as each of rows, cols, slices)
        flat_data = np.ndarray.ravel(data)
        flat_data = flat_data[flat_data != 0]
        
        # Weight CoM by voxel values
        com[0] = sum(rows * flat_data) / total_sum
        com[1] = sum(cols * flat_data) / total_sum
        com[2] = sum(slices * flat_data) / total_sum
    
    return com
    
def generate_inertia_matrix( nifti, centre_of_mass ):
    """Generate and return inertia tensor of a 3D NIfTI image."""
    
    data, hdr = nifti.get_data(), nifti.get_header()
    
    pixdims = hdr.structarr['pixdim'][1:4]
    
    # If we have a 4D image, only use the first timepoint.
    if np.size(data.shape) > 3:
        data = data[:,:,:,0]
    
    # indices of non-zero elements
    rows, cols, slices = np.ndarray.nonzero(data)
    
    # Add 1 to avoid beginning at 0.
    rows = rows + 1
    cols = cols + 1
    slices = slices + 1
    
    # linear list of actual elements (must be same length as each of rows, cols, slices)
    flat_data = np.ndarray.ravel(data)
    flat_data = flat_data[flat_data != 0]
    
    # translation to centroid (in mm)
    x_dash = (rows - centre_of_mass[0]) * pixdims[0]
    y_dash = (cols - centre_of_mass[1]) * pixdims[1]
    z_dash = (slices - centre_of_mass[2]) * pixdims[2]
    
    # Moments of inertia with weights (** == .^)
    Ixx = sum( flat_data * ((y_dash**2) + (z_dash**2)));
    Iyy = sum( flat_data * ((z_dash**2) + (x_dash**2)));
    Izz = sum( flat_data * ((x_dash**2) + (y_dash**2)));

    # Products of inertia with weights
    Ixy = sum( flat_data * x_dash * y_dash);
    Ixz = sum( flat_data * x_dash * z_dash);
    Iyz = sum( flat_data * y_dash * z_dash);

    I = np.array([[Ixx, -Ixy, -Ixz],
                    [-Ixy, Iyy, -Iyz],
                    [-Ixz, -Iyz, Izz]])
                    
    print("  determinant of I is {0}".format(np.linalg.det(I)))
    
    return I
    
def eig_vv_sorted( matrix ):
    """Calculates eigenvalues and eigenvectors; returns them sorted in a specific order.
    
    Quite a lot of effort is spent avoiding reflection (determinant -1).
    
    [1]: http://stackoverflow.com/questions/8092920/sort-eigenvalues-and-associated-eigenvectors-after-using-numpy-linalg-eig-in-pyt
    
    """
    
    # Get the eigenvalues and eigenvectors
    eigen_values, eigen_vectors = linalg.eig(matrix)
    
    print("  original eigen_vectors: {0}".format(eigen_vectors))
    print("  determinant of original eigen_vectors is {0}".format(np.linalg.det(eigen_vectors)))
    
    # Sort both in order of ascending eigenvalues
    idx = eigen_values.argsort()
    eigen_values_sorted = eigen_values[idx]
    eigen_vectors_sorted = eigen_vectors[:,idx]
    
    # sort in order *of the principal axes*, which are inverse of the eigenvalue magnitudes!
    # The order here determines:
    # 1) alignment with X,Y,Z cartesian axes in the image
    # 2) whether the determinant of eigen_vectors is +1 or -1.
    # The determinant must be +1; otherwise we have reflection (undesirable). As we are only testing rolls around the Y axis later, we also wish "A" in RAS (that is, AP) to be aligned with Y. The order is thus either [1,0,2] or [2,0,1], and via testing, it appears [2,0,1] is better for +ve determinant. Update: sometimes these are switched, so must test.
    # * Keep reporting, and checking determinants +/-.
    sort_1 = [1,0,2]
    sort_2 = [2,0,1]
    
    eigen_values_1 = eigen_values_sorted[sort_1]
    eigen_vectors_1 = eigen_vectors_sorted[:,sort_1]
    
    eigen_values_2 = eigen_values_sorted[sort_2]
    eigen_vectors_2 = eigen_vectors_sorted[:,sort_2]
    
    det_ev_1 = np.linalg.det(eigen_vectors_1)
    det_ev_2 = np.linalg.det(eigen_vectors_2)
    print("  determinants of two possible eigen-vector matrices are [1,0,2]: {0} and [2,0,1]: {1}.".format(det_ev_1, det_ev_2))
    
    if det_ev_1 > 0:
        eigen_values_chosen = eigen_values_sorted[sort_1]
        eigen_vectors_chosen = eigen_vectors_sorted[:,sort_1]
    elif det_ev_2 > 0:
        eigen_values_chosen = eigen_values_sorted[sort_2]
        eigen_vectors_chosen = eigen_vectors_sorted[:,sort_2]
    else:
        print("** Warning: could not find correct arrangement of eigen-vectors to give positive determinant. **")
    
    print("  eigen-values: {0}".format(eigen_values_chosen))
    print("  eigen-vectors:\n  {0}".format(eigen_vectors_chosen))
    print("  determinant of chosen eigen-vector matrix is {0}".format(np.linalg.det(eigen_vectors_chosen)))
    
    return eigen_values_chosen, eigen_vectors_chosen
    
def fill_homogeneous_matrix( matrix ):
    """Create a 3D homogeneous affine transformation matrix from a 3x3 matrix input."""
    
    homm = np.zeros(shape=(4,4))
    homm[3,3] = 1
    homm[0:3,0:3] = matrix
    return homm
    
def generate_translation_to_origin( pixdims, existing_affine, coordinates ):
    """Return the 4x4 matrix which translates the voxel coordinates to the origin.
    
    NB: working in voxels currently. Would probs be better, and more consistent, to move to mm.
    
    """
    
    # Get sform from header and convert the translation elements into voxels
    existing_sform_translation_vox = existing_affine[:3,3] / pixdims
    
    print("pixdims: {0}\nexisting_affine: {1}\ncoordinates: {2}\nexisting_sform_translation_vox: {3}\n".format(pixdims, existing_affine, coordinates, existing_sform_translation_vox))

    # NB this addition may fail if image dimensions don't agree, e.g. if coordinates is for some reason 4D rather than 3D...
    # existing_sform_translation_vox will be +ve (?)
    translation_voxels = coordinates + existing_sform_translation_vox
    
    t_diag = mmfn.create_affine(trans=translation_voxels)
    
    t = right_matrix_division( np.dot(existing_affine, linalg.inv(t_diag)), existing_affine )
    return t
    
def translate_to_apply_transformation( translation_matrix, transformation_matrix ):
    """Returns the homogeneous matrix result of translating to a point, applying another homogeneous (affine) transformation matrix, such as a rotation, and translating back to the original location."""
    # M = inv(T) * eVectsHom * T
    
    m = (np.linalg.inv(translation_matrix)).dot(transformation_matrix).dot(translation_matrix)
    # m = np.dot(np.dot( linalg.inv(translation_matrix), transformation_matrix ), translation_matrix)   # equivalent
    
    return m
    
def create_rotation_matrix( alpha, beta, gamma ):
    """Returns a 3x3 rotation matrix.
    
    See Arata et al. 1995 [1].
    
    For a 3D homogeneous result use fill_homogeneous_matrix(create_rotation_matrix(alpha, beta, gamma))
    
    [1] Arata, L. K., Dhawan, a P., Broderick, J. P., Gaskil-Shipley, M. F., Levy, a V, & Volkow, N. D. (1995). Three-dimensional anatomical model-based segmentation of MR brain images through Principal Axes Registration. IEEE transactions on bio-medical engineering, 42(11), 1069-78. doi:10.1109/10.469373
    """
    
    # Be consistent across naming schemes
    theta = alpha
    psi = beta
    phi = gamma
    
    # Rotate (roll) about x
    r_x = np.array([ [1, 0, 0],
                    [0, np.cos(theta), np.sin(theta)],
                    [0, -np.sin(theta), np.cos(theta)] ])
    
    # Rotate (roll) about y    
    r_y = np.array([ [np.cos(psi), 0, -np.sin(psi)],
                    [0, 1, 0],
                    [np.sin(psi), 0, np.cos(psi)] ])
    
    # Rotate (roll) about z in x-y plane, anticlockwise from x into y
    r_z = np.array([ [np.cos(phi), np.sin(phi), 0],
                    [-np.sin(phi), np.cos(phi), 0],
                    [0, 0, 1] ])
    
    rotation_matrix = r_z.dot(r_y).dot(r_x)
    
    print("  determinant of rotation_matrix is {0}".format(np.linalg.det(rotation_matrix)))
    
    return rotation_matrix
    # End of create_rotation_matrix() definition
    
def register_and_compare( reg_aladin_bin_path, reference_img_path, floating_img_path, resampled_output_path=None, output_affine_path=None, pc_ignore=0.0, ln=3, lp=3, rigid_only=False, reg_resample_path=None ):
    """Register floating_img to reference_img with NiftyReg's reg_aladin.
    
    Compare the resampled result with the reference_img using Pearson's product-moment correlation coefficient (r).
    Image histogram tails may be trimmed before comparison.
    * Assumes the reference image is correctly resampled into its own space *???
    
    """
    
    # Default save output to floating input directory
    if resampled_output_path is None:
        output_path = os.path.dirname(floating_img_path)
        output_name = os.path.basename(floating_img_path.split(os.extsep)[0])
        resampled_output_path = os.path.normpath(os.path.join(output_path, output_name + '_to_reference_res' + sargs.ext))
    else:
        output_path = os.path.dirname(resampled_output_path)
        output_name = os.path.basename(resampled_output_path.split(os.extsep)[0])
        
    if output_affine_path is None:
        output_affine_path = os.path.normpath(os.path.join(output_path, output_name + '_to_reference_affine.txt'))
    
    reg_aladin_calling_list = [reg_aladin_bin_path,
                                    '-ref', reference_img_path,
                                    '-flo', floating_img_path,
                                    '-aff', output_affine_path,
                                    '-res', resampled_output_path,
                                    '-ln', str(ln), '-lp', str(lp)]
    if rigid_only is True:
        reg_aladin_calling_list.append('-rigOnly')
        
    # Check for output
    if not os.path.isfile(output_affine_path):
        subprocess.call(reg_aladin_calling_list)
        
        # If it didn't work the first time, adjust parameters and try again ...
        if not os.path.isfile(output_affine_path):
            ln = ln + 1
            lp = lp + 1
            reg_aladin_calling_list = [reg_aladin_bin_path,
                                    '-ref', reference_img_path,
                                    '-flo', floating_img_path,
                                    '-aff', output_affine_path,
                                    '-res', resampled_output_path,
                                    '-ln', str(ln), '-lp', str(lp)]                                    
            reg_aladin_calling_list.append('-rigOnly')
            
            print("  Initial registration attempt failed; trying again with parameters: {0}".format(reg_aladin_calling_list))
            subprocess.call(reg_aladin_calling_list)
        
    else:
        print("  Output file, {0} already exists!".format(output_affine_path))
        
        # So we don't have to register. But do we still have to resample?
        if reg_resample_path is not None:
        
            # This is only run if reg_aladin isn't run, and if the path has been provided (suggesting that the resampled output is required)
            reg_resample_calling_list = [reg_resample_path,
                                            '-ref', reference_img_path,
                                            '-flo', floating_img_path,
                                            '-aff', output_affine_path,
                                            '-res', resampled_output_path]
        
            # Check for output
            if not os.path.isfile(resampled_output_path):
                subprocess.call(reg_resample_calling_list)
            else:
                print("  Output file, {0} already exists!".format(resampled_output_path))
    
    # Calculate the correlation coefficient of the two images.
    data_reference_image = (nib.load(reference_img_path)).get_data()
    data_resampled_output = (nib.load(resampled_output_path)).get_data()
    
    # Ignore the images' intensity extremes:
    if pc_ignore > 0:
        pc_maxmin = [pc_ignore, 100-pc_ignore]
        
        pc_intensities = np.percentile(data_reference_image, pc_maxmin)
        data_reference_image[data_reference_image < pc_intensities[0]] = 0
        data_reference_image[data_reference_image > pc_intensities[1]] = pc_intensities[1]
        
        pc_intensities = np.percentile(data_resampled_output, pc_maxmin)
        data_resampled_output[data_resampled_output < pc_intensities[0]] = 0
        data_resampled_output[data_resampled_output > pc_intensities[1]] = pc_intensities[1]
    
    pearson_r = stats.pearsonr(ndarray.ravel(data_reference_image), ndarray.ravel(data_resampled_output))[0]

    return pearson_r
    # End of register_and_compare() definition
    
def correct_slight_orientation_offset(nk, nf_path, final_output_path, res_output_affine_path, half_aff_path, reflection_output_filepath, reflection_affine_path, resampled_output_path, ln=3, lp=3, rigid_only=True ):
    """Correct a slight orientation offset from the Y axis, by reflecting.
    
    The input image is reflected, registered to its reflection, and the halfway affine transformation from that registration computed. This half-affine is then applied to the input image to update its header.
    
    Reflection currently applied in the YZ plane (i.e., along the X axis). When a brain is in RAS orientation, this is in the AP/IS plane, i.e. the mid-sagittal plane.
    
    """
    
    # If you have the affine, skip reflection and registration
    if not os.path.isfile(res_output_affine_path):
        
        # Apply reflection
        
        # Check for output
        if not os.path.isfile(reflection_output_filepath):
            subprocess.call([nk.reg_transform,
                                '-ref', nf_path,
                                '-updSform', nf_path, reflection_affine_path, reflection_output_filepath])
        else:
            print("  Output file, {0} already exists!".format(reflection_output_filepath))
            
        # Register original file to reflection
        
        reg_aladin_calling_list = [nk.reg_aladin,
                                    '-ref', reflection_output_filepath, 
                                    '-flo', nf_path,
                                    '-aff', res_output_affine_path,
                                    '-res', resampled_output_path,
                                    '-ln', str(ln), '-lp', str(lp)]
        if rigid_only is True:
            reg_aladin_calling_list.append('-rigOnly')
        
        # Check for output
        if not os.path.isfile(res_output_affine_path):
            subprocess.call(reg_aladin_calling_list)
        else:
            print("  Output file, {0} already exists!".format(res_output_affine_path))
    
    # Calculate half-way affine
    
    reg_aff = np.loadtxt(res_output_affine_path)

    # Calculate same (get the real part, otherwise the header causes the image to refuse to load in NiftyView)
    half_logm = linalg.logm(reg_aff) / 2.0
    half_aff_calc = np.real(linalg.expm(half_logm))
    half_aff = mmfn.create_affine(full_affine=half_aff_calc, output_path=half_aff_path)
    
    print("  Half the registration affine (calculated):\n {0}".format(half_aff_calc))
    
    # Apply halfway affine to original file
    
    if not os.path.isfile(final_output_path):
        subprocess.call([nk.reg_transform,
                            '-ref', nf_path,
                            '-updSform', nf_path, half_aff_path, final_output_path])
    else:
        print("  Output file, {0} already exists!".format(final_output_path))

    return final_output_path
    # End of correct_slight_orientation_offset() definition
    
def register_and_align(nk, reference_img_path, floating_img_path, resampled_output_path, res_output_affine_path, final_output_path, translation_affine_path=None, pc_ignore=0.0, ln=3, lp=3, rigid_only=True ):
    """Register a floating image to a reference, updating the header of the original.
    
    Can be used to specify a common coordinate space if the same reference image is used for multiple floating images.
    
    If translation_affine_path is not provided, the full affine from reg_aladin will be used for final alignment. This will include rotation (and possibly other affine DoF, if rigid_only=False). Thus, to just align images' centre of mass, it's better to provide translation_affine_path.
    
    """
    
    # Register original file to reflection
    reg_aladin_calling_list = [nk.reg_aladin,
                                '-ref', reference_img_path, 
                                '-flo', floating_img_path,
                                '-aff', res_output_affine_path,
                                '-res', resampled_output_path,
                                '-ln', str(ln), '-lp', str(lp)]
    if rigid_only is True:
        reg_aladin_calling_list.append('-rigOnly')
    
    # Check for output
    if not os.path.isfile(res_output_affine_path):
        subprocess.call(reg_aladin_calling_list)
    else:
        print("  Output file, {0} already exists!".format(res_output_affine_path))
        
    # If a path for a translation-only affine is provided, create it, and use it.
    if translation_affine_path is not None:
        # Get the translation elements only
        translation_elements = np.loadtxt(res_output_affine_path)[:3,3]
        translation_affine = mmfn.create_affine(trans=translation_elements, output_path=translation_affine_path)
        print("  translation affine: {0}".format(translation_affine))
    else:
        translation_affine_path = res_output_affine_path
    
    # Update header of the original image using the affine output
    if not os.path.isfile(final_output_path):
        subprocess.call([nk.reg_transform,
                            '-ref', floating_img_path,
                            '-updSform', floating_img_path, translation_affine_path, final_output_path])
    else:
        print("  Output file, {0} already exists!".format(final_output_path))

    return final_output_path
    # End of register_and_align() definition
    
def roll_and_compare_star(args):
    return roll_and_compare(*args)
    
def roll_and_compare(r, nk, sargs, temp_dir, nf_name, pa_output_filepath, reflection_affine_path):

    generic_roll_affine_path = os.path.normpath(os.path.join(temp_dir, 'roll_affine_' + str(r) + '.txt'))

    # Check for output
    roll_output_filepath = os.path.normpath(os.path.join(temp_dir, nf_name + '_pa_roll_' + str(r) + sargs.ext))
    if not os.path.isfile(roll_output_filepath):
        subprocess.call([nk.reg_transform,
                            '-ref', pa_output_filepath,
                            '-updSform', pa_output_filepath, generic_roll_affine_path, roll_output_filepath])
    else:
        print("  Output file, {0} already exists!".format(roll_output_filepath))
        
    # Check for output
    reflection_output_filepath = os.path.normpath(os.path.join(temp_dir, nf_name + '_pa_roll_' + str(r) + '_refl_rt' + sargs.ext))
    if not os.path.isfile(reflection_output_filepath):
        subprocess.call([nk.reg_transform,
                            '-ref', roll_output_filepath,
                            '-updSform', roll_output_filepath, reflection_affine_path, reflection_output_filepath])
    else:
        print("  Output file, {0} already exists!".format(reflection_output_filepath))
    
    # Apply reflection in YZ plane (along X axis) (resampling unnecessary, it turns out (?))
    # Looks like this isn't actually necessary for calculating the CC accurately(?)
    # NB didn't think I could use register_and_compare() here as the target image is not the resampled version - but it seems to work!!
    
    # Register roll to reflection
    reg_roll_to_reflection = os.path.normpath(os.path.join(temp_dir, nf_name + '_pa_roll_' + str(r) + '_to_refl_res' + sargs.ext))
    reg_roll_to_reflection_output_affine = os.path.normpath(os.path.join(temp_dir, nf_name + '_pa_roll_' + str(r) + '_to_refl_affine.txt'))
    
    # Calculate the correlation coefficient of both resampled files
    # Using function and non-resampled reflection
    pearson_r = register_and_compare(nk.reg_aladin, reflection_output_filepath, roll_output_filepath, resampled_output_path=reg_roll_to_reflection, output_affine_path=reg_roll_to_reflection_output_affine, ln=2, lp=2, rigid_only=True )
    
    # Delete almost everything
    mmfn.delete_files([roll_output_filepath, reflection_output_filepath, reg_roll_to_reflection])

    return pearson_r
    
def reg_rotations_and_compare_star(args):
    return reg_rotations_and_compare(*args)
    
def reg_rotations_and_compare(c, temp_dir, nf_name, sargs, reg_aladin_path, atlas_nf_path, ln=2, lp=2):

    # Test each of the candidates
    candidate_image_path = os.path.normpath(os.path.join(temp_dir, nf_name + '_corrected_' + str(c) + sargs.ext))
    registered_candidate_to_atlas = os.path.normpath(os.path.join(temp_dir, nf_name + '_corrected_' + str(c) + '_to_atlas_res' + sargs.ext))
    
    pearson_r = register_and_compare(reg_aladin_path, atlas_nf_path, candidate_image_path, resampled_output_path=registered_candidate_to_atlas, ln=2, lp=2 )
    
    return pearson_r
    
def right_matrix_division( top, bottom ):
    """Returns the matrix result X of when AX = B
    
    AX=B, so X=B/A. This is right division, requiring a least-squares approach, see references.
    
    This returns the same result X as if you had done X=B/A in Matlab.
    In Matlab this is just '/'; in Python it must be done this way.
    see: http://stackoverflow.com/questions/1001634/array-division-translating-from-matlab-to-python
    see: http://stackoverflow.com/a/1008869
    
    If you have image_a and image_b, the affine matrix which gives image_b when applied (using reg_transform -updSform) to image_a, is:
    m = linalg.inv( right_matrix_division( image_b_sform, image_a_sform ) )
    (see difference_affine())
    """
    
    # Must transpose the result
    result = linalg.lstsq(bottom.T, top.T)[0].T
    print("  determinant of right matrix division result is {0}".format(np.linalg.det(result)))
    return result
    # End of right_matrix_division() definition
    
def difference_affine( original_sform, final_sform ):
    """Given original and final sform matrices, find the affine matrix which, when applied to the original image with reg_transform, will return the final image. This works."""
    return linalg.inv( right_matrix_division( final_sform, original_sform ) )
    
def predict_final_sform( original_sform, transform_affine ):
    """Given the original sform in the header of a NIfTI image, predict the final sform after application of an affine matrix.
    
    As if you'd run reg_transform -ref original.nii -updSform original.nii transform_affine.txt final.nii
    
    You could potentially use this to create a final image directly, by inserting the final affine into its header:
        f = predict_final_sform(original_sform, transform_affine)
        hdr = old_nii.get_header()
        hdr.set_sform( f )
        new_nii = nib.Nifti1Image(old_nii.get_data(), affine=f, header=hdr)
    But the disadvantage of that is that you lose the handy affine.txt file of transform_affine.
    """
    
    predicted_final_sform = linalg.inv(transform_affine).dot(original_sform)
    print("  Predicted final sform: \n{0}".format(predicted_final_sform))
    return predicted_final_sform
    
def rotate_translate(pixdims, original_sform, single_orienting_affine, target_vox):
    """Returns the single affine which will apply any affine transformation, and then translate the image so that target_vox coordinates align with the origin.
    
    If using this function to resample (see mm_orient_corresponding_files.py), the original_sform is from the original image, and target_vox should be the centre voxel-coordinates of the original image.
    """
    
    # Get the translation elements from the existing orienting affine
    existing_trans_to_origin = -1 * single_orienting_affine[:3,3]
    trans_rotated_img_to_origin = mmfn.create_affine(trans=existing_trans_to_origin)
    
    # The affine to translate the corner of the oriented image to the origin
    corner_translation_affine = mmfn.create_affine(full_affine=trans_rotated_img_to_origin.dot(single_orienting_affine))
    
    corner_aligned_final_sform = predict_final_sform(original_sform, corner_translation_affine)
    
    # Calculate translation of centre of image to origin (in one go!)
    trans_to_origin_direct = np.linalg.inv( generate_translation_to_origin( pixdims, corner_aligned_final_sform, target_vox ) )
    final_single_orienting_affine = corner_translation_affine.dot(trans_to_origin_direct)
    
    return final_single_orienting_affine
    
def main():

    global sargs, nk
    
    # function decorator to process an entire directory, run a function, and save result??? (Multiple functions in sequence?)

    # 1 Downsample (save)
    # 2 get principal axis orientations (4), and rotations (14), and reflections (14)
    # 3 register reflections to originals (save)
    # 4 test CC & therefore best rotation
    # 5 get half the affine of the registration
    # 6 apply that half affine to all 4 pa + rotation possibilities
    # 7 register to template
    # 8 get CC & therefore best orientation
    # 9 combine affines & apply to original image (NB downsampling)
    
    # Parse input arguments and get sanitised version
    sargs = mm_parse_inputs.SanitisedArgs(mm_parse_inputs.parse_input_arguments())
    
    # Enable calling NiftyReg and NiftySeg, global so sub-functions can use it
    nk = mm_niftk.MM_Niftk()
    
    # Are all the input images in the same orientation, initially?
    all_same_orientation = sargs.true_false
    
    # Ignore the image's intensity extremes when calculating the principal axes.
    pc_ignore = 5.0
    
    # Flip this manual flag to true to turn on testing _all_ possible perpendicular orientations, initially. Use this when LR or IS are > AP.
    #############################
    # Not yet implemented or used
    # test_all_orientations = 0
    # if test_all_orientations is 1:
        # Create 90 degree rotation affines about the X and Z axes
        # rotation = fill_homogeneous_matrix(create_rotation_matrix( np.pi/2, 0, 0 ))
        # rotation_affine_path_x90 = os.path.normpath(os.path.join(temp_dir, 'rotation_affine_X90'+ '.txt'))
        # rotation_affine = mmfn.create_affine(full_affine=rotation, output_path=rotation_affine_path_x90)
        
        # rotation = fill_homogeneous_matrix(create_rotation_matrix( 0, 0, np.pi/2 ))
        # rotation_affine_path_z90 = os.path.normpath(os.path.join(temp_dir, 'rotation_affine_Z90'+ '.txt'))
        # rotation_affine = mmfn.create_affine(full_affine=rotation, output_path=rotation_affine_path_z90)
    # Not yet implemented or used
    #############################
    
    temp_dir = mmfn.create_temp_directories(sargs.output_directory)

    # Create generic affines for rolling and reflecting - these can be applied to any image
    
    # Create reflection affine
    reflection_affine_path = os.path.normpath(os.path.join(temp_dir, 'reflection_affine.txt'))
    reflection_affine = mmfn.create_affine(scale=[-1,1,1], output_path=reflection_affine_path)
    
    # Loop to create affines to roll about the Y axis
    max_range = np.pi
    num_rolls = 15
    for r in xrange(num_rolls):
        roll = fill_homogeneous_matrix(create_rotation_matrix( 0, r * max_range/num_rolls, 0 ))
        generic_roll_affine_path = os.path.normpath(os.path.join(temp_dir, 'roll_affine_' + str(r) + '.txt'))
        roll_affine = mmfn.create_affine(full_affine=roll, output_path=generic_roll_affine_path)
        
    # Create 180 degree rotation affines: rolls about nothing, Y, Z, X axes (the nothing included to make later steps easier)
    rotation = fill_homogeneous_matrix(create_rotation_matrix( 0, 0, 0 ))
    rotation_affine_path_1 = os.path.normpath(os.path.join(temp_dir, 'rotation_affine_1'+ '.txt'))
    rotation_affine = mmfn.create_affine(full_affine=rotation, output_path=rotation_affine_path_1)
    
    rotation = fill_homogeneous_matrix(create_rotation_matrix( 0, np.pi, 0 ))
    rotation_affine_path_2 = os.path.normpath(os.path.join(temp_dir, 'rotation_affine_2'+ '.txt'))
    rotation_affine = mmfn.create_affine(full_affine=rotation, output_path=rotation_affine_path_2)
    
    rotation = fill_homogeneous_matrix(create_rotation_matrix( 0, 0, np.pi ))
    rotation_affine_path_3 = os.path.normpath(os.path.join(temp_dir, 'rotation_affine_3'+ '.txt'))
    rotation_affine = mmfn.create_affine(full_affine=rotation, output_path=rotation_affine_path_3)
    
    rotation = fill_homogeneous_matrix(create_rotation_matrix( np.pi, 0, 0 ))
    rotation_affine_path_4 = os.path.normpath(os.path.join(temp_dir, 'rotation_affine_4' + '.txt'))
    rotation_affine = mmfn.create_affine(full_affine=rotation, output_path=rotation_affine_path_4)
    
    # Above all preserve the longest dimension along the Y axis, assuming it starts that way. If not, need to create 8 more...
    # rotation = fill_homogeneous_matrix(create_rotation_matrix( 0, 0, np.pi/2 ))
    # rotation_affine_path_5 = os.path.normpath(os.path.join(temp_dir, 'rotation_affine_5'+ '.txt'))
    # rotation_affine = mmfn.create_affine(full_affine=rotation, output_path=rotation_affine_path_5)
    
    # rotation = fill_homogeneous_matrix(create_rotation_matrix( np.pi, 0, np.pi/2 ))
    # rotation_affine_path_6 = os.path.normpath(os.path.join(temp_dir, 'rotation_affine_6'+ '.txt'))
    # rotation_affine = mmfn.create_affine(full_affine=rotation, output_path=rotation_affine_path_6)
    
    # rotation = fill_homogeneous_matrix(create_rotation_matrix( 0, 0, -1.0 * np.pi/2 ))
    # rotation_affine_path_7 = os.path.normpath(os.path.join(temp_dir, 'rotation_affine_7'+ '.txt'))
    # rotation_affine = mmfn.create_affine(full_affine=rotation, output_path=rotation_affine_path_7)
    
    # rotation = fill_homogeneous_matrix(create_rotation_matrix( np.pi, 0, -1.0 * np.pi/2 ))
    # rotation_affine_path_8 = os.path.normpath(os.path.join(temp_dir, 'rotation_affine_8'+ '.txt'))
    # rotation_affine = mmfn.create_affine(full_affine=rotation, output_path=rotation_affine_path_8)
    
    # rotation = fill_homogeneous_matrix(create_rotation_matrix( np.pi/2, 0, 0 ))
    # rotation_affine_path_9 = os.path.normpath(os.path.join(temp_dir, 'rotation_affine_9'+ '.txt'))
    # rotation_affine = mmfn.create_affine(full_affine=rotation, output_path=rotation_affine_path_9)
    
    # rotation = fill_homogeneous_matrix(create_rotation_matrix( np.pi/2, 0, np.pi ))
    # rotation_affine_path_10 = os.path.normpath(os.path.join(temp_dir, 'rotation_affine_10'+ '.txt'))
    # rotation_affine = mmfn.create_affine(full_affine=rotation, output_path=rotation_affine_path_10)
    
    # rotation = fill_homogeneous_matrix(create_rotation_matrix( -1.0 * np.pi/2, 0, 0 ))
    # rotation_affine_path_11 = os.path.normpath(os.path.join(temp_dir, 'rotation_affine_11'+ '.txt'))
    # rotation_affine = mmfn.create_affine(full_affine=rotation, output_path=rotation_affine_path_11)
    
    # rotation = fill_homogeneous_matrix(create_rotation_matrix( -1.0 * np.pi/2, 0, np.pi ))
    # rotation_affine_path_12 = os.path.normpath(os.path.join(temp_dir, 'rotation_affine_12'+ '.txt'))
    # rotation_affine = mmfn.create_affine(full_affine=rotation, output_path=rotation_affine_path_12)
    
    range_number_rotation_affines = len(glob.glob(os.path.join(temp_dir, 'rotation_affine_*' + '.txt'))) + 1
    
    # Get list of files (use file_name_filter)
    nifti_files_list = sorted(glob.glob(os.path.join(sargs.input_directory, sargs.file_name_filter + '.nii*')))
    print("  Processing {0} files: \n   - - -\n   \t{1}\n   - - -\n  ...".format(len(nifti_files_list), '\n\t'.join([ str(item) for item in nifti_files_list ])))
    
    for counter, nifti_file in enumerate(nifti_files_list):
        print "- - - - - - - - - - - \n  Processing {0} / {1}: {2} ...".format((counter + 1), len(nifti_files_list), nifti_file)
        
        # Get the input file and filename, removing path and 1+ extensions
        nf_path = os.path.normpath(os.path.join(nifti_file))
        nf_name = os.path.basename(nf_path).split(os.extsep)[0]
        
        final_output_path = os.path.normpath(os.path.join(sargs.output_directory, sargs.out_name_prepend + nf_name + sargs.out_name_append + sargs.ext))
        
        if counter == 0:
            # Set up atlas directory
            if not hasattr(sargs, 'atlas_directory'):
                # Create an atlas directory
                atlas_directory = os.path.normpath(os.path.join(sargs.output_directory, 'atlas'))
                print("  No atlas path specified! Creating one at {0} ...".format(atlas_directory))
            else:
                atlas_directory = sargs.atlas_directory
                if not os.path.exists(atlas_directory):
                    print("  Specified atlas path ({0}) is not a directory; creating it ...".format(atlas_directory))
            mmfn.check_create_directories([atlas_directory])
            
            # The first oriented image is used as the common coordinate space for the rest
            nf_path_first = final_output_path
        
        if not os.path.isfile(final_output_path):
        
            if counter == 0 or all_same_orientation is False:
                # Only do this once
                
                if sargs.do_downsample is True:
        
                    # # # # # # # # # # # # # #
                    #       Downsample       #
                    # # # # # # # # # # # # #
                    
                    print("mm_orient: Downsampling images ...")
                    
                    # Check if output_filename already exists
                    output_filename = os.path.normpath(os.path.join(temp_dir, nf_name + '_ds' + sargs.ext))
                    if not os.path.isfile(output_filename):
                    
                        print("  Loading {0} ...".format(nf_path))
                        img = nib.load(nf_path)
                        
                        # Downsample
                        ds_img = mmfn.downsample_image(img, sargs.downsample_factor)
                        
                        print("  Saving {0} ...".format(output_filename))
                        nib.save(ds_img, output_filename)
                    else:
                        print("  Output file, {0} already exists!".format(output_filename))
                        
                    original_nf_path = nf_path
                    nf_path = output_filename
                
                # # # # # # # # # # # # # # # # # # # # # # # # #
                #       Get principal axis transformation      #
                # # # # # # # # # # # # # # # # # # # # # # # #
                
                print("mm_orient: Generating principal axis transformation of images ...")
                
                # Check if output_filename already exists
                pa_affine_output_filename = os.path.normpath(os.path.join(temp_dir, nf_name + '_pa_rotation.txt'))
                if not os.path.isfile(pa_affine_output_filename):
                
                    print("  Loading {0} ...".format(nf_path))
                    img = nib.load(nf_path)
                    hdr = img.get_header()
                    pixdims = hdr.structarr['pixdim'][1:4]
                    
                    # Ignore intensity extremes
                    trimmed_nifti = mmfn.trim_intensity_extremes(img, percentile=pc_ignore )
                    
                    # Calculate CoM and inertia matrix
                    com = centre_of_mass(trimmed_nifti.get_data())
                    inertia = generate_inertia_matrix( trimmed_nifti, com )
                    
                    # Calculate, and sort, eigenvalues and vectors as I want them
                    eig_vals, eig_vects = eig_vv_sorted(inertia)
                    
                    # Create a homogeneous 3D transformation matrix
                    eig_vects_hom = fill_homogeneous_matrix(eig_vects)
                    print("  determinant of eig_vects_hom is {0}".format(np.linalg.det(eig_vects_hom)))
                    
                    # Calculate translation of centroid to origin
                    trans_to_origin = generate_translation_to_origin( pixdims, hdr.get_sform(), com )
                    print("  determinant of trans_to_origin is {0}".format(np.linalg.det(trans_to_origin)))
                    
                    # Apply the rotation matrix after doing this translation (then translate back)
                    pa_matrix = translate_to_apply_transformation(trans_to_origin, eig_vects_hom)
                    print("  determinant of pa_matrix is {0}".format(np.linalg.det(pa_matrix)))
                    
                    print("  Saving principal axis transformation matrix to {0} ...".format(pa_affine_output_filename))
                    pa_aff = mmfn.create_affine(full_affine=pa_matrix, output_path=pa_affine_output_filename)
                    
                else:
                    print("  Output file, {0} already exists!".format(pa_affine_output_filename))
                    pa_aff = np.loadtxt(pa_affine_output_filename)
                    
                print("  Principal axis affine:\n  {0}".format(pa_aff))
                
                # # # # # # # # # # # # # # # # # # # # # # # # # # #
                #       Roll about Y axis, reflect & register      #
                # # # # # # # # # # # # # # # # # # # # # # # # # #
                
                # We assume that the PA affine has successfully aligned the AP axis of the brain with the image Y axis.
                # The problem here is that we're testing for symmetry *first*, before testing the 12 possible 180-degree orientations with the atlas.
                
                final_affine_path = os.path.normpath(os.path.join(temp_dir, nf_name + '_final_affine.txt'))
                if not os.path.isfile(final_affine_path):
                
                    # Apply PA transformation to align PAs with image axes
                    
                    pa_output_filepath = os.path.normpath(os.path.join(temp_dir, nf_name + '_pa' + sargs.ext))
                    if not os.path.isfile(pa_output_filepath):
                        subprocess.call([nk.reg_transform,
                                            '-ref', nf_path,
                                            '-updSform', nf_path, pa_affine_output_filename, pa_output_filepath])
                    else:
                        print("  Output file, {0} already exists!".format(pa_output_filepath))

                    # Do the following loop in parallel, to save time
                    # Build the list of tasks, first
                    TASKS = [(r, nk, sargs, temp_dir, nf_name, pa_output_filepath, reflection_affine_path) for r in range(num_rolls)]
                    # Set the number of parallel processes to use
                    pool = multiprocessing.Pool(np.int(multiprocessing.cpu_count() / 2))
                    # The _star function will unpack TASKS to use in the actual function
                    # Using pool.map because we do care about the order of the results.
                    cc_results = pool.map(roll_and_compare_star, TASKS)
                    pool.close()
                    pool.join()
                                
                    print("  Pearson's r for each rotation with its reflection: {0}".format(cc_results))
                    
                    # Smooth the cc_results to improve robustness
                    smoothed_cc_results = []
                    for i in xrange(len(cc_results)):
                        if i == 0:
                            smoothed_cc_results.append( np.mean([cc_results[-1], cc_results[i], cc_results[1]]) )
                        elif i == len(cc_results) - 1:
                            smoothed_cc_results.append( np.mean([cc_results[0], cc_results[i], cc_results[-2]]) )
                        else:
                            smoothed_cc_results.append( np.mean([cc_results[i-1], cc_results[i], cc_results[i+1]]) )
                    
                    print("  The highest (averaged) Pearson's r is {0}, corresponding to rotation {1}".format(np.max(smoothed_cc_results), np.argmax(smoothed_cc_results)))
                    
                    # Get the best rotation for this image, and hence the registration affine from this roll to its reflection
                    best_roll_idx = np.argmax(smoothed_cc_results)
                    best_roll_affine_path = os.path.normpath(os.path.join(temp_dir, 'roll_affine_' + str(best_roll_idx) + '.txt'))
                    best_roll_aff = np.loadtxt(best_roll_affine_path)
                    
                    reg_roll_to_reflection_output_affine = os.path.normpath(os.path.join(temp_dir, nf_name + '_pa_roll_' + str(best_roll_idx) + '_to_refl_affine.txt'))
                    reg_roll_aff = np.loadtxt(reg_roll_to_reflection_output_affine)
                    
                    # Halve the registration affine from the rotation to its reflection
                    # Using NiftyReg to test Python's result in _half_reg_affine_calc.txt
                    half_reg_affine_filepath = os.path.normpath(os.path.join(temp_dir, nf_name + '_half_reg_affine_niftyreg.txt'))
                    if not os.path.isfile(half_reg_affine_filepath):
                        subprocess.call([nk.reg_transform,
                                            '-ref', pa_output_filepath,
                                            '-half', reg_roll_to_reflection_output_affine, half_reg_affine_filepath])
                    else:
                        print("  Output file, {0} already exists!".format(half_reg_affine_filepath))
                
                    # Calculate same (get the real part, otherwise the header causes the image to refuse to load in NiftyView)
                    half_logm = linalg.logm(reg_roll_aff) / 2.0
                    half_aff_calc = np.real(linalg.expm(half_logm))
                    half_aff = mmfn.create_affine(full_affine=half_aff_calc, output_path=os.path.normpath(os.path.join(temp_dir, nf_name + '_half_reg_affine_calc.txt')))

                    print("  Half the registration affine (calculated):\n  {0}".format(half_aff_calc))
                    
                    # Compose with principal axis transformation, and rotation
                    # final_affine = half_aff_calc.dot(best_roll_aff)            # applied to pa_output_filename, this works splendidly!
                    # final_affine = pa_aff.dot(half_aff_calc).dot(best_roll_aff)  # applied to nf_path, this worked splendidly, until I corrected the negative determinant!
                    final_affine = pa_aff.dot(best_roll_aff).dot(half_aff_calc)  # applied to nf_path, this works splendidly (and is in the expected order)!
                
                    final_affine = mmfn.create_affine(full_affine=final_affine, output_path=final_affine_path)
                    
                    # If every image is to be processed, great! However, if all images have the same initial orientation, only need to do the above once, and save the pa_aff and best_roll_aff, but not the half_aff_calc (as that will be very individual) (NO! It won't! You do want the half_aff_calc as well, as that corrects the roll! What will be individual is any further correction via reflection...)
                else:
                    # Quickly load final_affine so we can continue
                    final_affine = np.loadtxt(final_affine_path)
                        
                # # # # # # # # # # # # # # # # # # # # # # #
                #       Create candidate orientations      #
                # # # # # # # # # # # # # # # # # # # # # #

                # Create remaining candidates by composing with 180 degree rotation affines
                for rot in xrange(1,range_number_rotation_affines):
                    rotation_affine_path = os.path.normpath(os.path.join(temp_dir, 'rotation_affine_' + str(rot) + '.txt'))
                    
                    # Test application of final_affine (pa + roll + final correction) composed with the 180-degree rotation affine directly
                    # This does appear to work, and is what we want. Composing with rotation_1 is redundant, but more effort to avoid it.
                    rotation_affine = np.loadtxt(rotation_affine_path)
                    correction_affine_path = os.path.normpath(os.path.join(temp_dir, nf_name + '_correction_affine_' + str(rot) + '.txt'))
                    if not os.path.isfile(correction_affine_path):
                        correction_affine = mmfn.create_affine(full_affine=final_affine.dot(rotation_affine), output_path=correction_affine_path)
                    
                    final_image_filepath_rot = os.path.normpath(os.path.join(temp_dir, nf_name + '_corrected_' + str(rot) + sargs.ext))
                    if not os.path.isfile(final_image_filepath_rot):
                        subprocess.call([nk.reg_transform,
                                            '-ref', nf_path,
                                            '-updSform', nf_path, correction_affine_path, final_image_filepath_rot])
                    else:
                        print("  Output file, {0} already exists!".format(final_image_filepath_rot))
                        
                # # # # # # # # # # # # # # # # # # # # # # #
                #       Check for target atlas images      #
                # # # # # # # # # # # # # # # # # # # # # #
                
                # Some possibilities:
                # 1. user provides at least 3 string paths of correctly-aligned images
                # 2. user moves correctly-aligned images to new directory
                # 3. user deletes incorrectly aligned images
                # 4. in background, group all images by similarity. There will be 4 groups. User selects a group.
                # 5. user provides directory of existing atlas images
                
                # Get list of files (use file_name_filter)
                atlas_nifti_files_list = sorted(glob.glob(os.path.join(atlas_directory, sargs.atlas_name_filter + '.nii*')))
                
                n_desired_atlases = 1
                if len(atlas_nifti_files_list) < n_desired_atlases:
                    
                    # Get candidate list
                    candidate_list = sorted(glob.glob(os.path.join(temp_dir, '*_corrected_*.nii*')))
                    
                    # user_choice = raw_input("Type the number of the chosen 'corrected' image: ")
                    
                    mmfn.alert_user("Choose an atlas image from:\n{0},\n and copy it to your atlas directory, {1} !".format(candidate_list, atlas_directory))
                    mmfn.wait_for_files(atlas_directory, 1)
                
                # # # # # # # # # # # # # # # # # # # # #
                #       Compare with atlas images      #
                # # # # # # # # # # # # # # # # # # # #
                
                print("mm_orient: Comparing with atlas image(s) to get best candidate ...")
                
                # Re-list atlas images
                atlas_nifti_files_list = sorted(glob.glob(os.path.join(atlas_directory, '*.nii*')))
                
                print("Processing {0} atlas NIfTI files ...".format(len(atlas_nifti_files_list)))
                print atlas_nifti_files_list
                
                for atlas_counter, atlas_nifti_file in enumerate(atlas_nifti_files_list):
                    print(" Processing {0} / {1}: {2} ...".format((atlas_counter + 1), len(atlas_nifti_files_list), atlas_nifti_file))
                    
                    # Get the filename, removing path and 1+ extensions
                    atlas_nf_name = os.path.basename(atlas_nifti_file.split(os.extsep)[0])
                    
                    # Get the input file
                    atlas_nf_path = os.path.normpath(os.path.join(atlas_nifti_file))
                    
                    # print("  Loading {0} ...".format(atlas_nf_path))
                    # atlas_img = nib.load(atlas_nf_path)
                    
                    # atlas_data, atlas_hdr = mmfn.check_get_nifti_data(atlas_img)
                    
                    # Register and compare each candidate orientation with this atlas
                    # Do the following loop in parallel, to save time
                    # Build the list of tasks, first
                    TASKS = [(c, temp_dir, nf_name, sargs, nk.reg_aladin, atlas_nf_path, 2, 2) for c in range(1,range_number_rotation_affines)]
                    # Set the number of parallel processes to use
                    pool = multiprocessing.Pool(np.int(multiprocessing.cpu_count() / 2))
                    # The _star function will unpack TASKS to use in the actual function
                    # Using pool.map because we do care about the order of the results.
                    # A list for storing CC values: cc_results(c) will allow us to retrieve the best candidate orientation
                    cc_results = pool.map(reg_rotations_and_compare_star, TASKS)
                    pool.close()
                    pool.join()
                    
                    print("  Pearson's r for each candidate with this atlas: {0}".format(cc_results))
                        
                    # Get the best candidate for this atlas. NB + 1 because the names of candidates start at _1!
                    best_candidate_idx = np.argmax(cc_results) + 1
                    best_candidate_image_path = os.path.normpath(os.path.join(temp_dir, nf_name + '_corrected_' + str(best_candidate_idx) + sargs.ext))
                    best_rotation_affine_path = os.path.normpath(os.path.join(temp_dir, nf_name + '_rotation_affine_' + str(best_candidate_idx) + '.txt'))
                    best_correction_affine_path = os.path.normpath(os.path.join(temp_dir, nf_name + '_correction_affine_' + str(best_candidate_idx) + '.txt'))
                    
                    print("  Best candidate image: {0}".format(best_candidate_image_path))
                    
                    # To orient other images: If best c == 1, for the correct affine you just want final_affine_path; if c == {2,3,4}, you want final_affine_path and the rotation affine to compose it with. (Now updated.)
            
                if sargs.do_downsample is True:
                    # Restore original
                    nf_path = original_nf_path
            
            if not os.path.isfile(final_output_path):
                print("  Applying best alignment to full-size image ...")
                
                # Orient all images based only on a single best_candidate_rotation
                oriented_image_filepath = os.path.normpath(os.path.join(temp_dir, nf_name + '_oriented' + sargs.ext))
                if not os.path.isfile(oriented_image_filepath):
                    subprocess.call([nk.reg_transform,
                                        '-ref', nf_path,
                                        '-updSform', nf_path, best_correction_affine_path, oriented_image_filepath])
                else:
                    print("  Output file, {0} already exists!".format(oriented_image_filepath))
                        
                print("mm_orient: Correcting slight orientation offset ...")
                
                corr_output_path = os.path.normpath(os.path.join(temp_dir, nf_name + '-corr' + sargs.ext))
                res_output_affine_path = os.path.normpath(os.path.join(temp_dir, nf_name + '_to_refl_res_aff.txt'))
                half_aff_path = os.path.normpath(os.path.join(temp_dir, nf_name + '_to_refl_res_aff_half.txt'))
                reflection_output_filepath = os.path.normpath(os.path.join(temp_dir, nf_name + '_refl' + sargs.ext))
                resampled_output_path = os.path.normpath(os.path.join(temp_dir, nf_name + '_to_refl_res' + sargs.ext))
                
                corr_output_path = correct_slight_orientation_offset(nk, oriented_image_filepath, corr_output_path, res_output_affine_path, half_aff_path, reflection_output_filepath, reflection_affine_path, resampled_output_path, ln=2, lp=2)
                
                # Delete intermediate files
                mmfn.delete_files([oriented_image_filepath, reflection_output_filepath, resampled_output_path])
                
                if counter > 0:
                    print("mm_orient: Rigidly aligning image to common coordinate space ...")
                    # NB: as this may otherwise make correct_slight_orientation_offset slightly redundant, instead of using the whole reg_aladin affine here, a better way is (optionally) just to take the translation. If translation_affine_path is not provided, the full affine will be used and res_output_affine_path should be used, instead, to construct the single affine below (via common_coord_affine).
                    
                    resampled_output_path = os.path.normpath(os.path.join(temp_dir, nf_name + '_to_common_coord_res' + sargs.ext))
                    res_output_affine_path = os.path.normpath(os.path.join(temp_dir, nf_name + '_res_affine.txt'))
                    translation_affine_path = os.path.normpath(os.path.join(temp_dir, nf_name + '_to_common_coord.txt'))
                    final_output_path = register_and_align(nk, nf_path_first, corr_output_path, resampled_output_path, res_output_affine_path, final_output_path, translation_affine_path=translation_affine_path, ln=2, lp=2)    # should be able to use rigid_only=False here as just taking the translation elements anyway
                    
                else:
                    # So the below creation of a single affine also works for the first image (to which the others are all rigidly registered), need an affine here which does nothing
                    translation_affine_path = rotation_affine_path_1
                    
                # Delete intermediate files
                mmfn.delete_files([corr_output_path, resampled_output_path])
                    
            print("mm_orient: Creating single affine ...")
            # Combine all affines into a single transformation matrix which can be used to orient corresponding data (e.g. masks; phase images)
            
            single_orienting_affine_path = os.path.normpath(os.path.join(sargs.output_directory, nf_name + '_orienting_affine' + '.txt'))
            if not os.path.isfile(single_orienting_affine_path):
            
                best_correction_affine = np.loadtxt(best_correction_affine_path)
                slight_offset_affine = np.loadtxt(half_aff_path)
                common_coord_affine = np.loadtxt(translation_affine_path)
                single_orienting_affine = best_correction_affine.dot(slight_offset_affine).dot(common_coord_affine)   # original
                
                if counter == 0:
                    print("  Aligning first image, {0}, to origin ...".format(nf_path))
                    # Only need to do this for the first image as others are aligned to it in subsequent loops, with translation_affine (above)
                    # Ideally, this would be done in one step. As it is, it's done in two: (1) the affine to align the image origin (its corner) is applied to the original image with reg_transform, giving temp_im_a; (2) the affine to translate a position in the image is applied to temp_im_a, again with reg_transform.
                    # The good: you can choose whether you want the origin to be at the corner, the image centre, or the image centre of mass.
                    # The bad: I'm sure these two steps could be combined into one, but done that yet.
                    # If we could predict what the outcome header sform would be of temp_corner_nii, its generation would be unnecessary (that affine would just go straight into generate_translation_to_origin()).
                    # Here, if there are two or more .dot()s, the leftmost pair is multiplied first, and the result is multiplied by the rightmost. The order is ((1*2), then *3)
                    
                    img = nib.load(nf_path)
                    pixdims = img.get_header().structarr['pixdim'][1:4]
                    original_sform = img.get_header().get_sform()
                    
                    # Where might we want to put the origin (the mm x,y,z coordinates)? Find the translation from the corner to...
                    # (1) the image centre
                    centre_of_image_vox = np.array(img.get_shape()) / 2.0
                    # (2) the image centroid
                    centroid_vox = centre_of_mass(img.get_data())
                    # (3) an arbitrary position ...
                    target_vox = centre_of_image_vox
                    
                    final_single_orienting_affine = rotate_translate(pixdims, original_sform, single_orienting_affine, target_vox)
                    
                    # Save the affine which will orient the original image in one go
                    single_orienting_affine = mmfn.create_affine(full_affine=final_single_orienting_affine, output_path=single_orienting_affine_path)
                    
                    if not os.path.isfile(nf_path_first):
                        subprocess.call([nk.reg_transform,
                                            '-ref', nf_path,
                                            '-updSform', nf_path, single_orienting_affine_path, nf_path_first])
                    else:
                        print("  Output file, {0} already exists!".format(nf_path_first))
                else:
                    single_orienting_affine = mmfn.create_affine(full_affine=single_orienting_affine, output_path=single_orienting_affine_path)
                
        else:
            print("  Output file, {0} already exists!".format(final_output_path))
            
        # This should be a duplicate result; uncomment if you want to test that the single affine works
        # single_orienting_affine_path = mmfn.get_corresponding_file(sargs.output_directory, nf_name, name_filter='*_orienting_affine*', extension='*.txt', path_only=True)
        
        # filepath = os.path.normpath(os.path.join(sargs.output_directory, nf_name + '_via_single_affine' + sargs.ext))
        # if not os.path.isfile(filepath):
            # subprocess.call([nk.reg_transform,
                                # '-ref', nf_path,
                                # '-updSform', nf_path, single_orienting_affine_path, filepath])
        # else:
            # print("  Output file, {0} already exists!".format(filepath))
        
    # Above loop ends; do same for every image in the input directory
    
# def add_arguments(parser):
    # parser.add_argument('-v', '--verbose', action="store_true", help="Verbose output")
    # ...
    
def generate_principle_axis_transformation(nf_path, pa_affine_output_filename, pc_ignore=5.0):
    """
    
    """
    
    print("  Loading {0} ...".format(nf_path))
    img = nib.load(nf_path)
    hdr = img.get_header()
    pixdims = hdr.structarr['pixdim'][1:4]
    
    # Ignore intensity extremes
    trimmed_nifti = mmfn.trim_intensity_extremes(img, percentile=pc_ignore )
    
    # Calculate CoM and inertia matrix
    com = centre_of_mass(trimmed_nifti.get_data())
    inertia = generate_inertia_matrix( trimmed_nifti, com )
    
    # Calculate, and sort, eigenvalues and vectors as I want them
    eig_vals, eig_vects = eig_vv_sorted(inertia)
    
    # Create a homogeneous 3D transformation matrix
    eig_vects_hom = fill_homogeneous_matrix(eig_vects)
    print("  determinant of eig_vects_hom is {0}".format(np.linalg.det(eig_vects_hom)))
    
    # Calculate translation of centroid to origin
    trans_to_origin = generate_translation_to_origin( pixdims, hdr.get_sform(), com )
    print("  determinant of trans_to_origin is {0}".format(np.linalg.det(trans_to_origin)))
    
    # Apply the rotation matrix after doing this translation (then translate back)
    pa_matrix = translate_to_apply_transformation(trans_to_origin, eig_vects_hom)
    print("  determinant of pa_matrix is {0}".format(np.linalg.det(pa_matrix)))
    
    print("  Saving principal axis transformation matrix to {0} ...".format(pa_affine_output_filename))
    pa_aff = mmfn.create_affine(full_affine=pa_matrix, output_path=pa_affine_output_filename)
    
    return pa_aff
    
def go(args=None):
    """Actually run the expected script. main() should eventually just parse external input arguments and pass to this function. go() exists so mousemorph.py can call it directly.
    
    Run this with: mousemorph.py orient [arguments]
    """
    
    # How to call 'orient' with the below?
    if args is None:
        import mousemorph
        args = mousemorph.MouseMorph().args
        
    temp_dir = mmfn.create_temp_directories(args.output_directory)
    
    # Set up atlas directory
    if not hasattr(args, 'atlas_directory'):
        # Create an atlas directory
        atlas_directory = os.path.normpath(os.path.join(args.output_directory, 'atlas'))
        print("No atlas path specified! Creating one at {0} ...".format(atlas_directory))
    else:
        atlas_directory = args.atlas_directory
        if not os.path.exists(atlas_directory):
            print("Specified atlas path ({0}) is not a directory; creating it ...".format(atlas_directory))
    mmfn.check_create_directories([atlas_directory])
    
    # Ignore images' intensity extremes when calculating the principal axes.
    pc_ignore = 5.0
        
    # if not mmfn.voxels_are_isotropic(hdr):
        # If voxels are not isotropic, new image should be isotropic, with the smallest voxel dimension on each axis.
        
    # if args.allsame:
    
    # Create generic affines for rolling and reflecting - these can be applied to any image
    
    # Create reflection affine
    reflection_affine_path = os.path.normpath(os.path.join(temp_dir, 'reflection_affine.txt'))
    reflection_affine = mmfn.create_affine(scale=[-1,1,1], output_path=reflection_affine_path)
    
    # Loop to create affines to roll about the Y axis
    max_range = np.pi
    num_rolls = 15
    for r in xrange(num_rolls):
        roll = fill_homogeneous_matrix(create_rotation_matrix( 0, r * max_range/num_rolls, 0 ))
        generic_roll_affine_path = os.path.normpath(os.path.join(temp_dir, 'roll_affine_' + str(r) + '.txt'))
        roll_affine = mmfn.create_affine(full_affine=roll, output_path=generic_roll_affine_path)
        
    # Create 180 degree rotation affines: rolls about nothing, Y, Z, X axes (the nothing included to make later steps easier)
    mmfn.create_affine(full_affine=fill_homogeneous_matrix(create_rotation_matrix( 0, 0, 0 )), output_path=os.path.normpath(os.path.join(temp_dir, 'rotation_affine_1.txt')))
    mmfn.create_affine(full_affine=fill_homogeneous_matrix(create_rotation_matrix( 0, np.pi, 0 )), output_path=os.path.normpath(os.path.join(temp_dir, 'rotation_affine_2.txt')))
    mmfn.create_affine(full_affine=fill_homogeneous_matrix(create_rotation_matrix( 0, 0, np.pi )), output_path=os.path.normpath(os.path.join(temp_dir, 'rotation_affine_3.txt')))
    mmfn.create_affine(full_affine=fill_homogeneous_matrix(create_rotation_matrix( np.pi, 0, 0 )), output_path=os.path.normpath(os.path.join(temp_dir, 'rotation_affine_4.txt')))
    # Above all preserve the longest dimension along the Y axis, assuming it starts that way. If not, need to create 8 more...
    if args.allpa:
        mmfn.create_affine(full_affine=fill_homogeneous_matrix(create_rotation_matrix( 0, 0, np.pi/2 )), output_path=os.path.normpath(os.path.join(temp_dir, 'rotation_affine_5.txt')))
        mmfn.create_affine(full_affine=fill_homogeneous_matrix(create_rotation_matrix( np.pi, 0, np.pi/2 )), output_path=os.path.normpath(os.path.join(temp_dir, 'rotation_affine_6.txt')))
        mmfn.create_affine(full_affine=fill_homogeneous_matrix(create_rotation_matrix( 0, 0, -1.0 * np.pi/2 )), output_path=os.path.normpath(os.path.join(temp_dir, 'rotation_affine_7.txt')))
        mmfn.create_affine(full_affine=fill_homogeneous_matrix(create_rotation_matrix( np.pi, 0, -1.0 * np.pi/2 )), output_path=os.path.normpath(os.path.join(temp_dir, 'rotation_affine_8.txt')))
        mmfn.create_affine(full_affine=fill_homogeneous_matrix(create_rotation_matrix( np.pi/2, 0, 0 )), output_path=os.path.normpath(os.path.join(temp_dir, 'rotation_affine_9.txt')))
        mmfn.create_affine(full_affine=fill_homogeneous_matrix(create_rotation_matrix( np.pi/2, 0, np.pi )), output_path=os.path.normpath(os.path.join(temp_dir, 'rotation_affine_10.txt')))
        mmfn.create_affine(full_affine=fill_homogeneous_matrix(create_rotation_matrix( -1.0 * np.pi/2, 0, 0 )), output_path=os.path.normpath(os.path.join(temp_dir, 'rotation_affine_11.txt')))
        mmfn.create_affine(full_affine=fill_homogeneous_matrix(create_rotation_matrix( -1.0 * np.pi/2, 0, np.pi )), output_path=os.path.normpath(os.path.join(temp_dir, 'rotation_affine_12.txt')))
    
    range_number_rotation_affines = len(glob.glob(os.path.join(temp_dir, 'rotation_affine_*.txt'))) + 1
        
    for counter, nf_path in enumerate(args.input_files_list):
        print "- - - - - - - - - - - \n  Processing {0} / {1}: {2} ...".format((counter + 1), len(args.input_files_list), nf_path)
        
        # Get the input file and filename, removing path and 1+ extensions
        nf_path = os.path.normpath(os.path.join(nf_path))
        nf_name = os.path.basename(nf_path).split(os.extsep)[0]
        
        final_output_path = os.path.normpath(os.path.join(args.output_directory, args.out_name_prepend + nf_name + args.out_name_append + args.ext))
        
        if counter == 0:
            # The first oriented image is used as the common coordinate space for the rest
            nf_path_first = final_output_path
        
        # If we know AP is the longest axis, we can rely upon the principle axes and y-axis rolls to find a good atlas.
        # If we *don't* know which is the longest axis, principle axes' result could align any axis with the Y axis, so we can't roll yet. We need to compose with the rotation_affines first, then ask the user to pick the best one, then roll, then ask the user again. Alternatively could roll them all, but that seems like a lot.
        
        if not os.path.isfile(final_output_path):
        
            if counter == 0 or all_same_orientation is False:
                # Only do this once
                
                if args.downsample > 0:
                
                    original_nf_path = nf_path
                    
                    nf_path = mmfn.load_nifti_save(nf_path, function=mmfn.downsample_image, output_filepath=os.path.normpath(os.path.join(temp_dir, nf_name + '_ds' + args.ext)), factor=args.downsample)
                    
                # # # # # # # # # # # # # # # # # # # # # # # # #
                #       Get principal axis transformation      #
                # # # # # # # # # # # # # # # # # # # # # # # #
                
                print("mm_orient: Generating principal axis transformation of images ...")
                
                # Check if output_filename already exists
                pa_affine_output_filename = os.path.normpath(os.path.join(temp_dir, nf_name + '_pa_rotation.txt'))
                if not os.path.isfile(pa_affine_output_filename):
                
                    pa_aff = generate_principle_axis_transformation(nf_path, pa_affine_output_filename, pc_ignore)
                    
                else:
                    print("  Output file, {0} already exists!".format(pa_affine_output_filename))
                    pa_aff = np.loadtxt(pa_affine_output_filename)
                    
                print("  Principal axis affine:\n  {0}".format(pa_aff))
                
                # *ONLY* necessary to intervene with this here if we're not certain PA has aligned the AP axis with the image 'Y' axis. Otherwise, just do this *after* performing rolls.
                # NB asking for user input on every image would be awful - better to just apply all the rolls and hope that works (?)
                # Only need to test 3 of the 12 generated: (1, 5, 9) to find out which has the highest symmetry when it is rolled about Y...
                if args.allpa:
        
                    # # # # # # # # # # # # # # # # # # # # # # #
                    #       Create candidate orientations      #
                    # # # # # # # # # # # # # # # # # # # # # #

                    # Create remaining candidates by composing with 180 degree rotation affines
                    for rot in [1, 5, 9]:
                        rotation_affine_path = os.path.normpath(os.path.join(temp_dir, 'rotation_affine_' + str(rot) + '.txt'))
                        
                        # Test application of pa affine composed with the 180-degree rotation affine directly (skipping rolls)
                        # Composing with rotation_1 is redundant, but more effort to avoid it.
                        rotation_affine = np.loadtxt(rotation_affine_path)
                        correction_affine_path = os.path.normpath(os.path.join(temp_dir, nf_name + '_candidate_correction_affine_' + str(rot) + '.txt'))
                        if not os.path.isfile(correction_affine_path):
                            correction_affine = mmfn.create_affine(full_affine=pa_aff.dot(rotation_affine), output_path=correction_affine_path)
                        
                        final_image_filepath_rot = os.path.normpath(os.path.join(temp_dir, nf_name + '_pa_candidate_' + str(rot) + sargs.ext))
                        if not os.path.isfile(final_image_filepath_rot):
                            subprocess.call([nk.reg_transform,
                                                '-ref', nf_path,
                                                '-updSform', nf_path, correction_affine_path, final_image_filepath_rot])
                        else:
                            print("  Output file, {0} already exists!".format(final_image_filepath_rot))
                            
                    pa_candiate_paths = mmfn.get_files_list(temp_dir, '*_pa_candidate_*')
                    
                else:
                    # Still want to update an image
                    # Apply PA transformation to align PAs with image axes
                    
                    pa_output_filepath = os.path.normpath(os.path.join(temp_dir, nf_name + '_pa' + sargs.ext))
                    if not os.path.isfile(pa_output_filepath):
                        subprocess.call([nk.reg_transform,
                                            '-ref', nf_path,
                                            '-updSform', nf_path, pa_affine_output_filename, pa_output_filepath])
                    else:
                        print("  Output file, {0} already exists!".format(pa_output_filepath))
                    
                    pa_candiate_paths = [pa_output_filepath]
                    
                # # # # # # # # # # # # # # # # # # # # # # # # # # #
                #       Roll about Y axis, reflect & register      #
                # # # # # # # # # # # # # # # # # # # # # # # # # #
                
                # We assume that the PA affine has successfully aligned the AP axis of the brain with the image Y axis.
                # The problem here is that we're testing for symmetry *first*, before testing the 12 possible 180-degree orientations with the atlas.
                
                final_affine_path = os.path.normpath(os.path.join(temp_dir, nf_name + '_final_affine.txt'))
                if not os.path.isfile(final_affine_path):
                
                    max_cc_list = []
                    max_cc_idx_list = []
                
                    # There's just one pa_candiate_path if args.allpa is False
                    for pa_output_filepath in pa_candiate_paths:
                
                        # Do the following loop in parallel, to save time
                        # Build the list of tasks, first
                        
                        TASKS = [(r, nk, sargs, temp_dir, nf_name, pa_output_filepath, reflection_affine_path) for r in range(num_rolls)]
                        # Set the number of parallel processes to use
                        pool = multiprocessing.Pool(np.int(multiprocessing.cpu_count() / 2))
                        # The _star function will unpack TASKS to use in the actual function
                        # Using pool.map because we do care about the order of the results.
                        cc_results = pool.map(roll_and_compare_star, TASKS)
                        pool.close()
                        pool.join()
                                    
                        print("  Pearson's r for each rotation with its reflection: {0}".format(cc_results))
                        
                        # Smooth the cc_results to improve robustness
                        smoothed_cc_results = []
                        for i in xrange(len(cc_results)):
                            if i == 0:
                                smoothed_cc_results.append( np.mean([cc_results[-1], cc_results[i], cc_results[1]]) )
                            elif i == len(cc_results) - 1:
                                smoothed_cc_results.append( np.mean([cc_results[0], cc_results[i], cc_results[-2]]) )
                            else:
                                smoothed_cc_results.append( np.mean([cc_results[i-1], cc_results[i], cc_results[i+1]]) )
                        
                        print("  The highest (averaged) Pearson's r is {0}, corresponding to rotation {1}".format(np.max(smoothed_cc_results), np.argmax(smoothed_cc_results)))
                        
                        max_cc_list.append(np.max(smoothed_cc_results))
                        max_cc_idx_list.append(np.argmax(smoothed_cc_results))
                        
                    # Now not just comparing rolls; also comparing different initial orientations and their rolls, if args.allpa is True.
                    # This gives us the rotation_affine_path from above, by proxy.
                    best_candidate_idx = np.argmax(max_cc_list)
                    best_candidate_path = pa_candiate_paths[best_candidate_idx]
                    
                    # Get the best rotation for this image, and hence the registration affine from this roll to its reflection
                    best_roll_idx = max_cc_idx_list[best_candidate_idx]
                    best_roll_affine_path = os.path.normpath(os.path.join(temp_dir, 'roll_affine_' + str(best_roll_idx) + '.txt'))
                    best_roll_aff = np.loadtxt(best_roll_affine_path)
                    
                    reg_roll_to_reflection_output_affine = os.path.normpath(os.path.join(temp_dir, nf_name + '_pa_roll_' + str(best_roll_idx) + '_to_refl_affine.txt'))
                    reg_roll_aff = np.loadtxt(reg_roll_to_reflection_output_affine)
                    
                    # Halve the registration affine from the rotation to its reflection
                    # Using NiftyReg to test Python's result in _half_reg_affine_calc.txt
                    # (This just duplicates the result half_aff_calc below)
                    # half_reg_affine_filepath = os.path.normpath(os.path.join(temp_dir, nf_name + '_half_reg_affine_niftyreg.txt'))
                    # if not os.path.isfile(half_reg_affine_filepath):
                        # subprocess.call([nk.reg_transform,
                                            # '-ref', pa_output_filepath,
                                            # '-half', reg_roll_to_reflection_output_affine, half_reg_affine_filepath])
                    # else:
                        # print("  Output file, {0} already exists!".format(half_reg_affine_filepath))
                
                    # Calculate same (get the real part, otherwise the header causes the image to refuse to load in NiftyView)
                    half_logm = linalg.logm(reg_roll_aff) / 2.0
                    half_aff_calc = np.real(linalg.expm(half_logm))
                    half_aff = mmfn.create_affine(full_affine=half_aff_calc, output_path=os.path.normpath(os.path.join(temp_dir, nf_name + '_half_reg_affine_calc.txt')))

                    print("  Half the registration affine (calculated):\n  {0}".format(half_aff_calc))
                    
                    # Compose with principal axis transformation, and rotation
                    # final_affine = half_aff_calc.dot(best_roll_aff)            # applied to pa_output_filename, this works splendidly!
                    # final_affine = pa_aff.dot(half_aff_calc).dot(best_roll_aff)  # applied to nf_path, this worked splendidly, until I corrected the negative determinant!
                    final_affine = pa_aff.dot(best_roll_aff).dot(half_aff_calc)  # applied to nf_path, this works splendidly (and is in the expected order)!
                
                    final_affine = mmfn.create_affine(full_affine=final_affine, output_path=final_affine_path)
                    
                    # If every image is to be processed, great! However, if all images have the same initial orientation, only need to do the above once, and save the pa_aff and best_roll_aff, but not the half_aff_calc (as that will be very individual) (NO! It won't! You do want the half_aff_calc as well, as that corrects the roll! What will be individual is any further correction via reflection...)
                else:
                    # Quickly load final_affine so we can continue
                    final_affine = np.loadtxt(final_affine_path)
                    
                    
                    
                    

                    
                    # user_choice = raw_input("Candidate files generated are: \n{0} \nFrom the above list of files, please input the number (1 - 12) of the file whose antero-posterior axis (AP, or nose-tail) best-matches the image 'Y' axis, then press [Enter]: ".format())
        
                    # # # # # # # # # # # # # # # # # # # # # # #
                    #       Check for target atlas images      #
                    # # # # # # # # # # # # # # # # # # # # # #

                    
                    # Get list of files (use file_name_filter)
                    atlas_nifti_files_list = sorted(glob.glob(os.path.join(atlas_directory, sargs.atlas_name_filter + '.nii*')))
                    
                    n_desired_atlases = 1
                    if len(atlas_nifti_files_list) < n_desired_atlases:
                        
                        # Get candidate list
                        candidate_list = sorted(glob.glob(os.path.join(temp_dir, '*_corrected_*.nii*')))
                        
                        # user_choice = raw_input("Type the number of the chosen 'corrected' image: ")
                        
                        mmfn.alert_user("Choose an atlas image from:\n{0},\n and copy it to your atlas directory, {1} !".format(candidate_list, atlas_directory))
                        mmfn.wait_for_files(atlas_directory, 1)
        

        
        
def voxels_are_isotropic( nifti_header ):
    """Given just a NIfTI-1 header object, returns True if voxel pixdims (header.structarr['pixdim'][1:4]) are all the same; False otherwise."""
    
    pixdim = nifti_header.structarr['pixdim'][1:4]
    if pixdim[0] != pixdim[1] or pixdim[0] != pixdim[2] or pixdim[1] != pixdim[2]:
        return False
    else:
        return True
    # End voxels_are_isotropic() definition
    
if __name__ == '__main__':
    main()
    
# End