#!/usr/bin/python

"""MouseMorph, an automatic mouse MR image processor

Simple image processing functions, and tools for working with groups of NIfTI images.

With loop_and_save(), you can apply any compatible function to an entire directory of images.

Most of these were composed to fulfil a specific need. All should work as expected (let me know if not), but some are ugly hacks to quickly process large numbers of files.

Several return either a full NIfTI-1 image, just the 3D data, or just the header.

"""

import os
import csv
import sys
import copy
import glob
import argparse
import collections
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm

from scipy import ndimage, linalg

import mm_niftk

__author__ = 'Nick Powell (PhD student, CMIC & CABI, UCL, UK), nicholas.powell.11@ucl.ac.uk'
__version__ = '0.2.20150212'
__created__ = '2014-03-19, Wednesday'

def get_iterable( x ):
    """From: http://stackoverflow.com/a/6711233"""
    if isinstance(x, collections.Iterable) and not isinstance(x, basestring):
        return x
    else:
        return [x]
        
def get_nifti( input ):
    """Given a NIfTI, continues. Given a string, checks it's a path to a NIfTI image, and loads it.
    
    Returns a NIfTI.
    """
    
    if isinstance(input, nib.Nifti1Image):
        return input
    elif isinstance(input, str):
        npath = os.path.normpath(input)
        if os.path.isfile(npath):
            return nib.load(npath)
        else:
            raise IOError("Error: input '{0}' appears to be a string, but is not a path to an existing file.".format(input))
    else:
        print("  Error: either provide this function with a path to a NIfTI-1 file, or a NIfTI object.")
        return None
    # End check_get_nifti() definition
    
def check_get_nifti_data( input, hdr=None, verbose=False ):
    """If given a NIfTI image object, returns the data and hdr.
    
    If given just the data, returns just the data.
    If given a string, assumes it's a path to a NIfTI image, and loads it.
    
    """
    
    if type(input) is nib.nifti1.Nifti1Image:
        # We have a full NIfTI object and we need to get the data out
        data = input.get_data()
        hdr = input.get_header()
    elif type(input) is nib.nifti1.Nifti1Header:
        # We've just been given a NIfTI header
        print("  Only header provided ...")
        data = None
        hdr = input
    elif type(input) is str:
        if os.path.isfile(os.path.normpath(input)):
            # We might have a path to a NIfTI image
            nf_path = os.path.normpath(os.path.join(input))
            if verbose:
                print("  Input is a file: loading {0} ...".format(nf_path))
            img = nib.load(nf_path)
            data = img.get_data()
            hdr = img.get_header()
        else:
            print("  Input {0} does not exist.".format(input))
    else:
        # Assume type(input) is numpy.ndarray
        data = input

    return data, hdr
    # End of check_get_nifti_data() definition
        
# Files and folders
# -----------------

def get_folders_list( directory ):
    """Not yet complete. subfolder_list does work.
    
    Returns several lists of full paths to:
    
    - all immediate subdirectories
    - all directories (recursively)
    - all files
    
    all the immediate subfolders in the current directory (subfolders_list), and a list of all paths of all subfolders (allfolder_list).
    
    sublist, alllist, filelist = get_folders_list(root_dir)
    
    """
    
    root_dir = os.path.normpath(directory)
    
    # List only immediate subdirectories
    subfolder_list = [os.path.normpath(os.path.join(root_dir, subname)) for subname in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, subname))]
    
    allfolder_list = []
    allfile_list = []
    for rootname, dirnames, filenames in os.walk(root_dir):
        for subdirname in dirnames:
            allfolder_list.append(os.path.normpath(os.path.join(root_dir, subdirname)))

        for filename in filenames:
            allfile_list.append(os.path.normpath(os.path.join(root_dir, filename)))

    # List all folders in root_dir, recursively
    # allfolder_list = [os.path.normpath(os.path.join(root_dir, x[0])) for x in os.walk(root_dir)]
    
    # List all files
    # allfile_list = [os.path.normpath(os.path.join(root_dir, x[2])) for x in os.walk(root_dir)]
    
    return subfolder_list, allfolder_list, allfile_list

def get_files_list( directory, file_name_filter='*', extension='.nii*' ):
    """Returns a sorted list of files matching the input criteria"""
    if directory is not None:
        return sorted(glob.glob(os.path.join(directory, file_name_filter + extension)))
    else:
        return None
    # End get_files_list() definition
    
def get_names_list( directory, file_name_filter='*', extension='.nii*' ):
    """Returns a sorted list of file names matching the input criteria"""
    full_list = get_files_list( directory, file_name_filter, extension )
    return [get_file_name(file_path) for file_path in full_list]
    # End get_names_list() definition
    
def get_file_name( file_path ):
    return (os.path.basename(os.path.normpath(os.path.join(file_path)))).split(os.extsep)[0]
    # End get_file_name() definition
    
def get_corresponding_file( directory, original_name, name_filter='*', extension='*.nii*', path_only=False, verbose=False ):
    """Return the file most likely to be associated with the given original file.
    
    If there is more than one after filtering directory by original_name and name_filter, returns the first when sorted alphabetically.
    
    To avoid confusion between files like abc_1.nii and abc_10.nii, you can optionally specify extension='.nii' rather than the default ('*.nii').
    
    More generic than get_corresponding_mask().
    
    Optionally just return the path, without loading the NIfTI.
    
    """

    import fnmatch
    
    if original_name is None or original_name is np.NaN:
        print("Bad file name supplied: {0}".format(str(original_name)))
        return None
    else:
        # List the relevant file(s) for this original image, and sort in order of name
        corresponding_files_list = get_files_list(directory, '*' + original_name, extension)
    
    if not corresponding_files_list:
        print("There don't seem to be any corresponding files matching \'{0}\' in directory {1} ...! file_path set to None".format(original_name, directory))
        return None
    else:
        
        # In case there is >1 file per data image, filter the list by name_filter (include wildcards in argument if necessary; you must include the extension if not using a wildcard to finish), and (assuming we just have 1 now), get the 1st in the list
        if len(corresponding_files_list) > 1:
            print("  corresponding files found: : {0}".format(corresponding_files_list))
        file = fnmatch.filter(corresponding_files_list, '*' + name_filter + '*')[0]
        file_path = os.path.normpath(os.path.join(file))
        if verbose:
            print("  File corresponding to '{0}' found: {1}".format(original_name, file_path))
        
        if path_only is False:
            return nib.load(file_path)
        else:
            return file_path
        # End of get_corresponding_file() definition

def create_temp_directories(output_directory, temp_name="temp"):
    
    temp_directory = os.path.normpath(os.path.join(output_directory, temp_name))
    
    check_create_directories([output_directory, temp_directory])
    
    return temp_directory
    # End of create_temp_directories() definition
    
def check_create_directories( directory_list ):
    """Recursively creates directories input as a list."""
    # There are other ways; see http://stackoverflow.com/questions/273192/check-if-a-directory-exists-and-create-it-if-necessary
    
    for dir in directory_list:
        ndir = os.path.normpath(dir)
        if not os.path.exists(ndir):
            os.makedirs(ndir)
    # End of check_create_directories() definition
    
def check_for_output( file_path ):
    """Check if a file already exists."""
    if os.isfile():
        return True
    else:
        return False
    # End check_for_output() definition
    
def delete_files( file_list ):
    """Delete a list of files."""
    if type(file_list) is str:
        file_list = [file_list]
    for file_path in file_list:
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            print("  {0} does not exist, so cannot be deleted.".format(file_path))
    # End delete_files() definition
    
def copy_list_of_files(file_names_list, from_directory, to_directory, extension):
    import shutil
    
    if type(file_names_list) is str:
        file_names_list = [file_names_list]
    
    for file_name in file_names_list:
        files_list = sorted(glob.glob(os.path.join(from_directory, '*' + file_name + '*' + extension)))
        for counter, file in enumerate(files_list):
            full_file_name = os.path.basename(file)
            shutil.copy2(file, to_directory)
            if os.path.isfile(os.path.normpath(os.path.join(to_directory, full_file_name))):
                print("  successfully copied {3} / {4}, {0} from {1} to {2} ...".format(full_file_name, from_directory, to_directory, counter+1, len(files_list)))
    
def copy_files( from_directory, to_directory, file_names_list=None, csv_path=None, csv_cols=[0, 1], csv_group_filter=None, extension='.nii*', csv_name_filter=None ):
    """Copy a group of files from one directory to another.
    
    file_names_list should be a list of strings by which the files in from_directory are filtered.
    
    Alternatively, if file_names_list is left empty, provide a .CSV file containing file_names_list and groups, as well as a list of relevant columns csv_cols=[0,1] corresponding, respectively, to the file_names_list and the grouping columns, plus csv_group_filter, by which the grouping column should be filtered.
    
    So, given a .CSV with genotypes, you could copy all the 'wildtype' files from one dir to another.
    
    """
    
    # If the second dir doesn't exist, create it
    check_create_directories([to_directory])
    
    # Get list of files to copy
    if file_names_list is not None:
        copy_list_of_files([file_names_list], from_directory, to_directory, extension)
    elif csv_path is not None:
        
        group_names_list, _ = get_files_from_csv_by_group(csv_path, csv_cols[0], csv_cols[1], csv_group_filter, csv_name_filter)
        
        copy_list_of_files(group_names_list, from_directory, to_directory, extension)

    else:
        print("  Please provide either a file_names_list, or a .CSV with appropriate parameters.")
    # End copy_files() definition
    
    
def loop_csv( csv_path, input_directory, function, out_col_name='Values', name_column=0, group_column=1, csv_group_filter=None, extension='*', **kwargs ):
    """Opens or creates a .CSV with the given csvpath, and cycles through corresponding files in input_directory, performing function on each.
    
    Function should return a single value, to be entered into the corresponding row and given column.
    
    The existing .CSV is overwritten.
    
    """
    
    # file_path_list, csv_indices = get_existing_files_from_csv(csv_path, input_directory, name_column, group_column, csv_group_filter, extension)
    df, csv_path, file_path_list, csv_indices = check_create_csv(csv_path, input_directory, new_cols=[out_col_name], name_column=name_column, group_column=group_column, csv_group_filter=csv_group_filter, extension=extension)
    
    print("  Processing {0} files: \n   - - -\n   \t{1}\n   - - -\n  ...".format(len(file_path_list), '\n\t'.join([ str(item) for item in file_path_list ])))
    
    fn_name = function.__name__
    
    for counter, file_path in enumerate( file_path_list ):
        print "- - - - - - - - - - - \n  Applying function '{3}' to {0} / {1}: {2} ...".format((counter + 1), len(file_path_list), file_path, fn_name)
        
        # Get the file path
        file_path = os.path.normpath(os.path.join(file_path))
        
        # Get the row index of the (existing) .CSV to save the result
        idx = csv_indices[counter]
        
        df.ix[idx,out_col_name] = function(file_path, **kwargs)
        
        print("  Now DataFrame looks like this: \n{0}".format(df))
            
    df.to_csv(csv_path, index=False)
    # End loop_csv() definition

def loop_and_path( file_path_list, function=None, output_directory=None, mask_directory=None, out_name_prepend='', out_name_append='', out_ext='.nii.gz', **kwargs ):
    """loop_and_save, but pass the input path and output path to the function.
    
    ** Probably needs a lot of work.
    
    The function can accept masks, which are retrieved from mask_directory (if specified) using get_corresponding_file.
    
    """
    
    # import multiprocessing
    
    print("  Processing {0} files: \n   - - -\n   \t{1}\n   - - -\n  ...".format(len(file_path_list), '\n\t'.join([ str(item) for item in file_path_list ])))
    
    output_file_path_list = []
    
    for counter, file_path in enumerate( file_path_list ):
        print "- - - - - - - - - - - \n  Applying function '{3}' to {0} / {1}: {2} ...".format((counter + 1), len(file_path_list), file_path, function.__name__)
        
        # Get the file name and path
        file_path = os.path.normpath(os.path.join(file_path))
        file_name = (os.path.basename(file_path)).split(os.extsep)[0]
        
        # If no output extension or directory is provided, default to those of the input file
        if out_ext is None:
            out_ext = '.' + '.'.join((os.path.basename(file_path)).split(os.extsep)[1:])  # join, otherwise we have a list of extensions, if extension is '.nii.gz'.
        if output_directory is None:
            output_directory = os.path.dirname(file_path)
        else:
            check_create_directories([output_directory])
            
        if mask_directory:
            mask_path = get_corresponding_file(mask_directory, file_name, path_only=True)
        else:
            mask_path = None
        
        # Check if output_filepath already exists
        output_filepath = os.path.normpath(os.path.join(output_directory, out_name_prepend + file_name + out_name_append + out_ext))
        if not os.path.isfile(output_filepath):
        
            if function is not None:
                if mask_path is not None:
                    output_filepath = function(file_path, output_filepath, mask_path, **kwargs)
                else:
                    output_filepath = function(file_path, output_filepath, **kwargs)
            
            output_file_path_list.append(output_filepath)
            
        else:
            print("  Output file, {0} already exists!".format(output_filepath))
            
    print("- - - - - - - - - - - \n  Saved {0} files: \n   - - -\n   \t{1}\n   - - -".format(len(output_file_path_list), '\n\t'.join([ str(item) for item in output_file_path_list ])))
    return output_file_path_list
    # End loop_and_path() definition
    
def load_nifti_save_star(args):
    print 'args are:', args
    print("unpacked args are: {0}".format(*args))
    sys.stdout.flush()
    # Must use ** to unpack a dict, but * seems to work here
    return load_nifti_save(*args)
    
def load_nifti_save(input, function=None, output_filepath=None, overwrite=False, **kwargs):
    """Given a file_path, load a NIfTI, perform a function with **kwargs, and save to output_filepath."""
    
    nifti = get_nifti(input)
    
    if not os.path.isfile(output_filepath) and overwrite is False:
    
        # This allows us to replicate nifti_gzip by passing no function name as an argument
        if function is not None:
            new_nifti = function(nifti, **kwargs)
        else:
            new_nifti = nifti
        
        output_filepath = quick_save_nifti(new_nifti, output_filepath, overwrite=overwrite)
        
    else:
        print("  Output file, {0} already exists!".format(output_filepath))
    return output_filepath
    
def loop_and_save( file_path_list, function=None, output_directory=None, output_subdirectory=None, out_name_prepend='', out_name_append='', out_ext='.nii.gz', parallel=False, **kwargs ):
    """Loop through a list of files, apply a function (with a given format) to each, and save with the specified name.
    
    The function must take at least a NIfTI object as input, and return a NIfTI as well. This function loads the NIfTI from the input file_path_list, and saves the result.
    
    Other keyword arguments may be passed to the function using **kwargs (key-worded inputs to this function, e.g. 'factor=2').
    
    For example:
        ds_files_list = loop_and_save( input_files_list, downsample_image, output_directory=temp_dir, out_name_append='_ds', factor=3 )
        gz_files_list = loop_and_save( input_files_list, nifti_gzip, output_directory= )
        _ = loop_and_save( input_files_list, print_nifti_data, output_directory='', data_function=np.mean)   # must set output_directory='' so the existing file isn't skipped.
        
    Could use get_files_list(dir) to input the list of files.
    
    The output file path will be in output_directory, named: out_name_prepend + input_name + out_name_append + out_ext. If out_ext is None, the input extension will be used.
    
    If output_directory is left as None, output_subdirectory can be used instead to quickly send to a subdirectory of the existing file's path.
    
    The default behaviour of loop_and_save([file_path_list]) (if you leave function=None and out_ext='.nii.gz') is to replicate nifti_gzip.
    
    To do: check downsample_image (and other fns which use the header, like erode and dilate) will still work.
    
    """
    
    print("  Processing {0} files: \n   - - -\n   \t{1}\n   - - -\n  ...".format(len(file_path_list), '\n\t'.join([ str(item) for item in file_path_list ])))
    
    output_file_path_list = []
    
    if function is not None:
        fn_name = function.__name__
    else:
        fn_name = 'None specified'
        
    if parallel:
        import multiprocessing
        pool = multiprocessing.Pool(np.int(multiprocessing.cpu_count() / 2))
    
    for counter, file_path in enumerate( file_path_list ):
        print "- - - - - - - - - - - \n  Applying function '{3}' to {0} / {1}: {2} ...".format((counter + 1), len(file_path_list), file_path, fn_name)
        
        if counter == 0 and parallel:
            TASKS = []
        
        # Get the file name and path
        file_path = os.path.normpath(os.path.join(file_path))
        file_name = (os.path.basename(file_path)).split(os.extsep)[0]
        
        # If no output extension or directory is provided, default to those of the input file
        if out_ext is None:
            out_ext = '.' + '.'.join((os.path.basename(file_path)).split(os.extsep)[1:])  # join, otherwise we have a list of extensions, if extension is '.nii.gz'.
            
        if output_directory is None:
            output_directory = os.path.dirname(file_path)
            if output_subdirectory is not None:
                output_directory = os.path.normpath(os.path.join(output_directory, output_subdirectory))
                
        
        # Check if output_filepath already exists
        output_filepath = os.path.normpath(os.path.join(output_directory, out_name_prepend + file_name + out_name_append + out_ext))
        if not os.path.isfile(output_filepath):
        
            if parallel:

                d = {'file_path': file_path, 'function': function, 'output_filepath': output_filepath}
                # Expanding **kwargs will create a separate dict; merge them here:
                d.update(**kwargs)
                TASKS.append(d)
                
            else:
                output_filepath = load_nifti_save(file_path, function, output_filepath, **kwargs)
            
        else:
            print("  Output file, {0} already exists!".format(output_filepath))
            
        output_file_path_list.append(output_filepath)

    if parallel:
        print("Tasks to be run in parallel: \n{0}\n...".format('\n\t'.join([str(row) for row in TASKS])))
        # Force print
        sys.stdout.flush()
        output_file_path_list = pool.map(load_nifti_save_star, TASKS)
        # output_file_path_list = load_nifti_save_star(TASKS)
        pool.close()
        pool.join()
            
    print("- - - - - - - - - - - \n  Saved {0} files: \n   - - -\n   \t{1}\n   - - -".format(len(output_file_path_list), '\n\t'.join([ str(item) for item in output_file_path_list ])))
    return output_file_path_list
    # End loop_and_save() definition
            
    
def loop_and_print( file_path_list, function, **kwargs ):
    """Loop through a list of files, apply a function (with a given format) to each, print the result of that function, and return a list of all the results.
    
    Much like loop_and_save.
    
    ** Incomplete **
    
    Should work with: report_header, report_volume, report_number_clusters, etc.
    
    Aim for simplicity, so the printed output is clear: that's the point. But the called functions should do the printing, as each will require a different format.
    
    The function must take a Numpy array as input, and return something printable. This function loads the NIfTI from the input file_path_list, gets the data part (the Numpy array), and appends the function's output to a list.
    
    Other keyword arguments may be passed to the function using **kwargs (key-worded inputs to this function, e.g. 'axis=1').
    
    For example:
        _ = loop_and_print( input_files_list, print_nifti_data, data_function=np.mean)
    
    """
    
    print("  Processing {0} files: \n   - - -\n   \t{1}\n   - - -\n  ...".format(len(file_path_list), '\n\t'.join([ str(item) for item in file_path_list ])))
    
    output_list = []
    output_dict = {}
    
    for counter, file_path in enumerate( file_path_list ):
        # print "- - - - - - - - - - - \n  Applying function '{3}' to {0} / {1}: {2} ...".format((counter + 1), len(file_path_list), file_path, function.__name__)
        
        # Get the file name and path
        file_path = os.path.normpath(os.path.join(file_path))
        file_name = (os.path.basename(file_path)).split(os.extsep)[0]
        
        nifti = nib.load(file_path)
        
        result = function(nifti, **kwargs)
        
        output_list.append(result)
        output_dict[file_name] = result
        
    # output_dict = {file_path_list[c] : output_list[c] for c in output_list}
    return output_dict
    # End loop_and_print() definition
    
def print_nifti_data( nifti, data_function ):
    """Perform any function on the data part of a NIfTI object (which is a Numpy array), print and return the output.
    
    E.g. print_nifti_data( nifti, np.std )  # to print the standard deviation.
    
    Works with loop_and_print(): E.g. loop_and_print(input_files_list, print_nifti_data, data_function=np.std)
    
    """
    nf_path = os.path.normpath(os.path.join(nifti.get_filename()))
    nf_name = (os.path.basename(nf_path)).split(os.extsep)[0]
    
    result = data_function( nifti.get_data() )
    
    print("  {0} : \n\t\t{1}".format(nf_name, result))
    
    return result
    # End print_nifti_data() definition
    
def nifti_gzip( nifti, output_filepath=None, delete_original=False, new_extension='.nii.gz' ):
    """Compress a NIfTI file.
    
    Works with loop_and_save(), but a better way would be to run loop_and_save([file_path]), as that function's defaults will end up replicating this function.
    """
    
    nf_path = os.path.normpath(os.path.join(nifti.get_filename()))
    
    if output_filepath is None:
        nf_name = os.path.basename(nf_path).split(os.extsep)[0]
        dir = os.path.dirname(nf_path)
        output_filepath = os.path.normpath(os.path.join(dir, nf_name + new_extension))
    
    if not os.path.isfile(output_filepath):
        nib.save(nifti, output_filepath)
    
    if delete_original is True:
        delete_files([nf_path])
        
    return nifti
    # End nifti_gzip() definition
    
def wait_for_files( directory, number, name_filter='*', extension='.nii*', n_seconds=15 ):
    """Wait for a given number of files matching certain criteria."""
    
    import time
    
    file_list = glob.glob(os.path.join(directory, name_filter + extension))
    num_files = len(file_list)
    
    while num_files < number:
        time.sleep(n_seconds)
        
        file_list = glob.glob(os.path.join(directory, name_filter + extension))
        new_num_files = len(file_list)
        
        if new_num_files > num_files:
            num_files = new_num_files
            
    time.sleep(n_seconds)
    print("  ... Waited until these {1} files appeared: {0}".format(file_list, num_files))
    # Force print
    sys.stdout.flush()
    # End wait_for_files() definition
    
def quick_save_nifti( nifti, save_path, overwrite=False, default_name='output.nii.gz' ):
    """Given a NIfTI-1 object and a save_path, checks whether the file already exists, saves the NIfTI if not.
    
    Creates directories recursively, if necessary.
    
    Returns save_path
    """

    save_path = os.path.normpath(save_path)
    save_directory = os.path.dirname(save_path)
    check_create_directories([save_directory])
    
    if os.path.isdir(save_path):
        save_path = os.path.normpath(os.path.join(save_path, default_name))
        print("  Given path is a directory; saving as default filename there: {0}".format(save_path))
    
    if not os.path.isfile(save_path):
        print("  Saving {0} ...".format(save_path))
        nib.save(nifti, save_path)
    else:
        print("  Output file, {0} already exists!".format(save_path))
        if overwrite is True:
            print("  Overwriting anyway ...")
            nib.save(nifti, save_path)
    
    return save_path
    # End quick_save_nifti() definition
    
def mass_rename( input_directory, file_name_filter="", replace_string="", extension='.nii*', output_directory='', prepend='', append=''):
    """In input_directory, for files with file_name_filter in their name, replace file_name_filter with replace_string, and prepend and append strings if desired.
    
    If files have periods in their names, this goes wrong. But you can use it to fix that.
    
    If output_directory is specified and is not the same as input_directory, a copy will be made at the new location.
    
    """
    
    if output_directory == '':
        output_directory = input_directory
    else:
        import shutil
    
    # Get list of files (use file_name_filter)
    files_list = sorted(glob.glob(os.path.join(input_directory, '*' + file_name_filter + '*' + extension)))
    
    print("Mass-renaming '{0}' to '{1}' in {2} files ...".format(file_name_filter, replace_string, len(files_list)))
    print files_list
    
    for counter, file in enumerate(files_list):
        print "- - - - - - - - - - - \n  Processing {0} / {1}: {2} ...".format((counter + 1), len(files_list), file)
        
        # Get the filename, removing path and 1+ extensions
        full_file_name = os.path.basename(file)
        
        # Get the input file
        file_path = os.path.normpath(os.path.join(file))
        
        # This done in case there is a period in the file_path above, which confuses things.
        new_full_file_name = full_file_name.replace(file_name_filter, replace_string)
        
        file_ext = (os.extsep).join(new_full_file_name.split(os.extsep)[1:])     # join each part of the extension with the '.' separator to form a string rather than a list
        
        new_file_name = new_full_file_name.split(os.extsep)[0]
        
        print("  extension will be: '.{0}', from: {1}".format(file_ext, new_full_file_name.split(os.extsep)[1:]))
        
        # Refit the original extension
        destination = prepend + new_file_name + append + '.' + file_ext
        
        if output_directory == input_directory:
            os.rename(file_path, os.path.normpath(os.path.join(output_directory, destination)))
        else:
            # If the output directory is different, assume we want to copy - rather than move - the file
            shutil.copyfile(file_path, os.path.normpath(os.path.join(output_directory, destination)))
        
        print("  renamed {0} --> {1}".format(full_file_name, destination))
    # End mass_rename() definition
    
# File Metadata
# -------------

def report_header( nifti ):
    """Print details from the header.
    
    Works with loop_and_print().
    
    """
    
    hdr = nifti.get_header()
    nf_path = os.path.normpath(os.path.join(nifti.get_filename()))
    nf_name = (os.path.basename(nf_path)).split(os.extsep)[0]
    
    print("* * * * * * * * * * * * * * * * * * * * * * * *\n")
    print("  Header details for {0} ...".format(nf_name))
    print("  volume shape: \t{0}".format(hdr.get_data_shape()))
    print("  datatype: \t{0}".format(hdr.structarr['datatype']))
    print("  qform code: \t{0}".format(hdr.structarr['qform_code']))
    print("  sform code: \t{0}".format(hdr.structarr['sform_code']))
    print("  sform: \n  {0}".format(hdr.get_sform()))
    print("  qform: \n  {0}".format(hdr.get_qform()))
    print("  pixdim: \t{0}".format(hdr.structarr['pixdim']))
    print("  hdr.structarr: \t{0}".format(hdr.structarr))
    print("\n* * * * * * * * * * * * * * * * * * * * * * * *")
    
    return hdr.structarr
    # End of report_header() definition

def normalise_header( input ):
    """Strip all but the resolution and units (set to mm) from a NIfTI-1 header.
    Once done, you might need to perform orientation correction.
    See http://nipy.org/nibabel/gettingstarted.html for notes.
    
    """
    
    data, hdr = check_get_nifti_data(input)
    
    # Create brand new header
    hdr_new = nib.Nifti1Header()

    # set the units
    hdr_new.set_xyzt_units(('mm'))

    # Set some new values
    # sets 2nd, 3rd and 4th values; the last number in the list is excluded
    hdr_new['pixdim'][1:4] = hdr['pixdim'][1:4]

    # Translation elements are the offset from the origin and can be 0 (but in Matlab, are set to the pixdim values, I think, causing confusion).
    # Diagonals
    #    - these must be set to the respective pixdim values; if they're set to 1, registration to other images will fail
    hdr_new['srow_x'][0] = hdr_new['pixdim'][1]
    hdr_new['srow_y'][1] = hdr_new['pixdim'][2]
    hdr_new['srow_z'][2] = hdr_new['pixdim'][3]
    
    # hdr_new.structarr['qform_code'] = 1
    # hdr_new.structarr['sform_code'] = 1
    
    # Change the default header dtype if necessary (problems are caused if there's a mismatch)
    hdr_new.set_data_dtype( input.get_data_dtype() )
    
    return nib.Nifti1Image(data, hdr_new.get_sform(), header=hdr_new)
    # End of normalise_header() definition
    
def copy_header( nifti_1, nifti_2 ):
    """Apply the header from nifti_2 to nifti_1. Returns nifti_1 with the header of nifti_2.
    """
    return nib.Nifti1Image(nifti_1.get_data(), nifti_2.get_header().get_sform(), header=nifti_2.get_header())
    # End copy_header() definition
    
def convert_mm_to_voxels( mm_input, hdr=None ):
    """Returns the integer number of voxels closest to the mm input, for this particular image header.
    
    If no header is supplied or mm_input is negative, assume we have voxels anyway and return the absolute nearest integer.
    
    """

    # if mm_input is negative, we've actually been given voxels (as per NiftyReg's handling of input arguments), so don't perform a conversion
    if mm_input < 0:
        # also need to handle being given negative decimal numbers: need to round
        voxels = int(round(abs(mm_input)))
        # print "  No mm --> voxel conversion required; using {0} voxels ...".format(voxels)
    elif hdr is None:
        print("No header given for conversion from mm to voxels; returning absolute value of input.")
        voxels = abs(mm_input)
    else:
        
        # if len(mm_input) is 1:
            
        # Get the first dimension's resolution (assume isotropic for now)
        resn = hdr['pixdim'][1]
        voxels_per_mm = 1. / resn
        voxels = mm_input * voxels_per_mm
        
        # else:
        
            # resn = np.asarray(hdr['pixdim'][1:4])
            # voxels_per_mm = 1. / resn
            # voxels = tuple([int(round(x)) for x in mm_input * voxels_per_mm])
        
        print("  Given Header and desired mm input {0}, voxels are: {1}".format(mm_input, voxels))
    return voxels
    
def get_centre_of_mass( input, hdr=None, do_weighted=True, use_resolution=True ):
    """Returns the centre of mass (centroid) of a 3D NIfTI object, as a Numpy array.
    
    ** NB check this works as expected. **
    
    If do_weighted is True (default yes), voxels are weighted by their value (intensity).
    If use_resolution is True (default yes), position is in the same units as the header pixdims (likely mm).
    
    """
    
    data, hdr = check_get_nifti_data(input, hdr)
    
    if use_resolution is True:
        pixdims = hdr.structarr['pixdim'][1:4]
    else:
        pixdims = [1,1,1]
    
    # indices of non-zero elements
    rows, cols, slices = np.ndarray.nonzero(data)
    
    # Add 1 to avoid beginning at 0.
    rows = rows + 1
    cols = cols + 1
    slices = slices + 1
    
    total_sum = np.ndarray.sum(data)
    
    # linear list of actual elements (must be same length as each of rows, cols, slices)
    flat_data = np.ndarray.ravel(data)
    flat_data = flat_data[flat_data != 0]
    
    # Unweighted centre of mass
    com = np.array([pixdims[0] * np.mean(rows), pixdims[1] * np.mean(cols), pixdims[2] * np.mean(slices)])
    
    if do_weighted is True:
        # Weight CoM by voxel values
        com[0] = pixdims[0] * sum(rows * flat_data) / total_sum
        com[1] = pixdims[1] * sum(cols * flat_data) / total_sum
        com[2] = pixdims[2] * sum(slices * flat_data) / total_sum
        
    print("  Image centre of mass is at X = {0}, Y = {1}, Z = {2}".format(com[0], com[1], com[2]))
    
    return com
    # End of get_centre_of_mass() definition
    
def get_peak_location( input, hdr=None, mask=None, minimum=False ):
    """Given an ND NIfTI image (input), returns the full coordinates (ND) of the maximum (or minimum) voxel value, optionally within a (binarised) mask.
    
    """
    
    data, hdr = check_get_nifti_data(input, hdr)
    
    if mask is not None:
        data = apply_mask(data, mask)
        
    if minimum is False:
        linear_index = np.argmax(data)
    else:
        linear_index = np.argmin(data)
    
    return np.unravel_index(linear_index, data.shape)
    # End of get_peak_location() definition
    
def get_nifti_object_volumes( nifti_files_list ):
    """Returns a list of lists containing object volumes in each NIfTI file.
    
    Output
    ------
    object_volumes[0] is itself a list of volumes of all the objects in the 1st NIfTI file in nifti_files_list.
    
    Sorting occurs outside this function; output will be sorted in the order of nifti_files_list. Individual objects, however, will be sorted by volume in descending order, as per label_concomp().
    
    """
    
    # List of empty lists (thanks, http://span.ece.utah.edu/python-nested-lists)
    object_volumes = map(list, [[]] * len(nifti_files_list))
    
    print ("Processing %i NIfTI files ..." % len(nifti_files_list) )
    print nifti_files_list
    
    for counter, nifti_file in enumerate(nifti_files_list):
        print " Processing {0} / {1}: {2} ...".format((counter + 1), len(nifti_files_list), nifti_file)
        
        # Get the filename, removing path and 1+ extensions
        nf_name = os.path.basename(nifti_file.split(os.extsep)[0])
        
        # Get the input file
        nf_path = os.path.normpath(os.path.join(nifti_file))
    
        print ("  Loading %s ..." % nf_path )
        
        # Get the header from the input file
        img = nib.load(nf_path)
        hdr = img.get_header()
        
        # Get the 3D data
        data = img.get_data()
        
        # Re-label connected components (binarises first) (label 1 is largest, 2 the next-largest, etc.)
        labeled_image, num_concomp = label_concomp(data)
        print("Image {0}, {1} has {2} objects.".format((counter + 1), nf_name, num_concomp))
        
        for label_num in range(1, num_concomp+1):
        
            print("  Getting properties of object {0} of {1} ...".format(label_num, num_concomp))
            
            object = binarise(labeled_image == label_num)
            voxvol, physicalvol = get_binary_volume(object, hdr)
            object_volumes[counter].append(physicalvol)
    
    return object_volumes
    # End of get_nifti_object_volumes() definition
    
def add_nifti_list(input_path_list):
    """Add NIfTIs given as a list of image file paths, into the space of the first. Returns the total NIfTI (does not save).
    
    Images must be the same shape.
    
    """
    
    for counter, input_path in enumerate(input_path_list):
        print("  Adding file {0} / {1} ...".format(counter+1, len(input_path_list)))
        file_path = os.path.normpath(os.path.join(input_path))
        if counter is 0:
            total_nii = nib.load(file_path)
        else:
            total_nii = add(total_nii, nib.load(file_path))
            
    return total_nii
    # End add_nifti_list() definition
    
def average_nifti_list(input_path_list):
    """Averages NIfTIs given as a list of image file paths, into the space of the first. Returns the average NIfTI (does not save).
    
    Images must be the same shape.
    
    """
    
    return divide(add_nifti_list(input_path_list), np.float(len(input_path_list)))
    # End average_nifti_list() definition
    
def median_nifti_list(input_path_list):
    """Finds the voxel-wise median for a group of NIfTI files.
    Images must be the same shape.
    The output image header will be based upon the first NIfTI in the list.
    """
    
    ex_nii = nib.load(input_path_list[0])
    stack_dim = len(ex_nii.get_data().shape)
    median_data = np.median(mmfn.stack_nifti_list(input_path_list).get_data(), stack_dim)
    
    return nib.Nifti1Image(median_data, ex_nii.get_affine(), ex_nii.get_header())
    # End median_nifti_list() definition
    
def stack_nifti_list(input_path_list):
    """Based upon the first NIfTI in a list of files (given file paths), stacks NIfTIs in the N+1th dimension.
    
    All input NIfTIs must be the same shape.
    
    NB 2D inputs are a special case.
    
    """
    
    for counter, input_path in enumerate(input_path_list):
        print("  File {0} / {1} ...".format(counter+1, len(input_path_list)))
        file_path = os.path.normpath(os.path.join(input_path))
        
        if counter is 0:
            shape_1 = nib.load(file_path).shape
            stacked_nii = nib.load(file_path)
            
        else:
            if len(shape_1) == 2:
                # 2D images can use this function
                stacked_nii = concatenate_niftis(stacked_nii, nib.load(file_path))
                
            else:
                # If we have ND images where N>=3, create the 4D image and then begin stacking
                nii_i = nib.load(file_path)
                
                if counter is 1:
                    # Create an N+1D image
                    data_0 = stacked_nii.get_data()
                    hdr_0 = stacked_nii.get_header()
                    new_axis_num = len(shape_1)
                    
                    data_1 = nii_i.get_data()
                    
                    stacked_data = np.concatenate((data_0[..., np.newaxis], data_1[..., np.newaxis]), axis=new_axis_num)
                    
                    stacked_nii = nib.Nifti1Image(stacked_data, hdr_0.get_sform(), hdr_0)
                else:
                    # Already have the N+1D image; just stack into it
                    stacked_nii = concatenate_niftis_nd(stacked_nii, nii_i)
            
    return stacked_nii
    # End stack_nifti_list() definition
    
def concatenate_niftis(nii_1, nii_2):
    """Stack two 2D NIfTIs in the 3rd dimension.
    
    For ND NIfTIs, use concatenate_niftis_nd().
    
    """
    
    # data_1, hdr_1 = check_get_nifti_data(input_1)
    # data_2, hdr_2 = check_get_nifti_data(input_2)
    
    hdr_1 = nii_1.get_header()
    data_pointer_1 = np.asarray(nii_1.dataobj)
    data_pointer_2 = np.asarray(nii_2.dataobj)
    
    # return nib.Nifti1Image(np.dstack((data_1, data_2)), hdr_1.get_sform(), hdr_1)
    return nib.Nifti1Image(np.dstack((data_pointer_1, data_pointer_2)), hdr_1.get_sform(), hdr_1)
    # End concatenate_niftis() definition
    
def concatenate_niftis_nd(nii_1, nii_2):
    """Stack an ND NIfTI (nii_2) onto the end of the N+1D NIfTI nii_1.
    
    So if you're stacking a list of 3D NIfTIs, this assumes you already have the 4D NIfTI to begin with. Use stack_nifti_list() to call this function.
    
    """
    
    shape_1 = nii_1.shape
    new_axis_num = len(shape_1) -1   # Don't need to +1 as 1st axis is 0.
    
    hdr_1 = nii_1.get_header()
    
    data_1 = nii_1.get_data()
    data_2 = nii_2.get_data()
    
    new_data = np.concatenate((data_1, data_2[..., np.newaxis]), axis=new_axis_num)
    
    return nib.Nifti1Image(new_data, hdr_1.get_sform(), hdr_1)
    # End concatenate_niftis_nd() definition
    
def add( input_1, input_2=None, hdr_1=None, hdr_2=None, verbose=False ):
    """Adds two input images (either NIfTI objects or the 3D data).
    
    In the case of headers being included, the header of input_1 is preserved.
    Images must be the same size.
    
    """
    
    data_1, hdr_1 = check_get_nifti_data(input_1, hdr_1)
    
    if input_2 is not None:
        
        data_2, hdr_2 = check_get_nifti_data(input_2, hdr_2)
    
        if verbose:
            print("  Summing image data ...")
        
        data_out = np.add(data_1, data_2)
        
    else:
        # To make loops easier
        data_out = data_1

    if hdr_1:
        return nib.Nifti1Image(data_out, hdr_1.get_sform(), header=hdr_1)
    else:
        return data_out
    # End of add() definition
    
def subtract( input_1, input_2, hdr_1=None, hdr_2=None, verbose=False ):
    """Subtract input_2 from input_1 (either NIfTI objects or the 3D data).
    
    In the case of headers being included, the header of input_1 is preserved.
    Images must be the same size.
    
    """
    
    data_1, hdr_1 = check_get_nifti_data(input_1, hdr_1)
    data_2, hdr_2 = check_get_nifti_data(input_2, hdr_2)
    
    if verbose:
        print("  Subtracting image data ...")
    
    data_out = np.subtract(data_1, data_2)
    
    if hdr_1:
        return nib.Nifti1Image(data_out, hdr_1.get_sform(), header=hdr_1)
    else:
        return data_out
    # End of subtract() definition
    
def multiply( input_1, factor=1.0, hdr_1=None, hdr_2=None, verbose=False ):
    """Multiplies an input image with either a NIfTI object or a float.
    
    factor can either be a float, a NIfTI object, or just 3D data the same shape as that of input_1.
    
    In the case of headers being included, the header of input_1 is preserved.
    Images must be the same size.
    
    """
    
    data_1, hdr_1 = check_get_nifti_data(input_1, hdr_1)
    
    if verbose:
        print("  Multiplying image data ...")
    
    if type(factor) is np.float:
        data_out = data_1 * factor
    else:
        data_2, hdr_2 = check_get_nifti_data(factor, hdr_2)
        
        data_out = np.multiply(data_1, data_2)
    
    if hdr_1 is not None:
        return nib.Nifti1Image(data_out, hdr_1.get_sform(), header=hdr_1)
    elif hdr_2 is not None:
        # If input_1 is a float (for example) but factor (input_2) is a NIfTI image, use that one's header!
        return nib.Nifti1Image(data_out, hdr_2.get_sform(), header=hdr_2)
    else:
        return data_out
    # End of multiply() definition   
    
def divide( input_1, factor=1.0, hdr_1=None, hdr_2=None, verbose=False ):
    """Divide an input image by either a NIfTI object or a float.
    
    factor can either be a float, a NIfTI object, or just 3D data the same shape as that of input_1.
    
    In the case of headers being included, the header of input_1 is preserved.
    Images must be the same size.
    
    """
    
    data_1, hdr_1 = check_get_nifti_data(input_1, hdr_1)
    
    if verbose:
        print("  Dividing image data ...")
    
    # what if it's an int?
    if type(factor) is np.float:
        data_out = data_1 / factor
    else:
        data_2, hdr_2 = check_get_nifti_data(factor, hdr_2)
        
        data_out = np.divide(data_1, data_2)
    
    if hdr_1 is not None:
        return nib.Nifti1Image(data_out, hdr_1.get_sform(), header=hdr_1)
    elif hdr_2 is not None:
        # If input_1 is a float (for example) but factor (input_2) is a NIfTI image, use that one's header!
        return nib.Nifti1Image(data_out, hdr_2.get_sform(), header=hdr_2)
    else:
        return data_out
    # End of divide() definition
    
def difference_from_mean_difference( image_1, image_2, mask, output_path ):
    """For a pair of images, calculate the mean of the absolute difference image (within a mask), and subtract this value from the difference image at every voxel.
    
    Highlights where a pair of images are *most* different from one another.
    
    Intensities in the result image are arbitrary, however. Input images should be of the same modality.
    
    """
    
    data_1, hdr_1 = mmfn.check_get_nifti_data(image_1)
    data_2, hdr_2 = mmfn.check_get_nifti_data(image_2)
    data_mask, _ = mmfn.check_get_nifti_data(mask)
    
    data_2 = mmfn.remove_nans(data_2)
    
    diff = mmfn.absolute(mmfn.subtract(data_1, data_2))
    mean_diff = np.ndarray.mean(mmfn.constrain_to_mask(diff, data_mask))
    
    diffdiff = mmfn.apply_mask(mmfn.absolute(mmfn.subtract(diff, mean_diff)), data_mask)
    
    dd_nii = nib.Nifti1Image(diffdiff, hdr_1.get_sform(), hdr_1)
    mmfn.quick_save_nifti(dd_nii, output_path)
    # End of difference_from_mean_difference() definition
    
def binary_intersection( input_1, input_2 ):
    """Return a NIfTI containing the binary intersection of NIfTIs 1 and 2."""
    
    data_1, hdr_1 = check_get_nifti_data(input_1)
    data_2, hdr_2 = check_get_nifti_data(input_2)
    
    intersect_data = binarise(data_1) * binarise(data_2)
    
    if hdr_1:
        return nib.Nifti1Image(intersect_data, hdr_1.get_sform(), hdr_1)
    else:
        return intersect_data
    # End binary_intersection() definition
    
def lconcomp( input ):
    """Returns a binary ndarray the same shape as input, containing only the largest 6-connected component.
    
    Works with NIfTI-1 images as well as 3D data input.
    
    """
    
    labeled_image, num_concomp = label_concomp( input, max_to_get=1 )
    
    print("  Getting largest connected component only ...")
    
    labeled_data, hdr = check_get_nifti_data(labeled_image)
    
    largest_object = binarise(labeled_data == 1)
    
    # If we were given a NIfTI-1 image, return a NIfTI; otherwise return just a data cuboid.
    if hdr:
        return nib.Nifti1Image(largest_object, hdr.get_sform(), header=hdr)
    else:
        return largest_object
    # End of lconcomp() definition
    
def label_concomp( input, max_to_get=1000 ):
    """Label each 6-connected object in the image by volume. Returns labels in increasing volume: 0 is background; 1 is the largest, 2 the next-largest, etc.
    
    If more than one object has the same volume, sequence is determined by the order of np.sort.
    
    If max_to_get=N (default is 1000, which seems sensible), only the largest N objects are labelled. Useful for saving time if you only want the very largest N object(s).
    
    """

    # If hdr is None, we've just been given a data structure and that is all that should be returned. If hdr is not None, we've been given a NIfTI-1 image as the input, and we should return one.
    data, hdr = check_get_nifti_data(input)
    
    print("  Labelling connected components ...")

    # Ensure we're working with binary data (convert to int afterwards, just in case)
    mask = data > 0
    mask = mask.astype(int)

    # This is the travelling 'mask' which covers each voxel and is used to search for connections. 3,1 is a 3D cross, ie, 6-connected component. 3,2 includes the diagonals and only excludes the 8 corners, ie, 18-connected component. The latter is more inclusive.
    structure = ndimage.generate_binary_structure(3,1)
    
    # Label each object
    label_im, numpatches = ndimage.label(mask, structure)
    
    print("  {0} 6-connected objects labelled ...".format(numpatches))
    
    if numpatches > 0:
    
        print("  At most, the largest {0} objects will be returned ...".format(max_to_get))
        
        # Sum the values under each object in the original binary mask, thus getting the size
        # So, numpatches[0] has size == sizes[0]
        sizes = ndimage.sum(mask, label_im, range(1, numpatches+1))
        
        sorted_labels = np.zeros_like(label_im)
        patch_list = range(1, numpatches+1)
        sort_idx = np.argsort(sizes)[::-1]
        
        for counter, size in enumerate(np.sort(sizes)[::-1]):
        
            # Be more efficient
            if (counter + 1) > max_to_get:
                break
            else:
                sorted_labels[label_im == patch_list[sort_idx[counter]]] = counter + 1
            
            # Ignoring small objects would mess with numpatches count
            # if size > 1:
                # print patch_list[sort_idx[counter]]
            
        # Return a sensible number of objects, to save time. This must be _after_ we've got all the sizes, as initially they're not in size order
        if numpatches > max_to_get:
            numpatches = max_to_get
            
    else:
        print("Image is empty; no binary connected components found!")
        sorted_labels = mask
        
    # If we were given a NIfTI-1 image, return a NIfTI; otherwise return just a data cuboid.
    if hdr:
        return nib.Nifti1Image(sorted_labels, hdr.get_sform(), header=hdr), numpatches
    else:
        return sorted_labels, numpatches
    # End of label_concomp() definition
    
def get_roi_about_facet( facet, exclude ):
    """Test voxels within the cuboid defined by a 3D triangular facet.
    
    Inputs
    ------
    facet:  3 x 3 array of vertex coordinates: [[x,y,z], [x,y,z], [x,y,z]]
    
    exclude: 3D volume wherein voxels that should be excluded from testing == 1.
    
    """
    
    # Get the coordinates of the simplex vertices
    # v0 = point_coords[hull.simplices[:,0],:]
    # v1 = point_coords[hull.simplices[:,1],:]
    # v2 = point_coords[hull.simplices[:,2],:]
    
    # Each row of [v0, v1, v2] gives the point_coords of the vertices
    # for simplex_coordinates in [v0, v1, v2]:
    
    # Get the max and mins of the cuboid bounding the triangular simplex
    # xyz_min = np.min([v0, v1, v2], 0)
    # xyz_max = np.max([v0, v1, v2], 0)
    xyz_min = np.min(facet, 0)
    xyz_max = np.max(facet, 0)
    
    # Exclude voxels outside this cuboid. Subtract and add 1 to be slightly conservative.
    xyz_min = xyz_min - 1
    xyz_max = xyz_max + 1
    
    # Avoid exceeding boundaries
    for i in xrange(3):
        # if xyz_min[0,i] < 0: xyz_min[0,i] = 0
        if xyz_min[i] < 0: xyz_min[i] = 0
        # if xyz_max[0,i] >= exclude.shape[i]: xyz_max[0,i] = exclude.shape[i] - 1
        if xyz_max[i] >= exclude.shape[i]: xyz_max[i] = exclude.shape[i] - 1
    
    # Create a copy of exclude to modify separately
    exclude_here = np.empty_like(exclude)
    exclude_here[:] = exclude
    exclude_here[:xyz_min[0],:,:] = 1
    exclude_here[:,:xyz_min[1],:] = 1
    exclude_here[:,:,:xyz_min[2]] = 1
    exclude_here[xyz_max[0]:,:,:] = 1
    exclude_here[:,xyz_max[1]:,:] = 1
    exclude_here[:,:,xyz_max[2]:] = 1
    
    # Get indices of zero-voxels in exclude: these are the voxels to test!
    vox_to_test = (exclude_here == 0).astype(int)
    coords_to_test = np.asarray(np.ndarray.nonzero(vox_to_test)).T
    # print("{0} voxels to test for this simplex ...".format(np.count_nonzero(vox_to_test)))
    
    # hull_mask = test_voxels_within_hull( hull, hull_mask, normed_normals, normal_directions, coords_to_test )
    
    # After the above loop is complete, we can add the entire cuboid to exclude
    # exclude[hull_mask > 0] = 1
    
    return coords_to_test
    # End of get_roi_about_facet() definition
    
def get_binary_hull( data, min_volume=100 ):
    """Return the 3D binary hull around a 3D set of points.
    
    Input may have any number of objects; in the output they should all be connected within the hull.
    
    References
    ----------
    [1] http://www.mathworks.co.uk/matlabcentral/answers/9298
    
    """
    # Matlab's convhulln breaks if there are fewer than about 4 voxels. I believe there is a similar issue here. Set an arbitrary minimum object volume; return a blank image if the minimum isn't exceeded
    if np.count_nonzero(data) >= min_volume:
    
        from scipy.spatial import ConvexHull
        
        # Ensure we're working with binary data (convert to int afterwards, just in case)
        mask = (data > 0).astype(int)
        mask = fill(mask)
        
        hull_mask = mask
        
        # point_coords contains x,y,z coords
        # Transpose so point_coords[:,0] is x, [:,1] is y, etc.
        point_coords = np.asarray(np.ndarray.nonzero( mask )).T
    
        hull = ConvexHull(point_coords)
        
        # simplices are similar to Matlab's convhulln output K: they're indices of point_coords. Each row of simplices is thus a facet - a triangular plane. row 1, simplices[1], contains 3 elements: references to the rows of point_coords, each of which contain 3D coordinates. So point_coords[simplices[1]] gives 3 3D coordinates: corners of the triangle. This is very similar to Matlab's convhulln output, and a great explanation of that is in [1].
        
        # For the moment, see Matlab niftiHull_everything.m for a more complete explanation of what I'm doing here.
        
        # num voxels: np.shape(point_coords)[0])
        print("Convex hull facets: {0}".format(len(hull.simplices)))
        
        # Compute normal vectors to each facet, using two facet edges as the vectors
        print("Computing all normal vectors ...")
        vec1 = point_coords[hull.simplices[:,0],:] - point_coords[hull.simplices[:,1],:]
        vec2 = point_coords[hull.simplices[:,0],:] - point_coords[hull.simplices[:,2],:]
        normals = np.cross(vec1, vec2)
        
        # Normalise each normal vector to unit length
        # Matlab: normednormals = bsxfun(@times, normals, 1 ./ sqrt( sum( normals .^2, 2 )));
        # The transposing here is just to get the multiplication to work properly, as everything after the first '*' below produces a 1D row vector. The final result is again transposed so it looks like the normals input.
        normed_normals = (normals.T * 1 / np.sqrt(np.sum(normals**2, 1))).T
        
        # Test to check the above produces the same as Matlab:
        # Matlab:
        #   normals = [1,2,3;4,5,6]
        #   normednormals = bsxfun(@times, normals, 1 ./ sqrt( sum( normals .^2, 2 )))
        # Python:
        #   normals = np.array([[1,2,3], [4,5,6]])
        #   normed_normals = (normals.T * 1 / np.sqrt(np.sum(normals**2, 1))).T
        
        # Find centralish coordinate
        central_point = np.mean(point_coords, 0)
        print("central point: {0}".format(central_point))
        
        # Find vectors from one point in each facet to this central point. We know this is inward-pointing as it is an outer point minus a central point.
        print("Computing inward vectors ...")
        inward_vectors = point_coords[hull.simplices[:,0],:] - central_point
        
        # The dot product between two vectors projects one vector onto another. If the result is negative, the two vectors are pointing in opposite directions. We need to do this along the 2nd dimension (along columns rather than down them, so we get a value for each row). Matlab: D = dot( normednormals, inwardvectors, 2 )
        # Thanks, http://stackoverflow.com/questions/6229519/numpy-column-wise-dot-product
        normal_directions = np.array([np.dot(normed_normals[i,:], inward_vectors[i,:]) for i in xrange(normed_normals.shape[0])])
        
        # Now comes the intensive testing of each voxel within the image, to see whether it is inside the hull. We can use various speedup techniques.
        
        # First, find voxels to exclude from testing.
        # Exclude voxels already in the mask
        # (Yes, making a copy of mask like this is important)
        print("Excluding external voxels ...")
        exclude = np.empty_like(mask)
        exclude[:] = mask
        
        # Crop region of interest to tight cuboid bounding box around the voxels. Subtract and add 1 to be slightly conservative.
        xmin = np.min(point_coords[:,0]) - 1
        ymin = np.min(point_coords[:,1]) - 1
        zmin = np.min(point_coords[:,2]) - 1
        xmax = np.max(point_coords[:,0]) + 1
        ymax = np.max(point_coords[:,1]) + 1
        zmax = np.max(point_coords[:,2]) + 1
        # Avoid exceeding image boundaries
        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0
        if zmin < 0: zmin = 0
        if xmax >= mask.shape[0]: xmax = mask.shape[0] - 1
        if ymax >= mask.shape[1]: ymax = mask.shape[1] - 1
        if zmax >= mask.shape[2]: zmax = mask.shape[2] - 1
        # Set everything outside this to 1 in exclude.
        
        # print xmin, xmax, ymin, ymax, zmin, zmax
        
        exclude[:xmin,:,:] = 1
        exclude[:,:ymin,:] = 1
        exclude[:,:,:zmin] = 1
        exclude[xmax:,:,:] = 1
        exclude[:,ymax:,:] = 1
        exclude[:,:,zmax:] = 1
        
        # Speedup: we can selectively test voxels around the edge of the hull, by finding the cuboids defined by each triangular simplex. For each simplex, test all voxels within the surrounding cuboid (will be slower for larger simplices). Once all respective cuboids have been tested, we should have a hollow shape, which we can fill, thereby omitting to test all the internal voxels.
        
        print("Testing voxels around each simplex ...")
        
        facet_single_point = point_coords[hull.simplices[:,0],:]
        
        for simplex_counter, row in enumerate(hull.simplices):
            
            simplex_vertices = point_coords[row]
            
            coords = get_roi_about_facet( simplex_vertices, exclude )
            
            if len(coords) > 0:
                # print("{1} total remaining voxels: testing {0} around simplex {2} of {3} ...".format(len(coords), np.count_nonzero((exclude == 0).astype(int)), simplex_counter+1, len(hull.simplices)))
                # print("... testing voxels around simplex {0} of {1} ...".format(simplex_counter+1, len(hull.simplices)))
                hull_mask = test_voxels_within_hull( facet_single_point, hull_mask, normed_normals, normal_directions, coords )
            
                # Avoid re-testing the same coordinates (whole cuboid)
                exclude[coords[:,0],coords[:,1],coords[:,2]] = 1
        
        # Fill the (hopefully now hollow) shape
        hull_mask = fill(hull_mask)
        
        # Exclude these voxels from future tests
        exclude[hull_mask > 0] = 1
        
        # Get indices of zero-voxels in exclude: these are the voxels to test!
        vox_to_test = (exclude == 0).astype(int)
        print("All hull facets tested; {0} voxels remain to test after filling ...".format(np.count_nonzero(vox_to_test)))
        # print "shape of vox_to_test is ", np.shape(vox_to_test)
        coords_to_test = np.asarray(np.ndarray.nonzero(vox_to_test)).T
        # print "shape of coords_to_test is ", np.shape(coords_to_test)
        
        hull_mask = test_voxels_within_hull( facet_single_point, hull_mask, normed_normals, normal_directions, coords_to_test )
    
    else:
        print("Only {0} voxels; skipping this object (excluded from output).".format(np.count_nonzero(data)))
        hull_mask = (data > 0).astype(int)
    
    return hull_mask
    # End of get_binary_hull() definition
    
def test_voxels_within_hull( facet_single_point, hull_mask, normed_normals, normal_directions, coords_to_test ):

    # Cycle through all coordinates (as rows)
    for row_counter, coordinate in enumerate(coords_to_test):
    
        # Vectors from one point on each facet to this coordinate
        vectors = facet_single_point - coordinate
        
        # Dot product again with the normal vectors from each facet
        directions = np.array([np.dot(normed_normals[i,:], vectors[i,:]) for i in xrange(normed_normals.shape[0])])
        
        # If all these vectors have the same sign as the vectors, this point is inside the hull!
        # The result of this is 1 if the signs are the same and 0 if not.
        sign_test = ((np.sign(normal_directions) * np.sign(directions)) > 0).astype(int)
        
        if np.sum(sign_test) == len(sign_test):
            hull_mask[coordinate[0],coordinate[1],coordinate[2]] = 1
            
        # Further speedup: generate internal cuboids;
        # Also: switch arrays with broadcasting after testing a range, rather than individual voxels

    return hull_mask
    # End of test_voxels_within_hull() definition
    
# DBM-like method?
    
# Dilation method
def dilate_old( input_nifti, iterations=-1 ):
    """Dilate a NIfTI image a number of iterations.
    
    Iterations:
        voxels: < 0
        mm:     > 0
        
    If output_filepath is provided, saves the NIfTI file.
    Binarises the input if necessary and returns a binary data volume the same shape as that of the input, dilated.
    If input is a NIfTI-1 object, returns one with the same header; if input is just the data, returns just data.
    Works with loop_and_save().
    """

    data, hdr = check_get_nifti_data(input_nifti)

    # Ensure we're working with binary data (convert to int afterwards, just in case)
    mask = data > 0
    mask = mask.astype(int)
    
    structure = ndimage.generate_binary_structure(3,1)
    
    iterations = convert_mm_to_voxels( iterations, hdr )
    
    # Slightly awkward as I'm not certain iterations == voxels
    print ("  Dilating {0} iterations ...".format(iterations))
    
    if iterations > 0:
        # Could convert mm to iterations by rounding to the nearest number of voxels (assuming isotropic)
        dilated_data = ndimage.binary_dilation( mask, structure, iterations )
    else:
        dilated_data = mask
    
    if hdr:
        out_nifti = nib.Nifti1Image(dilated_data, hdr.get_sform(), header=hdr)
        return out_nifti
    else:
        return dilated_data
    # End of dilate() definition
    
def erode_old( input, iterations=-1 ):
    """Erode an image a number of iterations.
    
    Binarises the input if necessary and returns a binary data volume the same shape as that of the input, eroded.
    If input is a NIfTI-1 object, returns one with the same header; if input is just the data, returns just data.
    Works with loop_and_save().
    """

    data, hdr = check_get_nifti_data(input)

    # Ensure we're working with binary data (convert to int afterwards, just in case)
    mask = data > 0
    mask = mask.astype(int)
    
    structure = ndimage.generate_binary_structure(3,1)
    
    iterations = convert_mm_to_voxels( iterations, hdr )
    
    # Slightly awkward as I'm not certain iterations == voxels
    print ("  Eroding {0} iterations ...".format(iterations))
    
    # Could convert mm to iterations by rounding to the nearest number of voxels (assuming isotropic)
    eroded_data = ndimage.binary_erosion( mask, structure, iterations )
    
    if hdr:
        out_nifti = nib.Nifti1Image(eroded_data, hdr.get_sform(), header=hdr)
        return out_nifti
    else:
        return eroded_data
    # End of erode() definition
    
def apply_threshold_to_another(input_1, input_2, threshold=0.5, mode='upper'):
    """Apply a threshold (mode='upper' or 'lower') from input_1 to input_2.
    
    Returns input_2 where the corresponding voxels in input_2 are 0.
    input_1 and input_2 must be the same shape.
    
    """
    
    data_1, hdr_1 = check_get_nifti_data(input_1)
    data_2, hdr_2 = check_get_nifti_data(input_2)
    
    # Where are the existing zeros in data_1?
    data_1_zeros = data_1 == 0
    
    if mode is 'lower':
        data_1_thr = lower_threshold(data_1, threshold)
    elif mode is 'upper':
        data_1_thr = upper_threshold(data_1, threshold)
    else:
        print("  mode was set to '{0}'; it needs to be either 'lower' or 'upper'".format(mode))
        
    # We don't want to mistake existing zeros for thresholded locations
    thr_mask = (data_1_thr == 0) - data_1_zeros
    thresholded_data_2 = multiply(data_2, invert(thr_mask))
    
    if hdr_2:
        return nib.Nifti1Image(thresholded_data_2, hdr_2.get_sform(), header=hdr_2)
    else:
        return thresholded_data_2
    # End of apply_threshold_to_another() definition
    
def report_volume( input, hdr=None, label=None, mask=None ):
    """Report (just print) the approximate volume of the input (convert to binary first).
    
    This wrapper for get_binary_volume() allows one to call that function to report (print) the current working volume, and return the input rather than the volumes: input = report_volume(input)
    
    Still not quite right.
    """
    
    mask, hdr = check_get_nifti_data(input, hdr)
    
    voxvol, physicalvol = get_binary_volume(mask, hdr, label)
    voxvol, physicalvol = get_probabilistic_volume(input, hdr, mask)
    
    return input
    # End of report_volume() definition
    
def report_number_clusters( img ):

    data, hdr = check_get_nifti_data(img)
    
    # Label each object
    structure = ndimage.generate_binary_structure(3,1)
    _, numpatches = ndimage.label(data, structure)
    
    print("  {0} objects (6-connected) in image.".format(numpatches))
    
    return
    
def get_mode_of_histogram( data, nbins=100 ):
    """Returns the mid-point value of the largest bin."""

    # Ensure we have a 1D array
    flat_data = np.ndarray.ravel(data)

    n, bin_edges = np.histogram(flat_data, bins=nbins, normed=0)
    bin_centres = 0.5*(bin_edges[1:] + bin_edges[:-1])
    
    # Find the largest histogram bin and its nearest bin_centre
    mode_index = np.argmax(n)
    mode_bin_centre = bin_centres[mode_index]

    return mode_bin_centre
    # End of get_mode_of_histogram() definition
    
def binarise_label_img( input, label_value ):

    data, hdr = check_get_nifti_data(input)
    
    data = binarise_label(data, label_value)

    return nib.Nifti1Image(data, hdr.get_sform(), header=hdr)
    
def binarise_label( data, label_value ):
    
    return (data == float(label_value)).astype(int)
    
def get_binary_volume( input, hdr=None, label_value=None ):
    """Returns either the volume of all voxels > 0, or the volume of a specific label.
    
    May be quite unreliable with non-binary input due to mask = data > 0.
    """
    
    # Just in case we have the hdr available
    data, hdr = check_get_nifti_data(input, hdr)
    
    # Convert to binary
    if label_value is None:
        mask = data > 0
        mask = mask.astype(int)
        voxvol = mask.sum(dtype=int)
        label_string=""
    else:
        voxvol = int(np.count_nonzero(data[data == float(label_value)]))
        label_string="(label = {0}) ".format(label_value)
   
    if hdr:
        physicalvol = voxvol * hdr['pixdim'][1:4].prod()
    else:
        physicalvol = 'unknown'
        # print("  Binary volume is {0} voxels.".format(voxvol))
        
    print("  Binary volume {2}is {0} voxels, approx. {1} mm^3.".format(voxvol, physicalvol, label_string))
    # for mass of water with this vol (~approx. mass of tissue), ...
    
    return voxvol, physicalvol
    # End of volumes() definition
    
def get_probabilistic_volume( input, hdr=None, mask=None ):
    """Returns the volume of 3D input (based on its probabilistic value), optionally within a mask region.
    
    """

    # Just in case we have the hdr available
    data, hdr = check_get_nifti_data(input, hdr)
    
    # If >3D, assume we want a particular 4D time-point
    # if len(np.shape(data)) > 3:
        # data = data[:,:,:,tp]
    
    # Mask, if a mask is provided
    if mask is not None:
        mask_data, hdr_mask = check_get_nifti_data(mask)
        data = data[mask_data > 0]
        
    if np.max(data) > 1.0:
        print("** Warning from get_probabilistic_volume: data appears to not be probabilistic; value(s) > 1 found ... **")
        
    total_prob_vol = np.sum(data)
    
    if hdr:
        physicalvol = total_prob_vol * hdr['pixdim'][1:4].prod()
    else:
        physicalvol = 'unknown'
        
    print("  Probabilistic volume is {0} voxels, approx. {1} mm^3.".format(total_prob_vol, physicalvol))
    # for mass of water with this vol (~approx. mass of tissue), ...
    
    return total_prob_vol, physicalvol
    # End of volumes() definition
    
def get_label_volume( input, index ):
    """Returns the volume (in voxels) of a particular integer label."""
    
    data, hdr = check_get_nifti_data(input)

    return int(ndimage.sum( data, data, index=index) / float(index))
    # End of get_label_volume() definition
    
def get_label_underlying_values( input_underlying, labels, index=1 ):
    """Given two images of the same size, returns only the data from image 1 underlying the specified label (index) in the label image.
    
    """
    
    under_data, _ = check_get_nifti_data(input_underlying)
    label_data, _ = check_get_nifti_data(labels)
    
    return np.ravel(under_data[label_data == index])
    # End of get_label_underlying_values() definition
    
def get_unique_labels( input, do_combine_lr=False, lr_cutoff=0 ):
    """Get the unique labels from a label image, allowing for combining left/right mirrored labels.
    
    """
    
    label_data, _ = check_get_nifti_data(input)
    
    # Get unique labels - ignore the first, background (0)
    unique_labels = np.unique(label_data)[1:]
    print("  unique labels in image: {0}".format(unique_labels))
    
    if do_combine_lr:
    
        matching_labels = unique_labels[unique_labels > lr_cutoff]
        labels_other_side_index = np.where(unique_labels > lr_cutoff)
        unique_labels = np.delete(unique_labels, labels_other_side_index)
        
        # Check that all matching_labels have a pair <= lr_cutoff
        for check_ml in matching_labels:
            if (check_ml - lr_cutoff) not in unique_labels:
                unique_labels = np.append(unique_labels, check_ml)
                print("  label {0} did not have a pair with value <= {1}!".format(check_ml, lr_cutoff))
                
    else:
        matching_labels = None
    
    return unique_labels, matching_labels
    # End of get_unique_labels() definition
    
def scale_data_volume( input, percentage ):
    """Scale an image, without changing the header.
    
    Use this function to simulate growth or atrophy: the final volume will change; the header will not.
    
    Input percentage is the total % by which the volume will be expanded (>0) or shrunk (<0).
    
    total atrophy % = 1 - (% to scale one side ^3)
    therefore,
    % to scale one side = cube root(1 - total atrophy %)
    
    """
    
    data, hdr = check_get_nifti_data(input)
    
    # Get the header and 3D data
    # hdr = img.get_header()
    # data = img.get_data()
    
    # report current volume
    oldvoxvol = report_volume(input)[0]
    
    sc_percentage = percentage / 100.0
    
    # 1 + percentage here so we can shrink the volume with -ve input
    single_side_scale_percentage = (1 + sc_percentage) ** (1 / 3.0)
    
    print ("  Scaling input data by {0}% ({1}% on one side)...".format(percentage, single_side_scale_percentage))
    
    # empty_cube = np.zeros(shape=data.shape)
    # scaled_data = ndimage.interpolation.zoom(data, zoom=single_side_scale_percentage, order=3, output=empty_cube)
    
    # below is based on: http://stackoverflow.com/questions/20161175/how-can-i-use-scipy-ndimage-interpolation-affine-transform-to-rotate-an-image-ab
    
    scale_factor = 1 / single_side_scale_percentage
    
    centre_in = 0.5 * np.array(data.shape)
    centre_out = centre_in                    # maintain centre
    output_shape = data.shape                # maintain shape
    
    # transposed below; I still think the column order is x,y,z, though.
    transform = np.array([[scale_factor, 0.0, 0.0], [0.0,scale_factor,0.0],[0.0,0.0,scale_factor]])
    
    print transform
    print centre_in.shape
    print centre_out.shape
    print centre_out.dot(transform)
    
    offset = centre_in - centre_out.dot(transform)
    
    scaled_data = ndimage.interpolation.affine_transform(data, transform.T, order=3, offset=offset, output_shape=output_shape, cval=0.0, output=np.float32)
    
    # report new volume - may have a header available
    newvoxvol = report_volume(scaled_data, hdr)[0]
    
    print "New volume is {0}% of old.".format(100.0*newvoxvol/oldvoxvol)
    
    if hdr:
        return nib.Nifti1Image(scaled_data, hdr.get_sform(), header=hdr)
    else:
        return scaled_data
    # End of scale_data_volume() definition
    
def downsample_image( input, factor=2, hdr=None ):
    """Downsample an image and update the header appropriately
    
    Use this function to downsample an image and directly save it (e.g.:
        ds_img = downsample_img( img, factor )
        nib.save(ds_img, save_path) ).
    Use the function downsample_data(data, hdr, factor) instead to just return the data and hdr parts
    
    Don't use this to just scale data as the header adjustments will attempt to keep the physical volume constant. Instead use scale_volume().
    
    factor is the amount by which you're scaling in each dimension.
    
    The new filesize will be approximately original_size / (factor^3)
    
    This function works with loop_and_save().
    
    """
    
    data, hdr = check_get_nifti_data(input, hdr)
    
    ds_data, ds_hdr = downsample_data( data, factor, hdr )
    ds_img = nib.Nifti1Image(ds_data, ds_hdr.get_sform(), header=ds_hdr)
    
    return ds_img
    # End of downsample_image() definition
    
def downsample_data( data, factor, hdr ):
    """Resample data and update the header appropriately
    
    If factor < 1, this is *upsampling*.
    
    Use this function to just return the data and hdr parts in case you want to do further operations prior to saving.
    
    order=3 appears to crash Python 64-bit on Windows when the image is very large (800x500x500) and the method is trilinear. Order=1 works.
    
    """
    
    fraction = 1.0 / factor
    # ds_data = ndimage.interpolation.zoom(data, zoom=fraction, order=1) # default order=3
    # order=3 default
    # order=1 for very high-resolution images (default crashes)
    # order=0 for nearest neighbour
    
    if len(data.shape) > 3:
        print("  Data shape is {0}. Only the first three dimensions will be considered! (The output will be 3D: data[:,:,:,0])".format(data.shape))
        ds_data = ndimage.interpolation.zoom(data[:,:,:,0], zoom=fraction, order=0)
    else:
        ds_data = ndimage.interpolation.zoom(data, zoom=fraction, order=0)
    
    ds_hdr = copy.deepcopy(hdr)
    ds_hdr.set_data_shape(ds_data.shape)
    
    new_pixdims = hdr.structarr['pixdim'][1:4] * factor
    print("Pixdims old: {0}, new: {1}.".format(hdr.structarr['pixdim'][1:4], new_pixdims))
    ds_hdr.structarr['pixdim'][1:4] = new_pixdims
    
    sform_old = hdr.get_sform()
    print sform_old
    
    resampling_affine = create_affine(trans=[factor,factor,factor], scale=[factor, factor, factor])

    # Create the new sform matrix
    sform_new = sform_old.dot(resampling_affine)
    
    # Keep the exact-same translation elements
    sform_new[0:3,3] = sform_old[0:3,3]
    
    print sform_new
    
    ds_hdr.set_sform(sform_new)
    
    # hdr_new.set_sform(np.eye(4))
    # hdr_new['srow_x'][0] = hdr_new['pixdim'][1]
    # hdr_new['srow_y'][1] = hdr_new['pixdim'][2]
    # hdr_new['srow_z'][2] = hdr_new['pixdim'][3]
    # hdr_new.get_sform()

    # hdr_new['srow_x'][3] = hdr_new['pixdim'][1]
    # hdr_new['srow_y'][3] = hdr_new['pixdim'][2]
    # hdr_new['srow_z'][3] = hdr_new['pixdim'][3]
    
    return ds_data, ds_hdr
    # End of downsample_data() definition
    
def get_corners( mask ):
    print("  Getting cuboid corners from mask ...")
    
    # Similar to Matlab's ind2sub (don't even need this; numpy.ndarray.nonzero does it)
    # row, col, sli = np.unravel_index([22, 41, 37], data.shape)
    rows, cols, slices = np.ndarray.nonzero( mask )
    
    # top left front
    tlf_x = min(rows)
    tlf_y = min(cols)
    tlf_z = min(slices)
    # bottom right back
    brb_x = max(rows)
    brb_y = max(cols)
    brb_z = max(slices)
    
    return [brb_x, tlf_x], [brb_y, tlf_y], [brb_z, tlf_z]
    
def crop( input, hdr=None, mask=None, dilation=-2, threshold=0, reset_origin=False, maintain_position=False ):
    """Crop a 3D data matrix to the tight cuboid defined by the extreme edges of a mask.
    
    Dilates the mask a given number of iterations, just to give some leeway.
    
    Also edits the given header to reflect the new dimensions.
    
    reset_origin is currently experimental: should put the origin at the centre of the new, cropped image.
    
    """
    
    data, hdr = check_get_nifti_data(input, hdr)
    
    # In case a mask isn't passed, define a threshold (default 0) to generate one automatically
    if mask is None:
        mask = lower_threshold_bin( data, threshold )    # this is a very roundabout way of doing what I do in two lines otherwise
        # mask = data > threshold   # this is slightly different as it omits the actual threshold value
    
    # Crop about the largest connected component only
    # mask = lconcomp(mask)
    
    # Speed up possibility: get cuboid corners first, then dilate those voxels only; don't fill, and get cuboid corners again after dilation.
    x, y, z = get_corners(mask)
    print("x {0}, y {0}, z {0}".format(x,y,z))
    
    new_mask = np.zeros(shape=mask.shape)
    new_mask[x[0], y[0], z[0]] = 1.0
    new_mask[x[1], y[1], z[1]] = 1.0
    
    # Dilation breaks if it is 0
    if dilation != 0:
        new_mask = dilate(new_mask, dilation)
    
        # Iterate again
        x, y, z = get_corners(new_mask)
        print("x {0}, y {0}, z {0}".format(x,y,z))
    
    # cuboid dimensions
    # dim_x = brb_x - tlf_x + 1
    # dim_y = brb_y - tlf_y + 1
    # dim_z = brb_z - tlf_z + 1
    dim_x = x[0] - x[1] + 1
    dim_y = y[0] - y[1] + 1
    dim_z = z[0] - z[1] + 1
    
    print("dim_x, dim_y, dim_z: {0}, {1}, {2}".format(dim_x, dim_y, dim_z))
    
    cropped_hdr = copy.deepcopy(hdr)
    cropped_hdr.set_data_shape([dim_x, dim_y, dim_z])
    
    # In Matlab:
    # source_img(tlf_x:brb_x, tlf_y:brb_y, tlf_z:brb_z);
    # In Python:
    # Here's how to (explicitly) get all the elements from the first row and first column onwards, of a 2D array:
    # a[1:len(a[1,:]),1:len(a[:,1])]
    # Here's how you'd get the central elements of a 2D array (removing the first and last rows and columns):
    # a[1:len(a[1,:])-1,1:len(a[:,1])-1]
    
    print "  Cropping data and mask ..."
    # not sure why the +1s are necessary, below, but they do appear to be, to get the final elements:
    # cropped_data = data[tlf_x:brb_x+1, tlf_y:brb_y+1, tlf_z:brb_z+1]
    # cropped_mask = mask[tlf_x:brb_x+1, tlf_y:brb_y+1, tlf_z:brb_z+1]
    cropped_data = data[x[1]:x[0]+1, y[1]:y[0]+1, z[1]:z[0]+1]
    cropped_mask = mask[x[1]:x[0]+1, y[1]:y[0]+1, z[1]:z[0]+1]
    
    # Multiply (* is element-wise, here) the data image by the mask, to exclude extraneous objects
    cropped_data = cropped_data * cropped_mask
    
    # To put new header's origin at the image centre, get the new dimensions and halve them...
    if reset_origin is True:
    
        pixdims = cropped_hdr.structarr['pixdim'][1:4]
        half_dim_mm = pixdims * [dim_x, dim_y, dim_z] / 2.0
        
        affine = cropped_hdr.get_sform()
        affine[:3,3] = [-half_dim_mm[0],-half_dim_mm[1],-half_dim_mm[2]]
        
        cropped_hdr.set_sform(affine)
        
        # Reset header entirely
        # new_img = nib.Nifti1Image(cropped_data, hdr_new.get_sform())
        
    elif maintain_position is True:
        # Keep the image in the same position, relative to the origin (mutually exclusive from reset_origin)
        
        pixdims = cropped_hdr.structarr['pixdim'][1:4]
        affine = cropped_hdr.get_sform()
        old_offset = affine[:3,3]
        
        # NB Relies on tlf (x[1] etc. vs. x[0]) being the correct side of the origin
        cropped_between_origin_and_edge = pixdims * [x[1], y[1], z[1]]
        
        new_trans = old_offset + cropped_between_origin_and_edge
        
        new_affine = affine
        new_affine[:3,3] = new_trans
        
        cropped_hdr.set_sform(new_affine)

    return cropped_data, cropped_mask, cropped_hdr
    # End of crop() definition
    
def write_csv_data( csv_path, rows_to_write ):
    # Format rows_to_write with strings as: rows_to_write = [['col 1 string'] + ['col 2 string'],['line2'], [another_string], ['line 3 numbers in cols', 1, 2, 3]]
    # 'a' appends; 'b' writes in binary mode (necessary on Win, harmless on Unix)
    # Can append even if file doesn't exist yet
    with open(csv_path, 'ab') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for row in rows_to_write:
            # csvwriter.writerow([string_to_write] + [string2] + [string3])
            # fullrow = []
            # for element in row:
                # fullrow += [element]
            # print fullrow
            csvwriter.writerow(row)
            
    # End of write_csv_data() definition
    
def apply_mask( input, mask=None ):
    """Returns array with the same shape as the input."""
    
    data, hdr = check_get_nifti_data( input )
    
    if mask is not None:
        mask, _ = check_get_nifti_data( mask )
    
        print("  Applying mask ...")
    
        # Apply mask
        try:
            masked_data = np.array(data)
            masked_data[mask == 0] = 0
            
        except ValueError:
            print("Mask and image are likely different shapes (check they correspond).")
            raise
            
        if hdr:
            return nib.Nifti1Image(masked_data, hdr.get_sform(), header=hdr)
        else:
            return masked_data
    else:
        print("  No mask provided.}")
        return data
    # End of apply_mask() definition
    
def constrain_to_linear_indices( image, lin_indices ):
    """Returns the image voxels corresponding to linear indices.
    
    """
    data, hdr = check_get_nifti_data(image)
    flat_data = np.ndarray.ravel(data)
    
    return flat_data[lin_indices]
    
def constrain_to_mask( data_input, mask_input=None ):
    """Returns only the voxels from data within the mask."""
    
    if mask_input is not None:
    
        data, hdr = check_get_nifti_data( data_input )
        mask_data, mask_hdr = check_get_nifti_data( mask_input )
        
        try:
            # mask_data = mask_data.astype(int)
            constrained_data = data[mask_data > 0]
            
        except ValueError:
            print("Mask and image are likely different shapes (check they correspond).")
            raise
            
    else:
        print("Mask not provided; output is same shape as input.")
        constrained_data = data_input
        
    return constrained_data
    # End of constrain_to_mask() definition
    
def hollow_box( nifti, thickness=1 ):
    """Generate a binary NIfTI whose shape is the same as the input NIfTI, but with a binary rim around the edges, forming a hollow box.
    
    """
    
    cuboid = np.ones(nifti.shape)
    internal = erode(cuboid, thickness)
    box_data = cuboid - internal
    
    return nib.Nifti1Image(box_data, nifti.get_sform(), nifti.get_header())
    # End hollow_box() definition
        
# Plots and images
# ----------------

def nice_figure_for_png( figname=None, suptitle=None, xSize=16.0 ):
    """Returns a nice, 1600x1000 px figure for saving to PNG.
    
    xSize is in inches (default 16.0 for nice and wide).
    
    """

    fig = plt.figure(figname)
    
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=16)
    
    # Set size and shape
    # This combination looks decent! (Thanks, http://wiki.scipy.org/Cookbook/Matplotlib/AdjustingImageSize )
    xPix = xSize * 100.0    # default 1600, then
    yPix = 1000
    # xSize = 16.0    # inches
    ySize = xSize/xPix*yPix
    fig.set_size_inches(xSize, ySize)

    return fig
    
def nice_figure_for_png_two( figname=None, suptitle=None, xSize=16.0, ySize=10.0, dpi=300.0, fontsize=10.0, dimunits='inches' ):
    """Returns a nice, 1600x1000 px figure for saving to PNG.
    
    xSize is in inches (default 16.0 for nice and wide).
    
    NB fontsize is the height in points. 1pt == 1/72 of an inch. So, if you specify a 16 inch high figure, a fontsize of 10.0 means the text will be (10/72)/16 of the total height.
    
    dimunits options are 'mm', 'in', 'px'.
    
    """
    
    font = {'size': fontsize}
    plt.rc('font', **font)

    fig = plt.figure(figname, dpi=dpi)
    
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=fontsize+2.0)
    # if title is not None:     # does not yet work
        # plt.title(title, fontsize=fontsize-2.0)
    
    # Set size and shape
    # xPix = xSize * dpi    # default 1600, then
    # yPix = ySize * dpi
    # xSize = 16.0    # inches
    # ySize = xSize/xPix*yPix
    
    # if 'in' in dimunits:
        # Default behaviour
        # fig.set_size_inches(xSize, ySize)
    if 'mm' in dimunits:
        xSize = xSize * 0.03937
        ySize = ySize * 0.03937
    elif 'px' in dimunits:
        xSize = xSize / dpi
        ySize = ySize / dpi
        
    fig.set_size_inches(xSize, ySize)

    return fig
    
def save_fig( directory, name, extension=".png" ):
    """Creates a full path to an image file, complete with current date and time.
    """
    import datetime
    
    if directory is None:
        directory = os.getcwd()
        
    if name is not None:
        name = " - " + name
    
    save_time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return os.path.normpath(os.path.join(directory, save_time + name + extension))
    # End save_fig() definition
    
def compare_image_to_multiple(reference_images, input_directory, file_name_filters, extension='.nii.gz'):
    """Use correlation to compare images (they must all be in the same space) in an input_directory with single input reference_image(s).
    
    reference_images should be a list of NIfTI-1s equal in length to file_name_filters. If you want to compare with just one reference image for all file name filters, repeat it in the list.
    
    The files in input_directory will all be compared and the PMCC output in a .CSV saved to the input_directory.
    
    If file_name_filters is a list, then all files in input_directory matching each element of file_name_filters will be compared. If the list has more than one element, there will be multiple columns in the output, corresponding to each filter.
    
    """
    import datetime
    from scipy import stats
    
    # Ensure we can iterate
    file_name_filters = get_iterable(file_name_filters)
    reference_images = get_iterable(reference_images)
    
    # Get the first filter, so we can put 'universal' file names in the first column of the .CSV.
    filter = file_name_filters[0]
    
    ## Create the .CSV
    
    # Get filenames without path
    filenames = get_names_list(input_directory, '*' + filter + '*' )
    
    # Get filenames without the filter part, for the first column
    filenames = [filename.replace(filter,'') for filename in filenames]
    
    # Create a new DataFrame and fill the first column with 'universal' filenames
    cols=['filename'].extend(file_name_filters)
    df = pd.DataFrame(index=range(len(filenames)), columns=cols)
    df['filename'] = filenames
    
    existing_file_indices = range(len(filenames))
    
    for fcount, filter in enumerate(file_name_filters):
    
        reference_image = get_nifti(reference_images[fcount])
        
        # Go through the rows of the new DataFrame and get the corresponding file, from this list
        for counter, fn in enumerate(filenames):
            
            nf_path = get_corresponding_file(input_directory, fn, name_filter=filter, path_only=True)
            nf_name = (os.path.basename(nf_path)).split(os.extsep)[0]
            print "- - - - - - - - - - - \n  Processing {0} / {1}: {2} ...".format((counter + 1), len(filenames), nf_name)
            
            # Get the row index of the .CSV to save the result
            idx = existing_file_indices[counter]

            pearson_r = stats.pearsonr(np.ndarray.ravel(reference_image.get_data()), np.ndarray.ravel(get_nifti(nf_path).get_data()))[0]
            
            df.ix[idx,filter] = pearson_r
            
    print("  DataFrame: \n{0}".format(df))
    save_time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    df.to_csv(os.path.normpath(os.path.join(input_directory, 'correlations_' + save_time + '.csv')))
    
    return df
    # End compare_image_to_multiple() definition
    
# Print information for the user
# ------------------------------

def notify_script_start( script_name='' ):
    """Give the user some helpful notes at the launch of the script."""
    import time
    time_format = '%H:%M:%S %p on %b %d, %Y (%Z)'
    time_start = time.strftime(time_format)
    alert_user("  MouseMorph script {0} starting at {1}!\n...".format(script_name, time_start))
    
    return time_start

def notify_script_complete( script_name, output="NA", time_start=None ):
    # Add note of log file location?
    import time
    from datetime import datetime
    time_format = '%H:%M:%S %p on %b %d, %Y (%Z)'
    if time_start is None:
        time_start = time.strftime(time_format)
    time_stop = time.strftime(time_format)
    time_diff = datetime.strptime(time_stop, time_format) - datetime.strptime(time_start, time_format)
    alert_user("  MouseMorph script {0} completed at {1}!\n  Time taken was {3}.\n  Please check your output ({2}).".format(script_name, time_stop, output, time_diff))
    return
    
def alert_user( string_to_print ):
    print("\n* * * * * * * * * * * * * * * * * * * * * * * *\n")
    print(string_to_print)
    print("\n* * * * * * * * * * * * * * * * * * * * * * * *\n")
    # Force print
    sys.stdout.flush()
    return
    
# CSV-related functions
# ---------------------

def check_create_csv(csvpath, input_directory, file_name_filter="*", out_name_preface="", out_name_append="", csv_name_string="csv_output", new_empty=False, new_cols=[], name_column=0, group_column=1, csv_group_filter=None, extension='*.nii*'):
    """Given a csvpath (an existing .CSV file or one to be created), checks input_directory for NIfTI files.
    
    If no .CSV exists, a Pandas DataFrame will be created containing a list of all NIfTI files in input_directory (nifti_files_list). If a .CSV already exists, the DataFrame will contain its contents.
    
    If new_empty=True, does not populate a new .CSV; returns an empty DataFrame
    Add new columns with new_cols=['one', 'two', 'three']. Will be filled with NaNs if DataFrame is not empty and these columns do not yet exist. Will be skipped if they exist already.
    
    """
    
    if os.path.isfile(csvpath):
        print("  CSV exists! Reading ...")
        csv_file_path = csvpath
        
        # Pandas: assume row 0 is the headers, (and assume the first column, index_col=0, holds the file names through which we wish to iterate)
        df = pd.read_csv(csv_file_path, header=0)
        
        # If the .CSV contains more rows than files in in_path, need the indices of the files which do exist
        nifti_files_list, existing_file_indices = get_existing_files_from_csv(csv_file_path, input_directory, name_column, group_column, csv_group_filter, extension)
        
    else:
        
        if os.path.isdir(csvpath):
            # Set a save name
            csv_file_path = os.path.normpath(os.path.join(csvpath, out_name_preface + csv_name_string + out_name_append + '.csv'))
        else:
            # Create a new .CSV
            csv_file_path = csvpath
        
        print("  No CSV exists! A .CSV will be created: {0}. Getting NIfTI files from {1} instead ...".format(csv_file_path, input_directory))
        
        if new_empty is False:
        
            # Get list of files (use file_name_filter)
            nifti_files_list = sorted(glob.glob(os.path.join(input_directory, file_name_filter + extension)))
            
            # Get filenames without path
            filenames = [(os.path.basename(filepath)).split(os.extsep)[0] for filepath in nifti_files_list]
            
            # Create a new DataFrame and fill the first column with filenames; the rest are checked and created below
            df = pd.DataFrame(index=range(len(filenames)), columns=['filename'])
            df['filename'] = filenames
            
            # Needed to be consistent with the above case of an existing .CSV; much simpler in this case
            existing_file_indices = range(len(filenames))
            
        else:
            df = pd.DataFrame()
            nifti_files_list = []
            existing_file_indices = []
            
        df.to_csv(csv_file_path, index=False)
            
    existing_cols = list(df.columns.values)
    for new_col in new_cols:
        if new_col not in existing_cols:
            df[new_col] = ''
        
    print("  Contents of DataFrame from .CSV at {0}: \n{1}".format(csv_file_path, df))
        
    return df, csv_file_path, nifti_files_list, existing_file_indices
    # End check_create_csv() definition
    
def get_files_from_csv_by_group(csv_path, name_column=0, group_column=1, csv_group_filter=None, csv_name_filter=None):
    """Use get_existing_files_from_csv() to call this function in order to omit non-extant files.
    
    Further filters the names themselves by csv_name_filter, if specified.
    """
    
    df = pd.read_csv(csv_path, header=0)
    columns_idx = np.array([name_column, group_column])
    column_names = df.columns.values
    
    print("  You specified the following columns from the .CSV {0}: {1} ...".format(csv_path, columns_idx))
    print("  File names from column {2} ({3}) will be grouped by column {0} ({1}):".format(columns_idx[1], column_names[columns_idx[1]], columns_idx[0], column_names[columns_idx[0]]))
    
    # Group by values in the first column specified by the user
    df_grouped = df.groupby(column_names[columns_idx[1]])
    df_single_group = df_grouped.get_group(csv_group_filter)
    
    if csv_name_filter is not None:
        print("  File names will be further filtered by '{0}'".format(csv_name_filter))
        # See http://stackoverflow.com/a/11872393/3329384
        
        df_single_group = df_single_group[df_single_group[column_names[columns_idx[0]]].str.contains(csv_name_filter)]
        
        # def mask_fn(dfm, key, value):
            # return dfm[dfm[key] == value]
            
        # pd.DataFrame.mask = mask_fn
        # df_single_group = df_single_group.mask(column_names[columns_idx[0]], csv_name_filter)
    
    # Get list of names to filter by
    file_name_list = list(df_single_group.ix[:,columns_idx[0]])
    row_indices = df_single_group.index.tolist()
    
    return file_name_list, row_indices
        
def get_existing_files_from_csv(csv_path, input_directory, name_column=0, group_column=1, csv_group_filter=None, extension='*.nii*'):
    """Given an existing .CSV file, returns a list of existing files in input_directory which match names in the given csv name_column (default 0).
    
    Assumes there is a header line.
    
    group_column (default 1) is only used if csv_group_filter (default None) is given. In which case, only files matching csv_group_filter in group_column will be returned.
    
    csv_cols can be an integer (to just check that column) or a list of two (if csv_group_filter is also specified), e.g. [0, 1]
    
    E.g., if a string in the csv_cols is "file_name", returns the path to the NIfTI named "*file_name*.nii*" in input_directory.
    
    If csv_group_filter is not None, returns only files matching those in csv_cols[0] which match csv_group_filter in csv_cols[1]
    
    Only returns paths to existing files, in nifti_files_list.
    Correspondingly, existing_file_indices is a list of the rows of the CSV corresponding to nifti_files_list, in order.
    
    """
    
    # Get list of NIfTI files through which to iterate from the first column
    nifti_files_list = []
    existing_file_indices = []
    
    if csv_group_filter is not None:
        complete_list, all_row_indices = get_files_from_csv_by_group(csv_path, name_column, group_column, csv_group_filter)
        
    else:
        df = pd.read_csv(csv_path, header=0)
        complete_list = df.ix[:,name_column].tolist()
        all_row_indices = range(len(complete_list))
        
    # Check file existence
    for i, fn in enumerate(complete_list):
        # requires str(fn)?
        potential_file_path = glob.glob(os.path.join(input_directory, '*' + fn + extension))
        
        if not potential_file_path:
            # In reporting the row number, allow for zero-indexing and the header line, which is excluded from the enumeration
            print("No file containing {0} (from row {2} of .CSV) in {1}; skipping ...".format(fn, input_directory, i))
        
        elif len(potential_file_path) > 1:
                print("Warning: there is more than one file matching {0} (from row {1} of .CSV) in {2} ...".format(fn, i+2, input_directory))
                sys.exit("This must be corrected; there is no way around it. Rename the file (and the corresponding entry in the .CSV) so that the string in the CSV is unique, then rerun this script.")
                print(df)
                print(complete_list)
        
        elif os.path.isfile(potential_file_path[0]):
            nifti_files_list.append(potential_file_path[0])
            existing_file_indices.append(all_row_indices[i])
            
        else:
            print("Error...")
            sys.exit(1)
            
    # We don't want to return a sorted list; we want the list as it appears in the CSV, so we can easily index back in later.
    return nifti_files_list, existing_file_indices
    # End get_existing_files_from_csv() definition
    
def filter_list_by_list(list_to_filter, list_to_filter_by):
    """Returns a list consisting of list_to_filter elements which contain text from some element of list_to_filter_by.
    
    E.g. list_to_filter = ['one', 'two', 'three']; list_to_filter_by = ['t', 'x']; returns ['two', 'three']
    
    """
    filtered_list = []
    for element in list_to_filter:
        for sub_element in list_to_filter_by:
            if sub_element in element:
                filtered_list.append(element)
    
    return list(set(filtered_list))
    # End filter_list_by_list() definition
    
# Text
# ----

def generate_affine_matrix( xx=1, xy=0, xz=0, yx=0, yy=1, yz=0, zx=0, zy=0, zz=1, tx=0, ty=0, tz=0 ):
    """Generate a 3D homogeneous affine matrix.
    By default, identity.
    You only need to specify the elements which differ from the identity matrix, e.g. generate_affine_matrix(yy=-1).
    """
    
    return np.array([ (xx, xy, xz, tx), (yx, yy, yz, ty), (zx, zy, zz, tz), (0, 0, 0, 1) ])
    # End of generate_affine_matrix() definition

def create_affine(trans=[0,0,0], scale=[1,1,1], full_affine=None, output_path=None):
    """Create a 3D homogeneous affine transformation matrix.
    
    Writes this to a text file if output_path is specified.
    
    """
    
    if full_affine is not None:
        aff = full_affine
    else:
        aff = np.eye(4)
        aff[:3,3] = trans
        np.fill_diagonal(aff[:3,:3], scale)
    
    if output_path is not None:
        np.savetxt(output_path, aff, fmt='%.8f')
        print("  Affine file (with determinant {1}) saved to: {0} \n{2} ...".format(output_path, np.linalg.det(aff), aff))
    
    return aff
    # End of create_affine() definition
    
def check_zero_off_diagonals( matrix, ignore_final_column=False ):
    """Returns False when any off-diagonal element of an NxN matrix is non-zero.
    
    If ignore_final_column is True, does not take translation elements (of a 4x4 homogeneous affine matrix) into account.
    
    """
    
    rows = np.shape(matrix)[0]
    cols = np.shape(matrix)[1]
    
    if ignore_final_column is True:
        cols -= 1
    
    total = 0.0
    for r in range(rows):
        for c in range(cols):
            if r != c:
                total = total + matrix[r,c]
                
    return total == 0.0
    # End of check_off_diagonal_elements() definition

def load_affine( input_path, affine_function=None ):
    """Load an affine transformation matrix from a file.
    
    Can perform a mathematical operation on it with affine_function if desired; otherwise returns the entire affine.
    
    """
    
    aff = np.loadtxt(input_path)
    
    if affine_function is None:
        return aff
    else:
        return affine_function(aff)
    # End of load_affine() definition
    
# Statistics
# ----------

def jaccard_index( nifti_1, nifti_2 ):
    """Return the Jaccard Index for two binary NIfTI images in the same space.
    
    NB Dice score is 2J/(1+J)
    """
    
    data_1 = np.ravel(nifti_1.get_data())
    data_2 = np.ravel(nifti_2.get_data())
    
    if not data_1.shape == data_2.shape:
        print("Error: NIfTI data objects have different shape: {0}, {1}".format(data_1.shape, data_2.shape))
        
    # NB see https://github.com/scikit-learn/scikit-learn/issues/3037 : The built-in jaccard_similarity_score is not valid for two binary NIfTIs.
    # from sklearn.metrics import jaccard_similarity_score
    # J = jaccard_similarity_score(data_1, data_2)
    
    J = np.sum(np.logical_and(data_1, data_2)) / np.sum(np.logical_or(data_1, data_2)).astype(float)
    # dice = 2. * J / (1. + J)
    return J
    # End of jaccard_index() definition
    
def dice_score( nifti_1, nifti_2 ):
    J = jaccard_index(nifti_1, nifti_2)
    return 2. * J / (1. + J)
    # End of dice_score() definition

def mean_stdev_overall( nifti_path_list, mask_directory=None, mask_threshold=0.5 ):
	"""Returns the mean and standard deviation values of all voxels across all images. Constrainable with a mask from mask_directory (file names must match).
	
	As each mask may be different, images may have different shapes and be in a different space.
	
	Unlike average_nifti_list() and group_standard_deviation(), which return voxel-wise results as a NIfTI image, this function returns single values (floats) for mean and standard deviation.
	
	"""
	
	all_data = []
	
	# Loop through files
	print("Looping through {0} files ...".format(len(nifti_path_list)))
	for i, nf_path in enumerate(nifti_path_list):
		nf_name = get_file_name(nf_path)
		print("  {0} / {1}: ".format(i+1, len(nifti_path_list), nf_name))
		
		data, hdr = check_get_nifti_data(nf_path)
		mask = get_corresponding_file(mask_directory, nf_name)
		
		if mask is not None:
			mask_data = mask.get_data() > mask_threshold
			constr_data = constrain_to_mask(data, mask_data)
			all_data.extend(constr_data)
		
	mean_value = np.mean(all_data)
	stdev_value = np.std(all_data)
	print("Mean: {0}; standard deviation: {1}".format(mean_value, stdev_value))
	
	return mean_value, stdev_value
	# mean_stdev_overall() definition

def group_standard_deviation( nifti_path_list, sample_stdev=True, mean=None ):
    """Given a list of file paths of NIfTI-1 images in the same space, generate their voxel-wise standard deviation.
    
    """
    
    # Calculate overall voxel-wise mean
    if mean is None:
        mean_data = average_nifti_list(nifti_path_list).get_data()
    else:
        mean_data, mean_hdr = check_get_nifti_data(mean)
    
    # Voxel-wise subtraction of the mean, squaring and summing
        
    # Loop through files in input_directory
    for counter, nifti_path in enumerate(nifti_path_list):
        print("File {0} / {1}: {2} ...".format((counter + 1), len(nifti_path_list), nifti_path))
        
        # Get the filename, removing path and 1+ extensions
        nf_name = os.path.basename(nifti_path.split(os.extsep)[0])
        
        print("  Subtracting the mean from {0}, and squaring the result ...".format(os.path.normpath(os.path.join(nifti_path))))
        
        # Load the image
        img = nib.load(os.path.normpath(os.path.join(nifti_path)))
        
        # Subtract and square
        sub_img = subtract(img, mean_data)
        squ_img = power_n(sub_img, power=2.0)
        
        if counter == 0:
            sum_img = squ_img
        else:
            sum_img = add(sum_img, squ_img)
            
    # Voxel-wise mean to calculate the standard deviation
    # Here calculating the sample standard deviation, so divide by N-1
    stdev_nii = divide(sum_img, np.float(len(nifti_path_list)-1))
    
    # The above will likely generate several NaNs from dividing regions by zero
    stdev_nii_nn = remove_nans(stdev_nii)
    
    return stdev_nii_nn
    # End group_standard_deviation() definition
    
def group_standard_deviation_pooled( nifti_path_list_experimental=None, nifti_path_list_control=None, stdev_exp=None, stdev_con=None, exp_n=None, con_n=None ):
    """Given two lists of file paths of two groups of NIfTI-1 files, calculate the pooled voxel-wise standard deviation, and return as a NIfTI-1 image.
    
    Potentially speedup if either of the stdevs have already been calculated, by providing them directly.
    """
    
    if stdev_exp is None:
        stdev_exp = group_standard_deviation(nifti_path_list_experimental)
    if stdev_con is None:
        stdev_con = group_standard_deviation(nifti_path_list_control)
    
    experimental_var = power_n(stdev_exp, power=2.0)
    control_var = power_n(stdev_con, power=2.0)
    
    if exp_n is None:
        exp_n = len(nifti_path_list_experimental)
    if con_n is None:
        con_n = len(nifti_path_list_control)
    
    numerator = add(multiply(experimental_var, exp_n - 1), multiply(control_var, con_n - 1))
    denom = exp_n + con_n - 2.0
    
    f = divide(numerator, denom)
    
    std_pooled = power_n(f, 0.5)
    
    return std_pooled
    # End group_standard_deviation_pooled() definition

def fdr_correction( mask, p_data, q_value=0.05 ):
    """Perform FDR correction on a set of p-values. Returns an array the same (full-size) shape as the input mask.
    
    Inputs
        mask:       (NIfTI, np.array, or string path to a NIfTI) Must be full-size.
        p_data:     (NIfTI, np.array, or string path to a NIfTI) Can be full-size, or only the size of non-zero mask voxels.
        q_value:    (float or int)
    
    """
    
    from statsmodels.stats import multitest
    
    mask_data, _ = check_get_nifti_data( mask )
    p_data, _ = check_get_nifti_data( p_data )
    
    # Get relevant voxels
    all_nonzero_indices = np.flatnonzero(mask_data)
    
    # Get the mask region
    p_values = np.ndarray.ravel(p_data)[all_nonzero_indices]
    
    fdr_reject_hyp, pvals_corr_fdr = multitest.fdrcorrection(p_values, alpha=q_value)
    
    # Put back into a full 3D shape
    fdr_data = (np.ndarray.ravel(np.zeros_like(mask_data))).astype(np.float32)
    fdr_data[all_nonzero_indices] = fdr_reject_hyp
    fdr_reject_rs = np.ndarray.reshape(fdr_data, mask_data.shape)
    
    return fdr_reject_rs
    
def fdr_correction_twostage( mask, p_data, q_value=0.05 ):
    """Perform two-stage FDR correction on a set of p-values. Returns an array the same (full-size) shape as the input mask.
    
    """
    
    from statsmodels.stats import multitest
    
    mask_data, _ = check_get_nifti_data( mask )
    p_data, _ = check_get_nifti_data( p_data )
    
    # Get relevant voxels
    all_nonzero_indices = np.flatnonzero(mask_data)
    
    # Get the mask region
    p_values = np.ndarray.ravel(p_data)[all_nonzero_indices]
    
    fdr_reject_hyp, pvals_corr_fdr, _, _ = multitest.fdrcorrection_twostage(p_values, alpha=q_value)
    
    # Put back into a full 3D shape
    fdr_data = (np.ndarray.ravel(np.zeros_like(mask_data))).astype(np.float32)
    fdr_data[all_nonzero_indices] = fdr_reject_hyp
    fdr_reject_rs = np.ndarray.reshape(fdr_data, mask_data.shape)
    
    return fdr_reject_rs
    
def n_per_arm(std_nii, mean_nii, effect=0.25, sig=0.05, power=0.8, tails='two', roundup=True):
    """Calculate the N per arm required to detect a significant effect, given standard deviation and mean at each voxel.
    
    effect:     the fraction of the mean considered significant
    sig:        alpha, e.g. 0.05
    power:      e.g. 0.8
    tails:      if not 'two' (the default), one-tailed only.
    roundup:    True (default) if you want the output rounded up to the nearest integer.
    
    """
    from scipy import stats
    
    if tails is 'two':
        std_from_norm_mean_sig = np.abs(stats.norm.ppf(1. - (sig/2.)))
    else:
        std_from_norm_mean_sig = np.abs(stats.norm.ppf(1. - sig))
        
    std_from_norm_mean_pow = np.abs(stats.norm.ppf(1. - power))
        
    f_value = (std_from_norm_mean_pow + std_from_norm_mean_sig)**2.
    numerator = multiply(power_n(std_nii, 2.), f_value * 2.)
    denominator = power_n(multiply(mean_nii, effect), 2.)
    n_arm = divide(numerator, denominator)
    
    if roundup is False:
        return n_arm
    else:
        ru_data = np.ceil(n_arm.get_data())
        return nib.Nifti1Image(ru_data, n_arm.get_sform(), n_arm.get_header())
    # End of n_per_arm() definition
    
# Header-only functions
# ---------------------
# Take a NIfTI and return a new NIfTI with the header altered.

def apply_affine(nifti, transform_affine):
    """Apply an affine to a NIfTI-1 image.
    
    Should exactly replicate reg_transform -updSform.
    """
    
    hdr = nifti.get_header()
    original_sform = nifti.get_sform()
    final_sform = linalg.inv(transform_affine).dot(original_sform)
    hdr.set_sform(final_sform)
    
    return nib.Nifti1Image(nifti.get_data(), hdr.get_sform(), hdr)
    # End apply_affine() definition

    
# Data-only functions
# -------------------
# These are wrapped with niftify(nifti, function_name, **kwargs) to input and return a NIfTI object. NB this is done with a decorator, so each function can be called with just function(nifti). As used by loop_and_save().
    
def niftify(func):
    """Decorator function for any function requiring only the data part (NumPy ndarray) of a NIfTI image.
    
    The first argument must be recognisable by check_get_nifti_data().
    
    Returns a NIfTI image.
    """
    
    def niftify_and_call(*args, **kwargs):
        data, hdr = check_get_nifti_data(args[0])
        result_data = func(data, *args[1:], **kwargs)
        if result_data is not None:
            if hdr is not None:
                return nib.Nifti1Image(result_data, hdr.get_best_affine(), hdr)
            else:
                return result_data
    return niftify_and_call
    
@niftify
def absolute( data ):
    """Returns an absolute NIfTI or data object (always positive), i.e., |data|."""
    
    # data, hdr = check_get_nifti_data(input)
    # abs_data = np.absolute(data)
    
    # if hdr:
        # return nib.Nifti1Image(abs_data, hdr.get_sform(), header=hdr)
    # else:
    return np.absolute(data)
    # End of absolute() definition
    
@niftify
def binarise( input ):
    """Return binary mask where all voxels > 0.0 are 1."""
    return lower_threshold_bin( input, 0.0 )
    # End binarise() definition
    
@niftify
def binarise_using_padding( data, padding_value=0 ):
    """Return binary mask where all voxels not equal to the padding_value are 1.
    
    (The word "padding" inherited from ITK.)
    """
    
    data[data == padding_value] = 0
    data[data != padding_value] = 1
    
    return data
    # End binarise_using_padding() definition
    
@niftify
def dilate( data, iterations=1, structure=None ):
    """Dilate a binary ND array by a number of iterations."""
    
    # Convert to binary, just in case.
    mask = binarise(data).astype(int)
    
    if not structure:
        structure = ndimage.generate_binary_structure(3,1)
    
    # Check we have positive iterations - no header available here to convert from mm.
    iterations = np.abs(iterations)
    
    # Slightly awkward as I'm not certain iterations == voxels
    print ("  Dilating {0} iterations ...".format(iterations))
    
    if iterations > 0:
        dilated_mask = ndimage.binary_dilation( mask, structure, iterations )
    
    return dilated_mask
    # End of dilate() definition
    
@niftify
def erode( data, iterations=1, structure=None, nonbinary=False ):
    """Erode an ND array by a number of iterations.
    
    If nonbinary is True (default False), the input is binarised, the erosion applied, and the output is the original multiplied by the eroded binary.
    """
    
    # Convert to binary, just in case.
    mask = binarise(data).astype(int)
    
    if structure is None:
        structure = ndimage.generate_binary_structure(3,1)
    
    iterations = np.abs(iterations)
    
    # Slightly awkward as I'm not certain iterations == voxels
    print ("  Eroding {0} iterations ...".format(iterations))
    
    # Could convert mm to iterations by rounding to the nearest number of voxels (assuming isotropic)
    if iterations > 0:
        eroded_data = ndimage.binary_erosion( mask, structure, iterations )
        
    if nonbinary is True:
        return apply_mask(data, eroded_data)
    else:
        return eroded_data
    # End of erode() definition
    
@niftify
def fill( data ):
    """Convert an input image to binary, and fill any holes.
    """

    # data, hdr = check_get_nifti_data(input)
    # Ensure we're working with binary data (convert to int afterwards, just in case)
    mask = data > 0
    mask = mask.astype(int)
    
    print("  Filling holes ...")
    # Use default structure element for this 3D mask
    filled_data = ndimage.binary_fill_holes( mask ).astype(int)
    return filled_data
    # End of fill() definition
    
@niftify
def gaussian_smooth( data, sigma=-1.0 ):
    """Returns a smoothed image, having convolved with a Gaussian kernel with the given standard deviation (sigma).
    
    Input
    -----
    sigma:
        +ve for mm; -ve for voxels

    Works with loop_and_save().
    """
    # Replicating reg_tools -smoG
    
    sigma = convert_mm_to_voxels( sigma, hdr )
    
    # Ensure result will be non-binary
    data = data.astype('float')
    
    smoothed_data = ndimage.filters.gaussian_filter(data, sigma)
    return smoothed_data
    # End gaussian_smooth() definition
    
@niftify
def get_timepoint( data, tp=0 ):
    """Returns the timepoint (3D data volume, lowest is 0) from 4D input.
    
    You can save memory by using [1]:
    nifti.dataobj[..., tp]
    instead: see get_nifti_timepoint()
    
    Works with loop_and_save().
    Call directly, or with niftify().
    
    Ref:
    [1]: http://nipy.org/nibabel/images_and_memory.html
    """
    # Replicating seg_maths -tp

    tp = int(tp)
    
    if len(data.shape) < 4:
        print("Data has fewer than 4 dimensions. Doing nothing...")
        output = data
    else:
        if data.shape[3] < tp:
            print("Data has fewer than {0} timepoints in its 4th dimension.".format(tp))
            output = data
        else:
            output = data[:,:,:,tp]
    return output
        
    # elif len(data.shape) > 4:
        # print("Data has more than 4 dimensions! Assuming the 4th is time ...")
    # End get_timepoint() definition
    
@niftify
def grey_complement( data ):
    """Get the greyscale complement.
    
    Returns the max(data) - data at each voxel.
    """
    max = np.max(data)
    return max - data
    # End of grey_complement() definition
    
@niftify
def invert( data, verbose=False ):
    """Invert a binary NIfTI image or data array.
    """
    # Ensure we're working with binary data (convert to int afterwards, just in case)
    mask = data > 0
    mask = mask.astype(int)
    
    if verbose:
        print("Inverting image data ...")
    
    # Use default structure element for this 3D mask
    return np.logical_not(mask).astype(int)
    # End of invert() definition
    
@niftify
def linear_map( values, new_min=0.0, new_max=1.0 ):
    """Return a NumPy array of linearly scaled values between new_min and new_max.
    
    Equivalent to Matlab's mat2gray, I believe.
    """
    
    new_values = (((values - values.min()) * (new_max - new_min)) / (values.max() - values.min())) + new_min

    return new_values
    # End linear_map() definition
    
@niftify
def lower_threshold( data, threshold=0.0, new_value=0.0 ):
    """Set all voxels < threshold to a new_value (default 0) (does not binarise output).
    
    To do: accept percentages of the maximum.
    """

    print "  Thresholding at: {0}".format(threshold)
    low_indices = data < threshold
    data[low_indices] = new_value
    
    return data
    # End lower_threshold() definition
    
@niftify
def lower_threshold_bin( data, threshold=0.0 ):
    """Set all voxels < threshold to 0 (binary output).
    """
    
    # data, hdr = check_get_nifti_data(input)
    data = lower_threshold(data, threshold)
    
    # Convert to binary
    mask = data > 0.0
    mask = mask.astype(int)

    return mask
    # End lower_threshold_bin() definition
    
@niftify
def power_n( data, power=2.0, verbose=False ):
    """Returns the voxel-wise power of either a data array or a NIfTI-1 image object.
    
    To get the nth root, use 1/power.
    nth root of either a data array or a NIfTI image object.
    
    E.g. power_n(nii, 1./3) returns a NIfTI image whose voxel values are cube rooted.
    """
    
    if verbose:
        if power < 1.0:
            print("  Getting root-{0} of image data ...".format(power))
        else:
            print("  Getting image data^{0} ...".format(power))

    return data ** power
    # End of power_n() definition
    
@niftify
def log( data, base=np.e, verbose=False ):
    """Returns the voxel-wise logarithm (default natural log: base e) of either a data array or a NIfTI-1 image object.
    """
    
    if verbose:
        print("  Getting the logarithm (base {0}) at each voxel ...".format(base))
    if base is np.e:
        data_out = np.log(data)
    else:
        data_out = np.log(base ** data) / np.log(base)
        
    return data_out
    # End of log() definition
    
@niftify
def n_power( data, value=np.e ):
    """Just like power_n, except input value is raised to the power of the voxels in a NIfTI image.
    
    value^[NIfTI voxels] (voxel-wise).
    
    By default, value = np.e, so n_power(nifti) returns the inverse of log(nifti, base=np.e).
    """
    return value ** data
    # End of n_power() definition
    
@niftify
def remove_nans(data, replacement_value=0.0):
    """Remove all NaNs from a NIfTI image, and replace with another value (will convert to np.float).
    """
    
    print("  Replacing {0} NaNs with {1} ...".format(np.count_nonzero(np.isnan(data)), replacement_value))
    
    # Wash away the dirty NaNs
    data[np.isnan(data)] = np.float(replacement_value)
    
    return data
    # End remove_nans() definition
    
@niftify
def replace_values(data, search_value=np.nan, replacement_value=0.0):
    """Replace values matching search_value in data with replacement_value.
    """
    
    print("  Replacing all '{0}'s in data with '{1}' ...".format(search_value, replacement_value))
    
    data[data == search_value] = np.float(replacement_value)
    
    return data
    # End replace_values() definition
    
@niftify
def trim_intensity_extremes( data, percentile=5.0, max_to_zero=True ):
    """Sets the lower and upper percentile range of a NIfTI's intensity histogram to 0.
    
    If max_to_zero is False, sets the upper range to the new maximum, rather than 0.
    """
    
    pc_maxmin = [percentile, 100-percentile]
    
    pc_intensities = np.percentile(data, pc_maxmin)
    data[data < pc_intensities[0]] = 0
    
    if max_to_zero:
        data[data > pc_intensities[1]] = 0
    else:
        data[data > pc_intensities[1]] = pc_intensities[1]
        
    return data
    
@niftify
def upper_threshold( data, threshold=0.0, new_max=0.0 ):
    """Set all voxels > threshold to new_max (does not binarise output).
    
    new_max is the replacement maximum value.
    
    Works with loop_and_save().
    Call directly, or with niftify().
    """

    # data, hdr = check_get_nifti_data(input)
    print "  Thresholding at: {0}".format(threshold)
    high_indices = data > threshold
    data[high_indices] = new_max
    
    return data
    # End upper_threshold() definition
    
# Data-only functions with a little more complexity
# -------------------------------------------------

@niftify
def generate_rim_mask(data, iterations=1):
    """Generate a rim mask consisting of the XOR (exclusive OR) region between an input mask (binarised) dilated by iterations and eroded by iterations.
    
    """
    
    dilated = dilate(data, iterations)
    eroded = erode(data, iterations)
    rim_mask_data = np.subtract(dilated, eroded)
    
    return rim_mask_data
    # End generate_rim_mask() definition
    
@niftify
def threshold_concomp_volume( data, volume=100, uorl='lower', replace_value=0 ):
    """Given a binary image, remove all connected component objects with voxel volume either lower or greater than the input.
    
    If urol='upper', all objects larger than volume will be removed; default is 'lower'.
    
    """
    
    if uorl is not 'upper' and uorl is not 'lower':
        raise ValueError('uorl input must be "upper" or "lower"')
    
    # Make a binary copy
    mask = binarise(data)

    # This is the travelling 'mask' which covers each voxel and is used to search for connections. 3,1 is a 3D cross, ie, 6-connected component. 3,2 includes the diagonals and only excludes the 8 corners, ie, 18-connected component. The latter is more inclusive.
    structure = ndimage.generate_binary_structure(3,1)
    
    # Label each object
    label_im, numpatches = ndimage.label(mask, structure)
    print("{0} 6-connected objects labelled ...".format(numpatches))
    
    if numpatches > 0:
        # Sum the values under each object in the original binary mask, thus getting the size
        # So, numpatches[0] has size == sizes[0]
        sizes = ndimage.sum(mask, label_im, range(1, numpatches+1))
        
        labelled = np.zeros_like(label_im)
        patch_list = range(1, numpatches+1)
        
        for counter, size in enumerate(sizes):
            
            if uorl is 'lower':
                if size < volume:
                    labelled[label_im == patch_list[counter]] = replace_value
                else:
                    labelled[label_im == patch_list[counter]] = size
            else:
                if size > volume:
                    labelled[label_im == patch_list[counter]] = replace_value
                else:
                    labelled[label_im == patch_list[counter]] = size
                
            print("{0} / {1}".format(counter, len(patch_list)))
            sys.stdout.flush()
                
    bin_labelled = binarise(labelled)
    
    return apply_mask(data, bin_labelled)
    
# Functions which perform some analysis and return values
# -------------------------------------------------------
    
def cnr(foreground, background, noise):
    """Report CNR. Returns CNR value.
    
    Reference:
    Magnotta et al. (2006). Measurement of Signal-to-Noise and Contrast-to-Noise in the fBIRN Multicenter Imaging Study. Journal of Digital Imaging, June 2006, Volume 19, Issue 2, pp 140-147. DOI: 10.1007/s10278-006-0264-x
    """
    
    cnr = (np.mean(foreground) - np.mean(background)) / np.std(noise)
    
    print("  CNR: {0}".format(cnr))
    return cnr
    # End cnr() definition
    
def snr(signal, noise):
    """Report SNR. Returns 3 SNR values.
    
    References:
    [1] MRI From Picture to Proton, p. 204.
    [2] http://www.mr-tip.com/serv1.php?type=db1&dbs=SNR . Magnotta et al. (2006) calculate SNR with this formula, but stick to GM.
    """
    
    print("  Mean signal: {0}; mean noise: {1}".format(np.mean(signal), np.mean(noise)))
    # print("  signal shape: {0}; noise shape: {1}".format(signal_data.shape, noise.shape))
    
    snr_1 = 0.66 * np.mean(signal) / np.std(noise)           # [1]
    snr_2 = 0.66 * np.mean(signal) / (0.7979 * np.mean(noise))
    snr_MRTIP = np.mean(signal) / np.std(noise)              # [2]
    
    print("  SNR 1 (0.66): {0}; SNR 2 (0.66/0.7979): {1}; SNR (MR-TIP): {2}".format(snr_1, snr_2, snr_MRTIP))
    
    return snr_1, snr_2, snr_MRTIP
    # End snr() definition
    
# Functions which work with NIfTI objects directly
# ------------------------------------------------
    
def get_nifti_timepoint( nifti, tp=0 ):
    """Returns the timepoint (final dimension) from input NIfTI.
    
    If input is 4D, output will be 3D. If 3D, will be 2D (etc.)
    
    This saves memory by not loading the full ND data array; only the timepoint requested is loaded into memory.
    See also get_timepoint().
    Ref:
    [1]: http://nipy.org/nibabel/images_and_memory.html
    """
    # Replicating seg_maths -tp

    tp = int(tp)
    
    print("  NIfTI has {0} dimensions. Getting {1} from last dimension...".format(nifti.shape, tp))
    
    data = np.copy(nifti.dataobj[..., tp])
    
    return nib.Nifti1Image(data, nifti.get_sform(), nifti.get_header())
    # End get_nifti_timepoint() definition
    
# Functions which call NiftK programs
# -----------------------------------

def niftk_seg_EM(input_path, output_path, mask_path=None, string_args=None):
    """Just call seg_EM.
    
    Input, masking and output are handled. Bundle all other arguments into string_args, e.g.:
    string_args=r"-priors 1 C:\test.nii.gz"
    
    NB: since string_args is .split(',') on commas, if any paths in string_args contain a comma, they will be split and this won't work.
    """

    nk = init_niftk()
    
    calling_list = [nk.seg_EM,
                        '-in', input_path,
                        '-out', output_path]
                            
    if mask_path:
        calling_list.extend(['-mask', mask_path])
        
    if string_args:
        # split additional arguments by spaces, and extend the list (not append)
        calling_list.extend(string_args.split(','))
    
    if not os.path.isfile(output_path):
        subprocess_call_and_log(calling_list, os.path.dirname(output_path))
    else:
        print("  Output file, {0} already exists!".format(output_path))
        
    return output_path
    # End niftk_seg_EM() definition
    
def niftk_reg_aladin(reference_path, floating_path, output_path, affine_output_path=None, ref_mask_path=None, flo_mask_path=None, string_args=None):
    """Just call reg_aladin
    
    ! Not complete.
    """
    
    nk = init_niftk()
    
    calling_list = [nk.reg_aladin,
                        '-ref', reference_path,
                        '-flo', floating_path,
                        '-res', output_path]
                            
    if ref_mask_path:
        calling_list.extend(['-rmask', ref_mask_path])
    if flo_mask_path:
        calling_list.extend(['-fmask', flo_mask_path])
    if not affine_output_path:
        out_dir = os.path.dirname(output_path)
        out_name = (os.path.basename(output_path)).split(os.extsep)[0]
        affine_output_path = os.path.normpath(os.path.join(out_dir, out_name + '.txt'))
    calling_list.extend(['-aff', affine_output_path])
        
    if string_args:
        # split additional arguments by spaces, and extend the list (not append)
        calling_list.extend(string_args.split(','))
    
    if not os.path.isfile(output_path):
        subprocess_call_and_log(calling_list, os.path.dirname(output_path))
    else:
        print("  Output file, {0} already exists!".format(output_path))
        
    return output_path
    # End niftk_reg_aladin() definition

def niftk_reg_f3d(reference_path, floating_path, output_path, affine_input_path=None, ref_mask_path=None, flo_mask_path=None, string_args=None):
    """Call reg_f3d."""
    
    nk = init_niftk()
    
    calling_list = [nk.reg_f3d,
                        '-ref', reference_path,
                        '-flo', floating_path,
                        '-res', output_path]
                            
    if ref_mask_path:
        calling_list.extend(['-rmask', ref_mask_path])
    if flo_mask_path:
        calling_list.extend(['-fmask', flo_mask_path])
    if affine_input_path:
        calling_list.extend(['-aff', affine_input_path])
        
    if string_args:
        # split additional arguments by spaces, and extend the list (not append)
        calling_list.extend(string_args.split(','))
    
    if not os.path.isfile(output_path):
        subprocess_call_and_log(calling_list, os.path.dirname(output_path))
    else:
        print("  Output file, {0} already exists!".format(output_path))
        
    return output_path
    # End niftk_reg_f3d() definition
    
def niftk_reg_jacobian(reference_path, transformation_path, output_path, log=False):
    """Call reg_jacobian."""
    
    nk = init_niftk()
    
    calling_list = [nk.reg_jacobian,
                        '-ref', reference_path,
                        '-trans', transformation_path]
                            
    if not log:
        calling_list.extend(['-jac', output_path])
    else:
        calling_list.extend(['-jacL', output_path])
    
    if not os.path.isfile(output_path):
        subprocess_call_and_log(calling_list, os.path.dirname(output_path))
    else:
        print("  Output file, {0} already exists!".format(output_path))
        
    return output_path
    # End niftk_reg_jacobian() definition
    
def niftk_reg_resample(reference_path, floating_path, transformation_path, output_path, interpolation=1):
    """Call reg_resample.
    
    interpolation: 0: NN; 1: (tri)linear (default); 2: cubic
    """
    
    nk = init_niftk()
    
    calling_list = [nk.reg_resample,
                        '-ref', reference_path,
                        '-flo', floating_path,
                        '-trans', transformation_path,
                        '-res', output_path,
                        '-inter', str(interpolation)]
    
    if not os.path.isfile(output_path):
        subprocess_call_and_log(calling_list, os.path.dirname(output_path))
    else:
        print("  Output file, {0} already exists!".format(output_path))
        
    return output_path
    # End niftk_reg_resample() definition
    
def niftk_average_images(input_list, output_path):
    """A simple interface for NiftyReg's reg_average -avg for NIfTI images.
    
    reg_average will save to output_path; this function returns None.
    
    If you don't need to save output, use average_nifti_list()
    """
    
    nk = init_niftk()
    
    calling_list = [nk.reg_average,
                        output_path,
                        '-avg']
                        
    calling_list.extend(input_list)
    
    if not os.path.isfile(output_path):
        subprocess_call_and_log(calling_list, os.path.dirname(output_path))
    else:
        print("  Output file, {0} already exists!".format(output_path))
    # End niftk_average_images() definition
    
def niftk_make_isotropic( input_path, output_path ):
    """Given a path to a NIfTI, save a NIfTI with approximately isotropic voxels.
    
    Currently uses reg_tools.
    """
    
    nk = init_niftk()
    
    calling_list = [nk.reg_tools,
                        '-in',
                        input_path,
                        '-out',
                        output_path,
                        '-iso']
        
    if not os.path.isfile(output_path):
        subprocess_call_and_log(calling_list, os.path.dirname(output_path))
    else:
        print("  Output file, {0} already exists!".format(output_path))
        
    return output_path
    # End of niftk_make_isotropic() definition
    
def subprocess_call_and_log(calling_list, log_directory=None):
    """Call a subprocess and write log files in the given output directory."""
    
    import subprocess
    
    print("subprocess will be called with: \n\t{0}".format(calling_list))
    sys.stdout.flush()
    
    if log_directory is not None:
        log_directory = os.path.normpath(os.path.join(log_directory))
        check_create_directories([log_directory])
        
        output_logfile = os.path.normpath(os.path.join(log_directory, 'subprocess_log_output.txt'))
        f_log = open(output_logfile, 'w')
        error_logfile = os.path.normpath(os.path.join(log_directory, 'subprocess_log_error.txt'))
        f_errlog = open(error_logfile, 'w')
        
        subprocess.call(calling_list, stdout=f_log, stderr=f_errlog)
        f_log.close()
        f_errlog.close()
        
    else:
        subprocess.call(calling_list)
    # End subprocess_call_and_log() definition
    
def function_star(list):
    """Call a function (first item in list) with a list of arguments.
    
    The * will unpack list[1:] to use in the actual function
    """
    fn = list.pop(0)
    return fn(*list)
    
# Set up NiftK
def init_niftk():
    """Get paths to NiftK functions.
    
    Usage: nk = init_niftk()
    """
    return mm_niftk.MM_Niftk()

def main():
    """Mousemorph functions."""
    
    alert_user("There is not much point running this module directly; import it with \"import mm_functions as mmfn\" and use the functions defined within like this:\n\n\tnifti_c = mmfn.add(nifti_a, nifti_b)\n\t...")

if __name__ == '__main__':
    main()
    
# End