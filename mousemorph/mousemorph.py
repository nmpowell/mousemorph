#!/usr/bin/python

"""MouseMorph, an automatic mouse MR image processor

This is the base MouseMorph program. Run it with Python.

    Usage
        python mousemorph.py --help
        
        Orient brains to standard space
        -------------------------------
            python mousemorph.py orient <arguments>
            
        Non-uniformity correct brains
        -----------------------------
            python mousemorph.py nuc <arguments>
        
"""

# To do
# -----
# 1.    if -i has more than one argument following it, use as a list of input directories (or files) and combine all input files into the same list. Likewise for other directory arguments.

# corresponding == FSL FLIRT's "secondary"
# adapt to accept -corr [dir] [filter]
# if input is a single image and there's no filter here, all images in dir will be oriented as per the single input
# if input is a single image and there is a filter here, all images in dir matching the filter will be oriented as per the single input
# if input is a directory and there's no filter here, corresponding images in dir will be oriented
# if input is a directory and there is a filter here, corresponding images in dir which also matching the filter will be oriented

# Replace mm_multi.py with: run mousemorph.py multi action1 [-arg1 -arg2 -arg3 'param3'] action2 [-arg1 'param1' -arg2] action3 [-arg1 -arg2]

# Run any script from this one

import os
import sys
import glob
import time
import argparse
import subprocess
from itertools import chain
from datetime import datetime

# import test_go
import mm_functions as mmfn

__author__ = 'Nick Powell (PhD student, CMIC & CABI, UCL, UK), nicholas.powell.11@ucl.ac.uk'
__created__ = '2015-06-28'
    
def notify_start():
    """Give the user some helpful notes at the launch of the script."""
    time_format = '%H:%M:%S %p on %b %d, %Y (%Z)'
    time_start = time.strftime(time_format)
    mmfn.alert_user("Start time is {0} ...".format(time_start))
    return time_start

def notify_complete(time_start=None, log_location=''):
    time_format = '%H:%M:%S %p on %b %d, %Y (%Z)'
    time_stop = time.strftime(time_format)
    time_diff = datetime.strptime(time_stop, time_format) - datetime.strptime(time_start, time_format)
    mmfn.alert_user("Completion time is {0}. Took {1}.".format(time_stop, time_diff))
    return
    
class MouseMorph(object):
    """Define all the necessary arguments passed to MouseMorph programs.
    
    """

    def __init__(self):
        command_parser = argparse.ArgumentParser("argp_mousemorph", description = "MouseMorph, an automatic mouse MR image processor.",
        usage = """mousemorph <command> [<arguments>], where <command> can be any of: \n\textract, \n\torient, \n\tnuc, \n\tintstan, \n\t...""")
        
        command_parser.add_argument('command', help="MouseMorph program to be run.")
        first_arg = command_parser.parse_args(sys.argv[1:2])
        
        if not hasattr(self, first_arg.command):
            print("command '{0}' not recognised.".format(first_arg.command))
            command_parser.print_help()
            sys.exit(1)
            
        # Top-level parser with universal arguments
        top_parser = argparse.ArgumentParser("MouseMorph user input parser", description = "MouseMorph, an automatic mouse MR image processor.",
        usage = "mousemorph <command> [<arguments>]")
        
        # Input and output
        top_parser.add_argument('-i', '--input', dest='input', metavar='<directory> or <file path>', help='Input file (NIfTI-1, *.nii.gz or *.nii) or directory [current]')
        top_parser.add_argument('-o', '--output', dest='output', metavar='<directory>', help='Output directory [input directory]')
        top_parser.add_argument('-onp', '--out_name_prepend', metavar='<string>', help='Prepend this string to output name', default="")
        top_parser.add_argument('-ona', '--out_name_append', metavar='<string>', help='Append this string to output name', default="")
        top_parser.add_argument('-uz', '--unzip', action='store_true', help='Uncompressed output files [compressed]')
        top_parser.add_argument('-no', '--no_output', dest='no_output', action='store_true', help='Don\'t save output files [False]')
        top_parser.add_argument('-ow', '--overwrite', action='store_true', help='Overwrite existing output files [skip]')
        top_parser.add_argument('-dt', '--delete_temp', action='store_true', help='Delete temp files upon completion [False]')
        
        # Mask arguments
        top_parser.add_argument('-m', '--mask', dest='mask', metavar='<mask directory> or <mask file>', help='Mask directory or file', required=False)
        top_parser.add_argument('-mf', '--mask_filter', dest='mn_filter', metavar='<filter>', help="If mask is a directory, filter files ['']", default='')
        
        # Filters
        top_parser.add_argument('-if', '--in_filter', dest='in_filter', metavar='<filter>', help="If input is a directory, filter files ['']", default='')
        
        # Processing options
        top_parser.add_argument('-ds', '--downsample', dest='downsample', metavar='<factor>', help='Downsampling factor [0, off]\n\t(Downsampling input files may speed up processing, at the expense of accuracy.)', default=0, type=float)
        top_parser.add_argument('-par', '--parallel', dest='parallel', action="store_true", help='Use multiple cores to process in parallel using multiprocessing [off]')
        top_parser.add_argument('-rev', '--reverse_input', dest='reverse_input', action="store_true", help='Run through input files in reverse order')
        
        top_parser.add_argument('-v', '--verbose', action="store_true", help="Verbose output")
        
        # Add command-specific arguments
        parser = getattr(self, first_arg.command + '_args')(top_parser)
        self.args = parser.parse_args(sys.argv[2:])
        
        # Sanitise arguments & directories
        self.args = self.sanitise_arguments(self.args)
        if self.args.input:
            # Pre-populate a list of relevant files.
            self.args.input_files_list = mmfn.get_files_list(self.args.input_directory, self.args.input_name_filter, self.args.in_ext_filter)
            
            if self.args.reverse_input:
                self.args.input_files_list = self.args.input_files_list[::-1]
        
        # Run
        print("MouseMorph {0} will be run with arguments: \n\t{1}".format(first_arg.command, vars(self.args)))
        time_start = notify_start()
        
        getattr(self, first_arg.command)()
        
        notify_complete(time_start)
        
    # def sanitise_arguments(self, args):
        # """Windows: only necessary if the user has supplied directories ending with a "\" (os.sep), which argparse assumes was intentional and adds the final user quote to the end of the string. The user shouldn't supply that "\", really, but just in case..."""
        # for name in args.__dict__.keys():
            # try:
                # s = getattr(args, name)
                # setattr(args, name, s.rstrip(os.sep).rstrip('"'))
            # except AttributeError:
                # pass
            # else:
                # break
        # return args
        
    def add_arg_csv(self, parser, req=False):
        parser.add_argument('-csv', '--csv', dest='csv_path', metavar='<.CSV file path>', help='Path to .CSV file', required=req)
        parser.add_argument('-col', '--column', dest='column', metavar='<int int ... int>', nargs='+', help='Column number(s) [0 1]', default=1, type=int, required=req)
        return parser
        
    def add_arg_mask(self, parser, req=False):
        # Mask arguments
        parser.add_argument('-m', '--mask', dest='mask', metavar='<mask directory> or <mask file>', help='Mask directory or file', required=req)
        parser.add_argument('-mf', '--mask_filter', dest='mn_filter', metavar='<filter>', help="If mask is a directory, filter files ['']", default='')
        return parser
        
    def add_arg_list(self, parser, req=False):
        parser.add_argument('-l', '--list', dest='list', metavar='<directory> or <.CSV file path>', help='Either a directory containing files whose names are to be matched, or a .CSV file whose Nth column will be used as a list of names [current]', required=req)
        parser.add_argument('-lf', '--list_filter', dest='list_filter', metavar='<string>', help="String used to filter list input ['*']", default='*')
        return parser
        
    def bsi_args(self, parser):
        """These are the somewhat-unique requirements for BSI."""
        parser.add_argument('-b', '--baseline', dest='baseline_path', metavar='<baseline NIfTI file path>', help='Full path of the baseline NIfTI image', required=True)
        parser.add_argument('-r', '--repeat', dest='repeat_path', metavar='<repeat NIfTI file path>', help='Full path of the repeat NIfTI image', required=True)
        parser.add_argument('-bm', '--baseline_mask', dest='baseline_mask_path', metavar='<baseline mask NIfTI file path>', help='Full path of the baseline NIfTI image mask', required=True)
        parser.add_argument('-rm', '--repeat_mask', dest='repeat_mask_path', metavar='<repeat mask NIfTI file path>', help='Full path of the repeat NIfTI image mask', required=True)
        return parser
        
    def orient_args(self, parser):
        # Add specific arguments
        parser.add_argument('-at','--atlas', dest='atlas', metavar='<atlas>', help='Atlas directory containing NIfTIs, or a single file.', required=False)
        parser.add_argument('-corr','--corresponding', dest='corresponding', metavar='<corresponding>', help='NIfTI-1 file, or directory of files, to be oriented in the same manner as their correspondingly-named files in input_directory. (As per "secondary" in FSL FLIRT.)')
        parser.add_argument('-res','--resample', dest='resample', action='store_true', help='Also resample output files.')
        parser.add_argument('--allsame', dest='allsame', action='store_true', help='Flag to indicate that all brains are in approximately the same initial orientation. Only the first image will be compared with an atlas and the rest will have the same gross orientation applied. Final minor corrections will be performed individually.')
        parser.add_argument('--allpa', dest='allpa', action='store_true', help='Check all 12 possible principle axis orientations, in case AP is not the greatest dimension.')
        # parser.add_argument('--allpa', dest='allpa', action='store_true', help='Check all 12 possible principle axis orientations, in case AP is not the greatest dimension.')
        return parser
        
    def nuc_args(self, parser):
        # Add specific arguments
        parser.add_argument('-its','--iterations', dest='iterations', metavar='<iterations>', help='Iterations to run [200]', default=200, type=int)
        parser.add_argument('-fwhm', dest='fwhm', metavar='<fwhm>', help='Full width, half maximum [0.15]', default=0.15, type=float)
        parser.add_argument('-ss', dest='subsample', metavar='<factor>', help='Subsampling factor [4]', default=4, type=int)
        parser.add_argument('-nlevels', dest='nlevels', metavar='<nlevels>', help='Number of levels [4]', default=4, type=int)
        parser.add_argument('-conv', dest='convergence', metavar='<convergence>', help='Convergence threshold [0.001]', default=0.001, type=float)
        parser.add_argument('-nhb', dest='nhistbins', metavar='<nhistbins>', help='Number of histogram bins [256]', default=256, type=int)
        return parser
        
    def tails_type(self, str):
        acceptable = ['one', 'two']
        if str not in acceptable:
            raise argparse.ArgumentTypeError("--tails argument must be 'one' or 'two' (default is two, if omitted)")
        else:
            return str
        
    def power_args(self, parser):
        parser.add_argument('--power', dest='power', metavar='<0 < float < 1>', help='Desired power, 1-beta [0.8]', default=0.8, type=float)
        parser.add_argument('--significance', dest='significance', metavar='<0 < float < 1>', help='Desired significance level, alpha [0.05]', default=0.05, type=float)
        parser.add_argument('--detect_difference', dest='detect_difference', metavar='<0 < float < 1>', help='Fractional difference from the control mean to detect', default=0.25, type=float)
        parser.add_argument('--tails', dest='tails', metavar='<string>', help='Tails, one or two [two]', default='two', type=self.tails_type)
        parser = self.add_arg_csv(parser, req=True)
        parser.add_argument('--group', dest='csv_group_filter', metavar='<string>', help='Control group name filter [*]', default='')
        return parser
        
    def pair_args(self, parser):
        parser = self.add_arg_list(parser, req=False)
        parser.add_argument('-col', '--column', dest='column', metavar='<int>', help='Column number [0]', default=0, type=int, required=False)
        parser.add_argument('-i2','--input_2', dest='input_2', metavar='<directory> or <file path>', help='Second input directory containing NIfTIs, or a single file.', required=True)
        parser.add_argument('-fn', '--function', dest='function_name', metavar='<function>', help='MouseMorph function to run on each file', required=False)
        return parser
        
    def loop_args(self, parser):
        parser.add_argument('-fn', '--function', dest='function_name', metavar='<function>', help='MouseMorph function to run on each file', required=False)
        return parser
        
    def seg_EM_args(self, parser):
        # parser = self.add_arg_mask(parser, req=True)
        parser.add_argument('-t', '--tpm', dest='tpm', metavar='<tpm directory> or <tpm file>', help='TPM directory or file', required=False)
        parser.add_argument('-tf', '--tpm_filter', dest='tn_filter', metavar='<filter>', help="If tpm is a directory, filter files ['']", default='')
        parser.add_argument('--priors4D', dest='priors4D', action='store_true', help='Use this flag if the priors are all single 4D NIfTIs rather than individual files per class', required=False)
        parser.add_argument('--nopriors', dest='nopriors', metavar='<int>', help='Number of classes (no TPM inputs)', type=int)
        parser.add_argument('--mrf_beta', dest='mrf_beta', metavar='<0 < float < 1>', help='MRF prior strength [0.4]', default=0.4, type=float)
        parser.add_argument('--max_iter', dest='max_iter', metavar='<int>', help='Maximum number of iterations [100]', default=100, type=int)
        parser.add_argument('--rf_rel', dest='rf_rel', metavar='<0 < float < 1>', help='Prior relaxation factor [0.5]', default=0.5, type=float)
        parser.add_argument('--rf_gstd', dest='rf_gstd', metavar='<float>', help='Prior gaussian regularization [2.0]', default=2.0, type=float)
        return parser
        
    def sanitise_arguments(self, args):
        """ """
        
        args.in_ext_filter = '.nii*'
        
        if not args.unzip:
            args.ext = '.nii.gz'
        else:
            args.ext = '.nii'
        
        if args.input:
            if os.path.isdir(args.input):
                # print ("  Input {0} is a directory ...".format(args.input))
                args.input_name_filter = '*' + args.in_filter + '*'    # use wildcards if provided a directory alone
                args.input_name_filter_exact = args.in_filter
                args.input_directory = os.path.normpath(os.path.join(args.input))
                
            elif os.path.isfile(args.input):
                # print ("  Input {0} is a file ...".format(args.input))
                # Get the filename, removing path and 1+ extensions
                args.input_name_filter = os.path.basename(args.input).split(os.extsep)[0]
                args.input_directory = os.path.dirname(args.input)

            else:
                raise Exception("Input not recognised or does not exist: {0}".format(args.input))
        else:
            args.input_directory = os.getcwd()
            
        if not args.no_output:
            if args.output:
                if os.path.isdir(args.output):
                    # print ("  Output {0} is a directory ...".format(args.input))
                    args.output_directory = os.path.normpath(os.path.join(args.output))
                    
                else:
                    print "Specified output ({0}) is not a directory; creating it ...".format(args.output)
                    args.output_directory = os.path.normpath(os.path.join(args.output))
                    mmfn.check_create_directories([args.output_directory])
            else:
                print "No output directory specified. Setting to input directory ({0}) in case it is required.".format(args.input_directory)
                args.output_directory = args.input_directory
        
        if hasattr(args, 'input_2'):
            if args.input_2:
                if os.path.isdir(args.input_2):
                    # print ("  Input 2 {0} is a directory ...".format(args.input))
                    args.input_name_filter_2 = '*' + args.in_filter + '*'    # use wildcards if provided a directory alone
                    args.input_name_filter_exact_2 = args.in_filter
                    args.input_directory_2 = os.path.normpath(os.path.join(args.input_2))
                    
                elif os.path.isfile(args.input_2):
                    # print ("  Input 2 {0} is a file ...".format(args.input))
                    # Get the filename, removing path and 1+ extensions
                    args.input_name_filter_2 = os.path.basename(args.input_2).split(os.extsep)[0]
                    args.input_directory_2 = os.path.dirname(args.input_2)

                else:
                    raise Exception("Input 2 not recognised or does not exist: {0}".format(args.input_2))
            else:
                args.input_directory_2 = os.getcwd()
                
        if hasattr(args, 'mask'):
            if args.mask:
                args.mask = os.path.normpath(args.mask)
                if os.path.isdir(args.mask):
                    args.mask_name_filter = '*' + args.mn_filter + '*'    # use wildcards if provided a directory alone
                    args.mask_name_filter_exact = args.mn_filter
                    args.mask_directory = os.path.normpath(os.path.join(args.mask))
                    
                elif os.path.isfile(args.mask):
                    # Get the filename, removing path and 1+ extensions
                    args.mask_name_filter = os.path.basename(args.mask).split(os.extsep)[0]
                    args.mask_directory = os.path.dirname(args.mask)
                    
            else:
                args.mask_directory = None
            
        if hasattr(args, 'tpm'):
            if args.tpm:
                args.tpm = os.path.normpath(args.tpm)
                if os.path.isdir(args.tpm):
                    args.tpm_name_filter = '*' + args.tn_filter + '*'    # use wildcards if provided a directory alone
                    args.tpm_name_filter_exact = args.tn_filter
                    args.tpm_directory = os.path.normpath(os.path.join(args.tpm))
                    
                elif os.path.isfile(args.tpm):
                    # Get the filename, removing path and 1+ extensions
                    args.tpm_name_filter = os.path.basename(args.tpm).split(os.extsep)[0]
                    args.tpm_directory = os.path.dirname(args.tpm)
                    
            else:
                args.tpm_directory = None
            
        # Either get a list of strings as file names from a directory, or from a given column of a .CSV file
        if hasattr(args, 'list'):
            if args.list:
                args.list = os.path.normpath(args.list)
                if os.path.isdir(os.path.normpath(args.list)):
                    args.list_names = mmfn.get_names_list(args.list, args.list_filter, extension=args.in_ext_filter)
                # elif os.path.isfile(os.path.normpath(args.list)):
                    # args.column
                
        return args
        
    # Methods which actually run the command the user asked for
    def nuc(self):
        import mm_nuc_n4
        mm_nuc_n4.go(self.args)
    def orient(self):
        import mm_orient
        mm_orient.go(self.args)
    def pair(self):
        import mm_pair
        mm_pair.go(self.args)
    def power(self):
        import mm_powercalc
        mm_powercalc.go(self.args)
    def loop(self):
        import mm_loop
        mm_loop.go(self.args)
    def seg_EM(self):
        import mm_seg_EM_group
        mm_seg_EM_group.go(self.args)
    
def main():
    mm = MouseMorph()
    # print("{0}".format(mm.__dict__))
    print("{0}".format(mm.args))

if __name__ == '__main__':
    main()
    
# End