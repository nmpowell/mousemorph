#!/usr/bin/python

"""MouseMorph, an automatic mouse MR image processor

This Python script defines the MM_Niftk class, which reads the MouseMorph config file and gives paths to the most commonly-used NiftK programs, like reg_aladin and seg_maths. The config_file_path can either be given as an argument, or must be in the correct directory (default is the same as this script).

To use this class:
    from mm_niftk import *
    nk = MM_Niftk()
    
    # for non-default config_file_path:
    # nk = MM_Niftk(config_file_path=r'C:\Desktop\mm_config_user.ini')

The string paths are then accessible via nk.reg_aladin; nk.seg_maths and can be run via Python with, e.g.:
    import subprocess
    subprocess.call([nk.reg_aladin, '--help'])
    
Previous versions and other programs may optionally be accessible via, e.g.:
    nk.xxx_reg_aladin
where xxx is the variable name in config_file_path.

To do:
- Check whether essential apps exist, using niftk_apps

"""

import os
import sys
import glob
import ConfigParser

__author__ = 'Nick Powell (PhD student, CMIC & CABI, UCL, UK), nicholas.powell.11@ucl.ac.uk'
__version__ = '0.2.20150311'
__created__ = '2014-04-13, Sunday'

class MM_Niftk(object):

    # defaults
    __currentdir__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    config_file_path = os.path.join(__currentdir__, 'mm_config_user.ini')
    niftk_apps = ['reg_aladin', 'reg_average', 'reg_transform', 'reg_tools', 'reg_resample', 'reg_jacobian', 'reg_f3d', 'seg_maths', 'seg_EM', 'seg_LabFusion', 'seg_stats', 'seg_LoAd', 'niftkN4BiasFieldCorrection', 'niftkAtrophyCalculator', 'niftkMTPDbc']
    
    # File paths ending with the following will not be available
    exclude_tuple = ('.bmp', '.conf', '.dll', '.ico', '.lib', '.sh', '.so', '.txt', '.xml')
    
    def __init__(self, config_file_path=config_file_path, niftk_bin_path='', niftk_apps=niftk_apps, exclude_tuple=exclude_tuple):
    
        parser = ConfigParser.ConfigParser()
        parser.read(config_file_path)
        self.niftk_bin_path = niftk_bin_path
        
        # Check for, and keep, existing directories
        for option in parser.options('NiftKPath'):
        
            possible_niftk_path = os.path.normpath(os.path.join(parser.get('NiftKPath', option)))
            if os.path.isdir(possible_niftk_path):
                setattr(self, option, possible_niftk_path)
                
                all_files = glob.glob(os.path.normpath(os.path.join(possible_niftk_path, '*')))
                
                # Use the first working directory as default, so, e.g., nk.reg_aladin will always work if at least one directory is valid
                if self.niftk_bin_path is '':
                    print("  {0} is the first valid directory from settings file ({1})".format(possible_niftk_path, config_file_path))
                    self.niftk_bin_path = possible_niftk_path
                
                    for file_path in all_files:
                        # Check against a tuple of definite non-executables. Everything else will be accessible.
                        if not file_path.endswith(exclude_tuple):
                            app_name = os.path.basename(file_path).split(os.extsep)[0]
                            setattr(self, app_name, file_path)
                        
                # Define executable locations for non-default paths
                # You can have multiple working NiftK directories so as to use old installations
                for file_path in all_files:
                
                    # Check against a tuple of definite non-executables. Everything else will be accessible.
                    if not file_path.endswith(exclude_tuple):
                        app_name = os.path.basename(file_path).split(os.extsep)[0]
                        
                        # Prepend the config_file variable name
                        setattr(self, option + '_' + app_name, file_path)
                 
            else:
                print "Specified path to NiftK binaries, ({0}), is not a directory. Check you have NiftK installed there.".format(possible_niftk_path)
                
        if self.niftk_bin_path is '':
            print("Warning: no valid path to NiftK binaries found.")
            
def main():
    print("There's nothing to run here; use \'from mm_niftk import *\' instead...")
    print __doc__

if __name__ == '__main__':
    main()
    
# End