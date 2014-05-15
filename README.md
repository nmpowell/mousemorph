MouseMorph
==========

Tools for MRI mouse brain morphometry.

MouseMorph is a pipeline, or modular set of tools, for automatically analysing mouse brain MRI scans. It enables fully automatic Voxel- and Tensor-Based Morphometry (VBM, TBM) on large cohorts of high-resolution images. It is employed at the UCL Centre for Advanced Biomedical Imaging (CABI) for [phenotyping](http://en.wikipedia.org/wiki/Phenotype) mice based on both in-vivo and ex-vivo scans.

The primary distinction from clinically-focussed tools like [SPM]() and [FSL]() is a robust set of pre-processing steps, unique to the preclinical paradigm:
- Extraction of multiple subjects from a single scan image
- Orientation to a standard space, from any initial orientation
- Mouse brain extraction (skull stripping / brain masking)
- Tissue segmentation

Many of these steps are atlas-based, (requiring prior knowledge).

MouseMorph is open-source, cross-platform, and written in Python (it relies upon several separate tools).

Developed at the [UCL Centre for Medical Image Computing (**CMIC**)](http://cmic.cs.ucl.ac.uk/) and the [UCL Centre for Advanced Biomedical Imaging (**CABI**)](http://www.ucl.ac.uk/cabi) by Nick Powell (nicholas.powell.11@ucl.ac.uk) and others.

License
-------
MouseMorph is distributed under the BSD 3-clause license.

Getting Started
---------------

Links
-----
- [UCL Centre for Medical Image Computing (CMIC)](http://cmic.cs.ucl.ac.uk/)
- [UCL Centre for Advanced Biomedical Imaging (CABI)](http://www.ucl.ac.uk/cabi)

CMIC software:
- [NiftyReg](http://sourceforge.net/projects/niftyreg/)
- [NiftySeg](http://sourceforge.net/projects/niftyseg/)
- more [Nifty tools](http://cmic.cs.ucl.ac.uk/home/software/)

Mouse atlases:
- [NUS Mouse Atlas (Singapore)](http://www.bioeng.nus.edu.sg/cfa/mouse_atlas.html)
- [MRM NeAt Mouse Atlas (Florida)](http://brainatlas.mbi.ufl.edu/)