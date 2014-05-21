MouseMorph
==========

Tools for MRI mouse brain morphometry.

MouseMorph is a pipeline, or modular set of tools, for automatically analysing mouse brain MRI scans. It enables fully automatic [Voxel- and Tensor-Based Morphometry][ashb_vbm_2000] (VBM, TBM) on large cohorts of high-resolution images. It is employed at the UCL Centre for Advanced Biomedical Imaging (CABI) for [phenotyping](http://en.wikipedia.org/wiki/Phenotype) mice based on *in-vivo* and *ex-vivo* scans.

[!result image: TBM] [!result image: VBM]

The primary distinction from clinically-focussed tools like [SPM]() and [FSL]() is a robust set of pre-processing steps, unique to -- or with customisations for -- the preclinical paradigm:
- Extraction of multiple subjects from a single scan image
- Orientation to a standard space, from any initial orientation
- Mouse brain extraction (skull stripping / brain masking)
- Tissue segmentation

Many of these steps are atlas-based (requiring prior knowledge). A few mouse atlases are freely available to download (see links below). For a further introduction, see the [**poster**][mm_poster]. For a more detailed explanation, see the accompanying paper, [_**Powell, N.M., (2014) Fully-automated high-throughput phenotyping of mouse brains with µMRI, with application to the Tc1 model of Down syndrome**_][mm_paper].

MouseMorph is open-source, cross-platform, and written in Python. In addition to the code included in this repository, various elements from the corresponding [paper]() are open-source and available on [Figshare]().

Developed at the [UCL Centre for Medical Image Computing (**CMIC**)](http://cmic.cs.ucl.ac.uk/) and the [UCL Centre for Advanced Biomedical Imaging (**CABI**)](http://www.ucl.ac.uk/cabi) by Nick Powell (nicholas.powell.11@ucl.ac.uk) and others.

Paper: [![MouseMorph paper thumbnail; click for PDF](docs/mousemorph_poster_thumbnail.png "MouseMorph paper thumbnail; click for PDF")][mm_paper] Poster: [![MouseMorph poster thumbnail; click for PDF](docs/mousemorph_poster_thumbnail.png "MouseMorph poster thumbnail; click for PDF")][mm_poster]

## Getting Started
1. Set up Python
2. Download MouseMorph
3. Download a mouse atlas

## Download wild-type mouse brain data


## Links
- [UCL Centre for Medical Image Computing (CMIC)](http://cmic.cs.ucl.ac.uk/)
- [UCL Centre for Advanced Biomedical Imaging (CABI)](http://www.ucl.ac.uk/cabi)

CMIC software:
- [NiftyReg](http://sourceforge.net/projects/niftyreg/)
- [NiftySeg](http://sourceforge.net/projects/niftyseg/)
- more [Nifty tools](http://cmic.cs.ucl.ac.uk/home/software/)

Mouse atlases:
- [NUS Mouse Atlas (Singapore)](http://www.bioeng.nus.edu.sg/cfa/mouse_atlas.html)
- [MRM NeAt Mouse Atlas (Florida)](http://brainatlas.mbi.ufl.edu/)

## License
MouseMorph is distributed under the [BSD 3-clause license](LICENSE).

© 2014 Nick Powell and [University College London](http://www.ucl.ac.uk/), UK



[mm_paper]: docs/paper.pdf
[mm_poster]: docs/Nick_Powell-20131016-MouseMorph_MRI_Mouse_Phenotyping-Poster_A0_portrait.pdf
[ashb_vbm_2000]: http://www.fil.ion.ucl.ac.uk/~karl/Voxel-Based%20Morphometry.pdf "(PDF) Ashburner (2000): Voxel-Based Morphometry --- The Methods"