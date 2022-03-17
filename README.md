# PlotNIfTI

## Table of contents

* [General info](#general-info)
* [Example](#example)
* [Dependencies](#dependencies)

## General info

| Features                                    | Status              |
|---------------------------------------------|---------------------|

Bastien Rigaud, PhD Laboratoire Traitement du Signal et de l'Image (LTSI), INSERM U1099 Campus de Beaulieu, Université
de Rennes 1 35042 Rennes, FRANCE bastien.rigaud@univ-rennes1.fr

## Example

<p align="center">
<img src="example/screenshot_axial.png" height=200>    
</p>

<p align="center">
<img src="example/screenshot_sagittal.png" height=150>
<img src="example/screenshot_coronal.png" height=150>
</p>

```python
from PlotNIfTI import PlotNifti

def main():
    image_path = r"C:\Data\Data_test\plot\image.nii.gz"
    segmentation_paths = [r"C:\Data\Data_test\plot\Prostate.nii.gz",
                          r"C:\Data\Data_test\plot\Bladder.nii.gz",
                          r"C:\Data\Data_test\plot\Rectum.nii.gz"]

    for view in ['axial', 'sagittal', 'coronal']:
        output_path = r"example\screenshot_{}.png".format(view)
        PlotNifti(image_path=image_path, segmentation_paths=segmentation_paths, output_path=output_path, view=view,
                  get_at_centroid=True)
```

## Dependencies

```
pip install -r requirements.txt
```
