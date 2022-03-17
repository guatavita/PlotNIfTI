# Created by Bastien Rigaud at 17/03/2022
# Bastien Rigaud, PhD
# Laboratoire Traitement du Signal et de l'Image (LTSI), INSERM U1099
# Campus de Beaulieu, Université de Rennes 1
# 35042 Rennes, FRANCE
# bastien.rigaud@univ-rennes1.fr
# Description:

from PlotNIfTI import PlotNifti

def main():
    image_path = r"C:\Data\Data_test\plot\image.nii.gz"
    segmentation_paths = [r"C:\Data\Data_test\plot\Prostate.nii.gz",
                          r"C:\Data\Data_test\plot\Bladder.nii.gz",
                          r"C:\Data\Data_test\plot\Rectum.nii.gz"]

    for view in ['axial', 'sagittal', 'coronal']:
        output_path = r"example\screenshot_{}.png".format(view)
        PlotNifti(image_path=image_path, segmentation_paths=segmentation_paths, output_path=output_path, view=view,
                  show_contour=True, show_filled=True, transparency=0.20, get_at_centroid=True)

if __name__ == '__main__':
    main()
