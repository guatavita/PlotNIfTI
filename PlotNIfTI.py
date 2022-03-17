# Created by Bastien Rigaud at 17/03/2022
# Bastien Rigaud, PhD
# Laboratoire Traitement du Signal et de l'Image (LTSI), INSERM U1099
# Campus de Beaulieu, Université de Rennes 1
# 35042 Rennes, FRANCE
# bastien.rigaud@univ-rennes1.fr
# Description:

import numpy as np
import SimpleITK as sitk
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
from IOTools.IOTools import ImageReaderWriter

from Resample_Class.src.NiftiResampler.ResampleTools import ImageResampler


def compute_centroid(annotation):
    '''
    :param annotation: A binary image of shape [# images, # rows, # cols, channels]
    :return: index of centroid
    '''
    indexes = np.where(np.any(annotation, axis=(1, 2)) == True)[0]
    index_slice = int(np.mean(indexes))
    indexes = np.where(np.any(annotation, axis=(0, 2)) == True)[0]
    index_row = int(np.mean(indexes))
    indexes = np.where(np.any(annotation, axis=(0, 1)) == True)[0]
    index_col = int(np.mean(indexes))
    return (index_slice, index_row, index_col)


class PlotNifti(object):
    def __init__(self, image_path, segmentation_paths=None, output_path=None, show_contour=True, show_filled=True,
                 transparency=0.20, get_at_centroid=True, view='sagittal', intensity_range=[-1000, 1500],
                 segmentation_names=[]):
        '''
        :param image_path: image path
        :param segmentation_paths: list of segmentation nifti paths
        :param output_path: output path to save the image as png
        :param show_contour: True/False to show contour
        :param show_filled: True/False to show mask
        :param transparency: 0-1.0 filled segmentation transparency
        :param get_at_centroid: True to force the position to the centroid of the first segmentation
        :param view: view selection, ['axial', 'sagittal', 'coronal']
        :param intensity_range: list to arbitrary crop the intensity values
        :param segmentation_names: OPTIONAL, if provided will create a colormap with the correspondiung label name
        '''
        if segmentation_paths is None:
            segmentation_paths = []
        self.image_path = image_path
        self.segmentation_paths = segmentation_paths
        self.output_path = output_path
        self.show_contour = show_contour
        self.show_filled = show_filled
        self.transparency = transparency
        self.get_at_centroid = get_at_centroid
        if view not in ['axial', 'sagittal', 'coronal']:
            raise ValueError('View is not recognized, possible choice [axial,sagittal,coronal]')
        self.view = view
        self.intensity_range = intensity_range
        self.data_dict = {}
        self.dataloader = ImageReaderWriter

        self.load_data()
        self.resample_data()
        self.compute_contour()
        self.create_labels()
        self.convert_images()
        self.create_location()
        self.generate_plot()

    def load_data(self):
        image_loader = self.dataloader(filepath=self.image_path)
        self.data_dict['image'] = image_loader.import_data()

        if not isinstance(self.segmentation_paths, list):
            self.segmentation_paths = [self.segmentation_paths]

        for i, segmentation_path in enumerate(self.segmentation_paths):
            temp_loader = self.dataloader(filepath=segmentation_path)
            self.data_dict['segmentation_{}'.format(i)] = temp_loader.import_data()

    def resample_data(self):
        if self.data_dict['image'].GetSpacing() != (1.0, 1.0, 1.0):
            for img_key in ['image'] + [i for i in self.data_dict.keys() if 'segmentation' in i]:
                resampler = ImageResampler()
                if 'segmentation' in img_key:
                    interpolator = 'Nearest'
                else:
                    interpolator = 'Linear'
                empty_value = np.min(sitk.GetArrayFromImage(self.data_dict[img_key]))
                self.data_dict[img_key] = resampler.resample_image(self.data_dict[img_key],
                                                                   output_spacing=(1.0, 1.0, 1.0),
                                                                   interpolator=interpolator,
                                                                   empty_value=empty_value)

    def compute_contour(self):
        for seg_key in [i for i in self.data_dict.keys() if 'segmentation' in i]:
            contour_filter = sitk.BinaryContourImageFilter()
            contour_filter.SetFullyConnected(False)
            contour_filter.SetNumberOfThreads(0)
            self.data_dict[seg_key.replace('segmentation', 'contour')] = contour_filter.Execute(self.data_dict[seg_key])

    def create_labels(self):
        ref_size = self.data_dict['image'].GetSize()[::-1]
        segmentation_label = np.zeros(ref_size, dtype=np.int8)
        contour_label = np.zeros(ref_size, dtype=np.int8)

        for i, seg_key in enumerate([i for i in self.data_dict.keys() if 'segmentation' in i], start=1):
            segmentation_np = sitk.GetArrayFromImage(self.data_dict[seg_key])
            segmentation_label[segmentation_np > 0] = i
        self.data_dict['segmentation_label'] = segmentation_label

        for i, contour_key in enumerate([i for i in self.data_dict.keys() if 'contour' in i], start=1):
            contour_np = sitk.GetArrayFromImage(self.data_dict[contour_key])
            contour_label[contour_np > 0] = i
        self.data_dict['contour_label'] = contour_label

    def convert_images(self):
        image_np = sitk.GetArrayFromImage(self.data_dict['image'])
        image_np[image_np < self.intensity_range[0]] = self.intensity_range[0]
        image_np[image_np > self.intensity_range[1]] = self.intensity_range[1]
        image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))
        image_np *= 255
        self.data_dict['image_np'] = image_np

    def create_location(self):
        zsize, xsize, ysize = self.data_dict['image_np'].shape
        centroid = None
        if self.get_at_centroid:
            segmentation = self.data_dict.get('segmentation_0')
            if segmentation:
                centroid = compute_centroid(sitk.GetArrayFromImage(segmentation))
            else:
                print("WARNING: segmentation not found for centroid")

        if self.view is 'axial':
            axial_index = centroid[0] if centroid else int(zsize / 2)
            self.loc_tuple = axial_index, slice(0, xsize), slice(0, ysize)
            self.figsize = [xsize, ysize]
            self.imshow_option = {'origin': 'upper'}
        if self.view is 'sagittal':
            sagittal_index = centroid[2] if centroid else int(ysize / 2)
            self.loc_tuple = slice(0, zsize), slice(0, xsize), sagittal_index
            self.figsize = [xsize, zsize]
            self.imshow_option = {'origin': 'lower'}
        if self.view is 'coronal':
            coronal_index = centroid[1] if centroid else int(xsize / 2)
            self.loc_tuple = slice(0, zsize), coronal_index, slice(0, ysize)
            self.figsize = [ysize, zsize]
            self.imshow_option = {'origin': 'lower'}

    def generate_plot(self, dpi=45, margin=0.05):
        figsize = (1 + margin) * self.figsize[0] / dpi, \
                  (1 + margin) * self.figsize[1] / dpi
        fig = plt.figure(figsize=figsize)
        plt.axis("equal")

        plt.imshow(self.data_dict['image_np'][self.loc_tuple], cmap=cm.gray, vmin=0, vmax=255, **self.imshow_option)

        if self.show_filled:
            segmentation_label = self.data_dict['segmentation_label'][self.loc_tuple]
            segmentation_label = segmentation_label.astype(np.float)
            segmentation_label[segmentation_label == 0] = np.nan
            plt.imshow(segmentation_label, interpolation='none', cmap=cm.jet, alpha=self.transparency,
                       **self.imshow_option)

        if self.show_contour:
            contour_label = self.data_dict['contour_label'][self.loc_tuple]
            contour_label = contour_label.astype(np.float)
            contour_label[contour_label == 0] = np.nan
            plt.imshow(contour_label, interpolation='none', cmap=cm.jet, **self.imshow_option)

        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.margins(0, 0)
        # plt.show()

        if self.output_path is not None:
            fig.savefig(self.output_path, format='png')
