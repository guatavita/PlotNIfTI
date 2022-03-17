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


class PlotNifti(object):
    def __init__(self, image_path, segmentation_paths=None, output_path=None, show_contour=True, show_filled=True,
                 transparency=0.25, get_at_centroid=True, view='sagittal'):
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
        self.data_dict = {}
        self.dataloader = ImageReaderWriter

        self.load_data()
        self.compute_contour()
        self.create_labels()
        self.convert_images()
        self.create_location()
        self.generate_plot()
        xxx = 1

    def load_data(self):
        image_loader = self.dataloader(filepath=self.image_path)
        self.data_dict['image'] = image_loader.import_data()

        if not isinstance(self.segmentation_paths, list):
            self.segmentation_paths = [self.segmentation_paths]

        for i, segmentation_path in enumerate(self.segmentation_paths):
            temp_loader = self.dataloader(filepath=segmentation_path)
            self.data_dict['segmentation_{}'.format(i)] = temp_loader.import_data()

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
        image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))
        image_np *= 255
        self.data_dict['image_np'] = image_np

    def create_location(self):
        zsize, xsize, ysize = self.data_dict['image_np'].shape

        if self.view is 'axial':
            self.loc_tuple = int(zsize / 2), slice(0, xsize), slice(0, ysize)
            self.figsize = [xsize, ysize]
            self.imshow_option = {'origin': 'upper'}
        if self.view is 'sagittal':
            self.loc_tuple = slice(0, zsize), slice(0, xsize), int(ysize / 2)
            self.figsize = [xsize, zsize]
            self.imshow_option = {'origin': 'lower'}
        if self.view is 'coronal':
            self.loc_tuple = slice(0, zsize), int(xsize / 2), slice(0, ysize)
            self.figsize = [ysize, zsize]
            self.imshow_option = {'origin': 'lower'}

    def generate_plot(self, dpi=45, margin=0.05):
        figsize = (1 + margin) * self.figsize[0] / dpi, (1 + margin) * self.figsize[1] / dpi
        fig = plt.figure(figsize=figsize)
        plt.axis("equal")

        plt.imshow(self.data_dict['image_np'][self.loc_tuple], cmap=cm.gray, **self.imshow_option)

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
        plt.tight_layout()
        plt.show()

        if self.output_path is not None:
            fig.savefig(self.output_path)
