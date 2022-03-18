# Created by Bastien Rigaud at 17/03/2022
# Bastien Rigaud, PhD
# Laboratoire Traitement du Signal et de l'Image (LTSI), INSERM U1099
# Campus de Beaulieu, Université de Rennes 1
# 35042 Rennes, FRANCE
# bastien.rigaud@univ-rennes1.fr
# Description:

import sys, os
import gc
import numpy as np
import copy
import SimpleITK as sitk
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from IOTools.IOTools import ImageReaderWriter
from Resample_Class.src.NiftiResampler.ResampleTools import ImageResampler
from Image_Processors_Utils.Image_Processor_Utils import compute_centroid, create_external, compute_bounding_box

plt.style.use('dark_background')
mpl.use('Agg')


class PlotNifti(object):
    def __init__(self, image_path, segmentation_paths=None, output_path=None, show_contour=True, show_filled=True,
                 transparency=0.20, get_at_centroid=True, view='sagittal', intensity_range=None,
                 segmentation_names=None, crop_scan=True, crop_margin=2):
        """
        :param image_path: image path
        :param segmentation_paths: list of segmentation nifti paths
        :param output_path: output path to save the image as png
        :param show_contour: True/False to show contour
        :param show_filled: True/False to show mask
        :param transparency: 0-1.0 filled segmentation transparency
        :param get_at_centroid: True to force the position to the centroid of the first segmentation
        :param view: view selection, ['axial', 'sagittal', 'coronal']
        :param intensity_range: list to arbitrary crop the intensity values
        :param segmentation_names: OPTIONAL, if provided will create a colormap with the corresponding label name
        :param crop_scan: True/False to compute external body and crop the image
        :param crop_margin: crop_scan isotropic margin
        """
        if segmentation_names is None:
            segmentation_names = []
        if intensity_range is None:
            intensity_range = [-1000, 1500]
        if segmentation_paths is None:
            segmentation_paths = []
        assert isinstance(segmentation_names, list), "segmentation_names needs to be a list or None"
        assert isinstance(intensity_range, list), "intensity_range needs to be a list or None"
        assert isinstance(segmentation_paths, list), "segmentation_paths needs to be a list or None"
        assert view in ['axial', 'sagittal', 'coronal'], \
            'view is not recognized, possible choice [axial,sagittal,coronal]'
        self.image_path = image_path
        self.segmentation_paths = segmentation_paths
        self.output_path = output_path
        self.show_contour = show_contour
        self.show_filled = show_filled
        self.transparency = transparency
        self.get_at_centroid = get_at_centroid
        self.view = view
        self.intensity_range = intensity_range
        self.segmentation_names = segmentation_names
        self.crop_margin = crop_margin

        self.data_dict = {}
        self.dataloader = ImageReaderWriter

        self.load_data()
        self.resample_data()
        self.compute_contour()
        self.create_labels()
        if crop_scan:
            self.crop_by_external()
        self.convert_images()

    def set_output_path(self, output_path):
        self.output_path = output_path

    def set_view(self, view):
        self.view = view

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
                                                                   empty_value=int(empty_value))

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

    def crop_by_external(self):
        image_np = sitk.GetArrayFromImage(self.data_dict['image'])
        external_mask = create_external(image_np, threshold_value=-700, mask_value=1)
        bb_parameters = compute_bounding_box(external_mask, padding=self.crop_margin)
        self.data_dict['image_np'] = image_np[
                                     bb_parameters[0]:bb_parameters[1],
                                     bb_parameters[2]:bb_parameters[3],
                                     bb_parameters[4]:bb_parameters[5]]
        for key in ['segmentation_label', 'contour_label']:
            if key in self.data_dict.keys():
                temp = self.data_dict.get(key)
                self.data_dict[key] = temp[
                                      bb_parameters[0]:bb_parameters[1],
                                      bb_parameters[2]:bb_parameters[3],
                                      bb_parameters[4]:bb_parameters[5]]

    def convert_images(self):
        if 'image_np' in self.data_dict.keys():
            image_np = self.data_dict.get('image_np')
        else:
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
            segmentation = copy.deepcopy(self.data_dict.get('segmentation_label'))
            segmentation[segmentation > 1] = 0  # keep only the first label
            if self.data_dict.get('segmentation_0'):
                centroid = compute_centroid(segmentation)
            else:
                print("WARNING: segmentation not found for centroid")

        if self.view == 'axial':
            axial_index = centroid[0] if centroid else int(zsize / 2)
            self.loc_tuple = axial_index, slice(0, xsize), slice(0, ysize)
            self.figsize = [ysize, xsize]
            self.imshow_option = {'origin': 'upper'}
        if self.view == 'sagittal':
            sagittal_index = centroid[2] if centroid else int(ysize / 2)
            self.loc_tuple = slice(0, zsize), slice(0, xsize), sagittal_index
            self.figsize = [xsize, zsize]
            self.imshow_option = {'origin': 'lower'}
        if self.view == 'coronal':
            coronal_index = centroid[1] if centroid else int(xsize / 2)
            self.loc_tuple = slice(0, zsize), coronal_index, slice(0, ysize)
            self.figsize = [ysize, zsize]
            self.imshow_option = {'origin': 'lower'}

    def generate_plot(self, dpi=45, margin=0.05):
        self.create_location()
        figsize = (1 + margin) * self.figsize[0] / dpi, \
                  (1 + margin) * self.figsize[1] / dpi
        fig = plt.figure(figsize=figsize)
        plt.axis("equal")

        plt.imshow(self.data_dict['image_np'][self.loc_tuple], cmap=cm.gray, vmin=0, vmax=255, **self.imshow_option)

        if self.show_filled or self.show_contour or self.segmentation_names:
            # mmin = np.min(self.data_dict['segmentation_label'])
            mmax = np.max(self.data_dict['segmentation_label'])
            if self.segmentation_names and mmax != len(self.segmentation_names):
                print("WARNING, segmentation_names truncated because max value from segmentation map is different\n"
                      "Max value: {}, len of segmentation_names {}".format(mmax, len(self.segmentation_names)))
                self.segmentation_names = self.segmentation_names[0:mmax]
            base = plt.cm.get_cmap('jet')
            color_list = base(np.linspace(0, 1, mmax, 1))
            mask_cm = plt.cm.colors.ListedColormap(color_list, 'Segmentation', mmax)

        if self.show_filled:
            segmentation_label = self.data_dict['segmentation_label'][self.loc_tuple]
            segmentation_label = segmentation_label.astype(np.float)
            segmentation_label[segmentation_label == 0] = np.nan
            plt.imshow(segmentation_label, interpolation='none', cmap=mask_cm, alpha=self.transparency, vmin=1,
                       vmax=np.max(self.data_dict['segmentation_label']) + 1, **self.imshow_option)

        if self.show_contour:
            contour_label = self.data_dict['contour_label'][self.loc_tuple]
            contour_label = contour_label.astype(np.float)
            contour_label[contour_label == 0] = np.nan
            plt.imshow(contour_label, interpolation='none', cmap=mask_cm, vmin=1,
                       vmax=np.max(self.data_dict['segmentation_label']) + 1, **self.imshow_option, )

        if self.segmentation_names:
            cbar = plt.colorbar(shrink=0.5)
            cbar.ax.set_yticks(np.arange(1, mmax + 1, 1) + 0.5)
            cbar.ax.set_yticklabels(
                ['{} - {}'.format(i, name) for i, name in enumerate(self.segmentation_names, start=1)])
            cbar.ax.tick_params(labelsize=15)

        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.margins(0, 0)
        # plt.show()
        if self.output_path is not None:
            fig.savefig(self.output_path, format='png')
        # Clear the current axes.
        plt.cla()
        # Clear the current figure.
        fig.clf()
        plt.clf()
        # Closes all the figure windows.
        plt.close('all')
        plt.close(fig)
        gc.collect()
