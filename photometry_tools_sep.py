# Created on Wed Aug 03 15:57:43 2022
#
################## photometry_utils_1.0.py ##################
#
#
import os
import time
import warnings
# Plotting packages
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Circle
import cmasher as cmr
# Astronomy packages
import astropy
from astropy.io import fits
import astropy.units as u
from astropy.table import Table, QTable, Column, hstack, vstack
from astropy.wcs import WCS
from astropy.stats import SigmaClip
# Photometry packages
import sep
from photutils import Background2D, SExtractorBackground, MADStdBackgroundRMS, aperture_photometry, CircularAperture
from photutils.segmentation import SegmentationImage
# Calculation packages
import numpy as np
import random
from scipy import stats
from scipy.interpolate import interp1d
from scipy.ndimage import distance_transform_edt
from scipy.optimize import curve_fit



"""
This file holds useful functions for performing photometry, nestled under
classes called "Source_detector" and "Catalogor".
"""

def create_polygon_mask(shape, vertices, plotfig=False):
    """
    Creates np.array with dimensions defined by shape
    Fills polygon defined by vertices with ones, all other values zero
    Parameters
    ----------
    shape: tuple
        dimensions of full image within which to create the polygon mask
    vertices: list
        list of vertices - (x,y) tuples - for mask, starting at TOP LEFT 
        and going around the polygon in CLOCKWISE order; 
        NOTE: the polygon must be CLOSED (i.e. the first/last vertex 
        should be the same point)
    plotfig: bool
        whether or not to plot the created mask
    Returns
    -------
    grid: ndarray
        boolean array whose dimensions are defined by the "shape" input, 
        with True values set with a polygon defined by the "vertices" input
    """

    ny, nx = shape
    poly_verts = vertices

    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x,y)).T

    path = matplotlib.path.Path(poly_verts, closed=True)
    mask = path.contains_points(points)
    mask = mask.reshape((ny,nx))
    mask = np.array(mask,dtype=bool)

    if plotfig == True:
        plt.imshow(mask,cmap='Greys',origin='lower')
        plt.colorbar()
        plt.show()
    return mask


"""Main class # 1: Source_detecetor"""
class Source_detector(object):
    """
    Class to read in detection image and multiband data set. Performs source
    detection, aperture photometry, and error analysis.
    """

    def __init__(self,cluster_name,field_name,version):
        """
        Parameters
        ----------

        Attributes
        ----------
        """
        self.mask_fov = None
        self.cluster_name = cluster_name
        self.field_name = field_name
        self.version = version
        self.mask_cold = None

        return




    """MAIN body functions begin here"""

    def add_det_img(self, fn_det_im, pix_scale=0.040, create_fov_mask=False):
        """
        add a detection image, and optionally create a field of view ("fov")
        mask for the file. Data assumed to be in .fits format.
        Parameters
        ----------
        fn_det_in: str
            file name of the detection image
        create_fov_mask: bool
            whether or not a mask of the field of view should be created
        """
        self.pix_scale = pix_scale
        self.fn_det_im = fn_det_im
        self.det_im = fits.getdata(fn_det_im)
        self.det_im = self.det_im.byteswap(inplace=True).newbyteorder()
        if create_fov_mask:
            self.mask_fov = (self.det_im==0)
        return


    def calculate_bkg(self, box_size=64,kernel_size=3,mask=None,method='photutils',sub_bkg=True,plot_fig=False,stretch=['auto','auto']):
        """
        calculate (option: subtract from data) the 2D background.
        global bkg set as detection threshold.

        Parameters
        ----------
        box_size: int
            size of box within which to calculate background
        kernel_size: int
            size of kernel. must be an odd number
        method: str
            either 'sep' or 'photutils'
        sub_bkg: bool
            whether or not to subtract the background from the detection image
        plot_fig: bool
            whether or not to plot the 2D background & rms maps
        stretch: array
            [vmin,vmax] for plotting bkg & rms maps
        """
        start_time = time.time()
        if method == 'sep':
            self.bkg_codename = 'sep'
            # measure a spatially varying background on the image
            bkg = sep.Background(self.det_im, mask=mask, bw=box_size,
                                 bh=box_size, fw=kernel_size, fh=kernel_size)
            # get a "global" mean and noise of the image background:
            self.bkg_globalback = bkg.globalback
            self.bkg_globalrms = bkg.globalrms
            self.bkg_image = bkg.back()
            bkg_rms = bkg.rms()

            print('Median bkg: {:.5e}'.format(self.bkg_globalback))
            print('Median rms: {:.5e}'.format(self.bkg_globalrms))

        elif method == 'photutils':
            self.bkg_codename = 'photutils'
            # 2D background estimation
            sigma_clip = SigmaClip(sigma=3., maxiters=50)
            bkg_estimator = SExtractorBackground()
            bkg_rms_estimator = MADStdBackgroundRMS()

            #compute background and update user on key stats
            bkg = Background2D(self.det_im,box_size,filter_size=kernel_size,edge_method='pad',sigma_clip=sigma_clip,
                               bkg_estimator=bkg_estimator,bkgrms_estimator=bkg_rms_estimator,
                               coverage_mask=mask,  fill_value=0.)#coverage_mask=mask,
            # get a "global" mean and noise of the image background:
            self.bkg_globalback = bkg.background_median
            self.bkg_globalrms = bkg.background_rms_median
            self.bkg_image = bkg.background
            bkg_rms = bkg.background_rms

            print('Median bkg: {:.5e}'.format(self.bkg_globalback))
            print('Median rms: {:.5e}'.format(self.bkg_globalrms))

        if sub_bkg == True:
            self.det_im -= self.bkg_image #* (mask==0)

        if plot_fig == True:
            if stretch[0]=='auto':
                vmin_bkg = np.min(self.bkg_globalback)
                vmin_rms = np.min(self.bkg_globalrms)
            else:
                vmin_bkg = stretch[0]
                vmin_rms = stretch[0]
            if stretch[-1]=='auto':
                vmax_bkg = self.bkg_globalback*2
                vmax_rms = self.bkg_globalrms*2
            else:
                vmax_bkg = stretch[-1]
                vmax_rms = stretch[-1]
            
            # show the background and background rms
            fig, (ax1,ax2) = plt.subplots(1,2,figsize=(7,16))
            im1 = ax1.imshow(self.bkg_image, interpolation='nearest',vmin=vmin_bkg,vmax=vmax_bkg,cmap='gray', origin='lower')
            im2 = ax2.imshow(bkg_rms, interpolation='nearest',vmin=vmin_rms,vmax=vmax_rms, cmap='gray', origin='lower')
            ax2.get_yaxis().set_ticks([])
            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im1, cax=cax1, orientation='vertical')
            divider2 = make_axes_locatable(ax2)
            cax2 = divider2.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im2, cax=cax2, orientation='vertical')
            plt.show()

        print('\n\nFunction "calculate_bkg()" took {:.3} seconds to run'.format(time.time()-start_time))
        return


    def perform_source_detection(self,det_dict,mode='hot',mask=None,filter_kernel=None,diag=False):
        """
        Perform source detection on the detection image with the parameters provided.
        Parameters
        ----------
        det_dict: dict
            dictionary containing the following fields:
            nsigma: float
                multiple above detection threshold for a pixel to be considered
                a "detection"
            npix: int
                # of connected pixels for a source segmented to be counted as
                a "detection"
            nlevels: int
                # of deblending levels
            contrast: float
                deblend contrast
        mode: str, 'hot' or 'cold' or 'warm'
            the detection mode being performed; hot for aggressive settings, cold for coarse settings
        mask: array-like
            boolean mask to be applied when performing source detection
        filter_kernel: array-like
            filter_kernel to be applied in smoothing the detection image prior
            to source detection
        """
        start_time = time.time()
        # assign detection parameters
        nsigma = det_dict['nsigma']
        npixels = det_dict['npixels']
        nlevels = det_dict['nlevels']
        contrast = det_dict['contrast']
        
        objects, segm = sep.extract(self.det_im, nsigma, err=self.bkg_globalrms, mask=mask,
                              minarea=npixels, filter_kernel=filter_kernel, filter_type='matched',
                              deblend_nthresh=nlevels, deblend_cont=contrast, clean=True,
                              clean_param=1.0, segmentation_map=True)

        if diag==True and mask is not None:
            fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
            im1 = ax1.imshow(mask,vmin=0,vmax=1,cmap='Greys',origin='lower',interpolation='nearest')
            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im1, cax=cax1, orientation='vertical')
            ax2.imshow(segm,cmap='tab20',origin='lower',interpolation='nearest')
            plt.show()
            plt.close()
            
        obj_str = 'objects_{}'.format(mode)
        segm_str = 'segm_{}'.format(mode)
        setattr(self,obj_str, objects)
        setattr(self, segm_str, segm)
        print('# of objects detected/deblended in {} mode: {:d}'.format(mode,len(getattr(self,obj_str))))
        print('\n\nFunction "perform_source_detection()" took {:.3} seconds to run'.format(time.time()-start_time))
        return


    def remove_sources_near_mask(self,mask,dist_tol=0.5,mode='hot',update_segm=True,plotfig=False,diag=False):
        """
        Remove sources within a specified distance from a mask.
        Parameters
        ----------
        mask: 
            mask from which to determine distances and remove sources too close to said mask
        dist_tol: float
            max distance between centroids of hot mode/cold mode detections for 
            objects to be considered duplicate detections; in arcsec.
        mode: str
            'hot' or 'cold' for detection mode
        update_segm: bool
            whether or not to update the labels and segments in the segmentation map
        plotfig:
        Returns
        -------

        """
        mask = mask==0
        if plotfig == True:
            plt.title('Input mask')
            plt.imshow(mask,cmap='Greys',origin='lower',interpolation='nearest')
            plt.colorbar()
            plt.show()
        start_time = time.time()
        obj_str = 'objects_{}'.format(mode)
        kron_str = 'kronrad_{}'.format(mode)
        krflg_str = 'kronflag_{}'.format(mode)
        segm_str = 'segm_{}'.format(mode)
        objects = getattr(self,obj_str)
        kronrad = getattr(self,kron_str)
        krnflag = getattr(self,krflg_str)
        segm = getattr(self,segm_str)
        # compute map of distances from mask
        distances_to_mask_arr = distance_transform_edt(input=mask, return_distances=True)
        distances_to_mask_arr*=self.pix_scale
        if plotfig == True:
            plt.title('Distance transform')
            plt.imshow(distances_to_mask_arr,cmap='Greys',origin='lower',interpolation='nearest')
            plt.colorbar()
            plt.show()
        sources_to_remove = (distances_to_mask_arr < dist_tol)
        if plotfig == True:
            plt.title('map_to_remove')
            plt.imshow(map_to_remove,cmap='Greys',origin='lower',interpolation='nearest')
            plt.colorbar()
            plt.show()
        # identify & remove sources in segmentation map
        old_labels = np.unique(segm)
        old_labels= np.delete(old_labels,np.where(old_labels==0))
        print('Original # of sources: {}'.format(len(objects['x'])))
        if diag==True:
            print('\nOriginal # of segments (old_labels): {}\nold_labels: {}'.format(len(old_labels),old_labels))
        labels_to_remove = segm[sources_to_remove]
        labels_to_remove = np.unique(labels_to_remove)
        labels_to_remove= np.delete(labels_to_remove,np.where(labels_to_remove==0))
        print('# of sources to remove: {}'.format(len(labels_to_remove)))
        if diag==True:
            print('\nlabels_to_remove: {}'.format(labels_to_remove))
        if update_segm == True:
            if len(labels_to_remove) != 0:
                print('\nBeginning removal of segments from segmentation map...')
                for label in labels_to_remove:
                    indices = segm==label
                    segm[indices] = 0
                    old_labels= np.delete(old_labels,np.where(old_labels==label))           
                print('Removal of segments from segmentation map complete.')
            if diag==True:
                print('labels post-cleaning: {}'.format(old_labels))
            new_labels = np.arange(1,len(old_labels)+1,dtype='int32')
            if len(labels_to_remove) != 0:
                print('\nBeginning segm re-labelling...')
                if diag==True:
                    print('len(old_labels): {}\nlen(new_labels): {}\nold_labels: {}\n\nnew_labels to apply: {}'.format(len(old_labels),len(new_labels),old_labels,new_labels))
                for old_label,new_label in zip(old_labels,new_labels):
                    if new_label%1000==0:
                        print('New label: {}/{} ({:.2f}% complete)'.format(new_label,len(new_labels),new_label/len(new_labels)*100))
                    indices = segm==old_label
                    segm[indices] = new_label
                print('Re-labelling of segments in segmentation map complete.')
        new_labels = np.unique(segm)
        new_labels= np.delete(new_labels,np.where(new_labels==0))
        if diag==True:
            print('new_labels re-labelled: ',new_labels)
        # remove sources from "objects" list
        n_obj_initial = len(objects)
        print('# of sources within < {}" removed from segmentation map: {}'.format(dist_tol,len(labels_to_remove)))
        print('Original # of objects: ',n_obj_initial)
        if len(labels_to_remove) != 0:
            objects = np.delete(objects, (labels_to_remove-1), axis=0)
            kronrad = np.delete(kronrad, (labels_to_remove-1), axis=0)
            krnflag = np.delete(krnflag, (labels_to_remove-1), axis=0)
        setattr(self,segm_str,segm)
        setattr(self,obj_str,objects)
        setattr(self,kron_str,kronrad)
        setattr(self,krflg_str,krnflag)
        kronrad = getattr(self,kron_str)
        krnflag = getattr(self,krflg_str)
        if diag==True:
            print('Length of "kronrad": {}\nLength of "kronflag": {}'.format(len(kronrad),len(krnflag)))
        print('Final # of objects: {}\nFinal # of segments: {}\nTotal # of objects removed: {}'.format(len(objects),len(new_labels),n_obj_initial-len(objects)))
        print('\n\nFunction "remove_sources_near_mask()" took {:.3f} seconds to run'.format(time.time()-start_time))
        return


    def measure_kron_radius(self,mode='hot',diag=False):
        """Measure the kron radius from the detection image"""
        start_time = time.time()

        obj_str = 'objects_{}'.format(mode)
        segm_str = 'segm_{}'.format(mode)
        objects = getattr(self,obj_str)
        segm = getattr(self,segm_str)

        seg_id = np.arange(1,len(objects['x'])+1,1,dtype='int32')
        kronrad, krflag = sep.kron_radius(self.det_im, objects['x'], objects['y'], objects['a'],
                                          objects['b'], objects['theta'],seg_id=seg_id,segmap=segm,r=6.)

        kron_str = 'kronrad_{}'.format(mode)
        kronflag_str = 'kronflag_{}'.format(mode)
        setattr(self,kron_str, kronrad)
        setattr(self, kronflag_str, krflag)
        if diag==True:
            circ_kron_radius = 2.5*kronrad * np.sqrt(objects['a']*objects['b']) * self.pix_scale
            mask_nan = np.isnan(circ_kron_radius)
            circ_kron_radius=circ_kron_radius[~mask_nan]
            print('# of sources in "self.objects": {}\nlength of "kronrad" {}\n# of NaNs in "kronrad": {}\nMedian value of circ. "kronrad": {:.4f}"\nMax. value of circ. "kronrad": {:.4f}"\n'.format(len(objects),len(kronrad),np.sum(mask_nan),np.median(circ_kron_radius),np.max(circ_kron_radius)))
            # plot circ kron radius hist diagnostic
            plt.hist((circ_kron_radius),bins='auto')
            plt.show()
        print('\n\nFunction "measure_kron_radius()" took {:.3} seconds to run'.format(time.time()-start_time))
        return


    def remove_sources_too_large(self,circkronrad_tol=5.,update_segm=False,plotfig=False,diag=True):
        """
        Remove sources w/ too great a circular kron radius.
        Parameters
        ----------
        circkronrad_tol: float
            maximum allowed ciricularized kron radius; in arcsec
        Returns
        -------

        """
        start_time = time.time()
        # compute circularized kron radius. remove nans
        circ_kron_radius = 2.5 * self.kronrad * np.sqrt(self.objects['a']*self.objects['b']) * self.pix_scale
        mask_nan = np.isnan(circ_kron_radius)
        n_obj_initial = len(circ_kron_radius)
        if diag==True:
            print('\n# of NaNs in "kronrad": {}\nMedian value of circ. "kronrad": {:.4f}"\nMax. value of circ. "kronrad": {:.4f}"\n'.format(np.sum(mask_nan),np.median(circ_kron_radius[~mask_nan]),np.max(circ_kron_radius[~mask_nan])))
        # identify & remove sources in segmentation map
        sources_to_remove = np.where((circ_kron_radius > circkronrad_tol) | mask_nan)[0]
        print('# of sources to remove: ',len(sources_to_remove),'\nsources to remove: ',sources_to_remove)
        if update_segm == True:
            old_labels = np.unique(self.segm)
            old_labels= np.delete(old_labels,np.where(old_labels==0))
            print('Original # of sources: {}'.format(len(self.objects['x'])))
            if diag==True:
                print('\noriginal # of segments (old_labels): {}\nold_labels: {}'.format(len(old_labels),old_labels))
            labels_to_remove = sources_to_remove+1
            print('# of sources to remove: {}'.format(len(labels_to_remove)))
            if diag==True:
                print('\nlabels_to_remove: {}'.format(labels_to_remove))
            print('\nBeginning segm removal...')
            if len(labels_to_remove) != 0:
                for ii,label in enumerate(labels_to_remove):
                    if ii%500==0:
                        print('label: {} - {}/{} ({:.2f}% complete)'.format(label,ii,len(labels_to_remove),ii/len(labels_to_remove)*100))
                    indices = self.segm==label
                    self.segm[indices] = 0
                    old_labels = np.delete(old_labels,np.where(old_labels==label))
                if diag==True:
                    print('\n# of labels remaining: {}\nlabels post-cleaning: {}'.format(len(old_labels),old_labels))
            new_labels = np.arange(1,len(old_labels)+1,dtype='int32')
            print('\nBeginning segm re-labelling...')
            if diag==True:
                print('len(old_labels): {}\nlen(new_labels): {}\nnew_labels to apply: {}\n'.format(len(old_labels),len(new_labels),new_labels))
            for old_label,new_label in zip(old_labels,new_labels):
                if new_label%500==0:
                    print('New label: {}/{} ({:.2f}% complete)'.format(new_label,len(new_labels),new_label/len(new_labels)*100))
                indices = self.segm==old_label
                self.segm[indices] = new_label
            if diag==True:
                print('new_labels re-labelled: ',new_labels)
        # remove sources from "objects" list  
        print('# of sources > {}" removed from segmentation map: {}\nFinal # of segments: {}'.format(circkronrad_tol,len(sources_to_remove),len(np.unique(self.segm))-1))
        print('Original # of objects: ',n_obj_initial)
        if len(sources_to_remove) != 0:
            self.objects = np.delete(self.objects, (sources_to_remove), axis=0)
            self.kronrad = np.delete(self.kronrad, (sources_to_remove), axis=0)
            self.kronflag = np.delete(self.kronflag, (sources_to_remove), axis=0)
        print('Final # of objects: {}\nTotal # of objects removed: {}'.format(len(self.objects),n_obj_initial-len(self.objects)))
        circ_kron_radius = 2.5 * self.kronrad * np.sqrt(self.objects['a']*self.objects['b']) * self.pix_scale
        if diag==True:        
            mask_nan=np.isnan(circ_kron_radius)
            print('\nMedian value of circ. "kronrad": {:.4f}"\nMax. value of circ. "kronrad": {:.4f}"\n'.format(np.median(circ_kron_radius[~mask_nan]),np.max(circ_kron_radius[~mask_nan])))
        if plotfig == True:
           # plot circ kron radius hist diagnostic
            plt.hist((circ_kron_radius),bins='auto')
            plt.show()
        print('\n\nFunction "remove_sources_too_large()" took {:.3} seconds to run'.format(time.time()-start_time))
        return




    def write_ds9_reg_bad_kron(self, fn_ds9_reg, mode='hot', color='blue'):
        """
        Writes a ds9 region file:
        Parameters
        ----------
        fn_ds9_reg: str
            Filename to which the ds9 region will get written
        color: str
            Default color of ellipse to plot in DS9
        """
        start_time = time.time()
        kron_str = 'kronrad_{}'.format(mode)
        obj_str = 'objects_{}'.format(mode)
        kronrad = getattr(self,kron_str)
        objects = getattr(self,obj_str)
        r = 3.
        bad_kron_mask = np.isnan(kronrad)
        with open(fn_ds9_reg, 'w') as f:
            f.write("# Region file format: DS9\n")
            f.write(
                'global color={} dashlist=8 3 width=3 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n'.format(color))
            f.write("physical\n")  # This is required for newer versions of aplpy/pyregion
            for obj in objects[bad_kron_mask]:
                f.write(
#                         "ellipse {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:d} \n".format(
                        "ellipse {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} \n".format(
                        obj['x'] + 1, obj['y'] + 1, obj['a']*r,
                        obj['b']*r, obj['theta'] * (180/np.pi)))

        print('DS9 region file saved to: {}\n\nFunction "write_ds9_reg_bad_kron()" took {:.3} seconds to run'.format(fn_ds9_reg,time.time()-start_time))
        return


    def write_ds9_reg(self, fn_ds9_reg, mode='hot', color='blue'):
        """
        Writes a ds9 region file:
        Parameters
        ----------
        fn_ds9_reg: str
            Filename to which the ds9 region will get written
        mode: str, 'hot', 'cold', or 'comb'
            indicate whether you are saving the hot mode, cold mode, or combined detections
        color: str
            Default color of ellipse to plot in DS9
        """
        start_time = time.time()
        if mode == 'comb':
            obj_str = 'objects'
        else:
            obj_str = 'objects_{}'.format(mode)
        objects = getattr(self,obj_str)
        r = 3.

        with open(fn_ds9_reg, 'w') as f:
            f.write("# Region file format: DS9\n")
            f.write(
                'global color={} dashlist=8 3 width=3 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n'.format(color))
            f.write("physical\n")  # This is required for newer versions of aplpy/pyregion
            for obj in range(len(objects['x'])):
                f.write(
    #                         "ellipse {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:d} \n".format(
                        "ellipse {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} \n".format(
                        objects['x'][obj] + 1, objects['y'][obj] + 1, objects['a'][obj]*r,
                        objects['b'][obj]*r, objects['theta'][obj] * (180/np.pi)))

        print('DS9 region file saved to: {}\n\nFunction "write_ds9_reg()" took {:.3} seconds to run'.format(fn_ds9_reg,time.time()-start_time))
        return


    def write_segm_map(self,fn_save,mode='hot'):
        """
        Write the segmentation map to .fits file
        Parameters
        ----------
        fn_ds9_reg: str
            Filename to which the segmentation map will get written
        mode: str, 'hot', 'cold', or 'comb'
            indicate whether you are saving the hot mode, cold mode, or combined detections
        """
        start_time = time.time()
        if mode == 'comb':
            segm_str = 'segm'
        else:
            segm_str = 'segm_{}'.format(mode)
        segm = getattr(self,segm_str)

        header = fits.getheader(self.fn_det_im)
        hdul = fits.PrimaryHDU(segm, header=header)
        hdul.writeto(fn_save, overwrite=True)

        print('Segmentation map saved to: {}\n\nFunction "write_segm_map()" took {:.3} seconds to run'.format(fn_save,time.time()-start_time))
        return


    def make_detection_mode_mask(self,mode='cold',scale_detections=None,diag=False):
        """Make a mask from the previous detection mode segmentation map
        Parameters
        ----------
        mode: str
            the previous detection mode; 'cold' or 'hot'
        """
        start_time = time.time()
        if mode == 'cold':
            mask_str = 'mask_{}'.format(mode)
            segm = getattr(self,'segm_cold')
        elif mode == 'hot':
            mask_str = 'mask_cold_hot'
            segm_cold = getattr(self,'segm_cold')
            segm_hot = getattr(self,'segm_hot')
            segm = segm_cold + segm_hot
        mask = segm != 0
        
        mask = self.make_mask_bigger(mask,dist_tol=scale_detections,diag=False)
        
        setattr(self,mask_str,mask)
        if diag==True:
            plt.imshow(mask,vmin=0,vmax=1,cmap='Greys',origin='lower',interpolation='nearest')
            plt.colorbar()
            plt.show()
        print('\n\nFunction "make_detection_mode_mask()" took {:.3} seconds to run'.format(time.time()-start_time))
        return


    def make_stage0_cold_mode_mask(self,dist_tol=1.5,mask=None,diag=False):
        """Make a mask including a buffer distance around all cold mode detections"""
        start_time = time.time()
        if mask is not None:
            mask_temp = self.segm_cold != 0
            mask += (mask_temp == 0)
        else:
            mask_temp = self.segm_cold != 0
            mask = (mask_temp == 0)
        if diag==True:
            plt.imshow(mask,vmin=0,vmax=1,cmap='Greys',origin='lower',interpolation='nearest')
            plt.colorbar()
            plt.show()
        distances_to_mask_arr = distance_transform_edt(input=mask, return_distances=True)
        distances_to_mask_arr*=self.pix_scale
        if diag==True:
            plt.imshow(distances_to_mask_arr,cmap='Greys',origin='lower',interpolation='nearest')
            plt.colorbar()
            plt.show()
        new_mask = distances_to_mask_arr > dist_tol
        if diag==True:
            plt.imshow(new_mask,vmin=0,vmax=1,cmap='Greys',origin='lower',interpolation='nearest')
            plt.colorbar()
            plt.show()
        print('\n\nFunction "make_stage0_cold_mode_mask()" took {:.3} seconds to run'.format(time.time()-start_time))
        return new_mask
    

    def make_mask_bigger(self,mask,dist_tol=1.5,diag=False):
        """Make a mask including a buffer distance around bcg residual mask"""
        start_time = time.time()
        mask = mask == 0
        if diag==True:
            plt.imshow(mask,vmin=0,vmax=1,cmap='Greys',origin='lower',interpolation='nearest')
            plt.colorbar()
            plt.show()
        distances_to_mask_arr = distance_transform_edt(input=mask, return_distances=True)
        distances_to_mask_arr*=self.pix_scale
        if diag==True:
            plt.imshow(distances_to_mask_arr,cmap='Greys',origin='lower',interpolation='nearest')
            plt.colorbar()
            plt.show()
        new_mask = distances_to_mask_arr > dist_tol
        new_mask = new_mask==0
        if diag==True:
            plt.imshow(new_mask,vmin=0,vmax=1,cmap='Greys',origin='lower',interpolation='nearest')
            plt.colorbar()
            plt.show()
        print('\n\nFunction "make_cold_mode_mask()" took {:.3} seconds to run'.format(time.time()-start_time))
        return new_mask
    

    def combine_detection_modes(self,mode='hot',update_segm=False,diag=True):
        """
        Combine Cold mode and Hot mode segmentation maps and detection
        catalogs into a unified detection.
        Parameters
        ----------
        mode: str
            'cold', 'hot', or 'comb' (for combined)
        Returns
        -------
        """
        start_time = time.time()
        print('Mode: {}\n'.format(mode))
        # get new labels for combined segmentation map
        if mode == 'hot':
            segm_old = getattr(self,'segm_cold')
            segm_new = getattr(self,'segm_hot')
            segm_comb_str = 'segm_cold_hot'
            obj_comb_str = 'objects_cold_hot'
            obj_new_str = 'objects_hot'
            obj_old_str = 'objects_cold'
            kronrad_comb_str = 'kronrad_cold_hot'
            kronflag_comb_str = 'kronflag_cold_hot'
            kronrad_new_str = 'kronrad_hot'
            kronrad_old_str = 'kronrad_cold'
            kronflag_new_str = 'kronflag_hot'
            kronflag_old_str = 'kronflag_cold'
        elif mode == 'warm':
            segm_old = getattr(self,'segm_cold_hot')
            segm_new = getattr(self,'segm_warm')
            segm_comb_str = 'segm'
            obj_comb_str = 'objects'
            obj_new_str = 'objects_warm'
            obj_old_str = 'objects_cold_hot'
            kronrad_comb_str = 'kronrad'
            kronflag_comb_str = 'kronflag'
            kronrad_new_str = 'kronrad_warm'
            kronrad_old_str = 'kronrad_cold_hot'
            kronflag_new_str = 'kronflag_warm'
            kronflag_old_str = 'kronflag_cold_hot'
        elif mode == 'comb':
            segm_old = getattr(self,'segm_cold')
            segm_new = getattr(self,'segm_hot')
            segm_comb_str = 'segm'
            obj_comb_str = 'objects'
            obj_new_str = 'objects_hot'
            obj_old_str = 'objects_cold'
            kronrad_comb_str = 'kronrad'
            kronflag_comb_str = 'kronflag'
            kronrad_new_str = 'kronrad_hot'
            kronrad_old_str = 'kronrad_cold'
            kronflag_new_str = 'kronflag_hot'
            kronflag_old_str = 'kronflag_cold'
        kronrad_new = getattr(self,kronrad_new_str)
        kronrad_old = getattr(self,kronrad_old_str)
        kronflag_new = getattr(self,kronflag_new_str)
        kronflag_old = getattr(self,kronflag_old_str)
        obj_new = getattr(self,obj_new_str)
        obj_old = getattr(self,obj_old_str)
        obj = np.concatenate([obj_new,obj_old])
        for name in obj_new.dtype.names:
            kronrad = np.concatenate((kronrad_new,kronrad_old),axis=0)
            kronflag = np.concatenate((kronflag_new,kronflag_old),axis=0)
        setattr(self,obj_comb_str,obj)
        setattr(self,kronrad_comb_str,kronrad)
        setattr(self,kronflag_comb_str,kronflag)
        stage0_labels_old = np.unique(segm_old)
        stage0_labels_old = np.delete(stage0_labels_old,np.where(stage0_labels_old==0))
        stage1_labels = np.unique(segm_new)
        stage1_labels = np.delete(stage1_labels,np.where(stage1_labels==0))
        if (mode == 'hot') or (mode == 'comb'):
            print('Original # of sources (in detection cat):\nHot mode: {}\nCold mode: {}'.format(len(self.objects_hot['x']),len(self.objects_cold['x'])))
        elif mode == 'warm':
            print('Original # of sources (in detection cat):\nWarm mode: {}\nCold_Hot mode: {}'.format(len(self.objects_warm['x']),len(self.objects_cold_hot['x'])))
        print('Original # of segments (in segmentation map)\nnew_labels: {}\nold_labels: {}'.format(len(stage1_labels),len(stage0_labels_old)))
        if diag==True:
            print('\nstage1_labels: {}\nstage0_labels_old: {}'.format(stage1_labels,stage0_labels_old))
        stage0_labels_new = stage0_labels_old + len(stage1_labels)
        new_labels = np.concatenate((stage1_labels,stage0_labels_new),axis=None)
        if diag==True:
            print('\nnew_labels: [{} ... {} ... {}]'.format(new_labels[0:4],new_labels[len(stage1_labels)-4:len(stage1_labels)+4],new_labels[-4:]))
        # reassign labels in cold mode segmentation map to new combined label indices
        if update_segm == True:
            segm_old += len(stage1_labels)
            indices = segm_old == len(stage1_labels)
            segm_old[indices] = 0
        # combine hot/cold mode segmentation maps into a single map
        segm = segm_new + segm_old
        new_label_values = np.unique(segm)
        new_label_values= np.delete(new_label_values,np.where(new_label_values==0))
        if diag==True:
            print('\nnew_label_values: [{} ... {} ... {}]'.format(new_label_values[0:3],new_label_values[len(stage1_labels)-3:len(stage1_labels)+4],new_label_values[-4:]))
        # concatenate the two detection catalogs
            setattr(self,segm_comb_str,segm)
        if diag==True:
            print('Length of "kronrad": {}\nLength of "kronflag": {}'.format(len(kronrad),len(kronflag)))
        print('# of unique new labels: {}\nTotal # of segments in combined segmentation map: {}\nTotal # of sources in combined detection catalog: {}'.format(len(np.unique(new_label_values)),len(new_label_values),len(obj['x'])))
        print('\n\nFunction "combine_detection_modes()" took {:.3f} seconds to run'.format(time.time()-start_time))
        return

    


"""Main class # 2: Cataloger"""
class Catalogor(object):
    """
    Class to read in results of source detecetion and multiband data set.
    Performs aperture photometry, and error analysis.
    """

    def __init__(self,cluster_name,field_name,master_filter_dict=None,segm=None,pix_scale=0.040,version=1.0):
        """
        Parameters
        ----------

        Attributes
        ----------
        """
        self.cluster_name = cluster_name
        self.field_name = field_name
        self.master_filter_dict = master_filter_dict
        self.pix_scale = pix_scale
        self.segm = segm
        self.inst_zp_dict = None
        self.version = version
        self.ref_cat = None
        self.indices_cat= None
        return

    """Convenience functions"""
    def midbins(self,bins):
        """define a function to compute the mid-points of the bins from a histogram"""
        size = len(bins)-1
        x_midbins = np.empty(size,dtype='float64')
        for x in range(size):
            x_midbins[x] = np.mean([bins[x],bins[(x+1)]])
        return x_midbins


    def find_zeropoint_image(self,IM_NAME,AB=True,ST=False):

        """
        Calculates zeropoint of image in desired magnituded system
        Parameters
        ----------
        IM_NAME = name and path of image file
        AB = True (default) # AB magnitude system
        ST = False # ST magnitude system

        Returns
        -------
        ZP

        Notes
        -----
        Header must contain PHOTFLAM and PHOTPLAM
        Reference: https://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints#:~:text=The%20PHOTFLAM%20and%20PHOTPLAM%20header,TPLAM)%E2%88%92

        """
        IM = fits.open(IM_NAME)[0]
        if ST:
            ZP = -2.5 * np.log10(IM.header['PHOTFLAM']) - 21.1
        elif AB:
            ZP = -2.5 * np.log10(IM.header['PHOTFLAM']) - 5 * (np.log10(IM.header['PHOTPLAM'])) -2.408
        else:
            ZP = np.nan
            print('Magnitude system not defined')

        return ZP



    "MAIN functions"
    def perform_multiband_photometry(self,objects,aperture_diameters,mask=None,extinction_dict=None,
                            ref_band=None,aper_min_diameter=0.3,meas_kron_flux=False,kronrad=None,kronflag=None,
                            kron_scale_fact=2.5,unit_flux='nJy',extinction=False,psfmatched=False,bkg_sub=False,
                            lcl_bkg_sub=False,bkgann_width=None,zp=None,diag=True):
        """
        This is the script which creates the multi band catalog.
        Parameters
        ----------
        objects:
        aperture_diameters:
        mask:
        extinction_dict:
        det_im:
        ref_band: str
        aper_min_diameter:
        meas_kron_flux: bool
        kronrad:
        kronflag:
        kron_scale_fact: float
        unit_flux: str
        extinction: bool
        psfmatched: bool
        bkg_sub: bool
        """
        start_time = time.time()
        self.aperture_diameters = aperture_diameters    # in arcsec
        self.meas_catalogs_tbs = {}
        self.aper_col_names = []
        self.color_aper = aper_min_diameter
        self.inst_zp_dict = {}
        self.kronrad = kronrad
        self.unit_flux = unit_flux
        if ref_band is not None:
            self.ref_band = ref_band

        for i, filter in enumerate(self.master_filter_dict.keys()):
            filter_time = time.time()
            filter_dict = self.master_filter_dict[filter]

            if psfmatched==True:
                fn_data = filter_dict['fn_im_matchf444w_sub']
            elif psfmatched==False:
                if bkg_sub==True:
                    fn_data = filter_dict['fn_im_nomatch_sub']
                elif bkg_sub==False:
                    fn_data = filter_dict['fn_im_nomatch_nosub']

            print("\n\nNow processing filter: {:s}\n".format(filter))
            data_data = fits.getdata(fn_data)
            data_wcs = WCS(header=fits.getheader(fn_data))
            
            mask_phot = data_data == 0
            if mask is not None:
                mask_phot+=mask
            mask_phot = mask_phot.astype(bool)
            
            if zp is None:
                data_zp = self.find_zeropoint_image(fn_data,AB=True)
                self.inst_zp_dict[filter] = data_zp
            else:
                self.inst_zp_dict[filter] = zp
            if diag==True:
                print('Performing photometry on file: {:s}\nPixel scale: {} arcsec\nInstrumental zp: {:.5f}'.format(fn_data,self.pix_scale,data_zp))

            data_data = data_data.byteswap(inplace=True).newbyteorder()

            # Conversion factor to convert fluxes to certain unit
            fact_to = 1

            # >>>>> Important <<<<< ADU to flux density conversion 
            if unit_flux == 'nJy':
                unit_zp = 31.4
            elif unit_flux == 'uJy':
                unit_zp = 23.9
            elif unit_flux == '25ABmag':
                unit_zp = 25
            fact_to *= 10.**(0.4*(unit_zp - data_zp)) # 25 for AB 25 mag system;  23.9 for uJy; 31.4 for nJy
            if diag==True:
                print('\nADU-to-nJy flux conversion factor to units of {}, pre-extinction (10.**(0.4*(unit_zp - instr_zp))): {:.4f}'.format(unit_flux,fact_to))

            # >>>>> Important <<<<< Extinction 
            if extinction==True:
                fact_to *= 10.**(0.4 * extinction_dict[filter]) # for extinction correction
                if diag==True:
                    print('Extinction factor: 10.**(0.4 * extinction_dict[filter]) = {:.4f}'.format((10.**(0.4 * extinction_dict[filter]))))
                    # print('ADU-to-AB mag25 flux conversion factor post-extinction: {:.4f}\n'.format(fact_to))

            circ_kron_radius = kron_scale_fact * kronrad * np.sqrt(objects['a']*objects['b'])
            
            catalog_meas_tmp_tb = QTable()
            seg_id = np.arange(1, len(objects['x'])+1,1, dtype=np.int32)

            for aper_d_arcsec in aperture_diameters:
                aper_r = aper_d_arcsec / 2. / self.pix_scale # TODO: Should only accept 0.5
                if diag==True:
                    if i==0:
                        print('Fixed aperture diameter: {:.2f}"\nFixed aperture radius: {:.2f}" or {:.3f}pix'.format(aper_d_arcsec,(aper_d_arcsec/2),aper_r))
                if lcl_bkg_sub == True:
                    buffer = 0.0 / self.pix_scale
                    bkg_aper = aperture_diameters[-1]/2./self.pix_scale
                    width = bkgann_width / self.pix_scale
                    bkgann_inner = np.ones(len(objects['x']),dtype='float32') * (bkg_aper+buffer)
                    bkgann_outer = np.ones(len(objects['x']),dtype='float32') * (bkg_aper+buffer+width)
                    mask_bkg = circ_kron_radius > bkg_aper
                    bkgann_inner[mask_bkg] = circ_kron_radius[mask_bkg]+buffer
                    bkgann_outer[mask_bkg] = circ_kron_radius[mask_bkg]+buffer+width
                    bkgann = [bkgann_inner, bkgann_outer]
                else:
                    bkgann = None
                # fixed aperture photometry
                flux, flux_err, flag = sep.sum_circle(data_data,objects['x'],objects['y'],aper_r * np.ones(len(objects['x'])),
                                         err=None,mask=mask_phot,gain=None,seg_id=seg_id,segmap=self.segm,
                                         bkgann=bkgann,subpix=5)
                
                key_flux = 'FLUX_APER{:02.0f}'.format(aper_d_arcsec*10)
                key_flux_err = 'FLUXERR_APER{:02.0f}'.format(aper_d_arcsec*10)
                
                # add designation for sources outside FOV (i.e. no data coverage)
                flux[flux==0.] = np.nan
                
                if i == 0:
                    self.aper_col_names.append(key_flux)
                    self.aper_col_names.append(key_flux_err)
                if diag==True:
                    n_nan = np.sum(np.isnan(flux))
                    print('NaN check:\nAPER{:02.0f}; # of NaNs: {} ({:.2f}%)'.format(aper_d_arcsec*10,n_nan,n_nan/len(flux)*100))
                    is_masked = (flag & sep.APER_ALLMASKED) != 0
                    print('# of sources w/ ALL pixels masked: {}'.format(np.sum(is_masked)))
                    is_trunc = (flag & sep.APER_TRUNC) != 0
                    print('# of sources w/ truncated aperture: {}'.format(np.sum(is_trunc)))
                    

                catalog_meas_tmp_tb[key_flux] = flux * fact_to
                catalog_meas_tmp_tb[key_flux].unit = unit_flux

                catalog_meas_tmp_tb[key_flux_err] = flux_err * fact_to
                catalog_meas_tmp_tb[key_flux_err].unit = unit_flux

            if meas_kron_flux==False:
                key_flux_flag = 'PHOT_FLAG'
                catalog_meas_tmp_tb[key_flux_flag] = flag
            else:
                key_flux_auto = 'FLUX_AUTO'
                key_flux_auto_err = 'FLUXERR_AUTO'
                self.aper_col_names.append(key_flux_auto)
                self.aper_col_names.append(key_flux_auto_err)
                catalog_meas_tmp_tb[key_flux_auto] = flux_err
                catalog_meas_tmp_tb[key_flux_auto].unit = unit_flux
                catalog_meas_tmp_tb[key_flux_auto_err] = flux_err
                catalog_meas_tmp_tb[key_flux_auto_err].unit = unit_flux
                
            # isophotal flux photometry
            flux, flux_err, flag = sep.sum_circle(data_data,objects['x'],objects['y'],aper_r * np.ones(len(objects['x'])),
                                     err=None,mask=mask_phot,gain=None,seg_id=-seg_id,segmap=self.segm,
                                     bkgann=bkgann,subpix=5)

            # add designation for sources outside FOV (i.e. no data coverage)
            flux[flux==0.] = np.nan
                
            key_flux = 'FLUX_ISO'
            key_flux_err = 'FLUXERR_ISO'
            self.aper_col_names.append(key_flux)
            self.aper_col_names.append(key_flux_err)
            if diag==True:
                n_nan = np.sum(np.isnan(flux))
                print('NaN check:\nFLUX_ISO; # of NaNs: {} ({:.2f}%)'.format(n_nan,n_nan/len(flux)*100))

            catalog_meas_tmp_tb[key_flux] = flux * fact_to
            catalog_meas_tmp_tb[key_flux].unit = unit_flux

            catalog_meas_tmp_tb[key_flux_err] = flux_err * fact_to
            catalog_meas_tmp_tb[key_flux_err].unit = unit_flux
            
            # elliptical photometry
            if meas_kron_flux==True:
                if (filter==self.ref_band) or (self.ref_band=='all'):
                    # set minimum aperture for kron apertures
                    aper_min_pix = aper_min_diameter / 2. / self.pix_scale  # minimum diameter 0.35"; aper_min_pix in pixels
                    # Now do KRON apertures; start w/ min circ aperture to avoid sources w/ kronrad==nan
                    if diag==True:
                        print('\naper_min_pix: {}'.format(aper_min_pix))
                    use_kron_radius = np.ones_like(objects['x']) * aper_min_pix     # set min radius within circ. kron radius vector
                    use_kron =  (circ_kron_radius > aper_min_pix) | np.isnan(kronrad)
                    nan_mask = np.isnan(kronrad)
                    kronrad[nan_mask] = aper_min_pix
                    use_kron_radius[use_kron] = circ_kron_radius[use_kron]          # for diagnostic purposes only

                    if lcl_bkg_sub == True:
                        buffer = 1. / self.pix_scale
                        bkg_aper = aperture_diameters[-1]/2./self.pix_scale
                        width = (bkgann_width+0.5) / self.pix_scale
                        bkgann_inner_kron = np.ones(len(objects['x']),dtype='float32') * (bkg_aper+buffer)
                        bkgann_outer_kron = np.ones(len(objects['x']),dtype='float32') * (bkg_aper+buffer+width)
                        mask_bkg = circ_kron_radius > bkg_aper+buffer
                        bkgann_inner_kron[mask_bkg] = circ_kron_radius[mask_bkg]+buffer
                        bkgann_outer_kron[mask_bkg] = circ_kron_radius[mask_bkg]+buffer+width
                        bkgann_kron = [bkgann_inner_kron, bkgann_outer_kron]
                        bkgann_kron = None
                    else:
                        bkgann = None
                        bkgann_kron = None

                    flux, fluxerr, flag = sep.sum_circle(data_data, objects['x'],objects['y'],aper_min_pix,
                                                            mask=mask_phot,segmap=self.segm,seg_id=seg_id,
                                                            bkgann=bkgann,subpix=5)

                    kflux, kfluxerr, kflag = sep.sum_ellipse(data_data,objects['x'],objects['y'],
                                                             objects['a'],objects['b'],
                                                             objects['theta'],(kron_scale_fact*kronrad),
                                                             mask=mask_phot,seg_id=seg_id,segmap=self.segm,
                                                             bkgann=bkgann_kron,subpix=5)

                    flux[use_kron] = kflux[use_kron]
                    fluxerr[use_kron] = kfluxerr[use_kron]
                    flag[use_kron] = kflag[use_kron]

                    # add designation for sources outside FOV (i.e. no data coverage)
                    flux[flux==0.] = np.nan
                    
                    flag |= kronflag            # combine photometry & kron flags into 'flag'
                    flag |= objects['flag']     # combine photometry & source extraction flags into 'flag'

                    sel = flux>0
                    mag = -2.5*np.log10(flux[sel]) + data_zp

                    if diag==True:
                        mag_list = [22.,24.,26.,28.]
                        for mag_ref in mag_list:
                            index = self.find_nearest(mag, mag_ref)
                            print('\nFor filter {}:\nIndex: {};   Mag = {:.3f};   Circ. kron radius = {:.5f}"'.format(filter,index,mag[index],(use_kron_radius[sel][index]*self.pix_scale)))

                    key_flux = 'FLUX_AUTO'
                    key_flux_err = 'FLUXERR_AUTO'
                    key_flux_flag = 'PHOT_FLAG'
                    catalog_meas_tmp_tb[key_flux] = flux * fact_to
                    catalog_meas_tmp_tb[key_flux].unit = unit_flux
                    catalog_meas_tmp_tb[key_flux_err] = flux_err * fact_to
                    catalog_meas_tmp_tb[key_flux_err].unit = unit_flux
                    catalog_meas_tmp_tb[key_flux_flag] = flag

            self.meas_catalogs_tbs[filter] = catalog_meas_tmp_tb
            print('Filter {} took {:.3f} seconds to run.'.format(filter,time.time()-filter_time))
        print('\n\nFunction "perform_multiband_photometry()" took {:.3f} seconds to run.'.format(time.time()-start_time))
        return


    def find_nearest(self, list, value):
            """the idea: feed in list of magnitudes, and find_nearest tells you the
            index of the one closest to the limiting magnitude. that way if the clostest
            object's mag. is above the limiting mag., you can just choose the next object
            in the list, since the list is sorted."""
            array = np.asarray(list)
            index = (np.abs(array - value)).argmin()
            return index



    def create_cat(self,objects,obj_cols_to_use,cat_cols_names,kronrad=True):
        """Compile photometry in multiple bands into a single Astropy QTable
        Parameters
        ----------
        objects:
        obj_cols_to_use: list
            list of strings listing the columns to use from "objects"
        cat_cols_names: list
            list of names for final catalog corresponding to the columns used
            in "obj_cols_to_use"
        kronrad: bool
            whether or not a 'KRON_RADIUS' column is to be added
        ref_band: str
            filter in which total kron flux was measured
        """
        labels = np.arange(1,len(objects['x'])+1,dtype=np.int32).reshape(len(objects['x']),1)
        self.cat = QTable(labels)
        self.cat.rename_column('col0', 'NUMBER')
        # for (item,name,unit) in zip(obj_cols_to_use,obj_cols_names,obj_cols_units):
        for (item,name) in zip(obj_cols_to_use,cat_cols_names):
            self.cat[name] = objects[item]

        for filter,filter_dict in self.meas_catalogs_tbs.items():
            for col_name in self.aper_col_names:
                self.cat[col_name+'_'+filter] = filter_dict[col_name]
        self.cat['PHOT_FLAG'] = self.meas_catalogs_tbs[self.ref_band]['PHOT_FLAG']

        if kronrad==True:
            kron_index = self.cat.colnames.index('A')
            self.cat.add_column(self.kronrad,name='KRON_RADIUS',index=kron_index)
        return



    def add_total_fluxes(self,fixed_aper,aperture_diameters):
        """
        Add total fluxes for all filters, where the ratio of total-to-fixed_aper
        flux is equal in all bands and set by the ratio in the ref_band.
        Parameters
        ----------
        fixed_aper: float
            aperture size in arcsec which will set the total-to-fixed_aper flux ratio
        ref_band: str
            the filter which will set the total-to-fixed_aper flux ratio for all filters
        """
        start_time = time.time()
        ref_total_flux = self.cat['FLUX_AUTO_{}'.format(self.ref_band)]
        ref_fixed_aper_flux = self.cat['FLUX_APER{:02.0f}_{}'.format(fixed_aper*10,self.ref_band)]
        total_to_fixed_flux_ratio = (ref_total_flux / ref_fixed_aper_flux)

        for filter in self.master_filter_dict.keys():
            if filter == self.ref_band:
                pass
            else:
                filt_fixed_aper_flux = self.cat['FLUX_APER{:02.0f}_{}'.format(fixed_aper*10,filter)]
                filt_total_flux = filt_fixed_aper_flux * total_to_fixed_flux_ratio
                self.cat['FLUX_AUTO_{}'.format(filter)] = filt_total_flux
        print('\n\nFunction "add_total_fluxes()" took {:.3f} seconds to run.'.format(time.time()-start_time))
        return



    def find_empty_aperture_positions(self,det_dict,bkg_method='sep',box_size=75,kernel_size=3,filter_kernel=None,mask=None,n_apertures=2000,diag=False):
        """
        define a function to measure the flux in N empty apertures (where the
        apertures may vary in size). produce a histogram of the results.
        Parameters
        ----------
        det_dict: dict
            dictionary containing the following fields:
            nsigma: float
                multiple above detection threshold for a pixel to be considered
                a "detection"
            npix: int
                # of connected pixels for a source segmented to be counted as
                a "detection"
            nlevels: int
                # of deblending levels
            contrast: float
                deblend contrast
        bkg_method: str
            package used in calculating the background during source detection; either 'sep' or 'photutils'
        mask: bool array
            mask applied during source detection
        n_apertures: int
            # of empty aperture positions to find
        diag: bool
            produce a plot of mask, segm, and empty aperture positions on the
            combined segm+mask img
        """
        start_time = time.time()
        sep.set_extract_pixstack(1e7)
        sep.set_sub_object_limit(1e4)
        
        self.empty_aper_positions_x = {}
        self.empty_aper_positions_y = {}

        # assign detection parameters
        nsigma = det_dict['nsigma']
        npixels = det_dict['npixels']
        nlevels = det_dict['nlevels']
        contrast = det_dict['contrast']

        for filter in self.master_filter_dict.keys():
            print('\nWorking on filter: {}...'.format(filter))

            filter_time = time.time()
            self.empty_aper_positions_x[filter] = []
            self.empty_aper_positions_y[filter] = []
            filter_dict = self.master_filter_dict[filter]

            fn_data = filter_dict['fn_im_matchf444w_sub']

            data_data = fits.getdata(fn_data)
            data_data = data_data.byteswap(inplace=True).newbyteorder()

            mask_phot = data_data == 0
            if mask is not None:
                mask_phot+=mask

            if bkg_method == 'sep':
                # measure a spatially varying background on the image
                bkg = sep.Background(data_data, mask=mask_phot, bw=box_size,
                                     bh=box_size, fw=kernel_size, fh=kernel_size)
                # get a "global" mean and noise of the image background:
                bkg_globalback = bkg.globalback
                bkg_globalrms = bkg.globalrms
                bkg_image = bkg.back()
                bkg_rms = bkg.rms()

                print('Median bkg: {:.5e}'.format(bkg_globalback))
                print('Median rms: {:.5e}'.format(bkg_globalrms))

            elif bkg_method == 'photutils':
                # 2D background estimation
                sigma_clip = SigmaClip(sigma=3., maxiters=50)
                bkg_estimator = SExtractorBackground()
                bkg_rms_estimator = MADStdBackgroundRMS()

                #compute background and update user on key stats
                bkg = Background2D(data_data,box_size,filter_size=kernel_size,edge_method='pad',sigma_clip=sigma_clip,
                                   bkg_estimator=bkg_estimator,bkgrms_estimator=bkg_rms_estimator,
                                   coverage_mask=mask_phot,fill_value=0.)
                # get a "global" mean and noise of the image background:
                bkg_globalback = bkg.background_median
                bkg_globalrms = bkg.background_rms_median
                bkg_image = bkg.background
                bkg_rms = bkg.background_rms

                print('Median bkg: {:.5e}'.format(bkg_globalback))
                print('Median rms: {:.5e}'.format(bkg_globalrms))



            objects, segm = sep.extract(data_data, nsigma, err=bkg_globalrms, mask=mask_phot,
                              minarea=npixels, filter_kernel=filter_kernel, filter_type='matched',
                              deblend_nthresh=nlevels, deblend_cont=contrast, clean=True,
                              clean_param=1.0, segmentation_map=True)

            print('Beginning search for empty apertures...')

            segm_temp_arr = segm + mask_phot.astype('int32')
            segm_temp_img = SegmentationImage(segm_temp_arr)

            if diag==True:
                fig, ax1 = plt.subplots(1,1,figsize=(6,6))
                im1 = ax1.imshow(mask_phot,vmin=0,vmax=1,cmap='gray_r',origin='lower',interpolation='nearest')
                ax1.set_title('Mask being applied to segm')
                divider = make_axes_locatable(ax1)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im1, cax=cax, orientation='vertical')
                plt.show()

                fig, ax = plt.subplots(1, 1, figsize=(6,6))
                cmap = segm_temp_img.make_cmap(seed=123)
                ax.imshow(segm_temp_img, origin='lower', cmap=cmap, interpolation='nearest')
                ax1.set_title('segm + mask')
                ax.set_title('Empty apertures: 0/{}'.format(n_apertures))
                plt.show()
                plt.close()

            # store dimensions of segmentation map
            N, M = segm_temp_arr.shape
            # initialize a list to store position of empty apertures
            self.empty_aper_positions_x[filter] = []
            self.empty_aper_positions_y[filter] = []
            # store radius of aperture in pixels
            radius = np.max(self.aperture_diameters) / 2. / self.pix_scale
            if diag==True:
                print('\nMax radius of empty aperture: {:.2f}" or {:.3f}px'.format(radius*self.pix_scale,radius))

            n_apertures_found = 0
            while n_apertures_found < n_apertures:
                position_sum = 1
                while position_sum > 0:
                    pos_check = 1
                    while pos_check > 0:
                        # generate random position
                        x = random.randint(0, N-1)
                        y = random.randint(0, M-1)
                        #
                        pos_check = segm_temp_arr[x,y]
                    position = np.transpose([y,x])#(x,y)
                    aper = CircularAperture(position, r=radius)
                    temp_phot_table = aperture_photometry(segm_temp_img, aper)
                    position_sum = temp_phot_table['aperture_sum'][0]
                self.empty_aper_positions_x[filter].append(y)      # to correct for transpose line above, i.e. this is not a mistake. y appends to x_list, x to y_list
                self.empty_aper_positions_y[filter].append(x)
                if diag==True:
                    if len(self.empty_aper_positions_x[filter])%np.floor(n_apertures/4) == 0:
                        print('Empty positions found so far: {}/{}'.format(len(self.empty_aper_positions_x[filter]),n_apertures))
                n_apertures_found+=1
                #
            #
            if diag==True and filter == self.ref_band:

                fig, ax = plt.subplots(1, 1, figsize=(8,8))
                cmap = segm_temp_img.make_cmap(seed=123)
                ax.imshow(segm_temp_img, origin='lower', cmap=cmap, interpolation='nearest')
                ax.set_title('Empty apertures: {}'.format(n_apertures))
                for xx,yy in zip(self.empty_aper_positions_x[filter],self.empty_aper_positions_y[filter]):
                    circ = Circle((xx,yy),radius,edgecolor='r',fill=False,linewidth=2.)
                    ax.add_patch(circ)
                plt.show()
                plt.close()
            print('Filter {} took {:.3f} seconds to run.'.format(filter,time.time()-filter_time))
        #
        print('\n\nFunction "find_empty_aperture_positions()" took {:.3f} seconds to run.'.format(time.time()-start_time))
        return



    def measure_empty_aperture_fixed_apertures(self,mask,fit_param_guess=None,hist_n_bins=50,
                                            hist_x_lim=[-2e3,2e3],psfmatched=True,bkg_sub=True,use_wht=True,
                                            output_dir='./',diag=True):
        """
        define a function which takes in empty positions and measures the flux in
        circular apertures of varying sizes; produces a histogram of the result
        and fit a gaussian to it; report sigma of gaussian at each aperture size
        Parameters
        ----------
        mask: array-like, bool
            mask used for photometry, where True value indicates a pixel to be masked
        fit_param_guess: array
            initialize guesses for fitting gaussian parameters;
            format: [alpha_init_guess,beta_init_guess]
        hist_n_bins: int
            # of bins to use when plotting histograms of empty aperture
            fluxes
        hist_range: array
            min/max value for histogram
        psfmatched: bool
            whether the multiband data set has been convolved to a common
            psf; selects the colvolved files from self.master_filter_dict
        bkg_sub: bool
            whether to use the background-subtracted files for
            non-convolved imgs
        use_wht: bool
            whether to noise-normalize the image before measuring fluxes;
            noise_norm_data = data / (1 / np.sqrt(wht_file))
        """
        start_time = time.time()

        self.sigma_dict_final = {}
        self.alpha_dict = {}
        self.beta_dict = {}

        self.img_list= []
        self.img_err_list = []

        # convert aperture sizes from arcsec to pixels
        diameters_arcsec = np.round(np.arange(self.aperture_diameters[0],(self.aperture_diameters[-1]+0.1),0.1),decimals=1)    # in arcsec
        radii = diameters_arcsec / 2. / self.pix_scale    # in pix
        if diag:
            print('"radii" array in pixels: {}\n'.format(radii))
            print('\nAperture sizes (diameter) in arcsec: {}'.format(diameters_arcsec))



        """work on each filter, one at a time..."""
        for ii,filter in enumerate(self.master_filter_dict.keys()):
            print('\nNow working on filter: {}...'.format(filter))
            self.sigma_dict_final[filter] = {}
            sigma_list_filter_all_apers = []
            # sigma_list_filter_fixed_apers = []

            if psfmatched==True:
                img = fits.getdata(self.master_filter_dict[filter]['fn_im_matchf444w_sub'])
            elif psfmatched==False:
                if bkg_sub==True:
                    img = fits.getdata(self.master_filter_dict[filter]['fn_im_nomatch_sub'])
                elif bkg_sub==False:
                    img = fits.getdata(self.master_filter_dict[filter]['fn_im_nomatch_nosub'])
            img = img.byteswap(inplace=True).newbyteorder()

            mask = img==0.
            if use_wht == True:
                img_err = fits.getdata(self.master_filter_dict[filter]['fn_rms'])
                img_err = img_err.byteswap(inplace=True).newbyteorder()
                img[~mask] = ( img[~mask] / img_err[~mask] )
                if diag==True:
                    print('\nMedian value in noise-normalized img: {:.5e}'.format(np.median(img[img!=0.])))

            if diag==True:
                # now create a histogram of the results of each aperture measurement
                fig, ax = plt.subplots(1,1,figsize=(10,10))
                color = iter(cmr.neon(np.linspace(0.05,1.,len(self.aperture_diameters))))
                ax.set_title('Distribution of fluxes enclosed within empty apertures')
            #


            """for each filter, perform aperture photometry in each aperture size, fit a gaussian"""
            for aper_size in range(len(diameters_arcsec)):
                aper_r = radii[aper_size]     # in pix

                flux, flux_err, flag = sep.sum_circle(img, self.empty_aper_positions_x[filter],self.empty_aper_positions_y[filter],
                                         aper_r, err=None, mask=mask, gain=None)

                len_counts = len(flux)
                flux = flux[~np.isnan(flux) & np.isfinite(flux)]
                if diag==True:
                    if aper_size == len(diameters_arcsec)-1:
                        print('# of NaN/inf apertures removed: {}'.format(len_counts-len(flux)))
                # define histogram & bins
                y_hist, bins_hist = np.histogram(flux,bins=hist_n_bins,density=False)
                midbins_hist = self.midbins(bins_hist)
                # fit a gaussian best fit to the data
                amp_init = np.max(y_hist)
                mu_init = np.mean(flux,dtype='float32')
                sigma_init = np.std(flux,dtype='float64')

                popt, pcov = curve_fit(self.fit_Gaussian,midbins_hist, y_hist, p0=[amp_init,mu_init,sigma_init],maxfev=int(1e5))

                sigma = np.abs(popt[-1])
                # fwhm = 2 * np.sqrt(2*np.log(2)) * sigma

                if diag==True:
                    if diameters_arcsec[aper_size] in self.aperture_diameters:
                        print('\nAperture Size: {}"\n\ninitial gaussian estimate: [amp, mu, sigma]: [{:0.3e}, {:0.3e}, {:0.7e}]'.format(diameters_arcsec[aper_size],amp_init,mu_init, sigma_init))
                        print('Final gaussian fit:          [amp, mu, sigma]: [{:0.3e}, {:0.3e}, {:0.7e}]'.format(popt[0],popt[1],popt[2]))
                        print('Sigma: {:0.3e}'.format(sigma))

                y_gaussian_guess = self.fit_Gaussian(midbins_hist, amp_init, mu_init, sigma_init)
                y_gaussian_fit = self.fit_Gaussian(midbins_hist, *popt)

                sigma_list_filter_all_apers.append(sigma)

                if diameters_arcsec[aper_size] in self.aperture_diameters:
                    self.sigma_dict_final[filter]['APER{:02.0f}'.format(diameters_arcsec[aper_size]*10)] = sigma
                    # sigma_list_filter_fixed_apers.append(sigma)
                    if diag==True:
                        c = next(color)
                        ax.hist(flux,bins=hist_n_bins,histtype='step',linewidth=1.,label=str((diameters_arcsec[aper_size]))+'"',color=c)
                        ax.plot(midbins_hist,y_gaussian_fit,linewidth=2.,c=c)

            if diag==True:
                ax.set_title('Filter: {}'.format(filter),fontsize=15)
                ax.set_xlim(hist_x_lim)
                ax.legend(loc='upper left',frameon=False,fontsize=15)
                if use_wht==True:
                    ax.set_xlabel('Emtpy aper flux [ADU]',fontsize=15)
                elif use_wht==False:
                    ax.set_xlabel('Emtpy aper flux [AB flux]',fontsize=15)
                ax.set_ylabel('Count',fontsize=15)
                fn_save = os.path.join(output_dir,'empty_aper_hists_{}_v{}.png'.format(filter,self.version))
                plt.savefig(fn_save,format='png')
                plt.show()
                plt.close()

            if diag==True:
                print('# of fixed apertures: {}\n# of sigmas measured: {}'.format(len(self.aperture_diameters),len(self.sigma_dict_final[filter])))


            """store sigmas for fixed apertures in each filter in a dict for output"""
            # self.sigma_dict_final[filter] = sigma_list_filter_fixed_apers

            if filter==self.ref_band:
                # now fit a power law
                alpha_guess,beta_guess = fit_param_guess
                x0 = fit_param_guess
                x0_bounds = ([1e-5,-1],[5e4,5])     # bound on ([alpha_lower_bound,beta_lower_bound],[alpha_upper_bound,beta_upper_bound])
                if diag==True:
                    print('len(diameters_arcsec): {}\nlen(sigma_list_filter_all_apers): {}'.format(len(diameters_arcsec),len(sigma_list_filter_all_apers)))
                #
                fitting_method = 'scipy.optimize.curve_fit()'
                popt_sigma, pcov_sigma = curve_fit(self.error_power_law_scaling,diameters_arcsec,sigma_list_filter_all_apers,p0=x0,bounds=x0_bounds,maxfev=2e3)
                alpha_fit, beta_fit = popt_sigma
                if diag==True:
                    print('\n\nCurve-fitting method: {:s}\nparameters, initial guess - [alpha,beta]: [{:.3e},{:.3f}]\nparameters, best fit - [alpha,beta]: [{:.3e},{:.3f}]'.format(fitting_method,alpha_guess,beta_guess,alpha_fit,beta_fit))

                x_scaling = np.arange(np.min(diameters_arcsec),np.max(diameters_arcsec),0.001)
                y_scaling_fit = self.error_power_law_scaling(x_scaling,alpha_fit, beta_fit)#, std_07=std_07)
                y_scaling_fit_guess = self.error_power_law_scaling(x_scaling,alpha_guess, beta_guess)
                if diag==True:
                    # Visualize
                    fig, ax1 = plt.subplots(1,figsize=(8,8))
                    ax1.set_title('Noise scaling as a function of aperture: {}'.format(filter),fontsize=15)
                    ax1.scatter(diameters_arcsec,sigma_list_filter_all_apers,c='k',marker='^',s=100)
                    ax1.plot(x_scaling,y_scaling_fit,color='r',linestyle='solid',linewidth=2.,label='best-fit')
                    ax1.plot(x_scaling,y_scaling_fit_guess,color='grey',linestyle='dashed',linewidth=2.,label='initial guess')
                    string = 'alpha: {:.3e}\n\nbeta: {:.3f}'.format(alpha_fit,beta_fit)
                    ax1.legend(loc='upper left',frameon=False,fontsize=15)
                    ax1.set_xlabel('Aperture diameter [arcsec]',fontsize=15)
                    if use_wht==True:
                        ax1.set_ylabel('$\sigma$ [ADU]',fontsize=15)
                    elif use_wht==False:
                        ax1.set_ylabel('$\sigma$ [AB flux]',fontsize=15)
                    fn_save = os.path.join(output_dir,'sigma_v_aper_size_{}_v{}.png'.format(filter,self.version))
                    plt.savefig(fn_save,format='png')
                    plt.show()
                    plt.close()
                #
                self.alpha_dict[filter] = alpha_fit
                self.beta_dict[filter] = beta_fit

        # Visualize
        fig, ax1 = plt.subplots(1,figsize=(6,6))
        fig.suptitle('Noise scaling - by filter')

        color = iter(cm.tab10(np.linspace(0, 1, len(self.sigma_dict_final)+3)))

        for filter in self.sigma_dict_final.keys():
            sigma_filter_list= []
            for aper in self.aperture_diameters:
                sigma_filter_list.append(self.sigma_dict_final[filter]['APER{:02.0f}'.format(aper*10)])
            c = next(color)
            ax1.plot(self.aperture_diameters,sigma_filter_list,color=c,linestyle='solid',linewidth=1.,label='{}'.format(filter))
        ax1.legend(loc='upper left',frameon=False,fontsize=15,markerscale=3.)
        ax1.set_xlabel('aperture diameter [arcsec]',fontsize=15)
        ax1.set_ylabel('sigma [ADU]',fontsize=15)
        plt.show()
        plt.close()


        print('\n\nFunction "measure_empty_aperture_fixed_apertures()" took {:.3f} seconds to run.'.format(time.time()-start_time))
        return


        """Convenience functions for "measure_empty_apertures_fixed_apertures()" """

    def fit_Gaussian(self,x,a,mean,sigma):
        """define a function to fit a gaussian to a histogram"""
        return (a * np.exp(-1* ( (x-mean)**2 / (2 * sigma**2) ) ) )

    def error_power_law_scaling(self,aper_diameter,alpha,beta):
        """
        define a function to describe the standard deviation of the empty apertures
        distribution as a function of aperture size (entered as DIAMETER in ARCSEC)
        Parameters
        ----------
        aper_diameter: float
            aperture diameter in arcsec
        alpha, beta: float
            normalization constant and slope of power law describing growth
            in empty aperture errors
        """
        radius = aper_diameter / 2.
        N = np.sqrt(np.pi*(radius**2))
        return alpha * N**beta

    
    def assign_phot_errors_fixed_apertures(self,use_wht=True,diag=False,diag_II=False):
        """
        Assign empty aperture photometric errors for fixed apertures.
        Parameters
        ----------
        use_wht: bool
            whether or not to use the wht_file in computing
            position-specific errors
        diag: bool
            print general dianostic information
        diag_II: bool
            add diagnostic columns to catalog table for further investigation
        """
        start_time = time.time()
        aper_radii = (self.aperture_diameters / 2.) / self.pix_scale        # in pixels
        print('Apertures list in diameter ("): {}\nApertures list in radii (pix): {}'.format(self.aperture_diameters,aper_radii))


        """work on filters one at a time"""
        for ii,filter in enumerate(self.master_filter_dict.keys()):
            print('\n\nNow working on filter: {}...\n'.format(filter))
            # std_err_list_filter = self.sigma_dict_final[filter]

            time_filter = time.time()

            # Conversion factor to convert fluxes to certain unit
            if self.unit_flux == 'nJy':
                unit_zp = 31.4
            elif self.unit_flux == 'uJy':
                unit_zp = 23.9
            elif self.unit_flux == '25ABmag':
                unit_zp = 25
            fact_to = 1.
            fact_to = 10.**(0.4*(unit_zp - self.inst_zp_dict[filter])) # 23.9 for uJy; 31.4 for nJy
            print('ADU-to-nJy flux conversion factor: {}'.format(fact_to))

            if use_wht==True:

                img = fits.getdata(self.master_filter_dict[filter]['fn_rms'])
                img = img.byteswap(inplace=True).newbyteorder()

                mask = np.isinf(img)
                # img[~mask] = (1. / np.sqrt(img[~mask]) )
                plt.imshow(mask,vmin=0,vmax=1,cmap='Greys',origin='lower')
                plt.colorbar()
                plt.show()

                seg_id = np.arange(1, len(self.cat['X'])+1,1, dtype=np.int32)

                """for each filter, work on one aperture at a time"""
                for jj,aper in enumerate(self.aperture_diameters):
                    aper_r = aper / 2. / self.pix_scale        # in pix
                    print('aper_r used in sep.sum_cirlce() for APER{:02.0f}: {:.4f}px or {:.2f}"'.format(aper*10,aper_r,aper/2))

                    flux, flux_err, flag = sep.sum_circle(img,self.cat['X'],self.cat['Y'],
                                             aper_r, err=None, mask=mask, 
                                             segmap=self.segm,seg_id=seg_id,subpix=5)

                    std_err_filter_aper = self.sigma_dict_final[filter]['APER{:02.0f}'.format(aper*10)]
                    print('std_err_filter_aper: {}'.format(std_err_filter_aper))
                    print('# NaN fluxes from noise img: {}'.format(np.sum(np.isnan(flux))))
                    # std_err_filter_aperture = std_err_list_filter[jj]
                    is_masked = (flag & sep.APER_ALLMASKED) != 0
                    print('# of sources w/ ALL pixels masked: {}'.format(np.sum(is_masked)))
                    is_trunc = (flag & sep.APER_TRUNC) != 0
                    print('# of sources w/ truncated aperture: {}'.format(np.sum(is_trunc)))
                    is_trunc = (flag & sep.OBJ_TRUNC) != 0
                    print('# of sources w/ truncated "object" (source in segm?): {}'.format(np.sum(is_trunc)))

                    area = np.pi * aper_r**2        # compute area of aperture for calculation of avg noise at position of each source, in units of pixels

                    # store circular aperture photometry / area
                    # compute avg noise at position of each source and store in a list to be added to the catalogue
                    avg_noise = flux / area

                    if diag_II==True:
                        # add avg noise at position of each source, and another diagnostic column for the avg EA error for an aperature of the appropriate size
                        flux_col1 = Column(flux,name='FLUXERR_APER{:02.0f}_{}_total_noise'.format(aper,filter))
                        flux_col2 = Column(avg_noise,name='FLUXERR_APER{:02.0f}_{}_avg_noise'.format(aper,filter))
                        flux_col3 = Column(std_err_filter_aperture,name='FLUXERR_APER{:02.0f}_{}_std_error'.format(aper,filter))
                        try:
                            self.cat.add_column(flux_col1)
                            self.cat.add_column(flux_col2,index=-1)
                            self.cat.add_column(flux_col3,index=-1)
                        except:
                            self.cat['FLUXERR_APER{:02.0f}_{}_total_noise'.format(aper*10,filter)] = flux_col1
                            self.cat['FLUXERR_APER{:02.0f}_{}_avg_noise'.format(aper*10,filter)] = flux_col2
                            self.cat['FLUXERR_APER{:02.0f}_{}_std_error'.format(aper*10,filter)] = flux_col3

                    """assign the actual error"""
                    self.cat['FLUXERR_APER{:02.0f}_{}'.format(aper*10,filter)] = avg_noise * std_err_filter_aper * fact_to
                    self.cat['FLUXERR_APER{:02.0f}_{}'.format(aper*10,filter)].unit = u.nJy

                    # add designation for sources outside FOV (i.e. no data coverage) 
                    err_mask = np.isnan(self.cat['FLUX_APER{:02.0f}_{}'.format(aper*10,filter)])
                    self.cat['FLUXERR_APER{:02.0f}_{}'.format(aper*10,filter)][err_mask] = np.nan

                    fluxerr = self.cat['FLUXERR_APER{:02.0f}_{}'.format(aper*10,filter)] /u.nJy
                    flux = self.cat['FLUX_APER{:02.0f}_{}'.format(aper*10,filter)] /u.nJy
                    SNR = flux / fluxerr

                    mag = -2.5 * np.log10(flux) + 34.1
                    sel = (~np.isnan(SNR) & np.isfinite(SNR))
                    print('\nFor filter {} in APER{:02.0f}:'.format(filter,aper*10))
                    SNR_list = [50.,20.,10.,5.]
                    for SNR_ref in SNR_list:
                        index = self.find_nearest(SNR[sel], SNR_ref)
                        print('Index: {}; @ SNR = {:.2f};  mag = {:.2f}'.format(index,SNR[sel][index],mag[sel][index]))

                    if diag==True:
                        print('# NaN areas: {}'.format(np.sum(np.isnan(area))))
                        n_nan_flux = np.sum(np.isnan(self.cat['FLUX_APER{:02.0f}_{}'.format(aper*10,filter)]))
                        print('\nTotal # of NaN fluxes measured in APER{:02.0f} filter {}: {}'.format(aper*10,filter,n_nan_flux))
                        n_nan_err = np.sum(np.isnan(self.cat['FLUXERR_APER{:02.0f}_{}'.format(aper*10,filter)]))
                        print('Total # of NaN errors measured in APER{:02.0f} filter {}: {}\nTotal length of catalogue: {}'.format(aper*10,filter,n_nan_err,len(self.cat)))


                print('Computing errors in filter: {} took {:.3f}s'.format(filter,(time.time() - time_filter)))
            elif use_wht == False:
                for jj,aper in enumerate(self.aperture_diameters):
                    std_err_filter_aper = self.sigma_dict_final[filter]['APER{:02.0f}'.format(aper*10)]
                    print('\nstd_err_filter_aperture for filter {}, APER{:02.0f}": {}\n'.format(filter,aper*10,std_err_filter_aper))
                    print('\nstd_err_filter_aper: {:.4f}\nfact_to: {:.4f}'.format(std_err_filter_aper,fact_to))
                    self.cat['FLUXERR_APER{:02.0f}_{}'.format(aper*10,filter)] = std_err_filter_aper * fact_to * u.nJy
                    # self.cat['FLUXERR_APER{:02.0f}_{}'.format(aper,filter)].unit = u.nJy



        print('\n\nFunction "assign_phot_errors_fixed_apertures()" took {:.3f} seconds to run.'.format(time.time()-start_time))
        return
    


    def assign_phot_errors_kron_apertures(self,mask=None,kron_scale_fact=2.5,use_wht=True,diag=False,diag_II=False):
        """
        define a function to assign photometric errors to all sources in the
        self.catalogue table list of error images (these are weight maps where:
        noise = 1 / sqrt(weight) ) should correspond to filter_names order.
        Parameters
        ----------
        ref_band: str
            filter in which total flux was measured, which sets the
            total-to-fixed aperture flux ratio (or error ratio)
        use_wht: bool
            whether or not to use the wht_file in computing
            position-specific errors
        diag: bool
            print general dianostic information
        diag_II: bool
            add diagnostic columns to catalog table for further investigation
        """
        start_time = time.time()
        alpha = self.alpha_dict[self.ref_band]
        beta = self.beta_dict[self.ref_band]

        if diag_II==True:
            print('\nKron aperture error scalings for sigma = alpha * N**beta, where N = sqrt(pi*r**2):\nFilter: {};   alpha: {:.5f};   beta: {:.5f}'.format(self.ref_band,alpha,beta))
            try:
                new_col_name = 'FLUXERR_AUTO_{}_total_noise'.format(self.ref_band)
                new_col = Column(-99,name=new_col_name)
                index = self.cat.colnames.index('KRON_RADIUS_{}'.format(self.ref_band))
                self.cat.add_column(new_col, index=(index+1))

                new_col_name = 'FLUXERR_AUTO_{}_avg_noise'.format(self.ref_band)
                new_col = Column(-99,name=new_col_name)
                index = self.cat.colnames.index('FLUXERR_AUTO_{}'.format(self.ref_band))
                self.cat.add_column(new_col, index=(index+1))

                new_col_name = 'FLUXERR_AUTO_{}_std_error'.format(self.ref_band)
                new_col = Column(-99,name=new_col_name)
                index = self.cat.colnames.index('FLUXERR_AUTO_{}'.format(self.ref_band))
                self.cat.add_column(new_col, index=(index+1))
            except:
                pass

        fn_wht = self.master_filter_dict[self.ref_band]['fn_rms']
        if self.inst_zp_dict==None:
            data_zp = find_zeropoint_image(fn_wht,AB=True)
            print('zp: {:.5f}'.format(data_zp))
        else:
            data_zp = self.inst_zp_dict[self.ref_band]

        # Conversion factor to convert fluxes to certain unit
        if self.unit_flux == 'nJy':
            unit_zp = 31.4
        elif self.unit_flux == 'uJy':
            unit_zp = 23.9
        elif self.unit_flux == '25ABmag':
            unit_zp = 25
        fact_to = 1
        fact_to *= 10.**(0.4*(unit_zp - data_zp)) # 23.9 for nJy; 31.4 for nJy
        print('ADU-to-nJy flux conversion factor: {}'.format(fact_to))

        # compute errors in the reference band
        kronrad = self.cat['KRON_RADIUS']   # in pix
        circ_kron_radius = kron_scale_fact * kronrad * np.sqrt(self.cat['A']*self.cat['B'])  # in pix

        aper_min_pix = self.color_aper / 2. / self.pix_scale  # minimum diameter = 3.5
        print('Min aper radius: {:.3f}px of {:.3f}"'.format(aper_min_pix,self.color_aper/2.))
        use_circle = circ_kron_radius < aper_min_pix
        circ_kron_radius[use_circle] = aper_min_pix
        area = np.pi * circ_kron_radius**2  # in pix

        if use_wht == True:
            img = fits.getdata(fn_wht)
            img = img.byteswap(inplace=True).newbyteorder()
            # img[img!=0] = (1 / np.sqrt(img[img!=0]) )

            seg_id = np.arange(1, len(self.cat['X'])+1,1, dtype=np.int32)
            print('len of cat["X"]: {}'.format(len(self.cat['X'])))
            print('len of seg_id: {}'.format(len(seg_id)))

            flux, fluxerr, flag = sep.sum_ellipse(img,self.cat['X'],self.cat['Y'],self.cat['A'],
                                    self.cat['B'],self.cat['THETA'],(kron_scale_fact * kronrad),
                                    mask=mask,segmap=self.segm,seg_id=seg_id,subpix=5)
            cflux, cfluxerr, cflag = sep.sum_circle(img, self.cat['X'],
                                    self.cat['Y'],aper_min_pix,mask=mask,
                                    segmap=self.segm,seg_id=seg_id,subpix=5)
            
            print('\nLength of self.cat: {}\nLength of min aper fluxes ("use_circle"): {}'.format(len(self.cat),np.sum(use_circle)))

            flux[use_circle] = cflux[use_circle]
            fluxerr[use_circle] = cfluxerr[use_circle]
            flag[use_circle] = cflag[use_circle]

            # # add designation for sources outside FOV (i.e. no data coverage)
            # flux[flux==0.] = np.nan
            
            avg_aper_error = flux / area

            if diag_II==True:
                self.cat['FLUXERR_AUTO_{}_total_noise'.format(self.ref_band)] = flux
                self.cat['FLUXERR_AUTO_{}_avg_noise'.format(self.ref_band)] = avg_aper_error

        # now multiply by "standard error' based on size of kron radius of an empty aperture
        max_kron_error = 0
        max_kron_aperture = 0

        empty_aperture_radius_kron = circ_kron_radius * self.pix_scale                     # in arcsec
        empty_aperture_radius_ISO = np.sqrt(self.cat['AREA']/np.pi) * self.pix_scale       # equivalent circular radius for ISE AREA, in arcsec

        empty_aperture_error_kron = self.error_power_law_scaling((empty_aperture_radius_kron * 2.),alpha,beta)
        empty_aperture_error_ISO = self.error_power_law_scaling((empty_aperture_radius_ISO * 2.),alpha,beta)

        if use_wht == True:
            self.cat['FLUXERR_AUTO_{}'.format(self.ref_band)] = empty_aperture_error_kron * avg_aper_error * fact_to
            self.cat['FLUXERR_AUTO_{}'.format(self.ref_band)].unit = u.nJy
            if diag_II==True:
                self.cat['FLUXERR_AUTO_{}_std_error'.format(self.ref_band)] = empty_aperture_error_kron
            if diag==True:
                print('Median FLUXERR_AUTO_{}: {:.4e}'.format(self.ref_band,np.median(self.cat['FLUXERR_AUTO_{}'.format(self.ref_band)])))

        elif use_wht == False:
            self.cat['FLUXERR_AUTO_{}'.format(self.ref_band)] = empty_aperture_error_kron * fact_to
            self.cat['FLUXERR_AUTO_{}'.format(self.ref_band)].unit = u.nJy
            if diag==True:
                print('Median FLUXERR_AUTO_{}: {:.4e}'.format(self.ref_band,np.median(self.cat['FLUXERR_AUTO_{}'.format(self.ref_band)])))

        # add designation for sources outside FOV (i.e. no data coverage) 
        err_mask = np.isnan(self.cat['FLUX_AUTO_{}'.format(self.ref_band)])
        self.cat['FLUXERR_AUTO_{}'.format(self.ref_band)][err_mask] = np.nan
        
        nan_mask = np.isnan(self.cat['FLUXERR_AUTO_{}'.format(self.ref_band)])
        max_kron_error = np.max(self.cat['FLUXERR_AUTO_{}'.format(self.ref_band)][~nan_mask])
        index = np.where(self.cat['FLUXERR_AUTO_{}'.format(self.ref_band)]==max_kron_error)[0]
        max_kron_aperture = empty_aperture_radius_kron.data[index][0]
        if diag==True:
            print('\nMax_kron_error: {:.4f} on circularized kron_radius of {:.3f}"'.format(max_kron_error,max_kron_aperture))

        fluxerr = self.cat['FLUXERR_AUTO_{}'.format(self.ref_band)]/u.nJy
        flux = self.cat['FLUX_AUTO_{}'.format(self.ref_band)]/u.nJy
        SNR = flux / fluxerr
        mag = -2.5*np.log10(flux) + 34.1

        print('\nFor filter: {}'.format(self.ref_band))
        sel = (~np.isnan(SNR) & np.isfinite(SNR))
        SNR_list = [50.,20.,10.,5.]
        for SNR_ref in SNR_list:
            index = self.find_nearest(SNR[sel], SNR_ref)
            print('Index: {};   SNR = {:.2f};   mag = {:.3f};   Circ. kron radius = {:.5f}"'.format(index,SNR[sel][index],mag[sel][index],(circ_kron_radius[sel][index]*self.pix_scale)))
   
        if diag==True:
            print('# NaN areas: {}'.format(np.sum(np.isnan(area))))
            n_nan_flux = np.sum(np.isnan(self.cat['FLUX_AUTO_{}'.format(self.ref_band)]))
            print('\nTotal # of NaN fluxes measured in  AUTO filter {}: {}'.format(self.ref_band,n_nan_flux))
            n_nan_err = np.sum(np.isnan(self.cat['FLUXERR_AUTO_{}'.format(self.ref_band)]))
            print('Total # of NaN errors measured in  AUTO filter {}: {}\nTotal length of catalogue: {}'.format(self.ref_band,n_nan_err,len(self.cat)))
                        
        # now assign errors to all the other bands
        ref_total_flux = self.cat['FLUX_AUTO_{}'.format(self.ref_band)]
        ref_fixed_flux = self.cat['FLUX_APER{:02.0f}_{}'.format(self.color_aper*10,self.ref_band)]
        total_to_fixed_flux_ratio = np.abs(ref_total_flux / ref_fixed_flux)

        for ii,filter in enumerate(self.master_filter_dict.keys()):
            print('\n\nNow working on filter {}...\n'.format(filter))
            if filter==self.ref_band:
                pass
            else:
                self.cat['FLUXERR_AUTO_{}'.format(filter)] = self.cat['FLUXERR_APER{:02.0f}_{}'.format(self.color_aper*10,filter)]  * total_to_fixed_flux_ratio
                # self.cat['FLUXERR_AUTO_{}'.format(filter)].unit = u.nJy

                # add designation for sources outside FOV (i.e. no data coverage) 
                err_mask = np.isnan(self.cat['FLUXERR_AUTO_{}'.format(filter)])
                self.cat['FLUXERR_AUTO_{}'.format(filter)][err_mask] = np.nan
                
                fluxerr = self.cat['FLUXERR_AUTO_{}'.format(filter)]/u.nJy
                flux = self.cat['FLUX_AUTO_{}'.format(filter)]/u.nJy
                SNR = flux / fluxerr
                mag = -2.5*np.log10(flux) + 34.1

                print('\nFor filter: {}'.format(filter))
                sel = (~np.isnan(SNR) & np.isfinite(SNR))
                SNR_list = [50.,20.,10.,5.]
                for SNR_ref in SNR_list:
                    index = self.find_nearest(SNR[sel], SNR_ref)
                    print('Index: {};   SNR = {:.2f};   mag = {:.3f};   Circ. kron radius = {:.5f}"'.format(index,SNR[sel][index],mag[sel][index],(circ_kron_radius[sel][index]*self.pix_scale)))
                
                if diag==True:
                    print('# NaN areas: {}'.format(np.sum(np.isnan(area))))
                    n_nan_flux = np.sum(np.isnan(self.cat['FLUX_AUTO_{}'.format(filter)]))
                    print('\nTotal # of NaN fluxes measured in AUTO filter {}: {}'.format(filter,n_nan_flux))
                    n_nan_err = np.sum(np.isnan(self.cat['FLUXERR_AUTO_{}'.format(filter)]))
                    print('Total # of NaN errors measured in AUTO filter {}: {}\nTotal length of catalogue: {}'.format(filter,n_nan_err,len(self.cat)))

        print("\nLength of GS' self.cat: {}".format(len(self.cat)))

        print('\n\nFunction "assign_phot_errors_kron_apertures()" took {:.3f} seconds to run.'.format(time.time()-start_time))
        return

                  

    def assign_phot_errors_iso_apertures(self,mask=None,use_wht=True,diag=False):
        """
        define a function to assign photometric errors to all sources in the
        self.catalogue table list of error images (these are weight maps where:
        noise = 1 / sqrt(weight) ) should correspond to filter_names order.
        Parameters
        ----------
        ref_band: str
            filter in which total flux was measured, which sets the
            total-to-fixed aperture flux ratio (or error ratio)
        use_wht: bool
            whether or not to use the wht_file in computing
            position-specific errors
        diag: bool
            print general dianostic information
        """
        start_time = time.time()
        alpha = self.alpha_dict[self.ref_band]
        beta = self.beta_dict[self.ref_band]

        # now multiply by "standard error' based on size of kron radius of an empty aperture
        empty_aperture_radius_ISO = np.sqrt(self.cat['AREA']/np.pi) * self.pix_scale       # equivalent circular radius for ISO AREA, in arcsec

        empty_aperture_error_ISO = self.error_power_law_scaling((empty_aperture_radius_ISO * 2.),alpha,beta)

        for ii,filter in enumerate(self.master_filter_dict.keys()):
            print('\nFor filter: {}'.format(filter))
            if use_wht == True:
                fn_wht = self.master_filter_dict[filter]['fn_rms']
                img = fits.getdata(fn_wht)
                img = img.byteswap(inplace=True).newbyteorder()
                # img[img!=0] = (1 / np.sqrt(img[img!=0]) )
                # get zero point
                if self.inst_zp_dict==None:
                    data_zp = find_zeropoint_image(fn_wht,AB=True)
                    print('zp: {:.5f}'.format(data_zp))
                else:
                    data_zp = self.inst_zp_dict[self.ref_band]

                # Conversion factor to convert fluxes to certain unit
                if self.unit_flux == 'nJy':
                    unit_zp = 31.4
                elif self.unit_flux == 'uJy':
                    unit_zp = 23.9
                elif self.unit_flux == '25ABmag':
                    unit_zp = 25
                fact_to = 1
                fact_to *= 10.**(0.4*(unit_zp - data_zp)) # 23.9 for nJy; 31.4 for nJy
                print('ADU-to-nJy flux conversion factor: {}'.format(fact_to))
                
                seg_id = np.arange(1, len(self.cat['X'])+1,1, dtype=np.int32)

                # ISO error
                flux, fluxerr, flag = sep.sum_ellipse(img,self.cat['X'],self.cat['Y'],self.cat['A'],
                                        self.cat['B'],self.cat['THETA'],(2.5 * self.cat['KRON_RADIUS']),
                                        mask=mask,segmap=self.segm,seg_id=-seg_id,subpix=5)

                # # add designation for sources outside FOV (i.e. no data coverage)
                # flux[np.isnan(self.cat['FLUX_ISO_{}'.format(filter)])] = np.nan
                
                avg_ISO_aper_error = flux / self.cat['AREA']           

                self.cat['FLUXERR_ISO_{}'.format(filter)] = empty_aperture_error_ISO * avg_ISO_aper_error * fact_to
                self.cat['FLUXERR_ISO_{}'.format(filter)].unit = u.nJy
                if diag==True:
                    nan_mask = np.isnan(self.cat['FLUXERR_ISO_{}'.format(filter)])
                    print('Median FLUXERR_ISO_{}: {:.4e}'.format(filter,np.median(self.cat['FLUXERR_ISO_{}'.format(filter)][~nan_mask])))
                    print('# of NaN ISO errors: {}'.format(np.sum(nan_mask)))

            elif use_wht == False:
                self.cat['FLUXERR_ISO_{}'.format(filter)] = empty_aperture_error_ISO * fact_to
                self.cat['FLUXERR_ISO_{}'.format(filter)].unit = u.nJy
                if diag==True:
                    nan_mask = np.isnan(self.cat['FLUXERR_ISO_{}'.format(filter)])
                    print('Median FLUXERR_ISO_{}: {:.4e}'.format(filter,np.median(self.cat['FLUXERR_ISO_{}'.format(filter)][~nan_mask])))
                    print('# of NaN ISO fluxes: {}'.format(np.sum(np.isnan(self.cat['FLUX_ISO_{}'.format(filter)]))))
                    print('# of NaN ISO errors: {}'.format(np.sum(nan_mask)))

            # add designation for sources outside FOV (i.e. no data coverage) 
            err_mask = np.isnan(self.cat['FLUX_ISO_{}'.format(filter)])
            self.cat['FLUXERR_ISO_{}'.format(filter)][err_mask] = np.nan

            fluxerr = self.cat['FLUXERR_ISO_{}'.format(filter)]/u.nJy
            flux = self.cat['FLUX_ISO_{}'.format(filter)]/u.nJy
            SNR = flux / fluxerr
            mag = -2.5*np.log10(flux) + 34.1

            sel = (~np.isnan(SNR) & np.isfinite(SNR))
            SNR_list = [50.,20.,10.,5.]
            for SNR_ref in SNR_list:
                index = self.find_nearest(SNR[sel], SNR_ref)
                print('Index: {};   SNR = {:.2f};   mag = {:.3f};   Area = {}'.format(index,SNR[sel][index],mag[sel][index],self.cat['AREA'][sel][index]))

        print("\nLength of GS' self.cat: {}".format(len(self.cat)))

        print('\n\nFunction "assign_phot_errors_iso_apertures()" took {:.3f} seconds to run.'.format(time.time()-start_time))
        return



    def apply_aper_corr_factors(self,fn_ref_band_cog,apply_to_fixed_aper=False,apply_to_color_aper=False,apply_to_kron_aper=True,
                                cog_unit='arcsec',kron_scale_fact=2.5,save_fig=False,fn_save=None,diag=False):
        """
        Compute aperture corrections for fraction of light from psf which falls outside
        the aperture of a given size.
        Parameters
        ----------
        fn_ref_band_cog: str
            filename to the curve of growth of the psf for the reference band
        ref_band: str
            filter in which total fluxes are measured
        save_fig: bool
            whether or not to save figure
        fn_save: str
            filename of file to be saved; must be provided if save_fig==True
        apply_to_fixed_aper/apply_to_color_aper/apply_to_kron_aper: bool
            apertures to which the correction will be applied
        kron_scale_fact: float
            kron scale factor
        """
        start_time = time.time()    ###
        # load ref band curve of growth
        curve_of_growth_array = np.loadtxt(fn_ref_band_cog)
        if cog_unit == 'arcsec':
            x_cog_radii = curve_of_growth_array[0] / self.pix_scale   # array is given in arcsec. convert to pixels
        else:
            x_cog_radii = curve_of_growth_array[0]    # array is given in pixels
        y_cog_flux = curve_of_growth_array[1]
        if diag==True:
            print('Curve of growth\nx_min: {}\nx_max: {}'.format(np.min(x_cog_radii),np.max(x_cog_radii)))
        # create a function that interpolates ref band curve of growth via cubic spline; input to function is radius in pixels
        ref_band_curve_of_growth = interp1d(x_cog_radii,y_cog_flux,kind='cubic')
        
        if apply_to_fixed_aper==True:
        # calculate correction factors for each fixed aperture
            for aper in self.aperture_diameters:
                new_fixed_col_name = 'APER{:02.0f}_CORR_FACT'.format(aper*10)
                new_col = Column(np.nan,name=new_fixed_col_name)
                try:
                    self.cat.add_column(new_col, index=-1)
                except:
                    pass

                source_radius = (aper / 2 /self.pix_scale)
                if source_radius > x_cog_radii[-1]:
                    source_radius = x_cog_radii[-1]

                flux_enclosed = ref_band_curve_of_growth(source_radius)

                self.cat[new_fixed_col_name] = (1 / flux_enclosed)
        
        if apply_to_color_aper==True:
        # calculate correction factors for each fixed aperture
            new_fixed_col_name = 'APER{:02.0f}_CORR_FACT'.format(self.color_aper*10)
            new_col = Column(np.nan,name=new_fixed_col_name)
            try:
                self.cat.add_column(new_col, index=-1)
            except:
                pass

            source_radius = (self.color_aper / 2 /self.pix_scale)
            if source_radius > x_cog_radii[-1]:
                source_radius = x_cog_radii[-1]

            flux_enclosed = ref_band_curve_of_growth(source_radius)

            self.cat[new_fixed_col_name] = (1 / flux_enclosed)
            
        # calculate correction factors for kron apertures of all sources
        if apply_to_kron_aper==True:
            new_fixed_col_name = 'AUTO_CORR_FACT'
            new_col = Column(np.nan,name=new_fixed_col_name)
            try:
                self.cat.add_column(new_col, index=-1)
            except:
                pass

            aper_r_min = self.color_aper / 2 / self.pix_scale
            circ_kron_radius = 2.5*self.cat['KRON_RADIUS'] * np.sqrt(self.cat['A']*self.cat['B'])
            index_below_r_min = np.where((circ_kron_radius) < aper_r_min)[0]
            index_above_r_max = np.where((circ_kron_radius) > np.max(x_cog_radii))[0]
            if diag==True:
                print('\naper_r_min: {:.4f}\n# of sources below aper_r_min: {}\n# of sources above max for ref curve of growth: {}'.format(aper_r_min,len(index_below_r_min),len(index_above_r_max)))

            for source in range(len(self.cat)):
                if source in index_below_r_min:
                    flux_enclosed = ref_band_curve_of_growth((aper_r_min))
                elif source in index_above_r_max:
                    flux_enclosed = ref_band_curve_of_growth(np.max(x_cog_radii))
                else:
                    flux_enclosed = ref_band_curve_of_growth(circ_kron_radius[source])

                self.cat[new_fixed_col_name][source] = (1 / flux_enclosed)

        # store correction factor for fixed aperture of 0.7" diameter
        flux_enclosed_dict = {}
        x_corr_factors = []
        y_corr_factors = []
        key_list = []

        for aper in self.aperture_diameters:
            radius = aper/2.
            flux_enclosed_dict[str(radius)] = ref_band_curve_of_growth(radius/self.pix_scale)
            flux_enclosed_dict[str(radius)] = 1/flux_enclosed_dict[str(radius)]
            x_corr_factors.append(radius)
            y_corr_factors.append(flux_enclosed_dict[str(radius)])
            key_list.append(str(radius))

        gs = {"hspace":0, "height_ratios":[2,1]}   #make a tiled-plot like vdB2013 w/ fractions below, this line sets the proporitons of plots in the figure

        fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8,12),sharex=True,gridspec_kw=gs)
        fig.suptitle('{} PSF & Total Flux Correction Factors'.format(self.ref_band),fontsize=20)
        fig.patch.set_facecolor('beige')

        ax1.plot((x_cog_radii*self.pix_scale),y_cog_flux,linestyle='solid',linewidth=2.5,color='cornflowerblue',label='HST/WFC3 PSF - {}'.format(self.ref_band))
        ax1.plot([self.color_aper/2.,self.color_aper/2.],[0,1.01],linestyle='dotted',linewidth=1.5)
        ax1.text(self.color_aper/2.+0.02,0.25,'Catalog aperture',fontsize=14)
        ax1.set_xlabel('Radius [arcsec]',fontsize=16)
        ax1.set_ylabel('Normalized Flux Enclosed',fontsize=16)
        ax1.set_xlim([0,np.max((x_cog_radii*self.pix_scale))+0.1])
        ax1.set_ylim([0,1.01])
        ax1.legend(loc='lower right',frameon=False,fontsize=16)
        ax1.xaxis.set_label_position("top")
        ax1.tick_params(axis='both',which='both',direction='inout',length=7,labelbottom=False,labeltop=True,left=True,right=True,bottom=True,top=True,labelsize='large')
        if apply_to_kron_aper==True:
            ax2.scatter((self.pix_scale*circ_kron_radius[~np.isnan(self.cat[new_fixed_col_name])]),self.cat[new_fixed_col_name][~np.isnan(self.cat[new_fixed_col_name])],c='cornflowerblue',alpha=0.5,s=10,label='AUTO aperture correction factor')
        ax2.scatter(x_corr_factors,y_corr_factors,c='firebrick',alpha=1,s=50,label='Fixed aperture correction factor')
        ax2.plot([self.color_aper/2.,self.color_aper/2.],[1,2.05],linestyle='dotted',linewidth=1.5)
        ax2.set_xlabel('Radius [arcsec]',fontsize=16)
        ax2.set_ylabel('Total flux correction factor',fontsize=16)
        ax2.set_xlim([0,np.max((x_cog_radii*self.pix_scale))+0.1])
        ax2.set_ylim([1,2.05])
        ax2.legend(loc='upper right',frameon=False,fontsize=16,markerscale=2.)
        ax2.tick_params(axis='both',which='both',direction='inout',length=7,labelbottom=True,labeltop=False,left=True,right=True,bottom=True,top=True,labelsize='large')

        # string = 'Total flux correction factor\nr = 0.15": {:.2f}      r = 0.75": {:.2f}\nr = 0.25": {:.2f}      r = 1.50": {:.2f}\nr = 0.35": {:.2f}'.format(flux_enclosed_dict[key_list[0]],flux_enclosed_dict[key_list[1]],flux_enclosed_dict[key_list[2]],flux_enclosed_dict[key_list[3]],flux_enclosed_dict[key_list[4]])
        # ax2.text(0.82,1.5,string,fontsize=14)
        plt.subplots_adjust(hspace=0)
        plt.show()
        plt.close()

        # apply aperture correction factors to fluxes and errors
        for filter in self.master_filter_dict.keys():
            if apply_to_fixed_aper==True:
                for aper in self.aperture_diameters:
                    self.cat['FLUX_APER{:02.0f}_{:s}'.format(aper*10,filter)] *= self.cat['APER{:02.0f}_CORR_FACT'.format(aper*10)]
                    self.cat['FLUXERR_APER{:02.0f}_{:s}'.format(aper*10,filter)] *= self.cat['APER{:02.0f}_CORR_FACT'.format(aper*10)]
            if apply_to_color_aper==True:
                for aper in self.aperture_diameters:
                    self.cat['FLUX_APER{:02.0f}_{:s}_TOTAL'.format(self.color_aper*10,filter)] = self.cat['FLUX_APER{:02.0f}_{:s}'.format(self.color_aper*10,filter)] * self.cat['APER{:02.0f}_CORR_FACT'.format(self.color_aper*10)]
                    self.cat['FLUXERR_APER{:02.0f}_{:s}_TOTAL'.format(self.color_aper*10,filter)] = self.cat['FLUXERR_APER{:02.0f}_{:s}'.format(self.color_aper*10,filter)] * self.cat['APER{:02.0f}_CORR_FACT'.format(self.color_aper*10)]
            if apply_to_kron_aper==True:
                self.cat['FLUX_AUTO_{}'.format(filter)] *= self.cat['AUTO_CORR_FACT']
                self.cat['FLUXERR_AUTO_{}'.format(filter)] *= self.cat['AUTO_CORR_FACT']
        if apply_to_fixed_aper==True:
            print('Aperture correction applied to FIXED APER fluxes.')
        if apply_to_kron_aper==True:
            print('Aperture correction applied to TOTAL fluxes.')
        print("Length of GS' catalog: {}".format(len(self.cat)))

        print('\n\nFunction "apply_aper_corr_factors()" took {:.3f} seconds to run.'.format(time.time()-start_time))
        return



    def match_sources_by_RA_DEC(cat,cat_ids=['RA','DEC'],ref_cat=None,ref_cat_ids=['RA','DEC'],matching_tol=0.3,diag=True):
        """ Match sources in one catalog to sources from a reference catalog, for
        comparing properties.
        Parameters
        ----------
        ref_cat: astropy Table
            reference catalog to match to
        ref_cat_ids: array
            identifies the names of the RA/DEC column in the ref_cat. must be in
            the order [RA,DEC]
        matching_tol: float
            the maximum distance between two sources in order to be considered a match; in arcsec
        Returns
        ----------
        indices_cat_common, indices_ref_cat_common: array
            indices of sources in cat, ref_cat, which are matched between both catalogs
        unique_cat, unique_ref_cat: array
            indices of sources in cat, ref_cat which are unique to each catalog (i.e. not found in the other)
        """
        start_time = time.time()
        if diag==True:
            print('length of cat catalog: {} rows.\nlength of cat_to_match catalog: {} rows.'.format(len(cat),len(ref_cat)))
        
        n_matched = 0
        indices_cat_common = []
        indices_ref_cat_common = []
        
        # store coords of ref cat
        RA_to_match = ref_cat[ref_cat_ids[0]] /u.deg * np.pi / 180 * u.rad   # convert to rad
        DEC_to_match = ref_cat[ref_cat_ids[1]] /u.deg * np.pi / 180 * u.rad   # convert to rad
        # print('RA_to_match: ',RA_to_match)
            
        # loop through sources in cat one at a time to find matches
        for source1 in range(len(cat)):

            RA_cat = cat[cat_ids[0]][source1] * np.pi / 180 * u.rad   # convert to rad
            DEC_cat = cat[cat_ids[1]][source1] * np.pi / 180 * u.rad   # convert to rad
            # print('RA_cat: ',RA_cat)
            # print('DEC_cat: ',DEC_cat)

            if source1%int(len(cat)/4) == 0:
                frac_done = (source1 / len(cat)) * 100
                if diag==True:
                    print('\nMatching row {}\n% complete: {:.2f}\n# matched: {}\nTime elapsed: {}s'.format(source1,frac_done,n_matched,(time.time()-start_time)))


            dist_astropy = astropy.coordinates.angular_separation(RA_cat,DEC_cat,RA_to_match,DEC_to_match)
            dist_astropy = dist_astropy / u.rad * 180 / np.pi * 3600  # convert back to degrees then arcsec
            
            matches = dist_astropy < matching_tol

            if np.sum(matches) > 1:
                print('WARNING: {} matches found for cat source {}'.format(np.sum(matches),source1))
            
            if np.sum(matches) == 1:
                
                index = np.where(matches == True)[0][0]
                # print('source: {};   match index: {}'.format(source1,index))
                n_matched+=1
                indices_cat_common.append(source1)
                indices_ref_cat_common.append(index)
                    
            
        # determine indices unique to each catalog
        arr_cat = np.asarray(indices_cat_common.copy())
        arr_ref_cat = np.asarray(indices_ref_cat_common.copy())
        
        unique_cat = np.arange(len(cat['RA']))
        unique_ref_cat = np.arange(len(ref_cat[ref_cat_ids[0]]))
        
        test = np.isin(unique_cat,arr_cat)
        test_ref = np.isin(unique_ref_cat,arr_ref_cat)
        
        unique_cat = np.delete(unique_cat,test)
        unique_ref_cat = np.delete(unique_ref_cat,test_ref)

        print('\n\n# of matched objects between cat & ref_cat: {}\n# sources unique to cat: {}\n# of sources unique to ref cat: {}\n\nIndices for cat stored as "self.indices_cat".\nIndices for ref_cat stored as "self.indices_ref_cat".'.format(n_matched,len(unique_cat),len(unique_ref_cat)))
        print('\n\nFunction "match_sources_by_RA_DEC()" took {:.3f} seconds to run.'.format(time.time()-start_time))
        return indices_cat_common, indices_ref_cat_common, unique_cat, unique_ref_cat


    def plot_zphot_zspec(self,zmax=8.,outlier_limit=0.1,scale='log',field_name='Some field...',
                         save_fig=False,dir_out='./'):
        """Plot zpht v zspec.
        Parameters
        ----------
        zmax: float
            maximum redshift to be plotted
        outlier_limit: float
            define outliers as (zphot - zspec) / (1 + zspec) > outlier_limit
        scale: str - 'log' or 'linear'
            choose plotting in log scale or linear scale
        save_fig: bool
            whether or not to save the figure
        """
        if not os.path.isdir(dir_out):
            os.makedirs(dir_out)
            print("\nCreated folder : "+dir_out)

        fn_out = os.path.join(dir_out,field_name+'_zphot_v_zspec_sep_{}_v{}.png'.format(scale,self.version))


        ## derive the bound which separate outliers in terms of plotting arrays for X and Y
        x_bound = np.arange(0.0,zmax,0.01)
        #
        y_upper_bound, y_lower_bound = self.delz_bound(x_bound,outlier_limit)
        #

        zphot = self.cat['z_phot'][self.cat['z_spec']!=(-1)]
        zspec = self.cat['z_spec'][self.cat['z_spec']!=(-1)]

        delz = (zphot - zspec) / (1 + zspec)

        n_outliers = np.sum(np.abs(delz)>outlier_limit)
        frac_outliers = n_outliers / len(delz) * 100

        outliers = np.argwhere(np.abs(delz)>outlier_limit)
        not_outliers = np.argwhere(np.abs(delz)<outlier_limit)
        median_delz = np.median(np.abs(delz[not_outliers]))
        scatter_delz = np.std(np.abs(delz[not_outliers]))

        print('# source in catalog: {}\n# of z_spec in catalog: {}\n# of outliers: {}'.format(len(self.cat['z_phot']),len(delz),n_outliers))

        ## construct a string to plot the outlier fraction & std dev ('scatter')
        string = 'Median |$\Delta$z|: %.3f'%median_delz+'\n$\sigma_{|\Delta z|}$ = %.3f'%scatter_delz+'\nOutliers (|$\Delta$z|>{:.0f}%):'.format(outlier_limit*100)+' %.2f'%frac_outliers+'%'+'\n$N_{obj}$: %d'%len(zspec)
        ## add text to plot
        fig,ax1 = plt.subplots(1,1,figsize=(7,7))
        ax1.scatter(zspec[not_outliers],zphot[not_outliers],c='b',marker='^',alpha=0.55,linewidths=0,s=30)
        ax1.scatter(zspec[outliers],zphot[outliers],c='r',marker='v',alpha=0.55,linewidths=0,s=30)
        if scale == 'log':
            ax1.text(0.315,.13,'z$_{clu}$: 0.308', fontsize=15)
            ax1.text(0.105,2.5,string, fontsize=15)
        elif scale == 'linear':
            ax1.text(0.15,0.7*z_ax,string, fontsize=15)
        ax1.plot([0,zmax],[0,zmax],'--k', linewidth=2)
        ax1.plot(x_bound,y_upper_bound,':k', linewidth=1.5)
        ax1.plot(x_bound,y_lower_bound,':k', linewidth=1.5)
        ax1.plot([0.308,0.308],[y_lower_bound,y_upper_bound],'k',linestyle='dashed',linewidth=1.)
        ax1.set_xlabel("$z_{spec}$", fontsize=15)
        ax1.set_xscale(scale)
        ax1.set_xlim(0.1,zmax)
        ax1.set_xticks([0.1,0.3,1,2,4,zmax])
        ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax1.set_yscale(scale)
        ax1.set_ylim(0.1,zmax)
        ax1.set_yticks([0.1,0.3,1,2,4,zmax])
        ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax1.set_ylabel("$z_{phot}$", fontsize=15)
        ax1.set_title("%s"%field_name+" - v%s"%self.version+"\n$z_{spec}$ vs. $z_{phot}$",fontsize=20)
        ax1.tick_params(axis='both', which='both',direction='in',color='k',top=True,right=True,labelright=False,labelsize=15)
        ax1.minorticks_on()
        plt.savefig(fn_out,format='png')
        plt.show()
        return


    def delz_bound(self,x,outlier_limit):
        """define a function to compute the Y-arrays for the bounds
        defining spectroscopic outliers"""
        y_upper_bound = outlier_limit*(1+x) + x
        y_lower_bound = -outlier_limit*(1+x) + x
        return y_upper_bound, y_lower_bound


    def make_diagnostic_plots(self,ref_cat=None,ref_cat_name='Reference catalog',id_stars=False,save_fig=False,output_dir='./'):
        """
        Make some useful diagnostic plots
        Parameters
        ----------
        ref_cat: astropy Table
            catalog against which to compare against.
            "self.match_sources_by_RA_DEC()" should be run first if a ref_cat
            Table is provided.
        ref_cat_name: str
            name of ref_cat for plotting .
        save_fig: bool
            whether or not to save the diagnostic plots.
        output_dir: str
            filepath to directory where output is to be saved.
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        start_time = time.time()
        if not os.path.isdir(os.path.join(output_dir,'diagnostic_plots/')):
            os.makedirs(os.path.join(output_dir,'diagnostic_plots/'))
            print("\nCreated folder : "+os.path.join(output_dir,'diagnostic_plots/'))

        if ref_cat is not None:
            if self.indices_cat is None:
                self.match_sources_by_RA_DEC(ref_cat)

        # MAG_GS v MAG_ref hist
        x_range = [18,34]
        y_range = x_range

        fig,axs = plt.subplots(3,4,figsize=(16,8),sharex=True,sharey=True)

        for ii,filter in enumerate(self.master_filter_dict.keys()):

            if ref_cat is not None:
                flux_GS = self.cat['FLUX_AUTO_{}'.format(filter)][self.indices_cat] / u.nJy
                fluxerr_GS = self.cat['FLUXERR_AUTO_{}'.format(filter)][self.indices_cat] / u.nJy
            else:
                flux_GS = self.cat['FLUX_AUTO_{}'.format(filter)] / u.nJy
                fluxerr_GS = self.cat['FLUXERR_AUTO_{}'.format(filter)] / u.nJy

            mag_GS = -2.5 * np.log10(flux_GS) + 34.1
            SNR_GS = flux_GS / fluxerr_GS
            sel = ~np.isnan(SNR_GS)
            fluxerr_GS = fluxerr_GS[sel]
            mag_GS = mag_GS[sel]
            SNR_GS = SNR_GS[sel]

            index = self.find_nearest(SNR_GS, 5)
            mag_SNR5 = mag_GS[index]
            index = self.find_nearest(SNR_GS, 3)
            mag_SNR3 = mag_GS[index]

            if ref_cat is not None:
                mag_ref_cat = -2.5 * np.log10(ref_cat['f_{}'.format(filter)][self.indices_ref_cat]) + 34.1
                mag_ref_cat = mag_ref_cat[sel]
                axs.flat[ii].hist(mag_ref_cat,bins='auto',color='b',histtype='step',linewidth=1.5,label=ref_cat_name)
            axs.flat[ii].hist(mag_GS,bins='auto',color='r',histtype='step',linewidth=1.5,label='GS cat')
            axs.flat[ii].plot([mag_SNR5,mag_SNR5],[0,600],'--k',linewidth=1.5,label='SNR=5')
            axs.flat[ii].plot([mag_SNR3,mag_SNR3],[0,600],':k',linewidth=1.5,label='SNR=3')
            if ii in range(7,11):
                axs.flat[ii].set_xlabel("MAG_AUTO", fontsize=20)
            axs.flat[ii].text(18.5,140,filter, fontsize=20)
            axs.flat[ii].set_xscale('linear')
            axs.flat[ii].set_xlim(x_range)
            axs.flat[ii].set_xticks([18,20,22,24,26,28,30,32])
            axs.flat[ii].set_ylim([0,600])
            if (ii==0) or (ii==3) or (ii==4) or (ii==8):
                axs.flat[ii].set_ylabel("Count", fontsize=20)
            axs.flat[ii].set_yscale('linear')
            axs.flat[ii].tick_params(axis='both', which='both',direction='in',color='k',top=True,right=True,labelright=False,labelsize=15)
            if ii==7:
                axs.flat[ii].set_ylabel("Count", fontsize=20)
                axs.flat[ii].tick_params(axis='both', which='both',direction='in',color='k',top=True,right=True,labelbottom=True,labelright=True,labelsize=15)
                axs.flat[ii].yaxis.set_label_position("right")
            if ii==3:
                axs.flat[ii].tick_params(axis='both', which='both',direction='in',color='k',top=True,right=True,labelright=True,labelbottom=True,labelsize=15)
                axs.flat[ii].yaxis.set_label_position("right")
        axs.flat[0].legend(loc='upper right',frameon=False,fontsize=15)
        axs.flat[-1].axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        if save_fig==True:
            if ref_cat is not None:
                plt.savefig(os.path.join(output_dir,'diagnostic_plots/hist_MAG_v{}_ref.png'.format(self.version)),format='png')
            else:
                plt.savefig(os.path.join(output_dir,'diagnostic_plots/hist_MAG_v{}.png'.format(self.version)),format='png')
        plt.show()
        plt.close()


        # CLR-MAG scatter plot
        x_range = [18,32]
        y_range = [-2,2]

        if ref_cat is not None:
            H_ref_cat = -2.5*np.log10(ref_cat['f_F160W'][self.indices_ref_cat]) + 34.1
            JH_ref_cat = -2.5*np.log10(ref_cat['f_F140W'][self.indices_ref_cat]) + 34.1
            J_ref_cat = -2.5*np.log10(ref_cat['f_F125W'][self.indices_ref_cat]) + 34.1
            I_ref_cat = -2.5*np.log10(ref_cat['f_F814W'][self.indices_ref_cat]) + 34.1
            V_ref_cat = -2.5*np.log10(ref_cat['f_F606W'][self.indices_ref_cat]) + 34.1
            B_ref_cat = -2.5*np.log10(ref_cat['f_F435W'][self.indices_ref_cat]) + 34.1

            F444W_GS = -2.5 * np.log10(self.cat['FLUX_AUTO_F444W'][self.indices_cat] / u.nJy) + 34.1
            F356W_GS = -2.5 * np.log10(self.cat['FLUX_AUTO_F356W'][self.indices_cat] / u.nJy) + 34.1
            F277W_GS = -2.5 * np.log10(self.cat['FLUX_AUTO_F277W'][self.indices_cat] / u.nJy) + 34.1
            F200W_GS = -2.5 * np.log10(self.cat['FLUX_AUTO_F200W'][self.indices_cat] / u.nJy) + 34.1
            F150W_GS = -2.5 * np.log10(self.cat['FLUX_AUTO_F150W'][self.indices_cat] / u.nJy) + 34.1
            F090W_GS = -2.5 * np.log10(self.cat['FLUX_AUTO_F090W'][self.indices_cat] / u.nJy) + 34.1
        else:
            F444W_GS = -2.5 * np.log10(self.cat['FLUX_AUTO_F444W'] / u.nJy) + 34.1
            F356W_GS = -2.5 * np.log10(self.cat['FLUX_AUTO_F356W'] / u.nJy) + 34.1
            F277W_GS = -2.5 * np.log10(self.cat['FLUX_AUTO_F277W'] / u.nJy) + 34.1
            F200W_GS = -2.5 * np.log10(self.cat['FLUX_AUTO_F200W'] / u.nJy) + 34.1
            F150W_GS = -2.5 * np.log10(self.cat['FLUX_AUTO_F150W'] / u.nJy) + 34.1
            F090W_GS = -2.5 * np.log10(self.cat['FLUX_AUTO_F090W'] / u.nJy) + 34.1

        fig,axs = plt.subplots(2,2,figsize=(12,12),sharex=True,sharey=True)

        ax1 = axs.flat[0]
        if ref_cat is not None:
            ax1.scatter(H_ref_cat,(J_ref_cat-H_ref_cat),c='b',marker='.',alpha=0.35,linewidths=0,s=20,label='Deepspace')
        if id_stars == True:
            stars = self.cat['STAR_FLAG']==1
            ax1.scatter(F444W_GS[~stars],(F277W_GS[~stars]-F356W_GS[~stars]),c='k',marker='.',alpha=0.35,linewidths=0,s=20,label='GS cat')
            ax1.scatter(F444W_GS[stars],(F277W_GS[stars]-F356W_GS[stars]),c='r',marker='*',alpha=0.5,linewidths=0,s=25,label='Stars')
        else:
            ax1.scatter(F444W_GS,(F277W_GS-F356W_GS),c='k',marker='.',alpha=0.35,linewidths=0,s=20,label='GS cat')
        ax1.set_xlabel("F444W", fontsize=25)
        ax1.set_xscale('linear')
        ax1.set_xlim(x_range)
        ax1.set_ylim(y_range)
        ax1.set_ylabel("F277W - F356W", fontsize=25)
        ax1.set_yscale('linear')
        ax1.tick_params(axis='both', which='both',direction='in',color='k',top=True,right=True,labelright=False,labelsize=15)
        ax1.minorticks_on()
        ax1.legend(loc='lower left',frameon=False,fontsize=15,markerscale=5)

        ax2 = axs.flat[2]
        if ref_cat is not None:
            ax2.scatter(H_ref_cat,(JH_ref_cat-H_ref_cat),c='b',marker='.',alpha=0.35,linewidths=0,s=20)
        if id_stars == True:
            ax2.scatter(F444W_GS[~stars],(F356W_GS[~stars]-F444W_GS[~stars]),c='k',marker='.',alpha=0.35,linewidths=0,s=20)
            ax2.scatter(F444W_GS[stars] ,(F356W_GS[stars]-F444W_GS[stars]),c='r',marker='*',alpha=0.55,linewidths=0,s=25)
        else:
            ax2.scatter(F444W_GS,(F356W_GS-F444W_GS),c='r',marker='.',alpha=0.35,linewidths=0,s=20)
        ax2.set_xlabel("F444W_GS", fontsize=25)
        ax2.set_xscale('linear')
        ax2.set_xlim(x_range)
        ax2.set_ylim(y_range)
        ax2.set_ylabel("F356W - F444W", fontsize=25)
        ax2.set_yscale('linear')
        ax2.tick_params(axis='both', which='both',direction='in',color='k',top=True,right=True,labelright=False,labelsize=15)
        ax2.minorticks_on()

        ax1 = axs.flat[1]
        if ref_cat is not None:
            ax1.scatter(I_ref_cat,(V_ref_cat-I_ref_cat),c='b',marker='.',alpha=0.35,linewidths=0,s=20,label=ref_cat_name)
        if id_stars == True:
            ax1.scatter(F200W_GS[~stars],(F150W_GS[~stars]-F200W_GS[~stars]),c='k',marker='.',alpha=0.35,linewidths=0,s=20)
            ax1.scatter(F200W_GS[stars] ,(F150W_GS[stars]-F200W_GS[stars]),c='r',marker='*',alpha=0.55,linewidths=0,s=25)
        else:
            ax1.scatter(F200W_GS,(F150W_GS-F200W_GS),c='r',marker='.',alpha=0.35,linewidths=0,s=20)
        ax1.set_xlabel("F200W", fontsize=25)
        ax1.set_xscale('linear')
        ax1.set_xlim(x_range)
        ax1.set_ylim(y_range)
        ax1.set_ylabel("F150W - F200W", fontsize=25)
        ax1.set_yscale('linear')
        ax1.tick_params(axis='both', which='both',direction='in',color='k',top=True,right=True,labelright=True,labelsize=15)
        ax1.minorticks_on()
        # ax1.legend(loc='upper left', fontsize=15,markerscale=5)
        ax1.yaxis.set_label_position("right")

        ax2 = axs.flat[3]
        if ref_cat is not None:
            ax2.scatter(I_ref_cat,(B_ref_cat-V_ref_cat),c='b',marker='.',alpha=0.35,linewidths=0,s=20)
        if id_stars == True:
            ax2.scatter(F200W_GS[~stars],(F090W_GS[~stars]-F150W_GS[~stars]),c='k',marker='.',alpha=0.35,linewidths=0,s=20)
            ax2.scatter(F200W_GS[stars],(F090W_GS[stars]-F150W_GS[stars]),c='r',marker='*',alpha=0.55,linewidths=0,s=25)
        else:
            ax2.scatter(F200W_GS,(F090W_GS-F150W_GS),c='r',marker='.',alpha=0.35,linewidths=0,s=20)
        ax2.set_xlabel("F200W", fontsize=25)
        ax2.set_xscale('linear')
        ax2.set_xlim(x_range)
        ax2.set_ylim(y_range)
        ax2.set_ylabel("F090W - F150W", fontsize=25)
        ax2.set_yscale('linear')
        ax2.tick_params(axis='both', which='both',direction='in',color='k',top=True,right=True,labelright=True,labelsize=15)
        ax2.minorticks_on()
        ax2.yaxis.set_label_position("right")

        plt.subplots_adjust(wspace=0, hspace=0)
        if save_fig==True:
            if ref_cat is not None:
                plt.savefig(os.path.join(output_dir,'diagnostic_plots/scatter_clr-mag_v{}_ref.png'.format(self.version)),format='png')
            else:
                plt.savefig(os.path.join(output_dir,'diagnostic_plots/scatter_clr-mag_v{}.png'.format(self.version)),format='png')
        plt.show()
        plt.close()


        # CLR-CLR scatter plot
        x_range = [-2,2]
        y_range = [-2,2]

        if ref_cat is not None:
            H_ref_cat = -2.5*np.log10(ref_cat['f_F160W'][self.indices_ref_cat]) + 34.1
            JH_ref_cat = -2.5*np.log10(ref_cat['f_F140W'][self.indices_ref_cat]) + 34.1
            J_ref_cat = -2.5*np.log10(ref_cat['f_F125W'][self.indices_ref_cat]) + 34.1
            I_ref_cat = -2.5*np.log10(ref_cat['f_F814W'][self.indices_ref_cat]) + 34.1
            V_ref_cat = -2.5*np.log10(ref_cat['f_F606W'][self.indices_ref_cat]) + 34.1
            B_ref_cat = -2.5*np.log10(ref_cat['f_F435W'][self.indices_ref_cat]) + 34.1

            F444W_GS = -2.5 * np.log10(self.cat['FLUX_AUTO_F444W'][self.indices_cat] / u.nJy) + 34.1
            F356W_GS = -2.5 * np.log10(self.cat['FLUX_AUTO_F356W'][self.indices_cat] / u.nJy) + 34.1
            F277W_GS = -2.5 * np.log10(self.cat['FLUX_AUTO_F277W'][self.indices_cat] / u.nJy) + 34.1
            F200W_GS = -2.5 * np.log10(self.cat['FLUX_AUTO_F200W'][self.indices_cat] / u.nJy) + 34.1
            F150W_GS = -2.5 * np.log10(self.cat['FLUX_AUTO_F150W'][self.indices_cat] / u.nJy) + 34.1
            F090W_GS = -2.5 * np.log10(self.cat['FLUX_AUTO_F090W'][self.indices_cat] / u.nJy) + 34.1
        else:
            F444W_GS = -2.5 * np.log10(self.cat['FLUX_AUTO_F444W'] / u.nJy) + 34.1
            F356W_GS = -2.5 * np.log10(self.cat['FLUX_AUTO_F356W'] / u.nJy) + 34.1
            F277W_GS = -2.5 * np.log10(self.cat['FLUX_AUTO_F277W'] / u.nJy) + 34.1
            F200W_GS = -2.5 * np.log10(self.cat['FLUX_AUTO_F200W'] / u.nJy) + 34.1
            F150W_GS = -2.5 * np.log10(self.cat['FLUX_AUTO_F150W'] / u.nJy) + 34.1
            F090W_GS = -2.5 * np.log10(self.cat['FLUX_AUTO_F090W'] / u.nJy) + 34.1

        fig,axs = plt.subplots(1,2,figsize=(12,6),sharex=True,sharey=False)

        ax1 = axs.flat[0]
        if ref_cat is not None:
            ax1.scatter(H_ref_cat,(J_ref_cat-H_ref_cat),c='b',marker='.',alpha=0.35,linewidths=0,s=20,label='Deepspace')
        if id_stars == True:
            stars = self.cat['STAR_FLAG']==1
            ax1.scatter(F356W_GS[~stars]-F444W_GS[~stars],(F277W_GS[~stars]-F356W_GS[~stars]),c='k',marker='.',alpha=0.35,linewidths=0,s=20,label='GS cat')
            ax1.scatter(F356W_GS[stars]-F444W_GS[stars],(F277W_GS[stars]-F356W_GS[stars]),c='r',marker='*',alpha=0.5,linewidths=0,s=25,label='Stars')
        else:
            ax1.scatter(F444W_GS-F356W_GS,F356W_GS-F277W_GS,c='k',marker='.',alpha=0.35,linewidths=0,s=20,label='GS cat')
        ax1.set_xlabel("F356W-F444W", fontsize=25)
        ax1.set_xscale('linear')
        ax1.set_xlim(x_range)
        ax1.set_ylim(y_range)
        ax1.set_ylabel("F377W - F356W", fontsize=25)
        ax1.set_yscale('linear')
        ax1.tick_params(axis='both', which='both',direction='in',color='k',top=True,right=True,labelright=False,labelsize=15)
        ax1.minorticks_on()
        ax1.legend(loc='lower left',frameon=False,fontsize=15,markerscale=5)

        ax2 = axs.flat[1]
        if ref_cat is not None:
            ax2.scatter(H_ref_cat,(JH_ref_cat-H_ref_cat),c='b',marker='.',alpha=0.35,linewidths=0,s=20)
        if id_stars == True:
            ax2.scatter(F150W_GS[~stars]-F200W_GS[~stars],F090W_GS[~stars]-F150W_GS[~stars],c='k',marker='.',alpha=0.35,linewidths=0,s=20)
            ax2.scatter(F150W_GS[stars]-F200W_GS[stars],F090W_GS[stars]-F150W_GS[stars],c='r',marker='*',alpha=0.55,linewidths=0,s=25)
        else:
            ax2.scatter(F200W_GS-F150W_GS,F150W_GS-F090W_GS,c='r',marker='.',alpha=0.35,linewidths=0,s=20)
        ax2.set_xlabel("F150W-F200W", fontsize=25)
        ax2.set_xscale('linear')
        ax2.set_xlim(x_range)
        ax2.set_ylim([-1.5,2.5])
        ax2.set_ylabel("F090W - F150W", fontsize=25)
        ax2.set_yscale('linear')
        ax2.tick_params(axis='both', which='both',direction='in',color='k',top=True,right=True,labelleft=False,labelright=True,labelsize=15)
        ax2.minorticks_on()
        ax2.yaxis.set_label_position("right")

        plt.subplots_adjust(wspace=0, hspace=0)
        if save_fig==True:
            if ref_cat is not None:
                plt.savefig(os.path.join(output_dir,'diagnostic_plots/scatter_clr-clr_v{}_ref.png'.format(self.version)),format='png')
            else:
                plt.savefig(os.path.join(output_dir,'diagnostic_plots/scatter_clr-clr_v{}.png'.format(self.version)),format='png')
        plt.show()
        plt.close()


        # SNR v MAG & ERR v MAG scatter plot
        x_range = [18,32]
        y_range = [0,100]

        fig,axs = plt.subplots(2,11,figsize=(21,9),sharex=True,sharey=False)

        for ii,filter in zip(np.arange(len(self.master_filter_dict.keys())),self.master_filter_dict.keys()):

            if ref_cat is not None:
                flux_GS = self.cat['FLUX_AUTO_{}'.format(filter)][self.indices_cat] / u.nJy
                fluxerr_GS = self.cat['FLUXERR_AUTO_{}'.format(filter)][self.indices_cat] / u.nJy
            else:
                flux_GS = self.cat['FLUX_AUTO_{}'.format(filter)] / u.nJy
                fluxerr_GS = self.cat['FLUXERR_AUTO_{}'.format(filter)] / u.nJy
            mag_GS = -2.5 * np.log10(flux_GS) + 34.1
            SNR_GS = flux_GS / fluxerr_GS
            sel = ~np.isnan(SNR_GS)
            fluxerr_GS = fluxerr_GS[sel]
            mag_GS = mag_GS[sel]
            SNR_GS = SNR_GS[sel]

            if ref_cat is not None:
                flux_ref = ref_cat['f_{}'.format(filter)][self.indices_ref_cat]
                fluxerr_ref = ref_cat['e_{}'.format(filter)][self.indices_ref_cat]
                mag_ref = -2.5 * np.log10(flux_ref) + 34.1
                SNR_ref = flux_ref / fluxerr_ref
                fluxerr_ref = fluxerr_ref[sel]
                mag_ref = mag_ref[sel]
                SNR_ref = SNR_ref[sel]


                axs[0][ii].scatter(mag_ref,fluxerr_ref,c='b',marker='.',alpha=0.35,linewidths=0,s=20,label=ref_cat_name)
                axs[1][ii].scatter(mag_ref,SNR_ref,c='b',marker='.',alpha=0.35,linewidths=0,s=20,label=ref_cat_name)

            axs[0][ii].scatter(mag_GS,fluxerr_GS,c='r',marker='.',alpha=0.35,linewidths=0,s=20,label='GS cat')
            axs[1][ii].scatter(mag_GS,SNR_GS,c='r',marker='.',alpha=0.35,linewidths=0,s=20,label='GS cat')
            axs[1][ii].plot(x_range,[5,5],':k', linewidth=.5)
            axs[1][ii].plot(x_range,[10,10],':k', linewidth=.5)
            axs[1][ii].plot(x_range,[20,20],':k', linewidth=.5)
            axs[1][ii].plot(x_range,[50,50],':k', linewidth=.5)
            axs[1][ii].set_xlabel("MAG_AUTO", fontsize=20)
            axs[0][ii].text(19.,0.05,filter, fontsize=20)
            axs[1][ii].text(19.,85,filter, fontsize=20)
            axs[0][ii].set_xlim(x_range)
            axs[1][ii].set_xlim(x_range)
            axs[0][ii].set_ylim([1e-2,1e-1])
            axs[1][ii].set_ylim([0,100])
            axs[0][ii].tick_params(axis='both', which='both',direction='in',color='k',top=True,right=True,labelleft=False,labelright=False,labelsize=15)
            axs[1][ii].tick_params(axis='both', which='both',direction='in',color='k',top=True,right=True,labelleft=False,labelright=False,labelsize=15)
            if (ii==0):
                axs[0][ii].set_ylabel("FLUXERR_AUTO", fontsize=20)
                axs[1][ii].set_ylabel("SNR AUTO", fontsize=20)
                axs[0][ii].tick_params(axis='both', which='both',direction='in',color='k',top=True,right=True,labelleft=True,labelright=False,labelsize=15)
                axs[1][ii].tick_params(axis='both', which='both',direction='in',color='k',top=True,right=True,labelleft=True,labelright=False,labelsize=15)
            # axs[1][ii].minorticks_on()
            if (ii==0):
                axs[0][ii].legend(loc='upper left',frameon=False,fontsize=15,markerscale=5)
            if ii==len(self.master_filter_dict.keys()):
                axs[0][ii].set_ylabel("FLUXERR_AUTO", fontsize=20)
                axs[0][ii].tick_params(axis='both', which='both',direction='in',color='k',top=True,right=True,labelright=True,labelbottom=True,labelsize=15)
                axs[0][ii].yaxis.set_label_position("right")
                axs[1][ii].set_ylabel("SNR - {}".format(aper), fontsize=20)
                axs[1][ii].tick_params(axis='both', which='both',direction='in',color='k',top=True,right=True,labelright=True,labelbottom=True,labelsize=15)
                axs[1][ii].yaxis.set_label_position("right")
        plt.subplots_adjust(wspace=0, hspace=0)
        if save_fig==True:
            if ref_cat is not None:
                plt.savefig(os.path.join(output_dir,'diagnostic_plots/scatter_SNR_ERR_v_MAG_v{}_ref.png'.format(self.version)),format='png')
            else:
                plt.savefig(os.path.join(output_dir,'diagnostic_plots/scatter_SNR_ERR_v_MAG_v{}.png'.format(self.version)),format='png')
        plt.show()
        plt.close()


        # ERR_GS v ERR_ref hist
        x_range = [0,0.5]
        y_range = x_range

        fig,axs = plt.subplots(3,4,figsize=(16,8),sharex=True,sharey=True)

        for ii,filter in enumerate(self.master_filter_dict.keys()):

            if ref_cat is not None:
                fluxerr_GS = self.cat['FLUXERR_AUTO_{}'.format(filter)][self.indices_cat]
            else:
                fluxerr_GS = self.cat['FLUXERR_AUTO_{}'.format(filter)].data

            if ref_cat is not None:
                fluxerr_ref = ref_cat['e_{}'.format(filter)][self.indices_ref_cat]
                axs.flat[ii].hist(fluxerr_ref,bins=100,range=x_range,color='b',histtype='step',linewidth=1.5,label=ref_cat_name)

            axs.flat[ii].hist(fluxerr_GS,bins=100,range=x_range,color='r',histtype='step',linewidth=1.5,label='GS cat')
            if ii in range(7,11):
                axs.flat[ii].set_xlabel("FLUXERR_AUTO", fontsize=20)
            axs.flat[ii].text(0.01,350,filter, fontsize=20)
            axs.flat[ii].set_xscale('linear')
            axs.flat[ii].set_xlim(x_range)
            axs.flat[ii].set_ylim(0,1000)
            if (ii==0) or (ii==3) or (ii==4) or (ii==8):
                axs.flat[ii].set_ylabel("Count", fontsize=20)
            axs.flat[ii].set_yscale('linear')
            axs.flat[ii].tick_params(axis='both', which='both',direction='in',color='k',top=True,right=True,labelright=False,labelsize=15)
            if ii==7:
                axs.flat[ii].set_ylabel("Count", fontsize=20)
                axs.flat[ii].yaxis.set_label_position("right")
                axs.flat[ii].tick_params(axis='both', which='both',direction='in',color='k',top=True,right=True,labelright=True,labelbottom=True,labelsize=15)
            if ii==3:
                axs.flat[ii].tick_params(axis='both', which='both',direction='in',color='k',top=True,right=True,labelright=True,labelbottom=True,labelsize=15)
                axs.flat[ii].yaxis.set_label_position("right")
        axs.flat[0].legend(loc='upper right',frameon=False,fontsize=15)
        axs.flat[-1].axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        if save_fig==True:
            if ref_cat is not None:
                plt.savefig(os.path.join(output_dir,'diagnostic_plots/hist_ERR_v{}_ref.png'.format(self.version)),format='png')
            else:
                plt.savefig(os.path.join(output_dir,'diagnostic_plots/hist_ERR_v{}.png'.format(self.version)),format='png')
        plt.show()
        plt.close()


        # SNR_GS v SNR_ref hist
        fig,axs = plt.subplots(3,4,figsize=(16,8),sharex=True,sharey=True)

        for ii,filter in enumerate(self.master_filter_dict.keys()):
            if ref_cat is not None:
                flux_GS = np.asarray(self.cat['FLUX_AUTO_{}'.format(filter)][self.indices_cat])
                fluxerr_GS = np.asarray(self.cat['FLUXERR_AUTO_{}'.format(filter)][self.indices_cat])
            else:
                flux_GS = np.asarray(self.cat['FLUX_AUTO_{}'.format(filter)])
                fluxerr_GS = np.asarray(self.cat['FLUXERR_AUTO_{}'.format(filter)])

            SNR_GS = flux_GS / fluxerr_GS
            sel = ~np.isnan(SNR_GS) & np.isfinite(SNR_GS)
            SNR_GS = SNR_GS[sel]

            if ref_cat is not None:
                flux_ref = ref_cat['f_{}'.format(filter)][self.indices_ref_cat]
                fluxerr_ref = ref_cat['e_{}'.format(filter)][self.indices_ref_cat]
                SNR_ref = flux_ref / fluxerr_ref
                SNR_ref = SNR_ref[sel]
                axs.flat[ii].hist(SNR_ref,bins=100,range=[0,100],color='b',histtype='step',linewidth=1.5,label=ref_cat_name)

            axs.flat[ii].hist(SNR_GS,bins=100,range=[0,50],color='r',histtype='step',linewidth=1.5,label='GS cat')
            axs.flat[ii].plot([5,5],[0,900],':k',linewidth=1.5,label='SNR=5')
            if ii in range(7,11):
                axs.flat[ii].set_xlabel("SNR AUTO", fontsize=20)
            axs.flat[ii].text(10,160,filter, fontsize=20)
            axs.flat[ii].set_xscale('linear')
            axs.flat[ii].set_xlim([0,50])
            axs.flat[ii].set_ylim(0,900)
            if (ii==0) or (ii==3) or (ii==4) or (ii==8):
                axs.flat[ii].set_ylabel("Count", fontsize=20)
            axs.flat[ii].set_yscale('linear')
            axs.flat[ii].tick_params(axis='both', which='both',direction='in',color='k',top=True,right=True,labelright=False,labelsize=15)
            if ii==7:
                axs.flat[ii].set_ylabel("Count", fontsize=20)
                axs.flat[ii].yaxis.set_label_position("right")
                axs.flat[ii].tick_params(axis='both', which='both',direction='in',color='k',top=True,right=True,labelright=True,labelbottom=True,labelsize=15)
            if ii==3:
                axs.flat[ii].tick_params(axis='both', which='both',direction='in',color='k',top=True,right=True,labelright=True,labelbottom=True,labelsize=15)
                axs.flat[ii].yaxis.set_label_position("right")
        axs.flat[0].legend(loc='upper right',frameon=False,fontsize=15)
        axs.flat[-1].axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        if save_fig==True:
            if ref_cat is not None:
                plt.savefig(os.path.join(output_dir,'diagnostic_plots/hist_SNR_v{}_ref.png'.format(self.version)),format='png')
            else:
                plt.savefig(os.path.join(output_dir,'diagnostic_plots/hist_SNR_v{}.png'.format(self.version)),format='png')
        plt.show()
        plt.close()
        print('\n\nFunction "make_diagnostic_plots()" took {:.3f} seconds to run.'.format(time.time()-start_time))
        return






    """The following functions add flag columns to the catalog"""

    def add_star_flag(self,method='flux_radius',fn_star_coords=None,max_flux_radius=3.,
                  min_flux_radius=1.,max_mag=24.5,save_fig=False,fn_save=None,diag=True):
        """
        Add a column which identifies stars in the catalog.

        Parameters
        ----------
        method: str
            method by which to identify stars; must be either 'ext' or 'flux_radius'
        fn_star_coords: str
            filename to stars list identified during PSF matching; ust be provided if method=='ext'
        ref_band: str
            filter in which stars were identified
        max_flux_radius: float
            maximum FLUX_RADIUS for a source to be considered a star
        max_mag: float
            maximum ref_band magnitude for a source to be considered a star
        save_fig: bool
            whether or not to save the figure; if True fn_save must be provided
        fn_save: str
            filename to save output; must be provided if save_fig==True
        """
        start_time = time.time()
        try:
            self.cat.add_column(False,name='STAR_FLAG',index=-1,dtype='bool')
        except:
            self.cat['STAR_FLAG'] = False

        if method == 'ext':     #i.e. read in an external file of star coordinates
            ref_cat = Table.read(fn_star_coords)

            self.match_sources_by_RA_DEC(ref_cat,ref_cat_ids=['RA','DEC'],matching_tol=0.5,diag=diag)
            self.cat['STAR_FLAG'][self.indices_cat] = True

        elif method == 'flux_radius':       # identify stars based on FLUX_RADIUS v magnitude
            flux_radius = self.cat['FLUX_RADIUS']
            mag = -2.5 * np.log10(self.cat['FLUX_AUTO_{}'.format(self.ref_band)] / u.nJy) + 31.4

            star_sel = (min_flux_radius<flux_radius) & (flux_radius<max_flux_radius) & (mag<max_mag)
            self.cat['STAR_FLAG'][star_sel] = 1

            #VISUALIZE
            fig, ax = plt.subplots(1,1,figsize=(8,8))
            fig.suptitle('Star Identification in {}\nMethod: FLUX_RADIUS'.format(self.ref_band),fontsize=20)
            fig.patch.set_facecolor('white')
            
            ax.scatter(mag[~star_sel],flux_radius[~star_sel],color='k',marker='.',s=20,alpha=0.5)
            ax.scatter(mag[star_sel],flux_radius[star_sel],color='r',marker='*',s=20,alpha=0.5)
            ax.plot([16,max_mag],[max_flux_radius,max_flux_radius],':k',linewidth=1.)
            ax.plot([16,max_mag],[min_flux_radius,min_flux_radius],':k',linewidth=1.)
            ax.plot([max_mag,max_mag],[min_flux_radius,max_flux_radius],':k',linewidth=1.)
            ax.set_xlabel('MAG_AUTO_{}'.format(self.ref_band),fontsize=15)
            ax.set_ylabel('FLUX_RADIUS [pix]',fontsize=15)
            ax.set_xlim([16,34])
            ax.set_yscale('log')
            ax.set_ylim([5e-2,120])
            ax.tick_params(axis='both',which='both',direction='in',length=7,right=True,labelsize='large')
            ax.set_yticks([0.1,1,3,10,30,100])
            ax.set_yticklabels([0.1,1,3,10,30,100])
            if save_fig==True:
                plt.savefig(fn_save,format='png')
            plt.show()
            plt.close()

        nstars = np.sum(self.cat['STAR_FLAG']==1)
        if diag==True:
            print('N_stars identified in filter {}: {}\nMethod of identification: {}'.format(self.ref_band,nstars,method))
        print('\n\nFunction "add_star_flag()" took {:.3f} seconds to run.'.format(time.time()-start_time))
        return


    def add_flux_radius(self,flux_frac=[0.5],diag=True):
        """
        Compute and add the half-light radius (or some other fraction); equivalent
        to Source Extractor's "FLUX_RADIUS" parameter.

        Parameters
        ----------
        mask: array
            Mask which was used for photometry
        flux_frac: float or array
            fraction of light for which to compute the radius
        ref_band: str
            filter in which to make the measurement
        """
        start_time = time.time()

        data = fits.getdata(self.master_filter_dict[self.ref_band]['fn_im_matchf444w_sub'])
        data = data.byteswap(inplace=True).newbyteorder()
        
        mask = data==0.

        seg_id = np.arange(1,len(self.cat['X'])+1,1,dtype='int32')
        r, flag = sep.flux_radius(data,self.cat['X'],self.cat['Y'],6.*self.cat['A'],flux_frac,
                                  normflux=self.cat['FLUX_AUTO_{}'.format(self.ref_band)],mask=mask,
                                  seg_id=seg_id, segmap=self.segm,subpix=1)

        print(r)
        index = self.cat.colnames.index('AREA')
        if isinstance(flux_frac, (float,int))==1:
            try:
                self.cat.add_column(r,name='FLUX_RADIUS',index=index+1)
            except:
                self.cat['FLUX_RADIUS'] = r
        elif len(flux_frac)==1:
            try:
                self.cat.add_column(r,name='FLUX_RADIUS',index=index+1)
            except:
                self.cat['FLUX_RADIUS'] = r
        else:
            if diag==True:
                print(r)
            for ii,frac in enumerate(flux_frac):
                try:
                    self.cat.add_column(r[ii][:],name='FLUX_RADIUS_{:02.0f}'.format(frac*100),index=index+1)
                except:
                    self.cat['FLUX_RADIUS_{:02.0f}'.format(frac*100)] = r[ii][:]

        if diag==True:
            nan_mask = np.isnan(self.cat['FLUX_RADIUS'])
            circ_kron_radius = 2.5*self.cat['KRON_RADIUS'] * np.sqrt(self.cat['A']*self.cat['B'])
            print('Median FLUX_RADIUS: {:.3f}\nMin/Max FLUX_RADIUS: {:.3f}/{:.3f}\n# of NaN FLUX_RADIUS: {}\n\nMedian KRON_RADIUS: {:.3f}'.format(np.median(self.cat['FLUX_RADIUS'][~nan_mask]),np.min(self.cat['FLUX_RADIUS'][~nan_mask]),np.max(self.cat['FLUX_RADIUS'][~nan_mask]),np.sum(nan_mask),np.median(circ_kron_radius)))
            plt.hist(r,bins='auto')
            plt.show()
        print('Function "add_flux_radius()" took {:.3} seconds to run'.format(time.time()-start_time))
        return



    def add_n_bands_flag(self,aper='AUTO',diag=True):
        """
        Add a catalog flag indicating the number of filters in which each source
        has at least a 1-sigma detection.
        Parameters:
        -----------
        aper: str
            which aperture to evaluate SNR; either "AUTO" or "FIXED"

        """
        start_time= time.time()
        try:
            self.cat.add_column(0*len(salf.cat),name='N_bands',index=-1)
        except:
            self.cat['N_BANDS'] = 0
        # assign flag value for each source
        for source in range(len(self.cat)):
            nbands = 0
            for filter in self.master_filter_dict.keys():
                if aper=='AUTO':
                    SNR = self.cat['FLUX_AUTO_{}'.format(filter)][source] / self.cat['FLUXERR_AUTO_{}'.format(filter)][source]
                elif aper=='FIXED':
                    SNR = self.cat['FLUX_APER{:02.0f}_{}'.format(self.color_aper*10,filter)][source] / self.cat['FLUXERR_APER{:02.0f}_{}'.format(self.color_aper*10,filter)][source]
                if SNR > 1.:
                    nbands+=1
            self.cat['N_BANDS'][source] = nbands
        # print some useful diagnostics
        if diag==True:
            for nbad in [2,4]:
                nbands = np.sum(self.cat['N_BANDS'] <= nbad)
                print('Fraction of catalog with "N_bands" <= {}: {:.2f}%'.format(nbad,nbands/len(self.cat)*100))
            for ngood in [6,8,10]:
                nbands = np.sum(self.cat['N_BANDS'] >= ngood)
                print('Fraction of catalog with "N_bands" >= {}: {:.2f}%'.format(ngood,nbands/len(self.cat)*100))

        print('\n\nFunction "add_n_bands_flag()" took {:.3f} seconds to run.'.format(time.time()-start_time))
        return


    def add_phot_flags(self,diag=True):
        """Add flag columns based on flags generated while performing aperture photometry"""
        start_time = time.time()
        # add flag to identify deblended sources
        try:
            self.cat.add_column(False,name='DEBLEND_FLAG',index=-1,dtype='bool')
        except:
            self.cat['DEBLEND_FLAG'] = False
        sel_merged = (self.cat['PHOT_FLAG'] & sep.OBJ_MERGED) != 0
        self.cat['DEBLEND_FLAG'][sel_merged] = True
        if diag==True:
            print('Number of deblended sources in catalog: {} ({:.2f}%)'.format(np.sum(self.cat['DEBLEND_FLAG']==1),(np.sum(self.cat['DEBLEND_FLAG']==1)/len(self.cat)*100)))

        self.cat.remove_column('PHOT_FLAG')

        print('\n\nFunction "add_phot_flags()" took {:.3f} seconds to run.'.format(time.time()-start_time))
        return


    def flag_sources_near_mask(self,mask,dist_tol=0.5,flag_name=None,plotfig=False,diag=False):
        """
        flag sources within a specified distance from a mask.
        Parameters
        ----------
        mask:
        dist_tol:
        flag_name
        plotfig:
        """
        start_time = time.time()
        if plotfig == True:
            plt.imshow(mask,cmap='Greys',origin='lower',interpolation='nearest')
            plt.colorbar()
            plt.show()
        # compute map of distances from mask
        distances_to_mask_arr = distance_transform_edt(input=mask, return_distances=True)
        distances_to_mask_arr*=self.pix_scale
        map_to_flag = (distances_to_mask_arr < dist_tol)
        sources_to_flag = map_to_flag[self.cat['Y'].data.astype('int32'),self.cat['X'].data.astype('int32')]
        try:
            self.cat.add_column(False,name=flag_name,index=-1,dtype='bool')
        except:
            self.cat[flag_name] = False
        # assign flag
        self.cat[flag_name][sources_to_flag] = True
        if plotfig == True:
            plt.imshow(distances_to_mask_arr,vmin=0,vmax=dist_tol*3,cmap='Greys',origin='lower',interpolation='nearest')
            plt.colorbar()
            plt.show()
        print('# of sources < {}" from mask edge: {} ({:.2f}%)\n\nFunction "flag_sources_near_mask()" took {:.3f} seconds to run'.format(dist_tol,np.sum(self.cat[flag_name]==1),(np.sum(self.cat[flag_name]==1)/len(self.cat)*100),time.time()-start_time))
        return


    def flag_sources_too_large(self,kronrad_tol=5.,plotfig=False,diag=True):
        """
        flag sources with KRON_RADIUS greater than some threshold, or NaN.
        Parameters
        ----------
        kronrad_tol: float
            maximum allowed ciricularized kron radius
        """
        start_time = time.time()
        # compute circularized kron radius. remove nans
        circ_kron_radius = 2.5*self.kronrad * np.sqrt(self.cat['A']*self.cat['B']) * self.pix_scale
        mask_nan = np.isnan(circ_kron_radius)
        if diag==True:
            print('\n# of NaNs in "kronrad": {}\nMedian value of circ. "kronrad": {:.4f}"\nMax. value of circ. "kronrad": {:.4f}"\n'.format(np.sum(mask_nan),np.median(circ_kron_radius[~mask_nan]),np.max(circ_kron_radius[~mask_nan])))
        # identify & remove sources in segmentation map
        sources_to_flag = np.where(circ_kron_radius > kronrad_tol)[0]
        try:
            self.cat.add_column(False,name='KRON_FLAG',index=-1,dtype='bool')
        except:
            self.cat['KRON_FLAG'] = False
        # assign flag
        self.cat['KRON_FLAG'][sources_to_flag] = True
        self.cat['KRON_FLAG'][mask_nan] = True
        if diag==True:
            mask_nan=np.isnan(circ_kron_radius[~sources_to_remove])
            print('\nMedian value of circ. "kronrad": {:.4f}"\nMax. value of circ. "kronrad": {:.4f}"\n'.format(np.median(circ_kron_radius[~sources_to_remove][~mask_nan]),np.max(circ_kron_radius[~sources_to_remove][~mask_nan])))
        print('# of sources w/ kronrad > {}" or nan: {} ({:.2f}%)\n\nFunction "flag_sources_too_large()" took {:.3f} seconds to run'.format(kronrad_tol,np.sum(self.cat['KRON_FLAG']==1),(np.sum(self.cat['KRON_FLAG']==1)/len(self.cat)*100,time.time()-start_time)))
        return


    def write_cat(self,fn_save=None,fmt='fits'):
        """Save the catalog to file.
        Parameters
        ----------
        fn_save: str
            filepath and filename (with extension) of the catalog to be saved,
            e.g. "/mypath/mycat.fits"
        fmt: str,  'fits' or 'ascii'
            format to save the catalog in
        """
        start_time = time.time()
        self.cat.write(fn_save, format=fmt, overwrite=True)
        print('\nCatalog saved to: {}\nFunction "write_cat()" took {:.3f} seconds to run.'.format(fn_save,time.time()-start_time))
        return


    def add_master_flag(self,flag_list=None):
        """add a flag identifying sources not flagged by other flags.
        Parameters
        ----------
        flag_list: list of str
            list where each entry is the flag name of a flag in the catalog. "
            {flag_list_name}_FLAG" must be an exact match to the name of the
            flag column in the catalog.
        """
        start_time = time.time()
        nsources = len(self.cat['X'])
        try:
            self.cat.add_column(False,name='USE_PHOT',index=-1,dtype='bool')
        except:
            self.cat['USE_PHOT'] = False
        flag_sum = np.zeros(nsources)
        for flag_name in flag_list:
            flag_sum += self.cat['{}_FLAG'.format(flag_name)]==1
        self.cat['USE_PHOT'][flag_sum==0] = True
        USE_sum = np.sum(self.cat['USE_PHOT']==1)
        print('# of sources flagged as "USE_PHOT": {} ({:.2f}%)'.format(USE_sum,USE_sum/nsources*100))
        print('\n\nFunction "add_master_flag()" took {:.3f} seconds to run.'.format(time.time()-start_time))
        return

    
    def add_hst_flag(self,HST_bands=['F435W','F606W','F814W'],diag=False):
        """Add a flag to indicate whether or not a source has coverage 
        in HST"""
        start_time = time.time()
        try:
            self.cat.add_column(True,name='HST_FLAG',index=-1,dtype='bool')
        except:
            self.cat['HST_FLAG'] = True
        if len(HST_bands)==2:
            sel = np.isnan(self.cat['FLUX_APER07_{}'.format(HST_bands[0])]) & np.isnan(self.cat['FLUX_APER07_{}'.format(HST_bands[1])])# & np.isnan(self.cat['FLUX_APER07_F435W'])
        elif len(HST_bands)==3:
            sel = np.isnan(self.cat['FLUX_APER07_{}'.format(HST_bands[0])]) & np.isnan(self.cat['FLUX_APER07_{}'.format(HST_bands[1])]) & np.isnan(self.cat['FLUX_APER07_{}'.format(ACS_bands[2])])
        self.cat['HST_FLAG'][sel] = False
        if diag==True:
            print('# sources w/ HST coverage: {} ({:.2f}%)'.format(np.sum(self.cat['HST_FLAG']==1),np.sum(self.cat['HST_FLAG']==1)/len(self.cat['HST_FLAG']) * 100))
        print('\nFunction "add_hst_flag()" took {:.3} seconds to run'.format(time.time()-start_time))
        return
        
    
    def add_jwst_flag(self,JWST_bands=['F150W','F210M','F410M','F444W'],diag=False):
        """Add a flag to indicate whether or not a source has coverage 
        in JWST"""
        start_time = time.time()
        try:
            self.cat.add_column(True,name='JWST_FLAG',index=-1,dtype='bool')
        except:
            self.cat['JWST_FLAG'] = True
        sel = np.isnan(self.cat['FLUX_APER07_{}'.format(JWST_bands[0])]) & np.isnan(self.cat['FLUX_APER07_{}'.format(JWST_bands[1])]) & np.isnan(self.cat['FLUX_APER07_{}'.format(JWST_bands[2])]) & np.isnan(self.cat['FLUX_APER07_{}'.format(JWST_bands[3])])
        self.cat['JWST_FLAG'][sel] = False
        if diag==True:
            print('# sources w/ JWST coverage: {}({:.2f}%)'.format(np.sum(self.cat['JWST_FLAG']==1),np.sum(self.cat['JWST_FLAG']==1)/len(self.cat['JWST_FLAG']) * 100))
        print('\nFunction "add_jwst_flag()" took {:.3} seconds to run'.format(time.time()-start_time))
        return


    def add_hst_only_flag(self,diag=False):
        """Add a flag to indicate whether or not a source ONLY has coverage 
        in HST"""
        start_time = time.time()
        try:
            self.cat.add_column(True,name='HST_ONLY',index=-1,dtype='bool')
        except:
            self.cat['HST_ONLY'] = False
        sel = (self.cat['HST_FLAG']==True) & (self.cat['JWST_FLAG']==False)
        self.cat['HST_ONLY'][sel] = True
        if diag==True:
            print('# sources w/ ONLY HST coverage: {}({:.2f}%)'.format(np.sum(self.cat['HST_ONLY']==1),np.sum(self.cat['HST_ONLY']==1)/len(self.cat['HST_ONLY']) * 100))
        print('\nFunction "add_hst_only_flag()" took {:.3} seconds to run'.format(time.time()-start_time))
        return


    
    def compute_radial_dist_BCG(self,BCG_coords,output_dir='./',diag=True):
        """Compute the radial distance from each source to the cluster BCG; in arcmin
        Parameters
        ----------
        BCG_coords: array
            [RA, DEC] of cluster BCG; in decimal degrees
        """
        start_time = time.time()
        BCG_coords = BCG_coords * np.pi / 180 * u.rad  # convert to rad
        RA_BCG, DEC_BCG = BCG_coords

        RA_cat = self.cat['RA']/u.deg * np.pi / 180 * u.rad   # convert to rad
        DEC_cat = self.cat['DEC']/u.deg * np.pi / 180 * u.rad   # convert to rad
        dist_astropy = astropy.coordinates.angular_separation(RA_cat,DEC_cat,RA_BCG,DEC_BCG) / u.rad
        dist_astropy = dist_astropy * 180 / np.pi * 60
        # the commented out code produce identical results as the astropy function
        # dist = np.arccos( np.sin(DEC_cat)*np.sin(DEC_BCG) + np.cos(DEC_cat)*np.cos(DEC_BCG)*np.cos(np.abs(RA_cat-RA_BCG)) ) #* 60  # *3600 to convert degrees to arcsec
        # dist = dist * 180 / np.pi / u.rad    # in deg
        # dist *= 60                   # in arcmin

        dist_col = Column(dist_astropy * u.arcmin,dtype='float32',name='DIST_BCG')

        try:
            self.cat.add_column(dist_col)
        except:
            self.cat['DIST_BCG'] = dist_col 

        if diag == True:
            mode = stats.mode(np.round(self.cat['DIST_BCG'],decimals=2))
            print(mode)
            median = np.median(self.cat['DIST_BCG'])
            fig, ax = plt.subplots(1,1,figsize=(8,8))
            ax.hist(self.cat['DIST_BCG'].data,bins='auto',color='r',histtype='step',linewidth=1.5,label='dist_BCG')
            ax.plot([mode[0],mode[0]],[0,1200],':k',linewidth=1.5,label='mode')
            ax.plot([median,median],[0,1200],'--k',linewidth=1.5,label='median')
            ax.set_ylim([0,1200])
            ax.set_ylabel('Count',fontsize=15)
            ax.set_xlabel('Distance to BCG [arcmin]',fontsize=20)
            ax.tick_params(axis='both', which='both',direction='in',color='k',top=False,right=True,labelbottom=True,labelright=True,labelsize=15)
            ax.legend(loc='upper right',fontsize=20)
            fn_save = os.path.join(output_dir,'{}{}_BCG_radial_dist_astropy.png'.format(self.cluster_name,self.field_name))
            plt.savefig(fn_save,format='png')
            plt.show()
        print('\n\nFunction "compute_radial_dist_BCG()" took {:.3} seconds to run'.format(time.time()-start_time))
        return

