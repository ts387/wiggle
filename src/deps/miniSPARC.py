import mrcfile
import numpy as np
import time
import torch
from Qt import QtCore
from scipy.ndimage import zoom
from skimage.transform import pyramid_gaussian, pyramid_reduce
import itertools
import warnings
from .device_utils import get_available_device, safe_to_device

# self.thread5 = QtCore.QThread()
# self.miniSPARC.moveToThread(self.thread5)
# self.thread5.start()


class miniSPARC(QtCore.QObject):
    running = QtCore.Signal()
    done = QtCore.Signal()
    def __init__(self, ui, consensus, components, apix):
        super().__init__()
        self.ui = ui
        self.consensus_original = consensus
        self.components_original = components
        self.apix = apix
        self.apix_curr = apix
        self._c = len(components)
        self.max = consensus.shape[0]
        self.consensus = consensus
        self.components = components
        self.component_vector = components.reshape(self._c, consensus.shape[0] ** 3)

        # Detect best available device (CUDA > MPS > CPU)
        self.device, self.device_type, self.device_name = get_available_device(verbose=True)

    def crop_to_box(self):
        crop = self.ui.spinBox_2B.value()
        if crop >= self.max:
            return -1
        else:
            return int(crop / 2) if (crop % 2 == 0) else int((crop + 1) / 2)

    def _update(self):
        # print('Slot doing stuff in:', QtCore.QThread.currentThread())
        self.running.emit()

        if self.ui.spinBox_9.value() > 16:
            scale = self.ui.spinBox_9.value() / self.ui.spinBox_2B.value()
            self.apix_curr = self.apix / scale
        else:
            scale = 1
            self.apix_curr = self.apix


        C = Crop(self.consensus_original, self.components_original, self.crop_to_box(), scale)
        C.cropper()

        low_pass = self.ui.doubleSpinBox_9.value()
        self.thread_FFT = QtCore.QThread()
        F = FFT(C.consensus, C.components, self.apix_curr, low_pass, self.device)
        F.moveToThread(self.thread_FFT)
        self.thread_FFT.started.connect(F.filter)
        F.finished.connect(self.thread_FFT.quit)
        F.finished.connect(F.deleteLater)
        self.thread_FFT.finished.connect(self.thread_FFT.deleteLater)
        self.thread_FFT.start()

        def _refresh():
            self.consensus = F.consensus
            self.components = F.components
            self.component_vector = F.component_vector
            self.done.emit()

        F.finished.connect(_refresh)

    def generate(self, zdim, flip):
        box = self.consensus.shape[0]
        _apply = np.dot(zdim.reshape(1, self._c), self.component_vector).reshape(box, box, box)
        vol = self.consensus + _apply.astype(np.float32)
        if flip:
            vol = vol[::-1]
        return vol.astype(np.float32)

class Crop:
    def __init__(self, consensus, components, box, scale):
        self.consensus_OG = consensus
        self.components_OG = components
        self._c = len(components)
        self.box = box
        self.scale = scale

    def cropper(self):
        s = time.time()
        if self.box > 16:
            self.consensus = self.crop_array(self.consensus_OG, self.box)
            self.components = np.array([self.crop_array(component, self.box) for component in self.components_OG])
            self.component_vector = self.components.reshape(self._c, self.consensus.shape[0] ** 3)
        else:
            self.consensus = self.consensus_OG
            self.components = self.components_OG
            self.component_vector = self.components.reshape(self._c, self.consensus.shape[0] ** 3)
        print(f'crop time {time.time() - s}')

        s = time.time()
        if 0 < self.scale < 1:
            self.consensus = zoom(self.consensus, self.scale, order=0)
            self.components = np.array([zoom(component, self.scale, order=0) for component in self.components])
            self.component_vector = self.components.reshape(self._c, self.consensus.shape[0] ** 3)
        print(f'rescale time {time.time() - s}')

        # s = time.time()
        # if 0 < self.scale < 1:
        #     self.consensus = pyramid_reduce(self.consensus, self.scale**-1)
        #     self.components = np.array([pyramid_reduce(component, self.scale**-1) for component in self.components])
        #     self.component_vector = self.components.reshape(self._c, self.consensus.shape[0] ** 3)
        # print(f'rescale time {time.time() - s}')

        # s = time.time()
        # if 0 < self.scale < 1:
        #     self.consensus = next(itertools.islice(pyramid_gaussian(self.consensus, self.scale), 1, None)) # Ew... this is shocking, but still fast.
        #     self.components = np.array([next(itertools.islice(pyramid_gaussian(component, self.scale), 1, None)) for component in self.components])
        #     self.component_vector = self.components.reshape(self._c, self.consensus.shape[0] ** 3)
        # print(f'rescale time {time.time() - s}')

    def crop_array(self, map, crop):
        if map.shape[0] != map.shape[1] != map.shape[2]:
            print("Map is not cubic... taking the first dimension. This may fail.")
        if map.shape[0] % 2 == 0:
            m = int((map.shape[0] / 2) - 1)
            l = m + 1 - crop
            h = m + crop
            return map[l:h, l:h, l:h]
        else:
            m = int((map.shape[0] / 2) - 1)
            l = m - crop
            h = m + crop + 1
            return map[l:h, l:h, l:h]

class FFT(QtCore.QObject):
    finished = QtCore.Signal()
    def __init__(self, consensus, components, apix, filter_resolution, device=None):
        super().__init__()
        self.consensus_unfil = consensus
        self.components_unfil = components
        self.apix = apix
        self._c = len(components)
        self.filter_resolution = filter_resolution
        # Use provided device or default to CPU
        self.device = device if device is not None else torch.device('cpu')
        # Cache for numpy mask (used in CPU fallback)
        self._mask_numpy_cache = None

    def filter(self):
        if self.filter_resolution > 1:
            self.box = self.consensus_unfil.shape[0]
            nyquist_fraction = 2 * self.resolution_to_radius(self.apix, self.filter_resolution, self.box)
            mask = torch.from_numpy(self.generate_sphere2((self.box, self.box, self.box), nyquist_fraction)).to(self.device)
            self.consensus = self.fourier_filter_gpu(self.consensus_unfil, mask)
            self.components = np.array([self.fourier_filter_gpu(self.components_unfil[i], mask) for i in range(0, self._c)])
            self.component_vector = self.components.reshape(self._c, self.box ** 3)
        else:
            self.consensus = self.consensus_unfil
            self.components = self.components_unfil
            self.box = self.consensus.shape[0]
            self.component_vector = self.components.reshape(self._c, self.box ** 3)
        self.finished.emit()

    def fourier_filter(self, map, mask):
        """CPU-based Fourier filtering using NumPy."""
        fft_3d = np.fft.fftn(map)
        shift = np.fft.fftshift(fft_3d)
        return np.real(np.fft.ifftn(np.fft.ifftshift(np.multiply(shift, mask))))

    def fourier_filter_gpu(self, map, mask):
        """GPU-accelerated Fourier filtering using PyTorch.

        Supports both CUDA and Metal (MPS) backends with automatic CPU fallback
        if MPS operations are not supported.
        """
        try:
            # Convert NumPy array to PyTorch tensor and move to device
            map_tensor = safe_to_device(torch.from_numpy(map), self.device, "3D FFT input")

            # Perform 3D FFT
            fft_3d = torch.fft.fftn(map_tensor)

            # Shift zero frequency to center
            shift = torch.fft.fftshift(fft_3d)

            # Apply mask
            filtered = torch.multiply(shift, mask)

            # Inverse shift and inverse FFT
            filtered = torch.fft.ifftshift(filtered)
            result = torch.fft.ifftn(filtered)

            # Take real part and convert back to NumPy
            return result.real.cpu().numpy()

        except (RuntimeError, NotImplementedError) as e:
            # MPS FFT operations may not be fully supported in all PyTorch versions
            if self.device.type == 'mps':
                warnings.warn(
                    f"MPS FFT operation failed: {e}\n"
                    f"Falling back to CPU-based Fourier filtering. "
                    f"For better performance, upgrade to PyTorch 2.0+",
                    RuntimeWarning
                )
                # Fall back to CPU-based NumPy implementation with cached mask
                return self.fourier_filter(map, self._get_mask_numpy(mask))
            else:
                # For other devices, re-raise the exception
                raise

        except Exception as e:
            # Unexpected error - fall back to CPU implementation
            warnings.warn(
                f"GPU Fourier filtering failed unexpectedly: {e}\n"
                f"Falling back to CPU implementation.",
                RuntimeWarning
            )
            return self.fourier_filter(map, self._get_mask_numpy(mask))

    def _get_mask_numpy(self, mask):
        """Get numpy version of mask, using cache if available."""
        if self._mask_numpy_cache is None:
            if torch.is_tensor(mask):
                self._mask_numpy_cache = mask.cpu().numpy()
            else:
                self._mask_numpy_cache = mask
        return self._mask_numpy_cache

    # def generate_sphere(self, volumeSize, radius):
    #     x_ = np.linspace(0, volumeSize, volumeSize)
    #     y_ = np.linspace(0, volumeSize, volumeSize)
    #     z_ = np.linspace(0, volumeSize, volumeSize)
    #     center = int(volumeSize / 2)  # center can be changed here
    #     u, v, w = np.meshgrid(x_, y_, z_, indexing='ij')
    #     a = np.power(u - center, 2) + np.power(v - center, 2) + np.power(w - center, 2)
    #     b = np.where(a <= radius * radius, 1, 0)
    #     return b

    def generate_sphere2(self, volumeSize, diameter):
        """Generate an n-dimensional spherical mask."""
        position = (0.5*volumeSize[0], 0.5*volumeSize[1], 0.5*volumeSize[2])
        assert len(position) == len(volumeSize)
        n = len(volumeSize)
        position = np.array(position).reshape((-1,) + (1,) * n)
        arr = np.linalg.norm(np.indices(volumeSize) - position, axis=0)
        return (arr <= diameter).astype(int)

    def resolution_to_radius(self, apix, resolution, box_size):
        pwr = (np.log(resolution / apix) / np.log(2)) + 1
        return (2**-pwr * box_size)