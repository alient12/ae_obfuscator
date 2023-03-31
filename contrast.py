import cv2
import numpy as np
import time

block_size = 32
static_background_frame_number = 1
ContrastSelectorThreshold = 0.8
SmokeContrastThreshold = 0.8
buffer_len = 30

cv2_close_window_flag = False


def convert_to_vectorized_tiles(image, block_size):
    """
    A function to tile an input image and vectorize each tile and build a data
    matrix containing all the vectorized tiles.

    Inputs:
       image: the input RGB image
       block_size: the input block tuple of (tile) size, e.g. (32, 32)
    Output:
       V: the output data matrix
    """
    if len(image.shape) == 3:
        h, w, c = image.shape

        image = image[0 : h - h % block_size, 0 : w - w % block_size, :]

        tiled_arr = image.reshape(
            h // block_size,
            block_size,
            w // block_size,
            block_size,
            c,
        )
        tiled_arr = tiled_arr.swapaxes(1, 2)
        tiled_arr = np.reshape(
            tiled_arr,
            (h // block_size * w // block_size, block_size * block_size * c),
        )
    elif len(image.shape) == 2:
        h, w = image.shape
        image = image[0 : h - h % block_size, 0 : w - w % block_size]

        tiled_arr = image.reshape(
            h // block_size,
            block_size,
            w // block_size,
            block_size,
        )
        tiled_arr = tiled_arr.swapaxes(1, 2)
        tiled_arr = np.reshape(
            tiled_arr, (h // block_size * w // block_size, block_size * block_size)
        )

    return tiled_arr


def update_buffer(buffer, input_data):
    """
    This function implements a temporal buffer

    Inputs;
       buffer: initial buffer
       input_data: the input matrix or frame to be added to the initial buffer
    Output:
       new_buffer: the updated buffer after adding the input data to the end
       of the initial buffer. The size of the input initial buffer and the
       output buffer will be the same.
    """
    k, _ = buffer.shape
    new_buffer = buffer.copy()
    new_buffer[: k - 1, :] = buffer[1:, :]
    new_buffer[k - 1] = input_data
    return new_buffer


def draw_bbox(image, mask, bgr, kernel_size=5, color_layar=False):
    if color_layar:
        mask_indx = np.where(mask > 0)
        image = image.astype(np.float)
        image[:, :, 0][mask_indx] *= 1 + bgr[0] / 255
        image[:, :, 1][mask_indx] *= 1 + bgr[1] / 255
        image[:, :, 2][mask_indx] *= 1 + bgr[2] / 255
        image[image > 255] = 255
        image = image.astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel)
    dilated_mask[dilated_mask > 0] = 1
    mask[mask > 0] = 1
    bbox_mask = dilated_mask - mask
    mask_indx = np.where(bbox_mask > 0)
    image[:, :, 0][mask_indx] = bgr[0]
    image[:, :, 1][mask_indx] = bgr[1]
    image[:, :, 2][mask_indx] = bgr[2]
    return image


def get_contrast_arr(frame, block_size):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_kernel = 1 / 8 * np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    edge_mask = cv2.filter2D(gray_frame, -1, laplacian_kernel)
    edge_mask_arr = convert_to_vectorized_tiles(edge_mask, block_size)
    contrast_mean_arr = np.mean(np.abs(edge_mask_arr), axis=1).flatten()
    return contrast_mean_arr


class contrast_smoke_detector:
    def __init__(self):
        pass

    def setup(
        self,
        first_frame,
        block_size,
        static_background_frame,
        SmokeContrastThreshold=0.8,
        ContrastSelectorThreshold=0.8,
        buffer_len=30,
    ):
        self.SmokeContrastThreshold = SmokeContrastThreshold
        self.ContrastSelectorThreshold = ContrastSelectorThreshold
        self.block_size = block_size
        self.buffer_len = buffer_len
        self.buffer_cnt = 1

        h, w, c = first_frame.shape
        self.h_blocks_number = h // self.block_size
        self.w_blocks_number = w // self.block_size
        self.h, self.w = (
            self.h_blocks_number * self.block_size,
            self.w_blocks_number * self.block_size,
        )
        self.raw_contrast_buffer = np.empty(
            shape=(self.buffer_len, self.h_blocks_number * self.w_blocks_number)
        )

        self.static_background_frame = static_background_frame[
            : self.h_blocks_number * block_size,
            : self.w_blocks_number * block_size,
            :,
        ]
        self.current_frame = first_frame[
            : self.h_blocks_number * block_size,
            : self.w_blocks_number * block_size,
            :,
        ]
        self.prev_frame = self.current_frame

        contrast_arr = get_contrast_arr(self.static_background_frame, block_size)
        self.static_contrast_arr = contrast_arr
        self.selected_blocks = self.static_contrast_arr > self.ContrastSelectorThreshold

    def detect(self, frame): 
        frame = frame[
            : self.h_blocks_number * block_size,
            : self.w_blocks_number * block_size,
            :,
        ]
        self.prev_frame = np.copy(self.current_frame)
        self.current_frame = np.copy(frame)

        contrast_arr = get_contrast_arr(frame, block_size)

        fgMask = np.zeros((self.h, self.w))

        if self.buffer_cnt < self.buffer_len:
            self.raw_contrast_buffer[self.buffer_cnt, :] = contrast_arr
            self.buffer_cnt += 1
            return None
        else:
            self.raw_contrast_buffer = update_buffer(
                self.raw_contrast_buffer, contrast_arr
            )
            mean_contrast_arr = np.average(self.raw_contrast_buffer, axis=0)
            self.current_frame_contrast_arr = mean_contrast_arr
            blured_blocks_arr = np.logical_and(
                (
                    mean_contrast_arr
                    < self.SmokeContrastThreshold * self.static_contrast_arr
                ),
                (self.selected_blocks),
            )
            blured_blocks_arr_indx = np.where(blured_blocks_arr)

            contrast_mask = np.zeros_like(self.static_contrast_arr)
            contrast_mask[blured_blocks_arr_indx] = 1

            CM = np.reshape(
                contrast_mask, (self.h_blocks_number, self.w_blocks_number)
            ).astype(np.uint8)
            # fgMask = cv2.resize(CM, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
            fgMask = cv2.resize(CM, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
            fgMask[fgMask > 0] = 255

        fgMask = fgMask.astype(np.uint8)
        return fgMask

    def set_new_static_frame(self, new_static_frame):
        new_static_frame = new_static_frame[
            : self.h_blocks_number * block_size,
            : self.w_blocks_number * block_size,
            :,
        ]
        self.static_background_frame = new_static_frame
        contrast_arr = get_contrast_arr(self.static_background_frame, block_size)
        self.static_contrast_arr = contrast_arr
        self.selected_blocks = self.static_contrast_arr > self.ContrastSelectorThreshold
        self.buffer_cnt = 1

    def get_static_frame(self):
        static_danger_indx = np.where(
            self.static_contrast_arr < self.ContrastSelectorThreshold
        )

        static_danger = np.zeros_like(self.static_contrast_arr)
        static_danger[static_danger_indx] = 1

        CM = np.reshape(
            static_danger, (self.h_blocks_number, self.w_blocks_number)
        ).astype(np.uint8)
        static_danger_mask = cv2.resize(
            CM, (self.w, self.h), interpolation=cv2.INTER_NEAREST
        )
        static_danger_mask[static_danger_mask > 0] = 255
        static_danger_mask = static_danger_mask.astype(np.uint8)
        static_output = draw_bbox(
            self.static_background_frame, static_danger_mask, (0, 0, 255), 5, True
        )
        return static_output

    def get_current_frame(self):
        return self.current_frame

    def set_smoke_contrast_thresh(self, thresh):
        self.SmokeContrastThreshold = thresh

    def get_confidence(self):
        confidence_arr = (
            (self.static_contrast_arr - self.current_frame_contrast_arr)
            / (self.static_contrast_arr + 0.00000001)
            * 255
        )
        confidence_arr[self.current_frame_contrast_arr >= self.static_contrast_arr] = 0
        confidence_arr[self.selected_blocks == 0] = 0
        confidence_arr[confidence_arr < 0] = 0
        confidence_mask = np.reshape(
            confidence_arr, (self.h_blocks_number, self.w_blocks_number)
        ).astype(np.uint8)

        return confidence_mask

    def get_size(self):
        return (self.w, self.h)

    def get_motion_mask(self):
        prev_frame = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        # prev_frame = cv2.cvtColor(self.static_background_frame, cv2.COLOR_BGR2GRAY)
        current_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        frameDelta = cv2.absdiff(prev_frame, current_frame)
        thresh = cv2.threshold(frameDelta, 10, 255, cv2.THRESH_BINARY)[1]
        cv2.namedWindow("motion", cv2.WINDOW_NORMAL)
        cv2.imshow("motion", thresh)
        return thresh