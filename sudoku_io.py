import os
from typing import Optional
import cv2
import numpy as np

def show_image(img: np.ndarray, contours=None, scale: Optional[float] = 1):
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if contours:
        out = cv2.drawContours(img.copy(), contours, -1,(0,255,0), 3)
    else:
        out = img.copy()
    scaled_img = cv2.resize(out, (int(scale*img.shape[1]), int(scale*img.shape[0])))
    cv2.imshow("sudoku", scaled_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_sudoku(raw_img: np.ndarray, final_fig_size: Optional[int] = 300) -> np.ndarray:
    # Taking gray image, blurring and adding some thresholding to make it easier to find contours
    gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (7,7), 0)
    ret, thresh = cv2.threshold(blurred_img,127,255,0)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_areas = [(contour, cv2.contourArea(contour)) for contour in contours] # Extracting contour area
    contours_areas.sort(key = lambda contour: contour[1], reverse=True) # sorting them by area

    # Get the outer sudoku contour
    sudoku_contour = None
    img_area = thresh.shape[0] * thresh.shape[1]
    for c, area in contours_areas:
        if area/img_area > 0.99:
            continue
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            sudoku_contour = approx
            break

    pts1 = np.float32(sudoku_contour).reshape(4,-1)
    pts1 = pts1[np.argsort(np.sum(pts1, axis=-1))] # sort them by size of the sum of x and y
    pts2 = np.float32([[0,0],[final_fig_size, 0], [0, final_fig_size], [final_fig_size, final_fig_size]])
    if pts1[1,0] < pts1[2,0]: # Correct if lowerleft and upperright corners are swapped
        pts2 = np.float32([[0,0], [0, final_fig_size], [final_fig_size, 0], [final_fig_size, final_fig_size]])

    # Transform and return
    M = cv2.getPerspectiveTransform(pts1, pts2)
    final_img = cv2.warpPerspective(raw_img, M, (final_fig_size, final_fig_size))

    return final_img

def get_sudoku_images(sudoku_path: str, img_size: Optional[int] = 28, threshold: Optional[float] = 0.3) -> np.ndarray:
    """
    Returns the 9x9 images of the sudoku grid in an array with shape(9,9,img_size,img_size)
    """
    sudoku = np.zeros((9,9,img_size,img_size))
    raw_img = cv2.imread(sudoku_path)
    img = find_sudoku(raw_img, img_size*9)
    indeces = np.array(np.linspace(0, img_size*9, 10), dtype=int)

    for i in range(9):
        for j in range(9):
            _, sudoku[i,j] = cv2.threshold(cv2.cvtColor(img[indeces[i]:indeces[i+1], indeces[j]:indeces[j+1]], cv2.COLOR_RGB2GRAY)/255, threshold, 1, cv2.THRESH_BINARY)

    return sudoku


if __name__ == "__main__":
    FINAL_FIG_SIZE = 360

    raw_img = cv2.imread(os.path.join("img", "sudoku.jpeg"))
    img = find_sudoku(raw_img, final_fig_size=FINAL_FIG_SIZE)

    # thresh = cv2.threshold(img,0,255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    indeces = np.array(np.linspace(0, FINAL_FIG_SIZE,10), dtype=int)
    show_image(raw_img, scale =0.3)
    show_image(img)
    sudoku = np.zeros((9,9), dtype=int)
    for i in range(9):
        for j in range(9):
            show_image(img[indeces[i]:indeces[i+1], indeces[j]:indeces[j+1]])