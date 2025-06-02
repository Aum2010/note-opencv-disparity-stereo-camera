import cv2 as cv
import numpy as np
import glob
from matplotlib import pyplot as plt

# 画像ファイルの親ディレクトリのパスを指定．
folder_path = "20231211_calibration_image"
image_num = 15 # 画像の枚数．
CHECKERBOARD = (7, 7) # チェッカーボードの頂点数を(縦, 横)で指定．

# 繰り返し計算終了の基準を設定．
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 各頂点の三次元座標格納用の配列を作成．
# チェッカーボードの三次元座標は不変なので，この配列は使い回す．
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None
# print(f"objp = \n{objp}") # 作成した配列を確認したい場合はコメントアウトを外す．
objpoints = [] # 三次元座標格納用空配列．
imgpoints = [] # 二次元画像座標格納用空配列．

for img_i in range(image_num) : # 画像の枚数だけ繰り返し．
#    if img_i in [2, 10, 13] : # 除外する画像を指定．
#        continue
    file_name = glob.glob(f"{folder_path}/*.png")[img_i] # ディレクトリ内のファイル名を一括取得し，その中の 0 番目を取得．
    img = cv.imread(file_name) # 画像の読込．
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) # 画像を BGR カラー画像から白黒画像に変換．
    # img_for_disp = cv.cvtColor(img, cv.COLOR_BGR2RGB) # 画像を BGR から RGB に変換．
    # plt.show(plt.imshow(img_for_disp)) # 読み込んだ画像を確認したい場合はコメントアウトを外す．

    # チェッカーボードの頂点検出．
    # ret には指定された数の頂点が検出された場合に True が格納される．
    # corners には検出された頂点の座標が格納される．
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD,
              cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True : # 頂点が適切に検出された場合．
        objpoints.append(objp) # objpoints に objp を追加．

        # 頂点画像座標を高精度化．
        corners2 = cv.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)

        imgpoints.append(corners2) # imgpoints に corners2 を追加．

        # 画像に高精度化された頂点を描画．
        img = cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB) # 画像を BGR から RGB に変換．
        # plt.show(plt.imshow(img)) # 読み込んだ画像の確認．
        # plt.close()
        print(f"In No.{img_i} image, corners were detected correctly.")
    else : # 頂点が適切に検出されなかった場合．
        print(f"--- In No.{img_i} image, corners were not detected. ---")
    # 繰り返し終了．

# 各パラメータの推定．
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, # 各画像・各頂点の三次元座標．
                                                  imgpoints, # 各画像・各頂点の二次元画像座標．
                                                  gray.shape[::-1],
                                                  None,
                                                  None
                                                  )

print(f"ret = \n{ret}")
print(f"mtr = \n{mtx}") # カメラ行列の出力．
print(f"dist = \n{dist}") # 歪み係数の出力．
print(f"rvecs = \n{rvecs[0]}") # 回転ベクトルの出力．
print(f"tvecs = \n{tvecs[0]}") # 並進ベクトルの出力．

# 各パラメータをドライブ上に保存．
np.savez(f"{folder_path}/camera_parameter.npz", ret, mtx, dist, rvecs, tvecs)

# 歪みを除去した画像を出力．
for img_i in range(image_num) : # 画像の枚数だけ繰り返し．
    img = cv.imread(glob.glob(f"{folder_path}/*.png")[img_i]) # 歪みを除去した画像の作成用に再読込．
    h, w, color = img.shape # 画像の縦，横の画素数を取得．
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h)) # 最適なカメラ行列を取得．
    dst = cv.undistort(img, mtx, dist, None, newcameramtx) # 画像の歪みを除去．
    plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB)) # OpenCV は色がGBR順なのでRGB順に並べ替える
    plt.show()
    plt.close()

# 再投影誤差を計算．
mean_error = 0
for i in range(len(objpoints)) : # 頂点の数について繰り返し．
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i],imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print("total error: ", mean_error/len(objpoints)) # 再投影誤差を出力．
