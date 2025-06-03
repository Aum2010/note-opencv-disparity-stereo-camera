import cv2 as cv
import numpy as np
import glob
from matplotlib import pyplot as plt

# 画像ファイル，およびカメラパラメータファイルの親ディレクトリのパスを指定．
folder_path = "20231211_disparity_calc"
calb_folder_path = "20231211_calibration_image"

npz = np.load(f"{calb_folder_path}/camera_parameter.npz") # カメラパラメータの読込．
ret = npz["arr_0"]
mtx = npz["arr_1"] # カメラ行列．
dist = npz["arr_2"] # 歪み係数．
rvecs = npz["arr_3"] # 回転行列．
tvecs = npz["arr_4"] # 並進ベクトル．

print( mtx )

# print( mtx )

def plt_output(material, title) : # 描画用の関数を定義．
    plt.figure(figsize = (9, 6), dpi = 600)
    plt.imshow(cv.cvtColor(material, cv.COLOR_BGR2RGB)) # OpenCV は色がGBR順なのでRGB順に並べ替える
    plt.title(title)
    plt.show()
    plt.close()

def undistortion(img, mtx, dist, h, w) : # 歪み除去の関数を定義．
    h, w = img.shape # 画像の縦，横の画素数を取得．
    # 最適なカメラ行列を取得．
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, # カメラ行列と歪み係数．
                                                     (w,h),
                                                     0, # この引数を 1 にすると元画像の全画素が維持される．
                                                     (w,h))
    dst = cv.undistort(img, mtx, dist, None, newcameramtx) # 画像の歪みを除去．
    plt_output(dst, "Undistorted Image")
    return dst

img1 = cv.imread(f"{folder_path}/13.png", cv.IMREAD_GRAYSCALE) # 左画像を白黒で読込．
img2 = cv.imread(f"{folder_path}/14.png", cv.IMREAD_GRAYSCALE) # 右画像を白黒で読込．
# img1 = cv.imwrite(f"{calb_folder_path}/13.png", cv.IMREAD_GRAYSCALE) # 左画像を白黒で読込．
# img2 = cv.imwrite(f"{calb_folder_path}/14.png", cv.IMREAD_GRAYSCALE) # 右画像を白黒で読込．
h, w = img1.shape

img1 = undistortion(img1, mtx, dist, h, w)
img2 = undistortion(img2, mtx, dist, h, w)

cv.imwrite(f"00-13-undistortion.png",img1)

# 特徴量記述子とそれに対応する距離関数を定義．
# creater = cv.AKAZE_create() ; distance_func = cv.NORM_HAMMING # AKAZE を使用する場合．
creater = cv.SIFT_create() ; distance_func = cv.NORM_L1 # SIFT を使用する場合．

kp1, des1 = creater.detectAndCompute(img1, None) # 特徴点を検出．
kp2, des2 = creater.detectAndCompute(img2, None)
img1_with_kp = cv.drawKeypoints(img1, kp1, None, flags = 4) # 特徴点を描画．
img2_with_kp = cv.drawKeypoints(img2, kp2, None, flags = 4)
# plt_output(img1_with_kp, "Left Image with Key-Points")
print(f"The number of all key-points on the left image was {len(kp1)}")
# plt_output(img2_with_kp, "Right Image with Key-Points")
print(f"The number of all key-points on the right image was {len(kp2)}")

cv.imwrite(f"01-13-LeftImagewith-Key-Points.png",img1_with_kp)
cv.imwrite(f"01-14-RightImagewith-Key-Points.png",img2_with_kp)

matcher = cv.BFMatcher(distance_func, crossCheck = True) # マッチングアルゴリズムの定義．
if creater == cv.SIFT_create() : # 特徴量記述子が SIFT の場合．
    des1 = des1.astype(np.uint8) ; des2 = des2.astype(np.uint8) # 配列の型を変換．
matches = matcher.match(des1, des2) # 特徴点同士を総当りでマッチング．
matches = sorted(matches, key = lambda x:x.distance) # マッチングコスト順にソート．
src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2) # 特徴点からマッチングに成功した点を抽出
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
good_maches = matches[:round(len(matches) * 0.05)] # マッチングコストの高い上位 3 割を抽出．
img_with_matches = cv.drawMatches(img1, kp1, img2, kp2, good_maches, None, flags = 2)

# plt_output(img_with_matches, "Two Images with Matched-Points")
cv.imwrite(f"02-Two-Images-with-Matched-Points.png",img_with_matches)

# mask は extracted_matches と同じ要素数かつ要素が 0 or 1 の配列．
# 行列計算に使用された点は 1 で，使用されていない点は 0 で表現される．
H, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 1.0) # 射影変換行列の推定．
# Re_img2 = cv.warpPerspective(img2, H, (w, h)) # 右画像の位置合わせ．
Re_img2 = cv.warpPerspective(img2, H, (w, h)) # 右画像の位置合わせ．
# plt_output(Re_img2, "Rectified Right Image")
cv.imwrite(f"03-Rectified-Right-Image.png",Re_img2)

used_matches = [] # 行列算出に使用された対応点情報格納用の空配列を準備．
for i in range(mask.shape[0]) : # mask の要素について繰り返し．
    if mask[i][0] == 1 : # H の算出に使用した点である場合．
        used_matches.append(matches[i])
print(f"The number of used matched-points was {len(used_matches)}")
img_with_used_matches = cv.drawMatches(img1, kp1, img2, kp2, used_matches, None, flags = 2)
# plt_output(img_with_used_matches, "Two Images with Used Matched-Points")
cv.imwrite(f"04-Two-Images-with-Used-Matched-Points.png",img_with_used_matches)

w_img1 = cv.applyColorMap(img1, cv.COLORMAP_JET)
w_Re_img2 = cv.applyColorMap(Re_img2, cv.COLORMAP_HSV)
overlap_img = cv.addWeighted(w_img1,0.5,w_Re_img2,0.5,0) # addWeighted(画像1,重み,画像2,重み,γ値) で，画像を重ね合わせる．
# plt_output(overlap_img, "Overlapped Two Images")
cv.imwrite(f"05-Overlapped-Two-Images.png",overlap_img)

ret, thresh1 = cv.threshold(img1,    1, 255, cv.THRESH_BINARY) # img1 の画像領域の形をしたマスクを作成．
ret, thresh2 = cv.threshold(Re_img2, 1, 255, cv.THRESH_BINARY) # img2 の画像領域の形をしたマスクを作成．
thresh1 = thresh1 * 255
thresh2 = thresh2 * 255
trimmed_img1 = img1 * thresh2 # 右画像の存在範囲領域の左画像を抽出．
trimmed_img2 = Re_img2 * thresh1 # 左画像の存在範囲領域の右画像を抽出．

# SGBM 関数のパラメータを定義．
minDisparity  = -50 ; numDisparities = 100 ; blockSize       = 11 ; cost1 =   1 ; cost2        = 4
disp12MaxDiff =   0 ; preFilterCap   =   0 ; uniquenessRatio =  0 ; sWS   = 600 ; speckleRange = 2
P1 = cost1 * 3 * ( blockSize ** 2) ; P2 = cost2 * 3 * ( blockSize ** 2)
stereo = cv.StereoSGBM_create(minDisparity, numDisparities, blockSize, P1, P2,
                disp12MaxDiff, preFilterCap, uniquenessRatio, sWS, speckleRange, mode = cv.STEREO_SGBM_MODE_SGBM_3WAY)
# stereo = cv.StereoBM_create(numDisparities = 96, blockSize = 15)
### DI 算出．
DI = stereo.compute(trimmed_img1, trimmed_img2).astype(np.float32) / 16.0 # DI を算出．
DI = np.where((thresh1 == 1) & (thresh2 == 1), DI, np.nan) # 左右画像の重複領域以外の領域の視差を NaN に置換．
DI[np.where(DI <= minDisparity)] = np.nan # 無意味な最低視差値を NaN に置換．

plt.figure(figsize = (9, 6), dpi = 300)
# plt.rcParams["font.family"] = "Times New Roman"
plt.imshow(DI, cmap = plt.cm.jet, vmin = minDisparity, vmax = minDisparity + numDisparities)
plt.tick_params(labelsize = 20)
cbar = plt.colorbar(aspect = 40, pad = 0.02, shrink = 0.7, orientation = "vertical", extend = "both")
cbar.ax.tick_params(axis = "y", labelsize = 20)
cbar.set_label("Disparity", labelpad = 15, size = 30)
plt.xlabel("Width", fontsize = 30)
plt.ylabel("Height", fontsize = 30)

plt.savefig('06-Disparity.png')

plt.show()
plt.close()