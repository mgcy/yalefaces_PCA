from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from math import log10
import glob


def cal_snr(image_a, image_b):
    # calculate mean square error between two images
    var_a = np.var(image_a.astype(float))
    var_b = np.var(image_b.astype(float) - image_a.astype(float))
    snr = 10 * log10(var_a / var_b)
    return snr


'''
########################################################################################
# 1 part: train PCA using 15 images
# get data
# images are 320 * 243
num_train = 15
pose = 'glasses'

# initialize
im_data = np.zeros((1, 320 * 243))
for i in range(1, num_train + 1):
    if i <= 9:
        tmp_im_name = 'D:\\Users\\YangYifan\\PycharmProjects\\yalefaces_PCA\\yalefaces\\subject0' + str(
            i) + pose + '.gif'
    else:
        tmp_im_name = 'D:\\Users\\YangYifan\\PycharmProjects\\yalefaces_PCA\\yalefaces\\subject' + str(
            i) + pose + '.gif'

    # print(tmp_im_name)
    tmp_im = Image.open(tmp_im_name)
    tmp_im_data = np.array(tmp_im).reshape((1, 320 * 243))
    im_data = np.concatenate((im_data, tmp_im_data), axis=0)

im_data = np.delete(im_data, 0, 0)

# normailze
# im_data = normalize(im_data)

# shuffle the data
np.random.shuffle(im_data)

############################################################
# 95 PCA
pca95 = PCA(0.95)
lower_dimension_data = pca95.fit_transform(im_data)
print(lower_dimension_data.shape)

# reconstruction
approximation95 = pca95.inverse_transform(lower_dimension_data)
print(approximation95.shape)

approximation95 = approximation95.reshape(-1, 243, 320)
X_norm = im_data.reshape(-1, 243, 320)

# calculate avg snr and avg mse
total_snr = 0
for i in range(0, X_norm.shape[0]):
    total_snr = total_snr + cal_snr(X_norm[i,], approximation95[i,])
avg_snr = total_snr / num_train
print('95 SNR: ' + str(avg_snr))

######################################################################
# 97 PCA
pca97 = PCA(0.97)
lower_dimension_data = pca97.fit_transform(im_data)
print(lower_dimension_data.shape)

# reconstruction
approximation97 = pca97.inverse_transform(lower_dimension_data)
print(approximation97.shape)

approximation97 = approximation97.reshape(-1, 243, 320)
# X_norm97 = im_data.reshape(-1, 243, 320)

# calculate avg snr and avg mse
total_snr = 0
for i in range(0, X_norm.shape[0]):
    total_snr = total_snr + cal_snr(X_norm[i,], approximation97[i,])
avg_snr = total_snr / num_train
print('97 SNR: ' + str(avg_snr))
##########################################################################
# 99 PCA
pca99 = PCA(0.99)
lower_dimension_data = pca99.fit_transform(im_data)
print(lower_dimension_data.shape)

# reconstruction
approximation99 = pca99.inverse_transform(lower_dimension_data)
print(approximation99.shape)

approximation99 = approximation99.reshape(-1, 243, 320)
#X_norm99 = im_data.reshape(-1, 243, 320)

# calculate avg snr and avg mse
total_snr = 0
for i in range(0, X_norm.shape[0]):
    total_snr = total_snr + cal_snr(X_norm[i,], approximation99[i,])
avg_snr = total_snr / num_train
print('99 SNR: ' + str(avg_snr))
###########################################################################
# show some plot
fig1, axarr = plt.subplots(2, 2, figsize=(3, 3))
axarr[0, 0].imshow(X_norm[0,], cmap='gray')
axarr[0, 0].set_title('Original Image')
axarr[0, 0].axis('off')
axarr[0, 1].imshow(approximation99[0,], cmap='gray')
axarr[0, 1].set_title('\u03B7 = 0.99')
axarr[0, 1].axis('off')
axarr[1, 0].imshow(approximation97[0,], cmap='gray')
axarr[1, 0].set_title('\u03B7 = 0.97')
axarr[1, 0].axis('off')
axarr[1, 1].imshow(approximation95[0,], cmap='gray')
axarr[1, 1].set_title('\u03B7 = 0.95')
axarr[1, 1].axis('off')
plt.show()


'''
'''
###########################################################################
# do some test on other pose data base on sad97 training
# train pca97 on sad pose
num_train = 15
pose = 'sad'

# initialize
im_data = np.zeros((1, 320 * 243))
for i in range(1, num_train + 1):
    if i <= 9:
        tmp_im_name = 'D:\\Users\\YangYifan\\PycharmProjects\\yalefaces_PCA\\yalefaces\\subject0' + str(
            i) + pose + '.gif'
    else:
        tmp_im_name = 'D:\\Users\\YangYifan\\PycharmProjects\\yalefaces_PCA\\yalefaces\\subject' + str(
            i) + pose + '.gif'

    # print(tmp_im_name)
    tmp_im = Image.open(tmp_im_name)
    tmp_im_data = np.array(tmp_im).reshape((1, 320 * 243))
    im_data = np.concatenate((im_data, tmp_im_data), axis=0)

im_data = np.delete(im_data, 0, 0)

# fit 97 PCA
pca97 = PCA(0.97)
lower_dimension_data = pca97.fit_transform(im_data)
# get the data
other1 = 'D:\\Users\\YangYifan\\PycharmProjects\\yalefaces_PCA\\yalefaces\\subject03happy.gif'
other2 = 'D:\\Users\\YangYifan\\PycharmProjects\\yalefaces_PCA\\yalefaces\\subject06glasses.gif'
other3 = 'D:\\Users\\YangYifan\\PycharmProjects\\yalefaces_PCA\\yalefaces\\subject10surprised.gif'
other4 = 'D:\\Users\\YangYifan\\PycharmProjects\\yalefaces_PCA\\yalefaces\\subject11sleepy.gif'

im_test1 = np.array(Image.open(other1)).reshape((1, 320 * 243))
im_test2 = np.array(Image.open(other2)).reshape((1, 320 * 243))
im_test3 = np.array(Image.open(other3)).reshape((1, 320 * 243))
im_test4 = np.array(Image.open(other4)).reshape((1, 320 * 243))

# reconstruction
appro1 = pca97.inverse_transform(pca97.transform(im_test1)).reshape(243, 320)
appro2 = pca97.inverse_transform(pca97.transform(im_test2)).reshape(243, 320)
appro3 = pca97.inverse_transform(pca97.transform(im_test3)).reshape(243, 320)
appro4 = pca97.inverse_transform(pca97.transform(im_test4)).reshape(243, 320)

# find SNR
snr1 = cal_snr(im_test1.reshape(243, 320), appro1)
snr2 = cal_snr(im_test2.reshape(243, 320), appro2)
snr3 = cal_snr(im_test3.reshape(243, 320), appro3)
snr4 = cal_snr(im_test4.reshape(243, 320), appro4)

print(snr1, snr2, snr3, snr4)

# show some result
fig2, axarr = plt.subplots(2, 5, figsize=(8, 6))
axarr[0, 0].imshow(im_test1.reshape(243, 320), cmap='gray')
axarr[0, 0].set_title('Original Happy Image')
axarr[0, 0].axis('off')
axarr[0, 1].imshow(appro1, cmap='gray')
axarr[0, 1].set_title('Reconstruction with SNR ' + '%.2f' % snr1)
axarr[0, 1].axis('off')

axarr[0, 3].imshow(im_test2.reshape(243, 320), cmap='gray')
axarr[0, 3].set_title('Original Glasses Image')
axarr[0, 3].axis('off')
axarr[0, 4].imshow(appro2, cmap='gray')
axarr[0, 4].set_title('Reconstruction with SNR ' + '%.2f' % snr2)
axarr[0, 4].axis('off')

axarr[1, 0].imshow(im_test3.reshape(243, 320), cmap='gray')
axarr[1, 0].set_title('Original Surprised Image')
axarr[1, 0].axis('off')
axarr[1, 1].imshow(appro3, cmap='gray')
axarr[1, 1].set_title('Reconstruction with SNR ' + '%.2f' % snr3)
axarr[1, 1].axis('off')

axarr[1, 3].imshow(im_test4.reshape(243, 320), cmap='gray')
axarr[1, 3].set_title('Original Sleepy Image')
axarr[1, 3].axis('off')
axarr[1, 4].imshow(appro4, cmap='gray')
axarr[1, 4].set_title('Reconstruction with SNR ' + '%.2f' % snr4)
axarr[1, 4].axis('off')

axarr[0, 2].axis('off')
axarr[1, 2].axis('off')

plt.show()

'''
###################################################################################
#  Part 3: train PCA using all data

# load more data
file_name = 'D:\\Users\\YangYifan\\PycharmProjects\\yalefaces_PCA\\yalefaces\\*.gif'
file = glob.glob(file_name)
all_data = np.zeros((1, 243 * 320))
for i in range(len(file)):
    tmp_im = Image.open(file[i])
    tmp_im_data = np.array(tmp_im).reshape((1, 320 * 243))
    all_data = np.concatenate((all_data, tmp_im_data), axis=0)

all_data = np.delete(all_data, 0, 0)

# shuffle the data and split it into train and test data
np.random.shuffle(all_data)
num_train = int(len(all_data) / 3 * 2)
train_data = all_data[:num_train, ]
test_data = all_data[num_train:, ]
#################################################
# fit pca model
print('95: ')
pca95 = PCA(0.95)
train_pca_data = pca95.fit_transform(train_data)
print(train_pca_data.shape)
train_reconstruction95 = pca95.inverse_transform(train_pca_data).reshape(-1, 243, 320)
print(train_reconstruction95.shape)

####################################################
print('97: ')
pca97 = PCA(0.97)
train_pca_data = pca97.fit_transform(train_data)
print(train_pca_data.shape)
train_reconstruction97 = pca97.inverse_transform(train_pca_data).reshape(-1, 243, 320)
print(train_reconstruction97.shape)

###########################################
print('99: ')
pca99 = PCA(0.99)
train_pca_data = pca99.fit_transform(train_data)
print(train_pca_data.shape)
train_reconstruction99 = pca99.inverse_transform(train_pca_data).reshape(-1, 243, 320)
print(train_reconstruction99.shape)

train_data = train_data.reshape(-1, 243, 320)

# train SNR
train_snr = 0
for i in range(0, num_train):
    train_snr = train_snr + cal_snr(train_data[i,], train_reconstruction95[i,])
train_snr = train_snr / num_train
print('95 Train SNR: ' + str(train_snr))

train_snr = 0
for i in range(0, num_train):
    train_snr = train_snr + cal_snr(train_data[i,], train_reconstruction97[i,])
train_snr = train_snr / num_train
print('97 Train SNR: ' + str(train_snr))

train_snr = 0
for i in range(0, num_train):
    train_snr = train_snr + cal_snr(train_data[i,], train_reconstruction99[i,])
train_snr = train_snr / num_train
print('99 Train SNR: ' + str(train_snr))

# pick a image and show results
picked_index = 12
fig1, axarr = plt.subplots(2, 2, figsize=(3, 3))
axarr[0, 0].imshow(train_data[picked_index,], cmap='gray')
axarr[0, 0].set_title('Original Image')
axarr[0, 0].axis('off')
axarr[0, 1].imshow(train_reconstruction99[picked_index,], cmap='gray')
axarr[0, 1].set_title('\u03B7 = 0.99')
axarr[0, 1].axis('off')
axarr[1, 0].imshow(train_reconstruction97[picked_index,], cmap='gray')
axarr[1, 0].set_title('\u03B7 = 0.97')
axarr[1, 0].axis('off')
axarr[1, 1].imshow(train_reconstruction95[picked_index,], cmap='gray')
axarr[1, 1].set_title('\u03B7 = 0.95')
axarr[1, 1].axis('off')
# plt.show()

# test
test99 = pca99.inverse_transform(pca99.transform(test_data)).reshape(-1, 243, 320)
test97 = pca97.inverse_transform(pca97.transform(test_data)).reshape(-1, 243, 320)
test95 = pca95.inverse_transform(pca95.transform(test_data)).reshape(-1, 243, 320)

# snr
test_snr99 = 0
test_snr97 = 0
test_snr95 = 0
for i in range(0, len(all_data) - num_train):
    test_snr99 = test_snr99 + cal_snr(test_data[i,].reshape(243, 320), test99[i,])
    test_snr97 = test_snr97 + cal_snr(test_data[i,].reshape(243, 320), test97[i,])
    test_snr95 = test_snr95 + cal_snr(test_data[i,].reshape(243, 320), test95[i,])

test_snr99 = test_snr99 / (len(all_data) - num_train)
test_snr97 = test_snr97 / (len(all_data) - num_train)
test_snr95 = test_snr95 / (len(all_data) - num_train)

print('99,97,95 SNR: ' + str(test_snr99) + ' ' + str(test_snr97) + ' ' + str(test_snr95))

my_im = np.array(Image.open('D:\Courses\DIP\CA1\\my.png'))
my_rec = pca99.inverse_transform(pca99.transform(my_im.reshape((1, 243 * 320))))


# snr
testmy99 = pca99.inverse_transform(pca99.transform(my_im.reshape((1, 243 * 320)))).reshape(-1, 243, 320)
testmy97 = pca97.inverse_transform(pca97.transform(my_im.reshape((1, 243 * 320)))).reshape(-1, 243, 320)
testmy95 = pca95.inverse_transform(pca95.transform(my_im.reshape((1, 243 * 320)))).reshape(-1, 243, 320)
mysnr1 = cal_snr(my_im, testmy99)
mysnr2 = cal_snr(my_im, testmy97)
mysnr3 = cal_snr(my_im, testmy95)
print('99,97,95 SNR: ' + str(mysnr1) + ' ' + str(mysnr2) + ' ' + str(mysnr3))


# show some result
fig2, axarr = plt.subplots(2, 5, figsize=(6, 3))
axarr[0, 0].imshow(test_data[0,].reshape(243, 320), cmap='gray')
axarr[0, 0].set_title('Original Image')
axarr[0, 0].axis('off')
axarr[0, 1].imshow(test99[0,], cmap='gray')
axarr[0, 1].set_title('Reconstructed Image')
axarr[0, 1].axis('off')

axarr[0, 3].imshow(test_data[1,].reshape(243, 320), cmap='gray')
axarr[0, 3].set_title('Original Image')
axarr[0, 3].axis('off')
axarr[0, 4].imshow(test99[1], cmap='gray')
axarr[0, 4].set_title('Reconstructed Image')
axarr[0, 4].axis('off')

axarr[1, 0].imshow(test_data[2,].reshape(243, 320), cmap='gray')
axarr[1, 0].set_title('Original Image')
axarr[1, 0].axis('off')
axarr[1, 1].imshow(test99[2], cmap='gray')
axarr[1, 1].set_title('Reconstructed Image')
axarr[1, 1].axis('off')

axarr[1, 3].imshow(my_im, cmap='gray')
axarr[1, 3].set_title('Personal Image')
axarr[1, 3].axis('off')
axarr[1, 4].imshow(my_rec.reshape(243, 320), cmap='gray')
axarr[1, 4].set_title('Reconstructed Image')
axarr[1, 4].axis('off')

axarr[0, 2].axis('off')
axarr[1, 2].axis('off')
