# -*- coding: utf-8 -*-
"""
Original Code Author: Sudipan Saha.

"""
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import filters
from skimage import morphology
import cv2
from sklearn.metrics import confusion_matrix, f1_score
import time
from utilities import saturateSomePercentileBandwise, scaleContrast
from featureExtractionModule_RI import deepPriorCd
import argparse


time_start = time.time()

##Dataset details: https://citius.usc.es/investigacion/datasets/hyperspectral-change-detection-dataset

###As an example, the Santa Barbara scene, taken on the years 2013 and 2014 with the AVIRIS sensor over the Santa Barbara
### region (California) whose spatial dimensions are 984 x 740 pixels and includes 224 spectral bands.

### Santa Barbara: changed pixels: 52134   (label 1 in provided reference Map)
### Santa Barbara: unchanged pixels: 80418  (label 2 in provided reference Map)
### Santa Barbara: unknown pixels: 595608 (label 0 in reference Map)
### However we convert it in "referenceImageTransformed" and assign 0 to unchanged, 1 to changed and 2 to unknown pixels

### Imp link: https://aviris.jpl.nasa.gov/links/AVIRIS_for_Dummies.pdf

##Defining Parameters
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opt = parser.parse_args()
outputLayerNumbers = [5]
nanVar = float('nan')

# # # # ###############
# preChangeDataPath = 'G:/HSICD/datasets/015-HyperspectralCD-bayArea-Hermiston-santaBarbara/santaBarbara/mat/barbara_2013.mat'
# postChangeDataPath = 'G:/HSICD/datasets/015-HyperspectralCD-bayArea-Hermiston-santaBarbara/santaBarbara/mat/barbara_2014.mat'
# referencePath = 'G:/HSICD/datasets/015-HyperspectralCD-bayArea-Hermiston-santaBarbara/santaBarbara/mat/barbara_gtChanges.mat'
# resultPath = 'G:/HSICD/Result/1/santaBarbaraDeepImagePriorNonlinear'

# preChangeDataPath = 'G:/HSICD/datasets/015-HyperspectralCD-bayArea-Hermiston-santaBarbara/bayArea/mat/Bay_Area_2013.mat'
# postChangeDataPath = 'G:/HSICD/datasets/015-HyperspectralCD-bayArea-Hermiston-santaBarbara/bayArea/mat/Bay_Area_2015.mat'
# referencePath = 'G:/HSICD/datasets/015-HyperspectralCD-bayArea-Hermiston-santaBarbara/bayArea/mat/bayArea_gtChanges2.mat.mat'
# resultPath = 'G:/HSICD/Result/1/bayAreaDeepImagePriorNonlinear'
# # # # # # # # # # #
preChangeDataPath = 'G:/HSICD/datasets/015-HyperspectralCD-bayArea-Hermiston-santaBarbara/Hermiston/hermiston2004.mat'
postChangeDataPath = 'G:/HSICD/datasets/015-HyperspectralCD-bayArea-Hermiston-santaBarbara/Hermiston/hermiston2007.mat'
referencePath = 'G:/HSICD/datasets/015-HyperspectralCD-bayArea-Hermiston-santaBarbara/Hermiston/rdChangesHermiston_5classes.mat'
resultPath = 'G:/HSICD/Result/1/HermistonDeepImagePriorNonlinear'


# Reading images and reference
# preChangeImageContents = sio.loadmat(preChangeDataPath)
# preChangeImage = preChangeImageContents['HypeRvieW']
#
# postChangeImageContents = sio.loadmat(postChangeDataPath)
# postChangeImage = postChangeImageContents['HypeRvieW']
#
# referenceContents = sio.loadmat(referencePath)
# referenceImage = referenceContents['HypeRvieW']


# #############Reading images and reference for hermiston
preChangeImage=np.zeros((390, 200, 198))
postChangeImage=np.zeros((390, 200, 198))
preChangeImageContents=sio.loadmat(preChangeDataPath)
preChangeImage_all = preChangeImageContents['HypeRvieW'] ##HypeRvieW
preChangeImage[:, :, 0:50] = preChangeImage_all[:, :, 7:57]
preChangeImage[:, :, 50:198] = preChangeImage_all[:, :, 76:224]

postChangeImageContents=sio.loadmat(postChangeDataPath)
postChangeImage_all = postChangeImageContents['HypeRvieW']           ##HypeRvieW
postChangeImage[:, :, 0:50] = postChangeImage_all[:, :, 7:57]
postChangeImage[:, :, 50:198] = postChangeImage_all[:, :, 76:224]
# # #
referenceContents=sio.loadmat(referencePath)
referenceImage = referenceContents['gt5clasesHermiston']
# # # #


##Transforming the reference image
referenceImageTransformed = np.zeros(referenceImage.shape)
# # # ### We assign 0 to unchanged, 1 to changed and 2 to unknown pixels
# referenceImageTransformed[referenceImage==2] = 0
# referenceImageTransformed[referenceImage==1] = 1
# referenceImageTransformed[referenceImage==0] = 2
##Hermiston assign 0 to unchanged, 1  2 3 4 5 to changed
referenceImageTransformed[referenceImage==0] = 0
referenceImageTransformed[referenceImage==1] = 1
referenceImageTransformed[referenceImage==2] = 1
referenceImageTransformed[referenceImage==3] = 1
referenceImageTransformed[referenceImage==4] = 1
referenceImageTransformed[referenceImage==5] = 1


del referenceImage

###Pre-process/normalize the images
percentileToSaturate = 1
preChangeImage = saturateSomePercentileBandwise(preChangeImage, percentileToSaturate)
postChangeImage = saturateSomePercentileBandwise(postChangeImage, percentileToSaturate)
##Number of spectral bands
numSpectralBands = preChangeImage.shape[2]

Seed = [10, 20, 30, 40, 50]
pre = np.zeros(len(Seed))  # Precision
rec = np.zeros(len(Seed))  # Recall
oa = np.zeros(len(Seed))   # OA
kap = np.zeros(len(Seed))  # Kappa
f1 = np.zeros(len(Seed))   # F1


def kappa(confusion_matrix):
    """kappa"""
    pe_rows = np.sum(confusion_matrix, axis=0)
    pe_cols = np.sum(confusion_matrix, axis=1)
    sum_total = sum(pe_cols)
    pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
    po = np.trace(confusion_matrix) / float(sum_total)
    return (po - pe) / (1 - pe)

for i in range(5):
    manualSeed=Seed[i]
    print('Manual seed is '+str(manualSeed))
    ##setting manual seeds

    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    np.random.seed(manualSeed)

    ## Getting normalized CD map (magnitude map)
    # detectedChangeMapNormalized, timeVector1FeatureAggregated, timeVector2FeatureAggregated, Probabilisticblock1 = deepPriorCd(preChangeImage,postChangeImage, manualSeed, outputLayerNumbers)
    detectedChangeMapNormalized, timeVector1FeatureAggregated, timeVector2FeatureAggregated = deepPriorCd(preChangeImage, postChangeImage, manualSeed, outputLayerNumbers)
    ## Saving features for visualization
    # absoluteModifiedTimeVectorDifference=np.absolute(timeVector1FeatureAggregated-timeVector2FeatureAggregated)
    # print(absoluteModifiedTimeVectorDifference.shape)
    # for featureIter in range(absoluteModifiedTimeVectorDifference.shape[2]):
    #     detectedChangeMapThisFeature=absoluteModifiedTimeVectorDifference[:,:,featureIter]
    #     detectedChangeMapNormalizedThisFeature=(detectedChangeMapThisFeature-np.amin(detectedChangeMapThisFeature))/(np.amax(detectedChangeMapThisFeature)-np.amin(detectedChangeMapThisFeature))
    #     detectedChangeMapNormalizedThisFeature=scaleContrast(detectedChangeMapNormalizedThisFeature)
    #     plt.imsave('./savedFeatures/santaBarbara'+'FeatureBest'+str(featureIter)+'.png',np.repeat(np.expand_dims(detectedChangeMapNormalizedThisFeature,2),3,2))

    ## Getting CD map from normalized CD maps

    cdMap = np.zeros(detectedChangeMapNormalized.shape, dtype=bool)
    otsuThreshold = filters.threshold_otsu(detectedChangeMapNormalized)
    cdMap = detectedChangeMapNormalized > otsuThreshold
    cdMap = morphology.binary_erosion(cdMap)
    cdMap = morphology.binary_dilation(cdMap)



    ##Computing quantitative indices
    referenceImageTo1DArray = (referenceImageTransformed).ravel()
    cdMapTo1DArray = cdMap.astype(int).ravel()
    confusionMatrixEstimated = confusion_matrix(y_true=referenceImageTo1DArray, y_pred=cdMapTo1DArray, labels=[0,1])

    #getting details of confusion matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
    trueNegative, falsePositive, falseNegative, truePositive = confusionMatrixEstimated.ravel()
    TP = truePositive
    TN = trueNegative
    FP = falsePositive
    FN = falseNegative
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    Accuracy = (TP + TN) / (TP + FP + TN + FN)
    Error_rate = (FN + FP) / (TP + FP + TN + FN)
    F1_score = 2 * Precision * Recall / (Precision + Recall)
    Kappa = kappa(confusionMatrixEstimated)
    print("Precision:", Precision)
    print("Recall:", Recall)
    print("OA:", Accuracy)
    print("F1:", F1_score)
    print("Kappa:", Kappa)


    pre[i] = Precision  # Precision
    rec[i] = Recall    # Recall
    oa[i] = Accuracy   # OA
    kap[i] = Kappa     # Kappa
    f1[i] = F1_score   # F1

    ###########################################################################

    cv2.imwrite(resultPath + str(i) + '.png', ((1-cdMap)*255).astype('uint8'))

time_end = time.time()
time_sum = time_end - time_start
print('time used is: ' + str(time_sum))
print(pre)
print('mean precision is: ' + str(np.mean(pre)))
print(rec)
print('mean recall is: ' + str(np.mean(rec)))
print(oa)
print('mean OA is: ' + str(np.mean(oa)))
print(kap)
print('kappa kappa is: ' + str(np.mean(kap)))
print(f1)
print('mean f1 score is: ' + str(np.mean(f1)))












