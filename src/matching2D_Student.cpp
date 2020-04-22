
#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // FLANN workaround for types, it needs to be converted to CV_32F to make it work! Instead, it crashes during execution!
        // https://stackoverflow.com/questions/11565255/opencv-flann-with-orb-descriptors 

        if(descSource.type() != CV_32F ) {
            descSource.convertTo(descSource, CV_32F);
        }

        if (descRef.type() != CV_32F) {
            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(descSource, descRef, knnMatches, 2);
        std::for_each(knnMatches.begin(), knnMatches.end(), [&] (std::vector<cv::DMatch> m) { if ( m[0].distance < 0.8*m[1].distance) matches.push_back(m[0]); } );
    }
}

// Detect keypoints in image using the traditional Harry detector
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double& t, bool bVis)
{
    int blockSize = 4; // changed the blockSize to be equal with Shi-Tomasi given.
    int apertureSize = 3;
    int minResponse = 100;
    double k = 0.04;

    cv::Mat dst, dst_norm;//, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);

    t = (double)cv::getTickCount();
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    // cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    for (int i = 0; i < dst_norm.rows; i++) 
    {
        for (int j = 0; j < dst_norm.cols; j++) {
            // We only consider values higher than minResponse.
            if (dst_norm.at<float>(i,j) > minResponse) {
                cv::KeyPoint newKPoint;
                newKPoint.response = static_cast<int>(dst_norm.at<float>(i, j));
                newKPoint.size = 2 * apertureSize;
                newKPoint.pt = cv::Point2f(j, i);

                bool overlapped = false;
                for (auto kp: keypoints) {
                        double ratioOverlap = cv::KeyPoint::overlap(newKPoint, kp);
                        if (ratioOverlap > 0) { // if they overlap, not consider as a keypoint
                            overlapped = true;
                            break;
                        }
                }
                if (!overlapped)
                    keypoints.push_back(newKPoint);
            }
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harry Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        int bytes = 32;
        bool useOrientation = false;

        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, useOrientation);
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        int nFeatures = 500;
        float scaleFactor = 1.2f;
        int nLevels = 8;
        int threshold = 31;
        int firstLevel = 0;
        int WTA_K = 2;
        cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
        int patchSize = 31;
        int fastThreshold = 20;

        extractor = cv::ORB::create(nFeatures, scaleFactor, nLevels, threshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        bool orientationNormalized = true;
        bool scaleNormalized = true;
        float patternSize = 22.0f;
        int octaves = 4;

        extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternSize);
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        cv::AKAZE::DescriptorType descriptorType = cv::AKAZE::DESCRIPTOR_MLDB;
        int descriptorSize = 2;
        int descriptorChannels = 3;
        float threshold = 0.01;
        int octaves = 4;
        int octaveLayers = 4;
        cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;

        extractor = cv::AKAZE::create(descriptorType, descriptorSize, descriptorChannels, threshold, octaves, octaveLayers, diffusivity);
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        int nFeatures = 0;
        int octaveLayers = 3;
        double contrastThreshold = 0.04;
        double edgeThreshold = 10;
        double sigma = 1.6;

        extractor = cv::xfeatures2d::SiftDescriptorExtractor::create(nFeatures, octaveLayers, contrastThreshold, edgeThreshold, sigma);
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double& t, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detector keypoints in image using FAST detector
void detKeypointsFAST(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double& t, bool bVis)
{
    int threshold = 30;
    bool nonmaxSupression = true;
    cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;

    t = (double)cv::getTickCount();
    cv::FastFeatureDetector::create(threshold, nonmaxSupression, type)->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();


    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "FAST Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detector keypoints in image using BRISK detector
void detKeypointsBRISK(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double& t, bool bVis)
{
    int threshold = 30;
    int octaves = 3;
    float patternScale = 1.0f;

    t = (double)cv::getTickCount();
    cv::BRISK::create(threshold, octaves, patternScale)->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "BRISK Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detector keypoints in image using ORB detector
void detKeypointsORB(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double& t, bool bVis)
{
    int nFeatures = 500;
    float scaleFactor = 1.2f;
    int nLevels = 8;
    int threshold = 31;
    int firstLevel = 0;
    int WTA_K = 2;
    cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
    int patchSize = 31;
    int fastThreshold = 20;

    t = (double)cv::getTickCount();
    cv::ORB::create(nFeatures, scaleFactor, nLevels, threshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold)->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "ORB Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Dector keypoints in image using AKAZE detector
void detKeypointsAKAZE(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double& t, bool bVis)
{
    cv::AKAZE::DescriptorType descriptorType = cv::AKAZE::DESCRIPTOR_MLDB;
    int descriptorSize = 2;
    int descriptorChannels = 3;
    float threshold = 0.01;
    int octaves = 4;
    int octaveLayers = 4;
    cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;

    t = (double)cv::getTickCount();
    cv::AKAZE::create(descriptorType, descriptorSize, descriptorChannels, threshold, octaves, octaveLayers, diffusivity)->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "AKAZE Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detector keypoints in image using SIFT detector
void detKeypointsSIFT(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double& t, bool bVis)
{
    int nFeatures = 0;
    int octaveLayers = 3;
    double contrastThreshold = 0.04;
    double edgeThreshold = 10;
    double sigma = 1.6;

    t = (double)cv::getTickCount();
    cv::xfeatures2d::SIFT::create(nFeatures, octaveLayers, contrastThreshold, edgeThreshold, sigma)->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "SIFT Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
