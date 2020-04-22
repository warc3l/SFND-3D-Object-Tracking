
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // FP3. Associate Keypoint Correspondences with Bounding Boxes

    // Recommend to compute all Euclidean distance between keypoint matches
    std::vector<double> euclideanDistances = {}; 
    for (cv::DMatch kptMatch: kptMatches)  
    {
        // From all the matches, only consider the Euclidean distances that are inside the bounding Box
        if (boundingBox.roi.contains(kptsCurr[kptMatch.trainIdx].pt)) {
            euclideanDistances.push_back(cv::norm(kptsPrev[kptMatch.queryIdx].pt - kptsCurr[kptMatch.trainIdx].pt));
        }
    }

    // instead of using mean and std deviation, seems that TTC Time is better described using median instead
    std::sort(euclideanDistances.begin(), euclideanDistances.end());
    double medianEucledian = (euclideanDistances.size() % 2) != 0? euclideanDistances[euclideanDistances.size() / 2] : (euclideanDistances[euclideanDistances.size()/2] + euclideanDistances[euclideanDistances.size()/2 - 1])/2.0; 

    // Leave the follow lines commented as a reference to calcule the mean

    // double sumSquares = 0;
    // double meanEuclidean = std::accumulate(euclideanDistances.begin(), euclideanDistances.end(), 0.0)/static_cast<double>(euclideanDistances.size());
    // std::for_each(euclideanDistances.begin(), euclideanDistances.end(), [&](double x) { sumSquares += (x - meanEuclidean) * (x-meanEuclidean); });
    // double stdEuclidean = std::sqrt(sumSquares / static_cast<double>(euclideanDistances.size()));
    // std::cout << "(Mean, Std): (" << meanEuclidean << ", " << stdEuclidean << ") -> [" <<  meanEuclidean - 0.15*stdEuclidean << ", " << meanEuclidean + 0.15*stdEuclidean << "]" << std::endl;

    for (cv::DMatch kptMatch: kptMatches) 
    {
        if (boundingBox.roi.contains(kptsCurr[kptMatch.trainIdx].pt)) {
            double distance = cv::norm(kptsCurr[kptMatch.trainIdx].pt - kptsPrev[kptMatch.queryIdx].pt);
            // Remove those that are far away from the mean
            if (distance >= medianEucledian - 1.5 && distance <= medianEucledian + 1.5) {
                boundingBox.keypoints.push_back(kptsCurr[kptMatch.trainIdx]);
                boundingBox.kptMatches.push_back(kptMatch);
            }
        }   

    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // FP4. Compute Camera-based TTC
    std::vector<double> distRatios;
    for (auto kptMatch: kptMatches)
    {
        cv::KeyPoint kpOutCurr = kptsCurr[kptMatch.trainIdx]; // Get current keypoint
        cv::KeyPoint kpOutPrev = kptsPrev[kptMatch.queryIdx]; // Get its matched partner

        for (auto itMatch2 = kptMatches.begin() + 1; itMatch2 != kptMatches.end(); itMatch2++ )
        {
            double minDist = 100.0; // = 100.0 per default

            cv::KeyPoint kpInnerCurr = kptsCurr[itMatch2->trainIdx]; // Get next keypoint
            cv::KeyPoint kpInnerPrev = kptsPrev[itMatch2->queryIdx]; // Get next matched partner

            double distCurr = cv::norm(kpOutCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOutPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist) {
                distRatios.push_back(static_cast<double>(distCurr / distPrev));
            }
        }
    }

    if (!distRatios.empty()) {
        // Take the median value to make it statistically robust.
        std::sort(distRatios.begin(), distRatios.end());
        double medianDistRatio = (distRatios.size() % 2) != 0? distRatios[distRatios.size() / 2] : (distRatios[distRatios.size()/2] + distRatios[distRatios.size()/2 - 1])/2.0; 
        double dT = 1.0 / frameRate;

        // Calcule the TTC
        TTC = -dT / (1.0-medianDistRatio);
    }
    else {
        // Calcule the TTC
        TTC = NAN;
    }
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // FP2. Compute Lidar-based TTC
    double dT = 1.0 / frameRate;
    double laneWidth = 4.0;
    double minXPrev = 1e9, minXCurr = 1e9;
    double sumPrev = 0, sumCurr = 0;

    // Calculation of the mean and the standard deviation to extract possible outliers
    std::for_each(lidarPointsPrev.begin(), lidarPointsPrev.end(), [&](LidarPoint ldp) { sumPrev += ldp.x; } );
    std::for_each(lidarPointsCurr.begin(), lidarPointsCurr.end(), [&](LidarPoint ldp) { sumCurr += ldp.x; } );

    double ldrMeanPrev = sumPrev / static_cast<double>(lidarPointsPrev.size());
    double ldrMeanCurr = sumCurr / static_cast<double>(lidarPointsCurr.size());

    sumPrev = sumCurr = 0;
    std::for_each(lidarPointsPrev.begin(), lidarPointsPrev.end(), [&](LidarPoint ldp) { sumPrev += (ldp.x - ldrMeanPrev) * (ldp.x - ldrMeanPrev); } );
    std::for_each(lidarPointsCurr.begin(), lidarPointsCurr.end(), [&](LidarPoint ldp) { sumCurr += (ldp.x - ldrMeanCurr) * (ldp.x - ldrMeanCurr); } );

    double stdPrev = std::sqrt(sumPrev / static_cast<double>(lidarPointsPrev.size()));
    double stdCurr = std::sqrt(sumCurr / static_cast<double>(lidarPointsCurr.size()));

    // Find the min without the outliers
    for (LidarPoint lidarPrev: lidarPointsPrev) 
    {
        if (lidarPrev.y < laneWidth/2.0  && ( lidarPrev.x >=  ldrMeanPrev - stdPrev && lidarPrev.x <= ldrMeanPrev + stdPrev ) ) {
            minXPrev = minXPrev > lidarPrev.x ? lidarPrev.x : minXPrev;
        }
    }

    for (LidarPoint lidarCurr: lidarPointsCurr) 
    {
        if (lidarCurr.y < laneWidth/2.0 &&  ( lidarCurr.x >=  ldrMeanCurr - stdCurr && lidarCurr.x <= ldrMeanCurr + stdCurr ) ) {
            minXCurr = minXCurr > lidarCurr.x? lidarCurr.x : minXCurr;
        }
    }

    // Calcule TTC Lidar
    TTC = minXCurr * dT / (minXPrev - minXCurr);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // FP1. Match 3D objects
    std::multimap<int, int> multiMap;
    std::map<std::pair<int,int>, int> countMap;
    std::pair<int, int> bestMatch;
    
    for (BoundingBox currBox: currFrame.boundingBoxes) // Make sure to be 1:1, starting with boundingBoxes instead of matches loop...
    {
        for (cv::DMatch match: matches) 
        {
            for (BoundingBox prevBox: prevFrame.boundingBoxes) 
            {
                // Find all matches that are enclosed on both bounding box frames.
                if ( prevBox.roi.contains(prevFrame.keypoints[match.queryIdx].pt) && currBox.roi.contains(currFrame.keypoints[match.trainIdx].pt)) {
                    multiMap.insert(std::pair<int,int>(prevBox.boxID, currBox.boxID)); // Insert the potential candidates in a multimap
                }
            }
        }

        // Once completed the loop, find which shares the same pair and count them.
        std::for_each(multiMap.begin(), multiMap.end(), [&](std::pair<int,int> potentialCandidate) { countMap[potentialCandidate]++; });

        // Find the highest number of keypoint correspondence. Don't include the ones that has been already included in bbBestMatches to avoid "duplication" correspondences
        int highestNumber = 0;
        std::for_each(countMap.begin(), countMap.end(), [&](std::pair<std::pair<int,int>, int> potentialBestMatch) { if (potentialBestMatch.second > highestNumber && bbBestMatches.find(potentialBestMatch.first.first) == bbBestMatches.end() ) { highestNumber = potentialBestMatch.second; bestMatch = potentialBestMatch.first; }  });

        // Assign bbBestMatches
        bbBestMatches[bestMatch.first] = bestMatch.second;

        // Clean map structures for next current frame iteration
        multiMap.clear();
        countMap.clear();
    }
}
