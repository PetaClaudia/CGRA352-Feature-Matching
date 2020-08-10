
// std
#include <iostream>
#include <stdio.h>
#include <string>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;


// main program
// 
int main( int argc, char** argv ) {
    
    // check we have exactly one additional argument
    // eg. res/vgc-logo.png
    if( argc != 3) {
        cerr << "Usage: cgra352 <Image>" << endl;
        abort();
    }
    
    
    // read the file
    Mat input08, input11, output1, output2;
    input08 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    input11 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    
    // check for invalid input
    if(!input08.data || !input11.data) {
        cerr << "Could not open or find the image" << std::endl;
        abort();
    }
    
    //core 1
    SiftFeatureDetector detector(400);
    vector<KeyPoint> kpoints1, kpoints2;
    detector.detect(input08, kpoints1);
    detector.detect(input11, kpoints2);
    
    SiftDescriptorExtractor extractor;
    Mat des08, des11;
    extractor.compute(input08, kpoints1, des08);
    extractor.compute(input11, kpoints2, des11);
    
    BFMatcher matcher(NORM_L2, true);
    vector<DMatch> matches;
    matcher.match(des08, des11, matches);
    
    using Edge = pair<Point2f, Point2f>;
    vector<Edge> edges;
    for(DMatch m : matches){
        KeyPoint kp1 = kpoints1[m.queryIdx];
        KeyPoint kp2 = kpoints2[m.trainIdx];
        edges.push_back(Edge(kp1.pt, kp2.pt));
    }
    
    vconcat(input08, input11, output1);
    
    for(Edge e : edges){
        line(output1, Point(e.first), Point(e.second)+ Point(0, input08.rows), Scalar(0, 255, 0), 1, CV_AA);
    }
    
    //core 2
    vector<Edge> bestInliers;
    Mat bestHomography;
    srand (time(NULL));
    for(int i = 0; i<100; i++){
        cout<<i<<endl;
        
        // create the inlier list
        std::vector<Edge> inlierEdges;
        std::vector<Edge> outlierEdges;
        
        // select 4 random pairs
        int rand1 = rand() % edges.size();
        int rand2 = rand() % edges.size();
        int rand3 = rand() % edges.size();
        int rand4 = rand() % edges.size();
        while(rand2 == rand1 || rand2 == rand3 || rand2 == rand4)
        {
            rand2 = rand() % edges.size();
        }
        while(rand3 == rand1 || rand3 == rand2 || rand3 == rand4)
        {
            rand4 = rand() % edges.size();
        }
        while(rand4 == rand1 || rand4 == rand2 || rand4 == rand3)
        {
            rand4 = rand() % edges.size();
        }
        vector<Point2f> src;
        src.push_back(edges.at(rand1).first);
        src.push_back(edges.at(rand2).first);
        src.push_back(edges.at(rand3).first);
        src.push_back(edges.at(rand4).first);
        vector<Point2f> dst;
        dst.push_back(edges.at(rand1).second);
        dst.push_back(edges.at(rand2).second);
        dst.push_back(edges.at(rand3).second);
        dst.push_back(edges.at(rand4).second);
        
        //compute homography for paird
        Mat h = findHomography(src, dst,0);
        cv::Mat_<float> hMat = cv::Mat::eye(3,3,CV_32FC1);
        //compute inliers and outliers
        for(Edge e : edges){
            if(!h.empty()){
                hMat = h;
                Vec3f point(e.first.x, e.first.y, 1);
                Mat p(point);
                Mat q = hMat*p;
                Vec3f qVec = q;
                Point2f pointQ = Point(qVec[0], qVec[1]);
                float error = norm((Mat)e.second, (Mat)pointQ);
                int epsilon = 10;
                if(error < epsilon){
                    inlierEdges.push_back(e);
                }
            }
        }
        cout<<"inlier edge side "<<inlierEdges.size()<<"  best inlier size "<<bestInliers.size()<<endl;
        if(inlierEdges.size() > bestInliers.size()){
            bestInliers = inlierEdges;
            bestHomography = h.clone();
        }
    }
    
    std::vector<Edge> finalInliers;
    std::vector<Edge> finalOutliers;
    
    for(Edge e : edges){
        Mat_<float> homography = Mat::eye(3,3,CV_32FC1);
        homography = bestHomography;
        Vec3f point(e.first.x, e.first.y, 1);
        Mat p(point);
        Mat q = homography*p;
        Vec3f qVec = q;
        Point2f pointQ = Point(qVec[0], qVec[1]);
        float error = norm((Mat)e.second, (Mat)pointQ);
        int epsilon = 10;
        if(error < epsilon){
            finalInliers.push_back(e);
        }
        else{
            finalOutliers.push_back(e);
        }
    }
    cout<<"inlier size "<<finalInliers.size()<<"  outlier size "<<finalOutliers.size()<<endl;
    vconcat(input08, input11, output2);
    
    //draw inliners in green
    for(Edge in : finalInliers){
        line(output2, Point(in.first), Point(in.second)+ Point(0, input08.rows), Scalar(0, 255, 0), 1, CV_AA);
    }
    //draw outliers in red
    for(Edge out : finalOutliers){
        line(output2, Point(out.first), Point(out.second)+ Point(0, input08.rows), Scalar(0, 0, 255), 1, CV_AA);
    }
    
    //Core 3
    Mat input11Extended, input08Warped, input08Extended;
    copyMakeBorder(input11, input11Extended, 50, 50, 50, 50, BORDER_CONSTANT, Scalar(0, 255, 0));
    copyMakeBorder(input08, input08Extended, 50, 50, 50, 50, BORDER_CONSTANT, Scalar(0, 255, 0));
    cv::warpPerspective(input08Extended, input08Warped, bestHomography, Size(input11Extended.cols, input11Extended.rows), INTER_LINEAR, BORDER_CONSTANT, Scalar(0,255,0));
    
    Vec3b pixel;
    for(int i = 0; i< input11Extended.rows; i++){
        for(int j = 0; j< input11Extended.cols; j++){
            //if pixel is green
            if(input11Extended.at<cv::Vec3b>(i,j)[0] == 0
               && input11Extended.at<cv::Vec3b>(i,j)[1] == 255
               && input11Extended.at<cv::Vec3b>(i,j)[2] == 0){
                input11Extended.at<cv::Vec3b>(i,j)[0] = input08Warped.at<cv::Vec3b>(i,j)[0];
                input11Extended.at<cv::Vec3b>(i,j)[1] = input08Warped.at<cv::Vec3b>(i,j)[1];
                input11Extended.at<cv::Vec3b>(i,j)[2] = input08Warped.at<cv::Vec3b>(i,j)[2];
            }
        }
    }
    
    
    
    //Core 1
    // create a window for display and show our image inside it
    string img_display_core1 = "Core 1 Image Display";
    namedWindow(img_display_core1, WINDOW_AUTOSIZE);
    resizeWindow(img_display_core1, output1.cols/2, output1.rows/2);
    imshow(img_display_core1, output1);
    
    // save image
    imwrite("/output/core1.png", output1);
    
    //Core 2
    // create a window for display and show our image inside it
    string img_display_core2 = "Core 2 Image Display";
    namedWindow(img_display_core2, WINDOW_AUTOSIZE);
    resizeWindow(img_display_core2, output2.cols/2, output2.rows/2);
    imshow(img_display_core2, output2);
    
    // save image
    imwrite("/output/core2.png", output2);
    
    //Core 3
    // create a window for display and show our image inside it
    string img_display_core3 = "Core 3 Image Display";
    namedWindow(img_display_core3, WINDOW_AUTOSIZE);
    imshow(img_display_core3, input11Extended);
    
    // save image
    imwrite("/output/core3.png", input11Extended);
    
    
    // wait for a keystroke in the window before exiting
    waitKey(0);
    
    
}
