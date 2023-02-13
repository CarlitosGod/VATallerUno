#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc.hpp"
#include <iostream>
#include <string>

using namespace cv;
using namespace cv::ximgproc;
using namespace std;

Rect computeROI(Size2i src_sz, Ptr<StereoMatcher> matcher_instance);

const String keys =
   "{help h usage ? |                  | print this message                                                }"
   "{@left          |../data/aloeL.jpg | left view of the stereopair                                       }"
   "{@right         |../data/aloeR.jpg | right view of the stereopair                                      }"
   "{GT             |../data/aloeGT.png| optional ground-truth disparity (MPI-Sintel or Middlebury format) }"
   "{dst_path       |None              | optional path to save the resulting filtered disparity map        }"
   "{dst_raw_path   |None              | optional path to save raw disparity map before filtering          }"
   "{algorithm      |bm                | stereo matching method (bm or sgbm)                               }"
   "{filter         |wls_conf          | used post-filtering (wls_conf or wls_no_conf or fbs_conf)         }"
   "{no-display     |                  | don't display results                                             }"
   "{no-downscale   |                  | force stereo matching on full-sized views to improve quality      }"
   "{dst_conf_path  |None              | optional path to save the confidence map used in filtering        }"
   "{vis_mult       |1.0               | coefficient used to scale disparity map visualizations            }"
   "{max_disparity  |160               | parameter of stereo matching                                      }"
   "{window_size    |-1                | parameter of stereo matching                                      }"
   "{wls_lambda     |8000.0            | parameter of wls post-filtering                                   }"
   "{wls_sigma      |1.5               | parameter of wls post-filtering                                   }"
   "{fbs_spatial    |16.0              | parameter of fbs post-filtering                                   }"
   "{fbs_luma       |8.0               | parameter of fbs post-filtering                                   }"
   "{fbs_chroma     |8.0               | parameter of fbs post-filtering                                   }"
   "{fbs_lambda     |128.0             | parameter of fbs post-filtering                                   }"
   ;

int main(int argc, char** argv)
{
   CommandLineParser parser(argc,argv,keys);
   parser.about("Disparity Filtering Demo");
   if (parser.has("help"))
   {
       parser.printMessage();
       return 0;
   }

   String left_im = "../Data/l.png";
   String right_im = "../Data/r.png";
   String GT_path = parser.get<String>("GT");

   String dst_path = parser.get<String>("dst_path");
   String dst_raw_path = parser.get<String>("dst_raw_path");
   String dst_conf_path = parser.get<String>("dst_conf_path");
   String algo = parser.get<String>("algorithm");
   String filter = parser.get<String>("filter");
   bool no_display = parser.has("no-display");
   bool no_downscale = parser.has("no-downscale");
   int max_disp = parser.get<int>("max_disparity");
   double lambda = parser.get<double>("wls_lambda");
   double sigma  = parser.get<double>("wls_sigma");
   double fbs_spatial = parser.get<double>("fbs_spatial");
   double fbs_luma = parser.get<double>("fbs_luma");
   double fbs_chroma = parser.get<double>("fbs_chroma");
   double fbs_lambda = parser.get<double>("fbs_lambda");
   double vis_mult = parser.get<double>("vis_mult");

   int wsize;
   if(parser.get<int>("window_size")>=0) //user provided window_size value
       wsize = parser.get<int>("window_size");
   else
   {
       if(algo=="sgbm")
           wsize = 3; //default window size for SGBM
       else if(!no_downscale && algo=="bm" && filter=="wls_conf")
           wsize = 7; //default window size for BM on downscaled views (downscaling is performed only for wls_conf)
       else
           wsize = 15; //default window size for BM on full-sized views
   }

   if (!parser.check())
   {
       parser.printErrors();
       return -1;
   }

   //! [load_views]
   Mat left  = imread(left_im ,IMREAD_ANYCOLOR);
   if ( left.empty() )
   {
       cout<<"Cannot read image file: "<<left_im;
       return -1;
   }

   Mat right = imread(right_im,IMREAD_ANYCOLOR);
   if ( right.empty() )
   {
       cout<<"Cannot read image file: "<<right_im;
       return -1;
   }
   //! [load_views]

   bool noGT;
   Mat GT_disp;
   if (GT_path!="../Data/l.png" && left_im!="../Data/r.png")
           noGT=true;
       else
       {
           noGT=false;
           if(readGT(GT_path,GT_disp)!=0)
           {
               cout<<"Cannot read ground truth image file: "<<GT_path<<endl;
               return -1;
           }
       }


   Mat left_for_matcher, right_for_matcher;
   Mat left_disp,right_disp;
   Mat filtered_disp,solved_disp,solved_filtered_disp;
   Mat conf_map = Mat(left.rows,left.cols,CV_8U);
   conf_map = Scalar(255);
   Rect ROI;
   Ptr<DisparityWLSFilter> wls_filter;
   double matching_time, filtering_time;
   double solving_time = 0;
   if(max_disp<=0 || max_disp%16!=0)
   {
       cout<<"Incorrect max_disparity value: it should be positive and divisible by 16";
       return -1;
   }
   if(wsize<=0 || wsize%2!=1)
   {
       cout<<"Incorrect window_size value: it should be positive and odd";
       return -1;
   }

   if(filter=="wls_conf") // filtering with confidence (significantly better quality than wls_no_conf)
   {
       if(!no_downscale)
       {
           // downscale the views to speed-up the matching stage, as we will need to compute both left
           // and right disparity maps for confidence map computation
           //! [downscale]
           max_disp/=2;
           if(max_disp%16!=0)
               max_disp += 16-(max_disp%16);
           resize(left ,left_for_matcher ,Size(),0.5,0.5, INTER_LINEAR_EXACT);
           resize(right,right_for_matcher,Size(),0.5,0.5, INTER_LINEAR_EXACT);
           //! [downscale]
       }
       else
       {
           left_for_matcher  = left.clone();
           right_for_matcher = right.clone();
       }

       if(algo=="bm")
       {
           //! [matching]
           Ptr<StereoBM> left_matcher = StereoBM::create(max_disp,wsize);
           wls_filter = createDisparityWLSFilter(left_matcher);
           Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

           cvtColor(left_for_matcher,  left_for_matcher,  COLOR_BGR2GRAY);
           cvtColor(right_for_matcher, right_for_matcher, COLOR_BGR2GRAY);

           matching_time = (double)getTickCount();
           left_matcher-> compute(left_for_matcher, right_for_matcher,left_disp);
           right_matcher->compute(right_for_matcher,left_for_matcher, right_disp);
           matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();
           //! [matching]
       }
       else if(algo=="sgbm")
       {
           Ptr<StereoSGBM> left_matcher  = StereoSGBM::create(0,max_disp,wsize);
           left_matcher->setP1(24*wsize*wsize);
           left_matcher->setP2(96*wsize*wsize);
           left_matcher->setPreFilterCap(63);
           left_matcher->setMode(StereoSGBM::MODE_SGBM_3WAY);
           wls_filter = createDisparityWLSFilter(left_matcher);
           Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

           matching_time = (double)getTickCount();
           left_matcher-> compute(left_for_matcher, right_for_matcher,left_disp);
           right_matcher->compute(right_for_matcher,left_for_matcher, right_disp);
           matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();
       }
       else
       {
           cout<<"Unsupported algorithm";
           return -1;
       }

       //! [filtering]
       wls_filter->setLambda(lambda);
       wls_filter->setSigmaColor(sigma);
       filtering_time = (double)getTickCount();
       wls_filter->filter(left_disp,left,filtered_disp,right_disp);
       filtering_time = ((double)getTickCount() - filtering_time)/getTickFrequency();
       //! [filtering]
       conf_map = wls_filter->getConfidenceMap();

       // Get the ROI that was used in the last filter call:
       ROI = wls_filter->getROI();
       if(!no_downscale)
       {
           // upscale raw disparity and ROI back for a proper comparison:
           resize(left_disp,left_disp,Size(),2.1,2.1,INTER_LINEAR_EXACT);
           left_disp = left_disp*2.0;
           ROI = Rect(ROI.x*2,ROI.y*2,ROI.width*2,ROI.height*2);
       }
   }
   else if(filter=="fbs_conf") // filtering with fbs and confidence using also wls pre-processing
   {
       if(!no_downscale)
       {
           // downscale the views to speed-up the matching stage, as we will need to compute both left
           // and right disparity maps for confidence map computation
           //! [downscale_wls]
           max_disp/=2;
           if(max_disp%16!=0)
               max_disp += 16-(max_disp%16);
           resize(left ,left_for_matcher ,Size(),0.5,0.5);
           resize(right,right_for_matcher,Size(),0.5,0.5);
           //! [downscale_wls]
       }
       else
       {
           left_for_matcher  = left.clone();
           right_for_matcher = right.clone();
       }

       if(algo=="bm")
       {
           //! [matching_wls]
           Ptr<StereoBM> left_matcher = StereoBM::create(max_disp,wsize);
           wls_filter = createDisparityWLSFilter(left_matcher);
           Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

           cvtColor(left_for_matcher,  left_for_matcher,  COLOR_BGR2GRAY);
           cvtColor(right_for_matcher, right_for_matcher, COLOR_BGR2GRAY);

           matching_time = (double)getTickCount();
           left_matcher-> compute(left_for_matcher, right_for_matcher,left_disp);
           right_matcher->compute(right_for_matcher,left_for_matcher, right_disp);
           matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();
           //! [matching_wls]
       }
       else if(algo=="sgbm")
       {
           Ptr<StereoSGBM> left_matcher  = StereoSGBM::create(0,max_disp,wsize);
           left_matcher->setP1(24*wsize*wsize);
           left_matcher->setP2(96*wsize*wsize);
           left_matcher->setPreFilterCap(63);
           left_matcher->setMode(StereoSGBM::MODE_SGBM_3WAY);
           wls_filter = createDisparityWLSFilter(left_matcher);
           Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

           matching_time = (double)getTickCount();
           left_matcher-> compute(left_for_matcher, right_for_matcher,left_disp);
           right_matcher->compute(right_for_matcher,left_for_matcher, right_disp);
           matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();
       }
       else
       {
           cout<<"Unsupported algorithm";
           return -1;
       }

       //! [filtering_wls]
       wls_filter->setLambda(lambda);
       wls_filter->setSigmaColor(sigma);
       filtering_time = (double)getTickCount();
       wls_filter->filter(left_disp,left,filtered_disp,right_disp);
       filtering_time = ((double)getTickCount() - filtering_time)/getTickFrequency();
       //! [filtering_wls]

       conf_map = wls_filter->getConfidenceMap();

       Mat left_disp_resized;
       resize(left_disp,left_disp_resized,left.size());

       // Get the ROI that was used in the last filter call:
       ROI = wls_filter->getROI();
       if(!no_downscale)
       {
           // upscale raw disparity and ROI back for a proper comparison:
           resize(left_disp,left_disp,Size(),2.0,2.0);
           left_disp = left_disp*2.0;
           left_disp_resized = left_disp_resized*2.0;
           ROI = Rect(ROI.x*2,ROI.y*2,ROI.width*2,ROI.height*2);
       }

#ifdef HAVE_EIGEN
       //! [filtering_fbs]
       solving_time = (double)getTickCount();
       fastBilateralSolverFilter(left, left_disp_resized, conf_map/255.0f, solved_disp, fbs_spatial, fbs_luma, fbs_chroma, fbs_lambda);
       solving_time = ((double)getTickCount() - solving_time)/getTickFrequency();
       //! [filtering_fbs]

       //! [filtering_wls2fbs]
       fastBilateralSolverFilter(left, filtered_disp, conf_map/255.0f, solved_filtered_disp, fbs_spatial, fbs_luma, fbs_chroma, fbs_lambda);
       //! [filtering_wls2fbs]
#else
       (void)fbs_spatial;
       (void)fbs_luma;
       (void)fbs_chroma;
       (void)fbs_lambda;
#endif
   }
   else if(filter=="wls_no_conf")
   {
       /* There is no convenience function for the case of filtering with no confidence, so we
       will need to set the ROI and matcher parameters manually */

       left_for_matcher  = left.clone();
       right_for_matcher = right.clone();

       if(algo=="bm")
       {
           Ptr<StereoBM> matcher  = StereoBM::create(max_disp,wsize);
           matcher->setTextureThreshold(0);
           matcher->setUniquenessRatio(0);
           cvtColor(left_for_matcher,  left_for_matcher, COLOR_BGR2GRAY);
           cvtColor(right_for_matcher, right_for_matcher, COLOR_BGR2GRAY);
           ROI = computeROI(left_for_matcher.size(),matcher);
           wls_filter = createDisparityWLSFilterGeneric(false);
           wls_filter->setDepthDiscontinuityRadius((int)ceil(0.33*wsize));

           matching_time = (double)getTickCount();
           matcher->compute(left_for_matcher,right_for_matcher,left_disp);
           matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();
       }
       else if(algo=="sgbm")
       {
           Ptr<StereoSGBM> matcher  = StereoSGBM::create(0,max_disp,wsize);
           matcher->setUniquenessRatio(0);
           matcher->setDisp12MaxDiff(1000000);
           matcher->setSpeckleWindowSize(0);
           matcher->setP1(24*wsize*wsize);
           matcher->setP2(96*wsize*wsize);
           matcher->setMode(StereoSGBM::MODE_SGBM_3WAY);
           ROI = computeROI(left_for_matcher.size(),matcher);
           wls_filter = createDisparityWLSFilterGeneric(false);
           wls_filter->setDepthDiscontinuityRadius((int)ceil(0.5*wsize));

           matching_time = (double)getTickCount();
           matcher->compute(left_for_matcher,right_for_matcher,left_disp);
           matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();
       }
       else
       {
           cout<<"Unsupported algorithm";
           return -1;
       }

       wls_filter->setLambda(lambda);
       wls_filter->setSigmaColor(sigma);
       filtering_time = (double)getTickCount();
       wls_filter->filter(left_disp,left,filtered_disp,Mat(),ROI);
       filtering_time = ((double)getTickCount() - filtering_time)/getTickFrequency();
   }
   else
   {
       cout<<"Unsupported filter";
       return -1;
   }

   //collect and print all the stats:
   cout.precision(2);
   cout<<"Matching time:  "<<matching_time<<"s"<<endl;
   cout<<"Filtering time: "<<filtering_time<<"s"<<endl;
   cout<<"Solving time: "<<solving_time<<"s"<<endl;
   cout<<endl;

   double MSE_before,percent_bad_before,MSE_after,percent_bad_after;
   if(!noGT)
   {
       MSE_before = computeMSE(GT_disp,left_disp,ROI);
       percent_bad_before = computeBadPixelPercent(GT_disp,left_disp,ROI);
       MSE_after = computeMSE(GT_disp,filtered_disp,ROI);
       percent_bad_after = computeBadPixelPercent(GT_disp,filtered_disp,ROI);

       cout.precision(5);
       cout<<"MSE before filtering: "<<MSE_before<<endl;
       cout<<"MSE after filtering:  "<<MSE_after<<endl;
       cout<<endl;
       cout.precision(3);
       cout<<"Percent of bad pixels before filtering: "<<percent_bad_before<<endl;
       cout<<"Percent of bad pixels after filtering:  "<<percent_bad_after<<endl;
   }

   if(dst_path!="None")
   {
       Mat filtered_disp_vis;
       getDisparityVis(filtered_disp,filtered_disp_vis,vis_mult);
       imwrite(dst_path,filtered_disp_vis);
   }
   if(dst_raw_path!="None")
   {
       Mat raw_disp_vis;
       getDisparityVis(left_disp,raw_disp_vis,vis_mult);
       imwrite(dst_raw_path,raw_disp_vis);
   }
   if(dst_conf_path!="None")
   {
       imwrite(dst_conf_path,conf_map);
   }

   if(!no_display)
   {
       namedWindow("left", WINDOW_AUTOSIZE);
       imshow("left", left);
       namedWindow("right", WINDOW_AUTOSIZE);
       imshow("right", right);

       if(!noGT)
       {
           Mat GT_disp_vis;
           getDisparityVis(GT_disp,GT_disp_vis,vis_mult);
           namedWindow("ground-truth disparity", WINDOW_AUTOSIZE);
           imshow("ground-truth disparity", GT_disp_vis);
       }

       //! [visualization]
       Mat raw_disp_vis;
       getDisparityVis(left_disp,raw_disp_vis,vis_mult);
       namedWindow("raw disparity", WINDOW_AUTOSIZE);
       imshow("raw disparity", raw_disp_vis);
       Mat filtered_disp_vis;
       getDisparityVis(filtered_disp,filtered_disp_vis,vis_mult);
       namedWindow("filtered disparity", WINDOW_AUTOSIZE);
       imshow("filtered disparity", filtered_disp_vis);

       if(!solved_disp.empty())
       {
           Mat solved_disp_vis;
           getDisparityVis(solved_disp,solved_disp_vis,vis_mult);
           namedWindow("solved disparity", WINDOW_AUTOSIZE);
           imshow("solved disparity", solved_disp_vis);

           Mat solved_filtered_disp_vis;
           getDisparityVis(solved_filtered_disp,solved_filtered_disp_vis,vis_mult);
           namedWindow("solved wls disparity", WINDOW_AUTOSIZE);
           imshow("solved wls disparity", solved_filtered_disp_vis);
       }

       while(1)
       {
           char key = (char)waitKey();
           if( key == 27 || key == 'q' || key == 'Q') // 'ESC'
               break;
       }
       //! [visualization]
   }

   return 0;
}

Rect computeROI(Size2i src_sz, Ptr<StereoMatcher> matcher_instance)
{
   int min_disparity = matcher_instance->getMinDisparity();
   int num_disparities = matcher_instance->getNumDisparities();
   int block_size = matcher_instance->getBlockSize();

   int bs2 = block_size/2;
   int minD = min_disparity, maxD = min_disparity + num_disparities - 1;

   int xmin = maxD + bs2;
   int xmax = src_sz.width + minD - bs2;
   int ymin = bs2;
   int ymax = src_sz.height - bs2;

   Rect r(xmin, ymin, xmax - xmin, ymax - ymin);
   return r;
}



/*
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
using namespace cv;
using namespace std;
int main(int argc, char **argv)
{
   VideoCapture capture("../Data/vtest.avi");
   if (!capture.isOpened()){
       //error in opening the video input
       cerr << "Unable to open file!" << endl;
       return 0;
   }
   // Create some random colors
   vector<Scalar> colors;
   RNG rng;
   for(int i = 0; i < 100; i++)
   {
       int r = rng.uniform(0, 256);
       int g = rng.uniform(0, 256);
       int b = rng.uniform(0, 256);
       colors.push_back(Scalar(r,g,b));
   }
   Mat old_frame, old_gray;
   vector<Point2f> p0, p1;
   // Take first frame and find corners in it
   capture >> old_frame;
   cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
   goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);
   // Create a mask image for drawing purposes
   Mat mask = Mat::zeros(old_frame.size(), old_frame.type());
   while(true){
       Mat frame, frame_gray;
       capture >> frame;
       if (frame.empty())
           break;
       cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
       // calculate optical flow
       vector<uchar> status;
       vector<float> err;
       TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
       calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15,15), 2, criteria);
       vector<Point2f> good_new;
       for(uint i = 0; i < p0.size(); i++)
       {
           // Select good points
           if(status[i] == 1) {
               good_new.push_back(p1[i]);
               // draw the tracks
               line(mask,p1[i], p0[i], colors[i], 2);
               circle(frame, p1[i], 5, colors[i], -1);
           }
       }
       Mat img;
       add(frame, mask, img);
       String windowM = "Tracking vecindad";
       namedWindow(windowM, WINDOW_NORMAL);
       imshow(windowM, img);

       int keyboard = waitKey(30);
       if (keyboard == 'q' || keyboard == 27)
           break;
       // Now update the previous frame and previous points
       old_gray = frame_gray.clone();
       p0 = good_new;
   }
}

*/




/*

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
using namespace cv;
using namespace std;
int main()
{
   VideoCapture capture("../Data/vid1.mp4");
   if (!capture.isOpened()){
       //error in opening the video input
       cerr << "Unable to open file!" << endl;
       return 0;
   }
   Mat frame1, prvs;
   capture >> frame1;
   cvtColor(frame1, prvs, COLOR_BGR2GRAY);
   while(true){
       Mat frame2, next;
       capture >> frame2;
       if (frame2.empty())
           break;
       cvtColor(frame2, next, COLOR_BGR2GRAY);
       Mat flow(prvs.size(), CV_32FC2);
       calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
       // visualization
       Mat flow_parts[2];
       split(flow, flow_parts);
       Mat magnitude, angle, magn_norm;
       cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
       normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
       angle *= ((1.f / 360.f) * (180.f / 255.f));
       //build hsv image
       Mat _hsv[3], hsv, hsv8, bgr;
       _hsv[0] = angle;
       _hsv[1] = Mat::ones(angle.size(), CV_32F);
       _hsv[2] = magn_norm;
       merge(_hsv, 3, hsv);
       hsv.convertTo(hsv8, CV_8U, 255.0);
       cvtColor(hsv8, bgr, COLOR_HSV2BGR);
       String windowM = "Flujo optico denso";
       namedWindow(windowM, WINDOW_NORMAL);
       imshow(windowM, bgr);
       int keyboard = waitKey(30);
       if (keyboard == 'q' || keyboard == 27)
           break;
       prvs = next;
   }
}


*/


















/*

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/objdetect.hpp"


#include <iostream>

using namespace cv;
using namespace std;

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

void detectAndDisplay( Mat frame );

void drawText(Mat & image);

int main()
{


    String face_cascade_name =  "../Data/haarcascades/haarcascade_frontalface_default.xml";
    String eyes_cascade_name =  "../Data/haarcascades/haarcascade_eye.xml";


    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) )
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };
    if( !eyes_cascade.load( eyes_cascade_name ) )
    {
        cout << "--(!)Error loading eyes cascade\n";
        return -1;
    };



    cout << "Built with OpenCV " << CV_VERSION << endl;
    Mat image;
    VideoCapture capture;
    capture.open(2);
    if(capture.isOpened())
    {
        cout << "Capture is opened" << endl;
        for(;;)
        {
            capture >> image;
            if(image.empty())
                break;
            // ======================Pegar codigo de test ===============================================================
           detectAndDisplay( image );


            // =====================================================================================

           // drawText(image);
            //imshow("Sample", image);

            if(waitKey(10) >= 0)
                break;
        }
    }
    else
    {
        cout << "No capture" << endl;
        image = Mat::zeros(480, 640, CV_8UC1);
        drawText(image);
        imshow("Sample", image);
        waitKey(0);
    }
    return 0;
}

void drawText(Mat & image)
{
    putText(image, "Deteccion Rostro ",
            Point(20, 50),
            FONT_HERSHEY_COMPLEX, 1, // font face and scale
            Scalar(255, 255, 255), // white
            1, LINE_AA); // line thickness and type
}



void detectAndDisplay( Mat frame )
{
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    //-- Detect faces
    std::vector<Rect> faces;
    face_cascade.detectMultiScale( frame_gray, faces );
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4 );
        Mat faceROI = frame_gray( faces[i] );
        //-- In each face, detect eyes
        std::vector<Rect> eyes;
        eyes_cascade.detectMultiScale( faceROI, eyes );
        for ( size_t j = 0; j < eyes.size(); j++ )
        {
            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4 );
        }
    }
    //-- Show what you got
    imshow( "Capture - Face detection", frame );
}
*/
