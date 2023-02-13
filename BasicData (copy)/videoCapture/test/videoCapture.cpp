
#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"

#include <iostream>


#include "opencv2/calib3d.hpp"


#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"


using namespace cv;
using namespace std;

using namespace cv::xfeatures2d;
using std::cout;
using std::endl;



void drawText(Mat & image);



int main()
{
   Mat img_object,img_scene;
   string NamFil_objet,NamFil_scene;

   NamFil_objet="../Data/object.jpg";
   img_object=imread(NamFil_objet,IMREAD_GRAYSCALE);



   //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
   //parametros configuracion SURF
   double minHessian = 400;
   int nOctaves = 4;
   int nOctaveLayers = 3;
   Ptr<SURF> detector_SURF = SURF::create(minHessian,nOctaves,nOctaveLayers);

   int nfeatures = 0;
   int nOctaveLayers_sift = 3;
   double contrastThreshold = 0.04;
   double edgeThreshold = 10;
   double sigma = 1.6;

   Ptr<SIFT> detector_SIFT = SIFT::create(nfeatures,nOctaveLayers_sift,contrastThreshold,edgeThreshold,sigma);

   //parametros configuracion BRIS
   int thresh=30;
   int octaves=3;
   float patternScale=1.0f;
   Ptr<BRISK> detector_brisk = BRISK::create(thresh,octaves,patternScale);
  // Ptr<ORB>  detector_ORB=ORB::create();



   std::vector<KeyPoint> keypoints_object_sift, keypoints_scene_sift;
   std::vector<KeyPoint> keypoints_object_surf, keypoints_scene_surf;
   std::vector<KeyPoint> keypoints_object_brisk, keypoints_scene_brisk;



    Mat descriptors_object_sift_sift, descriptors_scene_sift_sift;
    Mat descriptors_object_surf_surf, descriptors_scene_surf_surf;
    // combinacion  kjeypoin sift extractor caract visules_surf
    //objeto_Algoritmo_keypoints_algoritm_extractorCarVIsuaUsado
    //Scene_Algoritmo_keypoints_algoritm_extractorCarVIsuaUsado
    Mat descriptors_object_sift_surf, descriptors_scene_sift_surf;
    Mat descriptors_object_surf_sift, descriptors_scene_surf_sift;
    Mat descriptors_object_surf_brisk, descriptors_scene_surf_brisk;
    Mat descriptors_object_brisk_brisk, descriptors_scene_brisk_brisk;
    Mat descriptors_object_sift_brisk, descriptors_scene_sift_brisk;

    // calculamos los keypoints del objeto par adiferentes combinaciones
    // detector->detectAndCompute( img_object, noArray(), keypoints_object, descriptors_object );
    //detector->detectAndCompute( img_scene, noArray(), keypoints_scene, descriptors_scene );
    detector_SURF->detect(img_object,keypoints_object_surf);
    detector_SIFT->detect(img_object,keypoints_object_sift);

    detector_SURF->detect(img_object,keypoints_object_surf,descriptors_object_surf_surf);
    detector_SIFT->detect(img_object,keypoints_object_sift,descriptors_object_sift_sift);


    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
   // Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    Ptr<DescriptorMatcher> matcher_fz_bt=DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    //Ptr<DescriptorMatcher> matcher_L1=DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_L1);
    std::vector< std::vector<DMatch> > knn_matches_sift_sif;
    std::vector< std::vector<DMatch> > knn_matches_sift_surf;
    std::vector< std::vector<DMatch> > knn_matches_sift_brisk;

    std::vector< std::vector<DMatch> > knn_matches_surf_surf;
    std::vector< std::vector<DMatch> > knn_matches_surf_sift;

    std::vector< std::vector<DMatch> > knn_matches_brisk_brisk;
    std::vector< std::vector<DMatch> > knn_matches_brisk_surf;
    std::vector< std::vector<DMatch> > knn_matches_brisk_sift;




    cout << "Built with OpenCV " << CV_VERSION << endl;
    VideoCapture capture;
    Mat image;
    capture.open(2);
    if(capture.isOpened())
    {
        cout << "Capture is opened" << endl;
        for(;;)
        {
            capture >> image;
            cvtColor(image,img_scene,COLOR_BGR2GRAY);

            /*if(image.empty())
                break;
            drawText(image);
            imshow("Sample", image);*/



           detector_SURF->detect(img_scene,keypoints_scene_surf,descriptors_scene_surf_surf);
           detector_SIFT->detect(img_scene,keypoints_scene_sift,descriptors_scene_sift_sift);

   //compute( InputArrayOfArrays images,C  keypoints,OutputArrayOfArrays descriptors );

           detector_SURF->compute(img_scene,keypoints_scene_surf,descriptors_scene_sift_surf);
           detector_SIFT->compute(img_scene,keypoints_scene_sift,descriptors_scene_sift_surf);

           // knnMatch( InputArray queryDescriptors, CV_OUT std::vector<std::vector<DMatch> >& matches, int k, InputArrayOfArrays masks=noArray(), bool compactResult=false );
           matcher_fz_bt->knnMatch(descriptors_object_sift_sift, descriptors_scene_sift_sift,knn_matches_sift_sif, 2 );



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
    putText(image, "Hello OpenCV",
            Point(20, 50),
            FONT_HERSHEY_COMPLEX, 1, // font face and scale
            Scalar(255, 255, 255), // white
            1, LINE_AA); // line thickness and type
}




/*
 //face detection
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

using namespace std;
using namespace cv;

void detectAndDisplay(Mat frame );


CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

int main()
{

    String face_cascade_name = samples::findFile( "../Data/haarcascades/haarcascade_frontalface_alt.xml");
    String eyes_cascade_name = samples::findFile( "../Data/haarcascades/haarcascade_eye_tree_eyeglasses.xml" );
      //-- 1. Load the cascades
      if( !face_cascade.load( face_cascade_name ) )
      {
          cout << "--(!)Error loading face cascade\n";
          return -1;
      }
      if( !eyes_cascade.load( eyes_cascade_name ) )
      {
          cout << "--(!)Error loading eyes cascade\n";
          return -1;
      }


      Mat ima_in;
      string nombre= "../Data/samples.png";

      ima_in=imread(nombre,IMREAD_COLOR);
      String windowName = "FACES IN";
      namedWindow(windowName, WINDOW_NORMAL); // Create a window
      imshow(windowName, ima_in); // Show our image inside the created window.
      detectAndDisplay(ima_in);

      waitKey(0); // Wait for any keystroke in the window
      destroyWindow(windowName); //destroy the created window


 return 0;
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
         rectangle(frame,faces[i], Scalar( 255, 0, 255 ), 4 ,LINE_8,0);
        Mat faceROI = frame_gray( faces[i] );
        //-- In each face, detect eyes
        std::vector<Rect> eyes;
        eyes_cascade.detectMultiScale( faceROI, eyes );
        for ( size_t j = 0; j < eyes.size(); j++ )
        {
            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, eye_center, radius, Scalar( 0, 255, 0 ), 4 );
        }
    }

    String winNam = "FACES DETECTED";
    namedWindow(winNam, WINDOW_NORMAL); // Create a window
    imshow(winNam, frame);
}
*/
