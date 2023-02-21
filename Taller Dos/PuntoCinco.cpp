#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

//http://opencv-tutorials-hub.blogspot.com/2015/10/splitting-colour-images-into-RGB-channels-split-merge-opencv-code-channels.pushback-src-mv.html

using namespace std;
using namespace cv;

int main()
{
    Mat image=imread("../Data/puntos.jpg",1);
    namedWindow("Original Image",1);
    imshow("Original Image",image);

    // Se separa cada uno de los diferentes canales de la imagen
    vector<Mat> rgbChannels(3);
    split(image, rgbChannels);

    // Se crea una matriz con las dimensiones de la imagen para luego almacenar sus canales rgb
    Mat g, fin_img;
    g = Mat::zeros(Size(image.cols, image.rows), CV_8UC1);

    // Canal rojo de la imagen
    {
    vector<Mat> channels;
    channels.push_back(g);
    channels.push_back(g);
    channels.push_back(rgbChannels[2]);

    /// Merge the three channels
    merge(channels, fin_img);
    namedWindow("Red",1);
 imshow("Red", fin_img);
    }

    // Canal de color verde de la imagen
    {
    vector<Mat> channels;
    channels.push_back(g);
    channels.push_back(rgbChannels[1]);
    channels.push_back(g);
    merge(channels, fin_img);
    namedWindow("Green",1);
 imshow("Green", fin_img);
    }

    // Canal de color azul de la imagen
    {
    vector<Mat> channels;
    channels.push_back(rgbChannels[0]);
    channels.push_back(g);
    channels.push_back(g);
    merge(channels, fin_img);
    namedWindow("Blue",1);
    imshow("Blue", fin_img);
    }
    waitKey(0);
    return 0;

}

