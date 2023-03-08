#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdarg.h>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui/highgui_c.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <numeric>
#include <typeinfo>
#include <opencv2/core/types.hpp>

using namespace std;
using namespace cv;
RNG rng(12345);

//----------------------------------------------------------TALLER 1-----------------------------------------------------------------//

void ejUnoTallerUno(string title, int nArgs, ...) {
    //Variable de escala de las imagenes
    float scale = 1.0;
    // Calculo de alto y ancho para el display asi como su tamaño
    const int w = 3, h = 2;
    const int size = 200;
    // Imagen de 3 canales RGB
    Mat DispImage = Mat::zeros(Size(100 + size*w, 60 + size*h), CV_8UC3);
    // Usamos los parametros de la lista
    va_list args;
    va_start(args, nArgs);
    for (int i = 0; i < nArgs; i++) {
        // Obtenemos la ubicacion de la imagen y lo guardamos en un objeto de tipo Mat
        Mat img = va_arg(args, Mat);
        // verificamos si la imagen que llega no es nulla en esa posicion
        if (img.empty()) {
            printf("Invalid arguments");
            va_end(args);
            return;
        }
        // obtenemos el tamaño de la imagen
        const int max_dim = std::max(img.cols, img.rows);
        // hallamos el factor de escalado de la imagen dividiendo el valor màximo de la imagen
        // ya sea su tamaño en columnas o en filas con un size predeterminado de 200
        scale = (float)max_dim / size;
        // Alineamos las imagenes sacando el mudulo de el ancho y el alto del tamaño del display
        const int row = i / w;
        const int col = i % w;
        const int x = 20 + col * (20 + size);
        const int y = 20 + row * (20 + size);
        // Redimensionar y copiar la imagen en la imagen de visualización
        Mat temp;
        resize(img, temp, cv::Size(), 1.0 / scale, 1.0 / scale);
        temp.copyTo(DispImage(cv::Rect(x, y, temp.cols, temp.rows)));
    }
    namedWindow(title, cv::WINDOW_NORMAL);
    imshow(title, DispImage);
    waitKey();
    va_end(args);
}

void ejDosTallerUno(){
       Mat img = imread("../Data/lena.jpg", IMREAD_COLOR);

       // Disminuir el tamaño de la imagen a la mitad
       Mat img_small;
       resize(img, img_small, Size(), 0.5, 0.5);

       // Definir la región de interés (ROI) como un rectángulo de 100x100 píxeles en el centro de la imagen
       int roi_x = img_small.cols / 2 - 50;
       int roi_y = img_small.rows / 2 - 50;
       Rect roi_rect(img_small.cols / 2 - 75, img_small.rows / 2 - 75, 150, 150);

       // Crear una copia de la imagen de entrada para mostrar los resultados
       Mat output_img = img_small.clone();

       // Convertir la región de interés a escala de grises
       Mat roi = output_img(roi_rect);
       cvtColor(roi, roi, COLOR_BGR2GRAY);

       // Copiar la región de interés en la imagen de salida en color
       Mat roi_color = output_img(roi_rect);
       cvtColor(roi, roi_color, COLOR_GRAY2BGR);

       // Mostrar la imagen de salida
       imshow("Output", output_img);
       waitKey(0);
}

void ejTresTallerUno() {
       Mat img_input_A = imread("../Data/triceratops.png", IMREAD_GRAYSCALE);
       Mat img_input_B = imread("../Data/triceratopsDos.png", IMREAD_GRAYSCALE);

       Mat src_gray = Mat(img_input_A.size(), CV_8UC1);

       //Realizar la resta de imagenes pixel por pixel con el fin de obtener en una matriz los pixeles con la diferencia entre las dos imagenes
       for (int i = 0; i < img_input_A.rows; i++) {
           for (int k = 0; k < img_input_A.cols; k++) {
               src_gray.at<uchar>(i, k) = abs((int)img_input_A.at<uchar>(i, k) - (int)img_input_B.at<uchar>(i, k));
           }
       }
       imshow("Resta de imagenes", src_gray);

           //Obtener los bordes que conforman el contorno del dinosaurio
           Mat canny_output;
           Canny(src_gray, canny_output, 70, 255);

           Mat kernel2 = getStructuringElement(MORPH_RECT, Size(3, 3));
           dilate(canny_output, canny_output, kernel2);

           // Binariza la imagen para obtener el contorno
           Mat thresh;
           threshold(src_gray, thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);

           // Encuentra el contorno de la imagen
           vector<vector<Point> > contours;
           findContours(canny_output, contours, RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
           vector<vector<Point>> contours_poly(contours.size());
           vector<Rect> boundRect(contours.size());

           // Encuentra las coordenadas del bounding box
           Rect bbox = boundingRect(contours[0]);

           // Recorta la región del bounding box de la imagen
           Mat image_roi = src_gray(bbox);

           // Calcula los momentos de la región recortada
           Moments m = moments(image_roi, true);

           // Calcula el centro de masa
           Point center(m.m10/m.m00 + bbox.x, m.m01/m.m00 + bbox.y);

           // Dibuja un círculo en el centro de masa
           circle(src_gray, center, 5, Scalar(255, 255, 255), -1);

           // Muestra la imagen con el centro de masa dibujado
           imshow("Imagen con centro de masa", src_gray);
           waitKey(0);
}

void ejCuatroTallerUno() {
    Mat img1 = imread("../Data/triceratops.png", IMREAD_COLOR);
    Mat img2 = imread("../Data/triceratopsDos.png", IMREAD_COLOR);
    resize(img2, img2, img1.size());

    // Convertir a escala de grises
    Mat gray1, gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);

    // Calcular la relación entre las dos imágenes
    Mat src_gray = (gray1 / gray2) * 255.0;

    const int max_thresh = 255;
        Mat kernel = getStructuringElement(MORPH_CROSS, Size(4, 4));

        Mat topHat;
        //Realizar un suavizado a la imagen con el fin de solo obtener los bordes, eliminando pixeles que le produzca ruido a la imagen
        GaussianBlur(src_gray, src_gray, cv::Size(5, 5), 0);
        imshow("1° Resta de imagenes ", src_gray);

    //Obtener los bordes que conforman el contorno del dinosaurio
    Mat canny_output;
    Canny(src_gray, canny_output, 70, 255);

    Mat kernel2 = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(canny_output, canny_output, kernel2);

    // Binariza la imagen para obtener el contorno
    Mat thresh;
    threshold(src_gray, thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // Encuentra el contorno de la imagen
    vector<vector<Point> > contours;
    findContours(canny_output, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    vector<vector<Point>> contours_poly(contours.size());
    vector<Rect> boundRect(contours.size());

    // Encuentra las coordenadas del bounding box
    Rect bbox = boundingRect(contours[0]);

    // Recorta la región del bounding box de la imagen
    Mat image_roi = src_gray(bbox);

    // Calcula los momentos de la región recortada
    Moments m = moments(image_roi, true);

    // Calcula el centro de masa
    Point center(m.m10/m.m00 + bbox.x, m.m01/m.m00 + bbox.y);

    // Dibuja un círculo en el centro de masa con color negro
    circle(src_gray, center, 5, Scalar(255, 255, 255), -1);

    // Muestra la imagen con el centro de masa dibujado
    imshow("Imagen con centro de masa", src_gray);
    waitKey(0);
}

//---------------------------------------------------------TALLER 2----------------------------------------------------------------//

void ejUnoTallerDos() {
    //Realizamos una dilatacion a la imagen
    Mat tuto1 = imread("../Data/tProgramacionUno.jpg");
    Mat kernel = getStructuringElement(MORPH_RECT, Size(8, 8));
    Mat tutoG;
    dilate(tuto1, tutoG, kernel);
    imshow("A", tuto1);
    imshow("B", tutoG);
    waitKey(0);
}

void ejDosTallerDos() {
    //Aplicamos erosion
    Mat tuto1 = imread("../Data/tProgramacionDos.jpg");
    Mat kernel = getStructuringElement(MORPH_RECT, Size(6, 4));
    Mat tutoDel;
    erode(tuto1, tutoDel, kernel);
    imshow("A", tuto1);
    imshow("B", tutoDel);
    waitKey(0);
}

void ejTresTallerDos() {
    //aplicamos la operacion morfologica de erosion
    Mat palomaS = imread("../Data/paloma.jpg");
    Mat kernel = getStructuringElement(MORPH_RECT, Size(9, 9));
    Mat palomaLimpia;
    morphologyEx(palomaS, palomaLimpia, MORPH_OPEN, kernel);
    imshow("A", palomaS);
    imshow("B", palomaLimpia);
    waitKey(0);
}

void ejCuatroTallerDos() {
    //Se aplica la operacion morfologica de cierre
    //Primero erosiona la imagen dilatada
    Mat palomaL = imread("../Data/palomaDos.jpg");
    Mat kernel = getStructuringElement(MORPH_RECT, Size(6, 7));
    Mat palomaBL;
    morphologyEx(palomaL, palomaBL, MORPH_CLOSE, kernel);
    imshow("A", palomaL);
    imshow("B", palomaBL);
    waitKey(0);
}

void ejCincoTallerDos() {
    //El arreglo channels se usa para almacenar las
    //diferentes imagenes de canales de color
    //el primer canal esta en la posicion 0
    //se usa la funcion merge para fusionar los
    //canales en una sola imagen
    Mat image=imread("../Data/puntos.jpg",1);
    namedWindow("A",1);
    imshow("A",image);
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
}

void ejSeisTallerDos() {
    //Se le aplica el filtro median para
    //suavizar la imagen y reducir el ruido
    //
    Mat lena = imread("../Data/lenaDos.jpg", IMREAD_GRAYSCALE);
    Mat lenaLimpia;
    medianBlur(lena, lenaLimpia, 5);
    imshow("A", lena);
    imshow("B", lenaLimpia);
     waitKey(0);
}
void ejSieteTallerDos() {
        Mat src = cv::imread("../Data/messi.jpg");
        // Conversión de la imagen de BGR a HSV
        Mat hsv;
        vector<Mat> hsv_channels;
        cvtColor(src, hsv, CV_BGR2HSV);
        split(hsv, hsv_channels);
        int channels = src.channels();
        int nRows = src.rows;
        int nCols = src.cols * channels;
        Mat dst;
        Mat window = Mat(hsv.rows + 40, hsv.cols * 3 + 6, CV_8UC3);
        dst.create(src.rows, src.cols, src.type());

        //Vector con el valor de H y r para cada una de las imagenes
        //{{H y r - Tonalidad verde},{H y r - Tonalidad azul},{H y r - Tonalidad roja}}
        vector<Point> H_r = { {-56,16},{100,48},{170,30} };
        uchar* ptr_dst;

        int col = 0;
        int row = 0;
        int aux = 0;
        Scalar black = cv::Scalar(0);
        for (int img = 0; img < 3; img++) {
            //Rango donde se encuentra la tonalidad de color que permanece en la imagen
            uchar h1 = (H_r[img].x - (H_r[img].y / 2) + 360) % 360;
            uchar h2 = (H_r[img].x + (H_r[img].y / 2) + 360) % 360;

            for (int i = 0; i < src.rows; i++) {
                ptr_dst = dst.ptr<uchar>(i);
                uchar* ptr_src = src.ptr<uchar>(i);
                for (int j = 0; j < src.cols; j++) {
                    col = src.cols*img + j;
                    bool in_range_color = false;

                    // obtener el valor hue (H)
                    uchar H = hsv_channels[0].at<uchar>(i, j);

                    // Verificar si el valor de H de la transformación HSV esta dentro del rango de h1 a h2.
                    if (h1 <= h2) {
                        if (H >= h1 && H <= h2)
                            in_range_color = true;
                    }
                    else if (H >= h1 || H <= h2) {
                        in_range_color = true;
                    }
                    // Si se encuentra dentro del rango se mantiene el color original para ese pixel en particular.
                    // En caso contrario, se convierte a escala de grises mediante el promedio de los canales

                    if (in_range_color == true) {
                        for (int k = 0; k < dst.channels(); k++) {
                            dst.ptr<uchar>(i, j)[k] = src.ptr<uchar>(i, j)[k];
                        }
                    }
                    else if (in_range_color == false) {

                        uchar sum = (src.ptr<uchar>(i, j)[0] + src.ptr<uchar>(i, j)[1] + src.ptr<uchar>(i, j)[2]) / 3;

                        // conversion a grises usando usando en metodo de promedio
                        dst.ptr<uchar>(i, j)[2] = dst.ptr<uchar>(i, j)[1] = dst.ptr<uchar>(i, j)[0] = sum;
                    }
                    window.ptr<uchar>(i, col)[0] = dst.ptr<uchar>(i, j)[0];
                    window.ptr<uchar>(i, col)[1] = dst.ptr<uchar>(i, j)[1];
                    window.ptr<uchar>(i, col)[2] = dst.ptr<uchar>(i, j)[2];
                }
                row = i + 1;
                aux = row;

            }
            // Permite asignar varios pixeles de color negro en las ultimas filas de la ventana
            while (aux < row + 40) {
                for (int column = 0; column < col; column++) {
                    window.ptr<uchar>(row, column)[2] = window.ptr<uchar>(row, column)[1] = window.ptr<uchar>(row, column)[0] = black.val[0];
                }
                aux++;
            }
        }
        cv::namedWindow("A", WINDOW_NORMAL);
        cv::imshow("B", window);
        cv::resizeWindow("C", 1500, 500);
        cv::moveWindow("D", 250, 250);
        waitKey();
}
int main() {
    VideoCapture c;
    int mPrincipal;
    int mTUno;
    int mTDos;
    do
    {
        cout << "...... Talleres vision por computador ......" << endl;
        cout << "1. Taller Uno" << endl;
        cout << "2. Taller Dos" << endl;
        cout << "0. Salir" << endl;
        cout << "Selecciona una Opcion: ";
        cin >> mPrincipal;
        if (mPrincipal == 1) {
            cout << "...... Taller Uno ......" << endl;
            cout << "1. Ejercicio Uno" << endl;
            cout << "2. Ejercicio Dos" << endl;
            cout << "3. Ejercicio Tres" << endl;
            cout << "4. Ejercicio Cuatro" << endl;
            cout << "0. Salir" << endl;
            cout << "Selecciona una Opcion: ";
            cin >> mTUno;
            if (mTUno == 1 ) {
                Mat img1 = imread("../Data/lena.jpg");
                Mat img2 = imread("../Data/finaldos.jpg");
                Mat img3 = imread("../Data/messi.jpg");
                Mat img4 = imread("../Data/puntos.jpg");
                Mat img5 = imread("../Data/triceratops.png");
                Mat img6 = imread("../Data/triceratopsDos.png");
                ejUnoTallerUno("Image", 6, img1, img2, img3, img4, img5, img6);
            }
            if (mTUno == 2 ) {
                ejDosTallerUno();
            }
            if (mTUno == 3) {
                ejTresTallerUno();
            }
            if (mTUno == 4) {
                ejCuatroTallerUno();
            }
        }
        if (mPrincipal == 2) {
            cout << "...... Taller Dos ......" << endl;
            cout << "1. Ejercicio Uno" << endl;
            cout << "2. Ejercicio Dos" << endl;
            cout << "3. Ejercicio Tres" << endl;
            cout << "4. Ejercicio Cuatro" << endl;
            cout << "5. Ejercicio Cinco" << endl;
            cout << "6. Ejercicio Seis" << endl;
            cout << "7. Ejercicio Siete" << endl;
            cout << "0. Salir" << endl;
            cout << "Selecciona una Opcion: ";
            cin >> mTDos;
            if (mTDos == 1 ) {
                ejUnoTallerDos();
            }
            if (mTDos == 2 ) {
                ejDosTallerDos();
            }
            if (mTDos == 3) {
                ejTresTallerDos();
            }
            if (mTDos == 4) {
                ejCuatroTallerDos();
            }
            if (mTDos == 5) {
                ejCincoTallerDos();
            }
            if (mTDos == 6) {
                ejSeisTallerDos();
            }
            if (mTDos == 7) {
                ejSieteTallerDos();
            }
        }
    } while (select != 0);
    return 1;
}

