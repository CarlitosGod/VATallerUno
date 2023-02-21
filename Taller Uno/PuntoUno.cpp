#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdarg.h>

//https://github-com.translate.goog/opencv/opencv/wiki/DisplayManyImages?_x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=wapp

using namespace std;
using namespace cv;

void mostrarImagenes(string title, int nArgs, ...) {
int size;
int m, n;
int x, y;

int w, h;

//Variable de escala de las imagenes
float scale;
int max;

// Calculo de alto y ancho para el display asi como su tamaño
w = 3; h = 2;
size = 200;

// Imagen de 3 canales RGB
Mat DispImage = Mat::zeros(Size(100 + size*w, 60 + size*h), CV_8UC3);

// Usamos los parametros de la lista
va_list args;
va_start(args, nArgs);

for (int i = 0, m = 20, n = 20; i < nArgs; i++, m += (20 + size)) {
    // Obtenemos la ubicacion de la imagen y lo guardamos en un objeto de tipo Mat
    Mat img = va_arg(args, Mat);

    // verificamos si la imagen que llega no es nulla en esa posicion
    if(img.empty()) {
        printf("Invalid arguments");
        return;
    }

    // obtenemos el tamaño de la imagen
    x = img.cols;
    y = img.rows;

    // validamos que la imagen sea cuadrada y asignamos su maximo valor a la variable max
    max = (x > y) ? x : y;

    // hallamos el factor de escalado de la imagen dividiendo el valor màximo de la imagen
    // ya sea su tamaño en columnas o en filas con un size predeterminado de 200
    scale = (float) ( (float) max / size );

    // Alineamos las imagenes sacando el mudulo de el ancho y el alto del tamaño del display
    if( i % w == 0 && m!= 20) {
        m = 20;
        n+= 20 + size;
    }

    // Por medio de Rect creamos el rectangulo con el fin de poner la imagen en cada uno
    // Luego hacemos el copyto hacia el disply ubicando la imagen con su rectangulo correspondiente
    Rect ROI(m, n, (int)( x/scale ), (int)( y/scale ));
    Mat temp; resize(img,temp, Size(ROI.width, ROI.height));
    temp.copyTo(DispImage(ROI));
}

namedWindow( title, 1 );
imshow( title, DispImage);
waitKey();

va_end(args);
}

int main(  )
{
    Mat img1 = imread("../Data/lena.jpg");
    Mat img2 = imread("../Data/lena.jpg");
    Mat img3 = imread("../Data/lena.jpg");
    Mat img4 = imread("../Data/lena.jpg");
    Mat img5 = imread("../Data/lena.jpg");
    Mat img6 = imread("../Data/lena.jpg");

    mostrarImagenes("Image", 6, img1, img2, img3, img4, img5, img6);

    return 0;
}

