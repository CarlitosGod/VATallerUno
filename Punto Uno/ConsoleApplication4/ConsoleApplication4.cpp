#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui/highgui_c.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>


using namespace std;
using namespace cv;

int main()
{

	//Tipos de datos:
	//Point:coordenadas de un punto (un pixel) en una imagén
	//o posición de una región(Rect) dentro de la misma:

	//puntos 2d
	int x = 0;
	int y = 0;
	Point pt(x, y);//(x,y)
	Point pt2 = Point(100, 100);
	Point pt3;
	Point coordenmas_imagen;
	pt3.x = 200;
	pt3.y = 300;

	coordenmas_imagen.x = 3;
	coordenmas_imagen.y = 73;
	//puntos 2d
	printf(" Puntos 2D ( %d , %d ) \n", int(pt2.x), int(pt2.y));

	//puntos 3d (x,y,z)
	Point3d pt3d = Point3d(100, 200, 300);
	printf(" Puntos 3D ( %d , %d , %d ) \n", int(pt3d.x), int(pt3d.y), int(pt3d.z));

	//variantes
	Point2f a(2.5, 6.5); //datos float
	Point2f b(233.533, 3333336.53); //datos double
	Point3i c(100, 100, 100);//datos int
	Point3f d(23.7, 56.9, 56.00);//datos float

	//Size: Sirve para indicar o representar el tamaño de
	//una región de pixeles en ancho y alto.

	Size dimensiones(800, 600);
	Size dimensiones2 = Size(1080, 720);//  hd // full hd 1080 1920
	Size dimensiones3;
	dimensiones3.width = 200;
	dimensiones3.height = 200;

	printf("dimensiones   =  (%d x %d) \n", dimensiones.width, dimensiones.height);
	printf("dimensiones2  = (%d x %d) \n", dimensiones2.width, dimensiones2.height);

	printf("area de la region: %d \n", dimensiones.area());

	//Rec:DefinE regiones rectangulares dentro de una imagen,
	//indiciando el ancho, el alto y su posición espacial en pixeles.

	Rect region = Rect(0, 0, 200, 200); // x : 0, y : 0, width: 200px, height: 200px
	Rect region2 = cvRect(0, 0, 200, 200);//usando la función cvRect
	Rect region4 = Rect(Point(200, 200), Size(800, 600));//usando las dos estructuras vistas

	//a partir de dos puntos, el primero indicará la posición y el segundo el tamaño
	Rect region5 = Rect(Point(400, 500), Point(100));
	Rect region7 = Rect(region);// a partir de otro rectangulo

	//modificando sus propiedades
	Rect region6;
	region6.x = 200;
	region6.y = 400;
	region6.width = 700;
	region6.height = 800;



	printf("el area del rectangulo es %d \n", region6.area());
	printf("el rectangulo esta ubicado en la pos (%d,%d) \n", region6.x, region6.y);
	printf("el tamaño del rectangulo es (%d,%d) \n", region6.width, region6.height);


	/*Mat : Partiendo de que una imagen, es una matriz o array bidimensional de números representada de forma digital
	Opencv usa este tipo de dato para guardarlas en memoria y así poderlas manipular programaticamente.
	A través de esta clase podemos acceder a las propiedades de dicha imagen,como por ejemplo:
	su tamaño en pixeles, el número de canales, el formato en el que fue comprimida (JPG, PNG, BMP)
	entre otras, además, acceder de forma indexada(x,y) a la información de cada pixel.
	Pero que representa un pixel??. Cada pixel representa una magnitud física:
	Cantidad de Luz en un punto de una escena, Valor de color(cantidad de radiación en la frecuencia del rojo, verde y azul),Nivel de radiación infrarroja, rayos x etc. EN general cualquier radiación electromagnética: profundidad(distancia) de una escena en una dirección, cantidad de presión ejercida en un punto,nivel de absorción de determinada radiación etc.
	Los formatos de imagen soportados por opencv son:
	Windows bitmaps – *.bmp, *.dib , JPEG files – *.jpeg, *.jpg, *.jpe , JPEG 2000 files – *.jp2 ,
	 Portable Network Graphics – *.png ,WebP – *.webp , Portable image format – *.pbm, *.pgm, *.ppm ,
	 Sun rasters – *.sr, *.ras , TIFF files – *.tiff, *.tif
	*/

	//Matriz de 4 filas y 5 columnnas, tipo CV_8U, valores permitidos
	//0 - 255, valor por defecto 23
	Mat A = Mat(4, 5, CV_8U, Scalar(23));

	// A partir de un vector de nxm dimensiones
	float m2[3][3] =
	{
	  { 2, 0, 1 },
	  { 3, 0, 0 },
	  { 5, 1, 1 }
	};

	float m3[3][3] =
	{
	{ 1, 0, 1 },
	{ 1, 2, 1 },
	{ 1, 1, 0 }
	};

	//matriz 3x3, inicializada con los valores de m2
	Mat B(3, 3, CV_32FC1, m2);

	//matriz 3x3, inicializada con los valores de m3
	Mat C(3, 3, CV_32FC1, m3);

	//A partir de un vector de valores separados por coma
	//Mat G = (Mat_&amp;lt;double&amp;gt;(3, 3) &amp;lt;&amp;lt; 1, 2, 3, 4, 5, 6, 7, 8, 9);

	// matriz identidad de 4x4
	Mat D = Mat::eye(4, 4, CV_64F);

	//Matriz de unos de 2x2
	Mat E = Mat::ones(2, 2, CV_32F);

	//Matriz de ceros de 3x3
	Mat F = Mat::zeros(3, 3, CV_8UC1);

	//A partir de otra matriz
	Mat H = B.clone();
	B.copyTo(H);

	//a partir de una imagen
	Mat I = imread("../Data/lena.jpg");

	printf("Ancho y alto (%d, %d) \n", I.cols, I.rows);
	printf("Ancho y alto (%d, %d) \n", I.size().width, I.size().height);
	printf("Numero de canales (%d) \n", I.channels());
	printf("Profundidad de pixeles (%d) \n", I.depth());
	printf("Número total de pixeles (%lu) \n", I.total());

	//Operaciones aritméticas:

	float m[3][3] =
	{
	{ 2, 0, 1 },
	{ 3, 0, 0 },
	{ 5, 1, 1 }
	};

	float n[3][3] =
	{
	{ 1, 0, 1 },
	{ 1, 2, 1 },
	{ 1, 1, 0 }
	};

	//matriz 3x3, inicializada con los valores de m2
	Mat M(3, 3, CV_32FC1, m);

	//matriz 3x3, inicializada con los valores de m3
	Mat N(3, 3, CV_32FC1, n);

	//Matrix entre Matrix
	Mat SUM = M + N; //suma entre dos matricez
	Mat SUB = M - N; //substracción de matricez
	Mat MUL = M * N; //multiplicación de matricez
	Mat DIV = M / N; //división de matricez

	//Operaciones Booleanas
	uchar alfa[2][2] =
	{
	{ 0, 1},
	{1 , 0}
	};

	uchar beta[2][2] =
	{
	{ 0, 0 },
	{ 0 , 1 }
	};

	//matriz 2x2, inicializada con los valores de a
	Mat Alfa(2, 2, CV_8U, alfa);

	//matriz 2x2, inicializada con los valores de b
	Mat Beta(2, 2, CV_8U, beta);

	Mat C1;
	bitwise_or(Alfa, Beta, C1);
	bitwise_and(Alfa, Beta, C1);
	bitwise_not(Alfa, Beta, C1);
	bitwise_xor(Alfa, Beta, C1);

	//Imprimir y recorrer Matrices
	uchar data[3][3] =
	{
	{ 3, 2, 1 },
	{ 5, 3, 4 },
	{ 1, 1,1 }
	};

	Mat R(3, 3, CV_8U, data);
	int rows = R.rows;
	int cols = R.cols;

	//recorrer matriz
	printf(" Matrix m: \n");
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{// Observe the type used in the template
			printf(" %d ", R.at<uchar>(i, j));
		}
		printf("\n");
	}

	// imprimir matriz
	cout << "R (default) = " << endl << R << endl << endl;
	cout << "R (matlab) = " << endl << format(R, Formatter::FMT_MATLAB) << endl << endl;
	cout << "R (python) = " << endl << format(R, Formatter::FMT_PYTHON) << endl << endl;
	cout << "R (csv) = " << endl << format(R, Formatter::FMT_CSV) << endl << endl;
	cout << "R (numpy) = " << endl << format(R, Formatter::FMT_NUMPY) << endl << endl;

}
