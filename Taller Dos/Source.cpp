#include <iostream>
#include <cstdarg>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//Punto uno, el de las 6 imagenes
/*


void getSquareImage(cv::InputArray img, cv::OutputArray dst, int size)
{
    if (size < 2) size = 2;
    int width = img.cols(), height = img.rows();

    cv::Mat square = dst.getMat();

    // si la imagen es cuadrada solo redimensionar
    if (width == height) {
        cv::resize(img, square, Size(size, size));
        return;
    }

    // establecer color de fondo del cuadrante
    square.setTo(Scalar::all(0));

    int max_dim = (width >= height) ? width : height;
    float scale = ((float)size) / max_dim;

    // calcular la region centrada 
    cv::Rect roi;

    if (width >= height)
    {
        roi.width = size;
        roi.x = 0;
        roi.height = (int)(height * scale);
        roi.y = (size - roi.height) / 2;
    }
    else
    {
        roi.y = 0;
        roi.height = size;
        roi.width = (int)(width * scale);
        roi.x = (size - roi.width) / 2;
    }

    // redimensionar imagen en la region calculada
    cv::resize(img, square(roi), roi.size());
}

void showImages(const String& window_name, int rows, int cols, int size, std::initializer_list<const Mat*> images, int pad = 1)
{
    if (pad <= 0) pad = 0;

    int width = size * cols + ((cols + 1) * pad);
    int height = size * rows + ((rows + 1) * pad);

    // crear la imagen de salida con un color de fondo blanco
    Mat dst = Mat(height, width, CV_8UC3, Scalar::all(255));

    int x = 0, y = 0, cols_counter = 0, img_counter = 0;

    // recorrer la lista de imagenes
    for (auto& img : images) {
        Mat roi = dst(Rect(x + pad, y + pad, size, size));

        // dibujar la imagen en el cuadrante indicado
        getSquareImage(*img, roi, size);

        // avanzar al siguiente cuadrante
        x += roi.cols + pad;

        // avanza a la siguiente fila
        if (++cols_counter == cols) {
            cols_counter = x = 0;
            y += roi.rows + pad;
        }

        // detener si no hay mas cuadrantes disponibles
        if (++img_counter >= rows * cols) break;
    }

    imshow(window_name, dst);
}

int main(int argc, char** argv)
{
    Mat image0 = imread("image/lena.png", IMREAD_COLOR);
    Mat image1 = imread("image/rawr.png", IMREAD_COLOR);
    Mat image2 = imread("image/boom.jpg", IMREAD_COLOR);

    Mat image4;
    cvtColor(image0, image4, COLOR_RGB2GRAY);
    cvtColor(image4, image4, COLOR_GRAY2BGR);

    Mat image5;
    filter2D(image1, image5, image5.depth(), Mat::eye(2, 2, CV_8UC1));

    Mat image6;
    Sobel(image2, image6, -1, 1, 1);

    if (!image0.empty() || !image1.empty() || !image2.empty())
    {
        namedWindow("Display Multiple Image", WINDOW_AUTOSIZE);
        showImages("Display Multiple Image", 2, 3, 240, { &image0, &image1, &image2, &image4, &image5, &image6 }, 5);
    }
    else cout << "No image data." << endl;

    waitKey(0);
    return 0;
}

*/

//Taller dos, el de messi
#include <iostream>
#include <cstdarg>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
int menu();

void tutoGordo() {
    Mat tuto1 = imread("../tallerVA/image/tutor.png");

    Mat kernel = getStructuringElement(MORPH_RECT, Size(8, 8));
    Mat tutoG;
    dilate(tuto1, tutoG, kernel);

    imshow("tutor", tuto1);
    imshow("Tutor gordo", tutoG);
    waitKey(0);
}

void tutoDelgado() {
    Mat tuto1 = imread("../tallerVA/image/tutor.png");

    Mat kernel = getStructuringElement(MORPH_RECT, Size(6, 4));
    Mat tutoDel;
    erode(tuto1, tutoDel, kernel);

    imshow("tutor", tuto1);
    imshow("Tutor delgado", tutoDel);
    waitKey(0);
}
void palomaSucia() {

    Mat palomaS = imread("../tallerVA/image/ave.jpg");
    Mat kernel = getStructuringElement(MORPH_RECT, Size(9, 9));
    Mat palomaBS;
    morphologyEx(palomaS, palomaBS, MORPH_CLOSE, kernel);

    imshow("Paloma sucia", palomaS);
    imshow("Paloma limpia", palomaBS);
    waitKey(0);
}
void palomaLimpia() {

    Mat palomaL = imread("../tallerVA/image/ave2.png");
    Mat kernel = getStructuringElement(MORPH_RECT, Size(6, 7));
    Mat palomaBL;
    morphologyEx(palomaL, palomaBL, MORPH_OPEN, kernel);

    imshow("Paloma", palomaL);
    imshow("Paloma limpia", palomaBL);
    waitKey(0);
}
void EjercicioRGB() {

    // Cargar la imagen original
    Mat image = imread("../tallerVA/image/imagenBGR.png", IMREAD_COLOR);

    // Dividir la imagen en sus tres canales de color
    Mat channels[3];
    split(image, channels);

    Mat blue = channels[0];

    //multiplicaciones

    // Mostrar cada canal en una ventana separada
    imshow("Canal Original", image);
    imshow("Canal Rojo", channels[2]);
    imshow("Canal Verde", channels[1]);
    imshow("Canal Azul", channels[0]);

    waitKey(0);
}

void EjercicioLena() {

    Mat lena = imread("../tallerVA/image/lenanoise.png", IMREAD_GRAYSCALE);

    Mat lenaLimpia;
    medianBlur(lena, lenaLimpia, 5);

    imshow("Lena", lena);
    imshow("Lena limpia", lenaLimpia);
    waitKey(0);
}

void Messi() {
    // Leer la imagen en formato Mat
    Mat image = imread("../tallerVA/image/mes.jpg", IMREAD_COLOR);

    // Convertir la imagen a espacio de color HSV
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);

    // Extraer la componente H (matiz) de la imagen HSV
    Mat h;
    std::vector<Mat> hsv_channels;
    split(hsv, hsv_channels);
    h = hsv_channels[0];

    // Crear una máscara para resaltar los píxeles que tienen el color rojo
    Mat mask;
    inRange(h, Scalar(0, 0, 155), Scalar(0, 0, 255), mask);
    inRange(h, Scalar(160, 100, 100), Scalar(179, 255, 255), mask);

    // Aplicar la máscara a la imagen original
    Mat result;
    image.copyTo(result, mask);

    // Rellenar las áreas de la imagen que no están resaltadas con un color gris
    Mat gray;
    cvtColor(result, gray, COLOR_BGR2GRAY);
    threshold(gray, gray, 1, 255, THRESH_BINARY_INV);
    cvtColor(gray, gray, COLOR_GRAY2BGR);
    result.setTo(Scalar(128, 128, 128), gray);

    // Mostrar la imagen resultante
    imshow("Imagen original", image);
    imshow("Imagen resaltada ROJO", result);
    waitKey(0);
}

int main() {
    VideoCapture c;
    int op;
    do
    {
        op = menu();
        switch (op) {
        case 0: cout << " Bye" << endl;
            break;
        case 1: tutoGordo();
            break;
        case 2: tutoDelgado();
            break;
        case 3: palomaSucia();
            break;
        case 4: palomaLimpia();
            break;
        case 5: EjercicioRGB();
            break;
        case 6: EjercicioLena();
            break;
        case 7: Messi();
            break;
        }
    } while (op != 0);
    return 1;
}

int menu() {
    int opcion;
    cout << "...... MENU ......" << endl;
    cout << "0. Salir" << endl;
    cout << "1. Ejercicio 1 - tutor gordo" << endl;
    cout << "2. Ejercicio 2 - tutor delgado" << endl;
    cout << "3. Ejercicio 3 - paloma limpia" << endl;
    cout << "4. Ejercicio 4 - paloma sucia" << endl;
    cout << "5. Ejercicio 5 - RGB" << endl;
    cout << "6. Ejercicio 6 - lena" << endl;
    cout << "7. Ejercicio 7 - Messi" << endl;
    cout << "Selecciona una Opcion: ";
    cin >> opcion;
    return opcion;
}
