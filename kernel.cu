#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <chrono> 


__global__ void convoluteGPU(int* pixeliIntrare, int* pixeliIesire, int linii, int coloane, int canaleCuloare) {
	int kernel[5][5] = {
	   { 0,  0, -1,  0,  0},
	   { 0, -1, -2, -1,  0},
	   {-1, -2, 16, -2, -1},
	   { 0, -1, -2, -1,  0},
	   { 0,  0, -1,  0,  0}
	};
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < linii * coloane * canaleCuloare) {
		//out[id] = in[id]+20;

		//apply kernel
		int linie = id / (coloane * canaleCuloare);
		int coloana = (id % (coloane * canaleCuloare)) / canaleCuloare;
		int canalCuloare = id % canaleCuloare;

		int pixel = 0;
		if (linie > 1 && linie < (linii - 2) && coloana > 1 && coloana < (coloane - 2)) {

			//mijloc
			pixel += kernel[2][2] * pixeliIntrare[id];

			//N
			pixel += kernel[1][2] * pixeliIntrare[id - coloane * canaleCuloare];

			//NE
			pixel += kernel[1][3] * pixeliIntrare[id - coloane * canaleCuloare + canaleCuloare];

			//E
			pixel += kernel[2][3] * pixeliIntrare[id + canaleCuloare];

			//SE
			pixel += kernel[3][3] * pixeliIntrare[id + coloane * canaleCuloare + canaleCuloare];

			//S
			pixel += kernel[3][2] * pixeliIntrare[id + coloane * canaleCuloare];

			//SV
			pixel += kernel[3][1] * pixeliIntrare[id + coloane * canaleCuloare - canaleCuloare];

			//V
			pixel += kernel[2][1] * pixeliIntrare[id - canaleCuloare];

			//NV
			pixel += kernel[1][1] * pixeliIntrare[id - coloane * canaleCuloare - canaleCuloare];

			//conturul kernelului
			//N
			pixel += kernel[0][0] * pixeliIntrare[id - 2 * coloane * canaleCuloare - 2 * canaleCuloare];
			pixel += kernel[0][1] * pixeliIntrare[id - 2 * coloane * canaleCuloare - canaleCuloare];
			pixel += kernel[0][2] * pixeliIntrare[id - 2 * coloane * canaleCuloare];
			pixel += kernel[0][3] * pixeliIntrare[id - 2 * coloane * canaleCuloare + canaleCuloare];
			pixel += kernel[0][4] * pixeliIntrare[id - 2 * coloane * canaleCuloare + 2 * canaleCuloare];

			//E
			pixel += kernel[1][4] * pixeliIntrare[id - 1 * coloane * canaleCuloare + 2 * canaleCuloare];
			pixel += kernel[2][4] * pixeliIntrare[id - 0 * coloane * canaleCuloare + 2 * canaleCuloare];
			pixel += kernel[3][4] * pixeliIntrare[id + 1 * coloane * canaleCuloare + 2 * canaleCuloare];

			//S
			pixel += kernel[4][0] * pixeliIntrare[id + 2 * coloane * canaleCuloare - 2 * canaleCuloare];
			pixel += kernel[4][1] * pixeliIntrare[id + 2 * coloane * canaleCuloare - canaleCuloare];
			pixel += kernel[4][2] * pixeliIntrare[id + 2 * coloane * canaleCuloare];
			pixel += kernel[4][3] * pixeliIntrare[id + 2 * coloane * canaleCuloare + canaleCuloare];
			pixel += kernel[4][4] * pixeliIntrare[id + 2 * coloane * canaleCuloare + 2 * canaleCuloare];

			//V
			pixel += kernel[1][0] * pixeliIntrare[id - 1 * coloane * canaleCuloare - 2 * canaleCuloare];
			pixel += kernel[2][0] * pixeliIntrare[id - 0 * coloane * canaleCuloare - 2 * canaleCuloare];
			pixel += kernel[3][0] * pixeliIntrare[id + 1 * coloane * canaleCuloare - 2 * canaleCuloare];

			pixel = pixel / 1;
		}
		else {
			pixel = 0;
		}
		pixeliIesire[id] = pixel;
	}
}

int* mapareMatricePixeliRGBLaVector(int*** imagine, int linii, int coloane, int canaleCuloare) {
	int* vector = (int*)malloc(linii * coloane * canaleCuloare * sizeof(int));
	int id = 0;
	for (int i = 0; i < linii; i++) {
		for (int j = 0; j < coloane; j++) {
			for (int c = 0; c < canaleCuloare; c++) {
				vector[id] = imagine[i][j][c];
				id++;
			}
		}
	}
	return vector;
}

int*** mapareVectorLaMatricePixeliRGB(int* vector, int linii, int coloane, int canaleCuloare) {
	int*** imagine = (int***)malloc(linii * sizeof(int**));
	int id = 0;

	for (int i = 0; i < linii; i++) {
		imagine[i] = (int**)malloc(coloane * sizeof(int*));

		for (int j = 0; j < coloane; j++) {
			imagine[i][j] = (int*)malloc(canaleCuloare * sizeof(int));

			for (int c = 0; c < canaleCuloare; c++) {
				imagine[i][j][c] = vector[id];
				id++;
			}
		}
	}
	return imagine;
}

void aplicareFiltru() {
	//citim matricea de pixeli RGB
	std::ifstream in("pixels.txt");
	int linii, coloane, canaleCuloare;
	in >> linii >> coloane >> canaleCuloare;

	int BLOCK_SIZE = 1000;

	int blockCount = ((linii * coloane * canaleCuloare) / BLOCK_SIZE) + 1;

	//citire in memorie
	int*** matrix = (int***)malloc(linii * sizeof(int**));
	for (int i = 0; i < linii; i++) {
		matrix[i] = (int**)malloc(coloane * sizeof(int*));

		for (int j = 0; j < coloane; j++) {
			int* line = (int*)malloc(canaleCuloare * sizeof(int));

			in >> line[0] >> line[1] >> line[2];

			matrix[i][j] = line;
		}
	}

	int dimensiune = linii * coloane * canaleCuloare;
	//maparea matricei la vector
	int* vector = mapareMatricePixeliRGBLaVector(matrix, linii, coloane, canaleCuloare);
	int* rezultat = (int*)malloc(dimensiune * sizeof(int));

	//copiem vectorul de pixeli in vectorDevice
	int* vectorDevice;
	int* rezultatDevice;
	cudaMalloc(&vectorDevice, dimensiune * sizeof(int));
	cudaMalloc(&rezultatDevice, dimensiune * sizeof(int));

	cudaMemcpy(
		vectorDevice, vector,
		dimensiune * sizeof(int),
		cudaMemcpyHostToDevice
	);

	//apelam filtrul convolutional (better: multiplu de 2 ca numar de thread-uri) (test: different block sizes)
	convoluteGPU <<< blockCount, BLOCK_SIZE >>> (vectorDevice, rezultatDevice, linii, coloane, canaleCuloare);

	//copiem rezultatDevice in rezultat
	cudaMemcpy(
		rezultat, rezultatDevice,
		dimensiune * sizeof(int),
		cudaMemcpyDeviceToHost
	);

	int*** imagine = mapareVectorLaMatricePixeliRGB(rezultat, linii, coloane, canaleCuloare);

	std::ofstream out("pixels.txt");
	out << linii << " " << coloane << " " << canaleCuloare << "\n";
	for (int i = 0; i < linii; i++) {
		for (int j = 0; j < coloane; j++) {
			for (int k = 0; k < canaleCuloare; k++) {
				out << imagine[i][j][k] << " ";
			}
			out << "\n";
		}
	}

	out.close();
}

int main() {
	char* pathFisierIntrare = "python in.py C:/Users/George/source/repos/P2/P2/landscape.png";
	char* pathFisierIesire = "python out.py C:/Users/George/source/repos/P2/P2/landscape1.png";

	system(pathFisierIntrare);			//citim si scriem valoarea pixelilor in pixels.txt

	auto start = std::chrono::steady_clock::now();
	aplicareFiltru();
	auto stop = std::chrono::steady_clock::now();

	system(pathFisierIesire);			//scriem pixelii in imaginea filtrata

	// we cuda've done that
	auto diff = stop - start;
	std::cout << std::chrono::duration <double, std::milli>(diff).count() << " ms" << std::endl;
	
	return 0;
}