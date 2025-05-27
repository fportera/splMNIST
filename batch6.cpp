//=======================================================================
// Copyright (c) 2017 Adrian Schneider
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <cstdio>
#include <vector>
#include <bits/stdc++.h>
#include <iostream>
#include <unordered_map>
#include <iostream>
#include <algorithm>
#include <random>

#include "mnist/mnist_reader.hpp"
#include "spline.h"

#define MNIST_DATA_LOCATION "./"
#define MNIST_NUM_CLASSES 10
#define NUM_FEATURES (28 * 28)

#define SQR(a) ((a) * (a))

using namespace std;

int dinp;
int d;
int trs;
int tes;

int bs = 64;

double a[MNIST_NUM_CLASSES*2][NUM_FEATURES];
double b;

double lr;

class Point {
public:
  float x;
  int y;

  Point *next;
};

bool comp(Point a, Point b) {
  return a.x < b.x;
}

class MapXY {
public:
  int np = 0;
  vector<Point*> mapxy;
  
  int binarySearch(vector<Point*> arr, int low, int high, double x)
  {
    while (low <= high) {
        int mid = low + (high - low) / 2;

        // Check if x is present at mid
        if (arr[mid]->x == x)
            return mid;

        // If x greater, ignore left half
        if (arr[mid]->x < x)
            low = mid + 1;

        // If x is smaller, ignore right half
        else
            high = mid - 1;
    }

    // If we reach here, then element was not present
    return -1;
  }

  void addPoint(float x, float y) {
    int posfound;
    
    if (np > 0)
      posfound = binarySearch(mapxy, 0, np - 1, x);
    else
      posfound = -1;
    
    if (posfound == -1) { 
      Point *p = new Point();
      p->x = x;
      p->y = y;
      p->next = NULL;
      
      mapxy.push_back(p);
      np++;
    } else { // Handle collisions
      // Add to bucket posfound
      Point *p = new Point();
      p->x = x;
      p->y = y;
      p->next = mapxy[posfound];
      mapxy[posfound] = p;
    }
  }
};

double Training(vector<vector<tk::spline>> &vsplines, int pos, vector<vector<double>> &matXtr, vector<int> &vecYtr) {

  int iter = 0;
  double BACC = 0;

  double loss = 0;
  for (int i = pos * bs; (i < (pos + 1) * bs) && (i < trs); i++) {

    for (int q = 0; q < MNIST_NUM_CLASSES; q++) {
      // Update weights for gradient descent
      double pred = 0;
      for (int j = 0; j < d; j++)
	pred += a[q][j] * vsplines[q][j](matXtr[j][i]);
	
      pred += b;

      loss += SQR(pred - (vecYtr[i] == q ? 1 : 0));
	  
      for (int j = 0; j < d; j++)
	a[q][j] -= lr * 2 * (pred - (vecYtr[i] == q ? 1 : 0)) * vsplines[q][j](matXtr[j][i]);

      b -= lr * 2 * (pred - (vecYtr[i] == q ? 1 : 0));
    }
  }

  printf("loss = %lf    ", loss);
  
  // Calculate Batch Training Accuracy

  for (int i = pos * bs; (i < (pos + 1) * bs) && (i < trs); i++) {
    double maxPred = -1E10;
    int predClass = -1; 
      
    for (int q = 0; q < MNIST_NUM_CLASSES; q++) {
      double pred = 0;
      for (int j = 0; j < d; j++) 
	pred += a[q][j] * vsplines[q][j](matXtr[j][i]);
      
      pred += b;
      
      if (pred > maxPred) {
	maxPred = pred;
	predClass = q;
      }
    }
      
    if ( predClass == vecYtr[i] )
      BACC += 1;
  }

  printf("Number of correct classified samples on currect batch = %d,   ", (int)BACC);
  
  BACC /= bs;
    
  //  printf("TRACC = %lf\n", TRACC);
    
  return BACC;
}

double Test(vector<vector<tk::spline>> &vsplines, vector<vector<double>> &matXte, vector<int> &vecYte) {
  double TEACC = 0;
  for (int j = 0; j < tes; j++) {
    double maxPred = -1E10;
    int predClass = -1; 

    for (int q = 0; q < MNIST_NUM_CLASSES; q++) {
      double pred = 0;
      for (int i = 0; i < d; i++)
	pred += a[q][i] * vsplines[q][i](matXte[i][j]);

      pred += b;
      
      if (pred > maxPred) {
	maxPred = pred;
	predClass = q;
      }
    }
      
    if ( predClass == vecYte[j] )
      TEACC += 1;    
  }
  
  TEACC /= tes;

  return TEACC;
}

mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset;

void shuffle() {
  for(int i = 0; i < trs; i++) {
    int p, q;
    p = (int)(rand() / (double) RAND_MAX * trs);
    do {
      q = (int)(rand() / (double) RAND_MAX * trs);
    } while(p == q);
    
    // swap
    vector<uint8_t> temp(NUM_FEATURES);
    temp = dataset.training_images[p];
    dataset.training_images[p] = dataset.training_images[q];
    dataset.training_images[q] = temp;
    uint8_t templ;
    templ = dataset.training_labels[p];
    dataset.training_labels[p] = dataset.training_labels[q];
    dataset.training_labels[q] = templ;
  }
}

int main(int argc, char* argv[]) {
  
  // MNIST_DATA_LOCATION set by MNIST cmake config
  std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

  // Load MNIST data
  dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

  std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
  std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
  std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
  std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

  // auto rng = default_random_engine {};
  // shuffle(std::begin(dataset.training_images), std::end(dataset.training_images), rng);

  dinp = NUM_FEATURES;
  d = (int)round(NUM_FEATURES);
  trs = dataset.training_images.size();
  tes = dataset.test_images.size();

  printf("Shuffling...\n");
  shuffle();
  
  vector<vector<tk::spline>> vsplines(MNIST_NUM_CLASSES, vector<tk::spline>(trs));
    
  vector<vector<double>> matXtr(dinp, vector<double>(trs));
  vector<int> vecYtr(trs);
  vector<vector<double>> matXte(dinp, vector<double>(tes));
  vector<int> vecYte(tes);

  // Building matXt and vecYtr
  for (int i = 0; i < dinp; i++) 
    for (int j = 0; j < trs; j++) 
      matXtr[i][j] = dataset.training_images[j][i] * 2;

  for (int j = 0; j < trs; j++)
    vecYtr[j] = dataset.training_labels[j];

  // Building maxTv and vecYte
  for (int i = 0; i < dinp; i++) 
    for (int j = 0; j < tes; j++) 
      matXte[i][j] = dataset.test_images[j][i] * 2;
    
  for (int j = 0; j < tes; j++) {
    vecYte[j] = dataset.test_labels[j];
    // printf("%d\n", vecYte[j]);
  }

  printf("Generating splines...\n");
  // Generate training splines
  for (int q = 0; q < MNIST_NUM_CLASSES; q++) {
      printf("Generating splines for class %d\n", q);
      for (int i = 0; i < d; i++) {
      if (i % 10 == 0)
	printf("Generating splines for input canal %d\n", i);
	
      vector<Point> tpoints(trs);

      for (int j = 0; j < trs; j++) {
	tpoints[j].x = matXtr[i][j];
	tpoints[j].y = (vecYtr[j] == q ? 1 : 0);
      }
      
      sort(tpoints.begin(), tpoints.end(), comp);

      MapXY *mxy = new MapXY();
      
      for (int j = 0; j < trs; j++) {
	mxy->addPoint(tpoints[j].x, tpoints[j].y);
      }

      // Build x vectors for i-th feature
      vector<float> x, y;
      for (int j = 0; j < mxy->mapxy.size(); j++) {
	int k = 1;
	float my = mxy->mapxy[j]->y;
	Point *np = mxy->mapxy[j]->next;
	
	while(np != NULL) {
	  my += np->y;
	  np = np->next;
	  k++;
	}
	my /= (float) k;
	x.push_back(mxy->mapxy[j]->x);
	y.push_back(my);
      }

      if (x.size() > 2) {
	tk::spline s(x, y);
	vsplines[q][i] = s;
      } else {
	tk::spline s({0, 1, 2}, {y[0], y[0], y[0]});
	vsplines[q][i] = s;
      }
     
      // x.clear();
      // y.clear();

      // for (int j = 0; j < mxy->mapxy.size(); j++) {	
      // 	int ze = 0, on = 0;

      // 	double my = mxy->mapxy[j]->y;
      // 	if (my == 0)
      // 	  ze++;
      // 	else
      // 	  on++;
	
      // 	Point *np = mxy->mapxy[j]->next;
      // 	while(np != NULL) {
      // 	  if (np->y == 0)
      // 	    ze++;
      // 	  else
      // 	    on++;
      // 	  np = np->next;
      // 	}
	
      // 	//	int k = 0;
      // 	// while(k < lp.size()) {
      // 	//   if (lp[k]->nSpline == i) {
      // 	//     my = lp[k]->y;
      // 	//     k = lp.size();
      // 	//   }
      // 	//   k++;
      // 	// } 
	
      // 	x.push_back(mxy->mapxy[j]->x);
      // 	y.push_back((ze > on ? 0 : 1));
	
      // }
      
      // if (x.size() > 2) {
      // 	tk::spline s(x, y);
      // 	vaddsplines[q][i] = s;
      // } else {
      // 	tk::spline s({0, 1, 2}, {y[0], y[0], y[0]});
      // 	vaddsplines[q][i] = s;
      // }
    }
  }

  double bestTEACC = 0;
    
  lr = 0.001;

  printf("Training...\n");
  // initialize predictor weights
  for (int q = 0; q < MNIST_NUM_CLASSES; q++)
    for (int i = 0; i < d; i++){
      a[q][i] = 0; // (rand() / (double) RAND_MAX) - 0.5; //1 / (double) ( q * d );
      a[q][d + i] = 0;
    }
  
  b = 0;

  int epoch = 1;
  while(epoch < 20) {
    // iter++;
    for (int i = 0; i < trs / bs; i++) {
      double BACC = Training(vsplines, i, matXtr, vecYtr);
      printf("BATCH_ACC = %lf, ", BACC);
      printf("To finish = %lf\n", (double) i / (double) (trs / (double) bs));
      fflush(stdout);
    }
      
    double TEACC = Test(vsplines, matXte, vecYte);      
    printf("TEACC = %lf, lr = %lf\n", TEACC, lr);
    fflush(stdout);

    if (TEACC > bestTEACC) {
      bestTEACC = TEACC;
      printf("IMPROVED TEACC = %lf\n", TEACC);
      fflush(stdout);
    }
      
    printf("epoch = %d\n", epoch);
      
    epoch++;
  }

  printf("bestTEACC = %lf\n", bestTEACC);
  fflush(stdout);
    
  return 0;
}
