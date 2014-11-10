#include "stdafx.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/features2d.hpp>

using namespace cv;

#define DICTIONARY_BUILD 1 // set DICTIONARY_BUILD 1 to do Step 1, otherwise it goes to step 2

int main(int argc, char* argv[])
{

#if DICTIONARY_BUILD == 1

	string dir = "/TRAIN", filepath;
	DIR *dp;
	struct dirent *dirp;
	struct stat filestat;

	dp = opendir( dir.c_str() );


}