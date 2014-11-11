#include <stdio.h>
#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <fstream>
#include <iostream>
#include <string>

#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace cv;
using namespace std;

// #define DICTIONARY_BUILD 1 // set DICTIONARY_BUILD 1 to do Step 1, otherwise it goes to step 2

int main(int argc, char* argv[])
{
	
	short flag = atoi(argv[1]);
	Mat dictionary;
if (flag == 1)
{
	cout << argv[1] << "\n";

	string dir = "TRAIN/", filepath, imgpath;
	DIR *dp;

	struct dirent *dirp, *dirp1;
	struct stat filestat;

	dp = opendir( dir.c_str() );

	// detecting keypoints
	SurfFeatureDetector detector(1000);
	vector<KeyPoint> keypoints;	

	// computing descriptors
	Ptr<DescriptorExtractor > extractor(new SurfDescriptorExtractor());//  extractor;
	Mat descriptors;
	Mat training_descriptors(1,extractor->descriptorSize(),extractor->descriptorType());
	Mat img;

	DIR* dp1;

	unsigned char isFile =0x8;

	cout << "------- build vocabulary ---------\n";

	cout << "extract descriptors.."<<endl;
	int count = 0;

	while (dirp = readdir( dp ))
	{
		filepath = dir + dirp->d_name;
		cout << filepath << endl;

		dp1 = opendir( filepath.c_str() );

		while (dirp1 = readdir( dp1 ))
		{

			filepath = dir + dirp->d_name;
			cout << filepath << endl;

			imgpath = filepath + "/"+dirp1->d_name;
			if (dirp1->d_type != isFile) continue;
			
			cout << imgpath << endl;

			img = imread(imgpath);
			detector.detect(img, keypoints);
			extractor->compute(img, keypoints, descriptors);
			
			training_descriptors.push_back(descriptors);



		}
		closedir(dp1);
		count ++;
	}

	cout << endl;
	closedir( dp );

	cout << "Total descriptors: " << training_descriptors.rows << endl;

	BOWKMeansTrainer bowtrainer(21); //num clusters
	bowtrainer.add(training_descriptors);
	cout << "cluster BOW features" << endl;
	dictionary = bowtrainer.cluster();
	
	FileStorage fs("dictionary.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();
	// bowtrainer.cluster();
}
else
{


    //prepare BOW descriptor extractor from the dictionary    
    Mat dictionary; 
    FileStorage fs("dictionary.yml", FileStorage::READ);
    fs["vocabulary"] >> dictionary;
    fs.release();    
    
    //create a nearest neighbor matcher
    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    //create Sift feature point extracter
    Ptr<FeatureDetector> detector(new SurfFeatureDetector());

    //create Sift descriptor extractor
    Ptr<DescriptorExtractor> extractor(new SurfDescriptorExtractor);  

    //create BoF (or BoW) descriptor extractor
    BOWImgDescriptorExtractor bowide(extractor,matcher);
    //Set the dictionary with the vocabulary we created in the first step
    bowide.setVocabulary(dictionary);
 
    //To store the image file name
    char * filename = new char[100];
    //To store the image tag name - only for save the descriptor in a file
    char * imageTag = new char[10];
 
    //open the file to write the resultant descriptor
    FileStorage fs1("descriptor.yml", FileStorage::WRITE);    
    
    //the image file with the location. change it according to your image file location
    sprintf(filename,"river08.tif");        
    //read the image
    Mat img = imread(filename,CV_LOAD_IMAGE_GRAYSCALE);        
    //To store the keypoints that will be extracted by SIFT
    vector<KeyPoint> keypoints;        
    //Detect SIFT keypoints (or feature points)
    detector->detect(img,keypoints);
    //To store the BoW (or BoF) representation of the image
    Mat bowDescriptor;        
    //extract BoW (or BoF) descriptor from given image
    bowide.compute(img,keypoints,bowDescriptor);
 
    //prepare the yml (some what similar to xml) file
    sprintf(imageTag,"img1");            
    //write the new BoF descriptor to the file
    fs1 << imageTag << bowDescriptor;        
 
    //You may use this descriptor for classifying the image.
            
    //release the file storage
    fs1.release();

}

return 0;
}