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
unsigned char isFile =0x8;

int main(int argc, char* argv[])
{
	
	short flag = atoi(argv[1]);
	Mat dictionary;
	//Construct BOWKMeansTrainer
	//the number of bags
	int dictionarySize=200;


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



	//define Term Criteria
	TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
	//retries number
	int retries=1;
	//necessary flags
	int flags=KMEANS_PP_CENTERS;
	//Create the BoW (or BoF) trainer
	BOWKMeansTrainer bowtrainer(dictionarySize,tc,retries,flags);

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
	string dir = "TRAIN/", filepath, imgpath;
	DIR *dp, *dp1;
	struct dirent *dirp, *dirp1;
	dp = opendir( dir.c_str() );


    //prepare BOW descriptor extractor from the dictionary    
    Mat dictionary; 
    FileStorage fs("dictionary.yml", FileStorage::READ);
    fs["vocabulary"] >> dictionary;
    fs.release();    
    
    //create a nearest neighbor matcher
    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    //create Sift feature point extracter
   SurfFeatureDetector detector(1000);
    //create Sift descriptor extractor
    Ptr<DescriptorExtractor> extractor(new SurfDescriptorExtractor);  


    //To store the keypoints that will be extracted by SIFT
    vector<KeyPoint> keypoints;  

    //create BoF (or BoW) descriptor extractor
    BOWImgDescriptorExtractor bowide(extractor,matcher);
    //Set the dictionary with the vocabulary we created in the first step
    bowide.setVocabulary(dictionary);
 
    //To store the image file name
    char * filename = new char[100];
    //To store the image tag name - only for save the descriptor in a file
    char * imageTag = new char[10];
 
    //open the file to write the resultant descriptor
    // FileStorage fs1("descriptor.yml", FileStorage::WRITE);

    Mat bowDescriptor; 
    Mat img;
    Mat labels(0, 1, CV_32FC1);
	Mat trainingData(0, dictionarySize, CV_32FC1);

	int count = 0;
	int a = 0;
    while (dirp = readdir( dp ))
	{
		filepath = dir + dirp->d_name;
		// cout << filepath << endl;

		dp1 = opendir( filepath.c_str() );

		while (dirp1 = readdir( dp1 ))
		{
			if (dirp1->d_type != isFile) continue;

			filepath = dir + dirp->d_name;
			// cout << filepath << endl;
			imgpath = filepath + "/"+dirp1->d_name;
			// cout << "img: " << imgpath << endl;

			img = imread(imgpath);
			if(img.data )                              // Check for invalid input
			{
				
				// cout << "label: " << dirp->d_name << endl;
			    img = imread(imgpath);
			    detector.detect(img,keypoints);
				bowide.compute(img,keypoints,bowDescriptor);
				
				trainingData.push_back(bowDescriptor);
				labels.push_back((float)count);
			}


		}
		closedir(dp1);
		// count ++;
	}  
    
	cout << "count: " << count << endl;
	cout << "a: " << a << endl;

 	//Setting up SVM parameters
	CvSVMParams params;
	params.kernel_type=CvSVM::RBF;
	params.svm_type=CvSVM::C_SVC;
	params.gamma=0.50625000000000009;
	params.C=312.50000000000000;
	params.term_crit=cvTermCriteria(CV_TERMCRIT_ITER,100,0.000001);
	CvSVM svm;


	printf("%s\n","Training SVM classifier");

	// bool res = svm.train(trainingData, labels, cv::Mat(), cv::Mat(), params);

	// cout<<"Processing evaluation data..."<<endl;

	// Mat groundTruth(0, 1, CV_32FC1);
	// Mat evalData(0, dictionarySize, CV_32FC1);
	// vector<KeyPoint> keypoints2;
	// Mat bowDescriptor2;

	// Mat results(0, 1, CV_32FC1);
	// Mat img2;

 //    while (dirp = readdir( dp ))
	// {
	// 	filepath = dir + dirp->d_name;
	// 	// cout << filepath << endl;

	// 	dp1 = opendir( filepath.c_str() );

	// 	while (dirp1 = readdir( dp1 ))
	// 	{
	// 		if (dirp1->d_type != isFile) continue;

	// 		filepath = dir + dirp->d_name;
	// 		// cout << filepath << endl;
	// 		imgpath = filepath + "/"+dirp1->d_name;
	// 		// cout << "img: " << imgpath << endl;

	// 		img = imread(imgpath);
	// 		if(img.data )                              // Check for invalid input
	// 		{
				
	// 			// cout << "label: " << dirp->d_name << endl;
	// 		    img2 = imread(imgpath);
	// 		    detector.detect(img2,keypoints2);
	// 			bowide.compute(img2,keypoints2,bowDescriptor2);
				
	// 			evalData.push_back(bowDescriptor2);
	// 			groundTruth.push_back((float) count);
	// 			float response = svm.predict(bowDescriptor2);
	// 			results.push_back(response);
	// 		}
			    

	// 	}
	// 	closedir(dp1);
	// 	count ++;
	// } 

	// double errorRate = (double) countNonZero(groundTruth- results) / evalData.rows;
	// printf("%s%f","Error rate is ",errorRate);

}

return 0;
}



//the image file with the location. change it according to your image file location
    // sprintf(filename,"river08.tif");        
    
    // //read the image
    // Mat img = imread(filename,CV_LOAD_IMAGE_GRAYSCALE);        
    // //To store the keypoints that will be extracted by SIFT
    // vector<KeyPoint> keypoints;        
    // //Detect SIFT keypoints (or feature points)
    // detector->detect(img,keypoints);
    // //To store the BoW (or BoF) representation of the image
    // Mat bowDescriptor;        
    // //extract BoW (or BoF) descriptor from given image
    // bowide.compute(img,keypoints,bowDescriptor);
 
    // //prepare the yml (some what similar to xml) file
    // sprintf(imageTag,"img1");            
    // //write the new BoF descriptor to the file
    // fs1 << imageTag << bowDescriptor;        
 
    // //You may use this descriptor for classifying the image.
            
    // //release the file storage
    // fs1.release();