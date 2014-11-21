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
	int dictionarySize = 180;

	char buf[255];
	ifstream ifs("trainpaths.txt");
	
	string trainPaths[22];
	string testPaths[22];
	
	short j = 0;
	while(!ifs.eof())
	{
		ifs.getline(buf, 255);
		string line(buf);
		if (line.compare("") != 0)
		{
			line = line + "/";
			// istringstream iss(line);
			trainPaths[j] = line;
			j++;
			cout << line << endl;
		}

	}
	ifs.close();

	ifs.open("testpaths.txt");
	j = 0;
	while(!ifs.eof())
	{
		ifs.getline(buf, 255);
		string line(buf);
		if (line.compare("") != 0)
		{
			// line = line + "/";
			// istringstream iss(line);
			testPaths[j] = line;
			j++;
			cout << line << endl;
		}

	}
	ifs.close();
	


if (flag == 1)
{
	cout << argv[1] << "\n";

	string dir = "train/", filepath, imgpath;
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

	// while (dirp = readdir( dp ))
	// {
	for(int i = 0; i < 21; i++)
	{
		filepath = trainPaths[i];
		// filepath = dir + dirp->d_name;
		cout << filepath << endl;

		dp1 = opendir( filepath.c_str() );

		while (dirp1 = readdir( dp1 ))
		{

			// filepath = dir + dirp->d_name;
			cout << filepath << endl;

			imgpath = filepath+dirp1->d_name;
			if (dirp1->d_type != isFile) continue;
			
			cout << imgpath << endl;

			img = imread(imgpath);
			detector.detect(img, keypoints);
			extractor->compute(img, keypoints, descriptors);
			
			training_descriptors.push_back(descriptors);

		}
		closedir(dp1);
	}

	cout << endl;
	closedir( dp );

	cout << "Total descriptors: " << training_descriptors.rows << endl;

	BOWKMeansTrainer bowtrainer(180); //num clusters
	bowtrainer.add(training_descriptors);
	cout << "cluster BOW features" << endl;
	dictionary = bowtrainer.cluster();
	
	FileStorage fs("dictionary.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();
}
else
{
	string dir = "train/", filepath, imgpath;
	DIR *dp, *dp1;
	struct dirent *dirp, *dirp1;
	// dp = opendir( dir.c_str() );


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
	Ptr<DescriptorExtractor > extractor(new SurfDescriptorExtractor());//  extractor;


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
 //    while (dirp = readdir( dp ))
	// {
	for(int i = 0; i < 21; i++){
		filepath = trainPaths[i];
		// filepath = dir + dirp->d_name;
		// cout << filepath << endl;

		dp1 = opendir( filepath.c_str() );

		while (dirp1 = readdir( dp1 ))
		{
			if (dirp1->d_type != isFile) continue;

			// filepath = dir + dirp->d_name;
			cout << filepath << endl;
			imgpath = filepath  +dirp1->d_name;
			cout << "img: " << imgpath << endl;

			img = imread(imgpath);
			if(img.data )                              // Check for invalid input
			{
				// cout << "label: " << dirp->d_name << endl;
			    detector.detect(img, keypoints);
				bowide.compute(img, keypoints, bowDescriptor);
				if(!bowDescriptor.empty() ){
					labels.push_back((float)i);
					trainingData.push_back(bowDescriptor);
				}
				
				
				
			}
		}
		closedir(dp1);
		
	}  

	// closedir( dp );

    
	cout << "train data: " <<  bowDescriptor.rows << " " << bowDescriptor.cols << endl;
	cout << "train data: " <<  trainingData.rows << " " << trainingData.cols << endl;
	cout << "labels data: " << labels.rows << " " << labels.cols << endl;
	
 	//Setting up SVM parameters
	CvSVMParams params;
	params.kernel_type = CvSVM::RBF;
	params.svm_type = CvSVM::C_SVC;
	params.gamma = 0.40425000000000009;
	params.C = 299.50000000000000;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER,100,0.000001);
	CvSVM svm;


	printf("%s\n","Training SVM classifier");

	bool res = svm.train(trainingData, labels, cv::Mat(), cv::Mat(), params);

	cout << "Processing evaluation data..." << endl;

	Mat groundTruth(0, 1, CV_32FC1);
	Mat evalData(0, dictionarySize, CV_32FC1);
	vector<KeyPoint> keypoints2;
	Mat bowDescriptor2;

	Mat results(0, 1, CV_32FC1);
	Mat img2;

	dir = "test/";

	// dp = opendir( dir.c_str() );
 //    while (dirp = readdir( dp ))
	// {
	for(int i = 0; i < 21; i++){
		filepath = testPaths[i];
		// filepath = dir + dirp->d_name;
		// cout << filepath << endl;

		dp1 = opendir( filepath.c_str() );

		while (dirp1 = readdir( dp1 ))
		{
			if (dirp1->d_type != isFile) continue;

			// filepath = testPaths[i] + dirp->d_name;
			cout << filepath << endl;
			imgpath = filepath + "/" + dirp1->d_name;
			cout << "img: " << imgpath << endl;

			img2 = imread(imgpath);
			if(img2.data )                              // Check for invalid input
			{
				// cout << "label: " << dirp->d_name << endl;
			    detector.detect(img2, keypoints2);
				bowide.compute(img2, keypoints2, bowDescriptor2);
				if(!bowDescriptor2.empty()){
					cout << "galo" << endl;

					evalData.push_back(bowDescriptor2);
					groundTruth.push_back((float) i);
					float response = svm.predict(bowDescriptor2);
					results.push_back(response);
				}
			}
			    

		}

		closedir(dp1);
	} 

	// closedir( dp );

	cout << " evalData.rows: " <<  evalData.rows << endl;

	cout << "(double) countNonZero(groundTruth - results): " << (double) countNonZero(groundTruth - results) << endl;

	double errorRate = (double) countNonZero(groundTruth - results) / evalData.rows;
	printf("%s%f","Error rate is ",errorRate);
	cout << endl;

}

return 0;
}