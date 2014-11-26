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

	int dictionarySize = 180;

	char buf[255];

	// reading trainning paths	
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
			trainPaths[j] = line;
			j++;
		}

	}
	ifs.close();

	// reading test paths	
	ifs.open("testpaths.txt");
	j = 0;
	while(!ifs.eof())
	{
		ifs.getline(buf, 255);
		string line(buf);
		if (line.compare("") != 0)
		{

			testPaths[j] = line;
			j++;
			// cout << line << endl;
		}

	}
	ifs.close();
	
// building the dictionary 
if (flag == 1)
{
	// cout << argv[1] << "\n";

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

	cout << "------- building vocabulary ---------\n";

	cout << "extracting descriptors.."<<endl;

	for(int i = 0; i < 21; i++)
	{
		filepath = trainPaths[i];

		dp1 = opendir( filepath.c_str() );

		while (dirp1 = readdir( dp1 ))
		{

			imgpath = filepath+dirp1->d_name;

			// checking files
			if (dirp1->d_type != isFile) continue;
			
			img = imread(imgpath);
			// detect feature points
			detector.detect(img, keypoints);
			// compute the descriptors for each keypoint
			extractor->compute(img, keypoints, descriptors);
			// put the all feature descriptors in a single Mat object 
			training_descriptors.push_back(descriptors);

		}
		closedir(dp1);
	}

	closedir( dp );

	cout << "Total descriptors: " << training_descriptors.rows << endl;
	
	// create the BoW trainer
	BOWKMeansTrainer bowtrainer(180); // num clusters
	bowtrainer.add(training_descriptors);
	cout << "cluster BOW features" << endl;

	// cluster the feature vectors
	dictionary = bowtrainer.cluster();
	
	FileStorage fs("dictionary.yml", FileStorage::WRITE);
	// store the vocabulary
	fs << "vocabulary" << dictionary;
	fs.release();
}

// training
else
{
	string dir = "train/", filepath, imgpath;
	DIR *dp, *dp1;
	struct dirent *dirp, *dirp1;

    // prepare BOW descriptor extractor from the dictionary    
    Mat dictionary; 
    FileStorage fs("dictionary.yml", FileStorage::READ);
    fs["vocabulary"] >> dictionary;
    fs.release();    
    
    // create a nearest neighbor matcher
    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);

    // create Surf feature point extracter
  	SurfFeatureDetector detector(1000);

    // create Surf descriptor extractor
	Ptr<DescriptorExtractor > extractor(new SurfDescriptorExtractor());

    // keypoints that will be extracted by Surf
    vector<KeyPoint> keypoints;  

    // create BoF descriptor extractor
    BOWImgDescriptorExtractor bowide(extractor,matcher);

    // set the dictionary with the vocabulary we created in the first step
    bowide.setVocabulary(dictionary);
 
	cout << endl <<"Extracting histograms in the form of BOW for each image "<< endl;

    Mat bowDescriptor; 
    Mat img;
    Mat labels(0, 1, CV_32FC1);
	Mat trainingData(0, dictionarySize, CV_32FC1);

	int count = 0;

	for(int i = 0; i < 21; i++){
		filepath = trainPaths[i];
		dp1 = opendir( filepath.c_str() );

		while (dirp1 = readdir( dp1 ))
		{
			// checking files
			if (dirp1->d_type != isFile) continue;

			imgpath = filepath  +dirp1->d_name;
			img = imread(imgpath);

			// check for invalid input
			if(img.data )       
			{
				// detect feature points
			    detector.detect(img, keypoints);
				bowide.compute(img, keypoints, bowDescriptor);
				if(!bowDescriptor.empty() ){
					// adding labels 
					labels.push_back((float)i);
					trainingData.push_back(bowDescriptor);
				}
				
				
				
			}
		}
		closedir(dp1);
		
	}  
	
 	// setting up SVM parameters
	CvSVMParams params;
	params.kernel_type = CvSVM::RBF;
	params.svm_type = CvSVM::C_SVC;
	params.gamma = 0.42545000000000009;
	params.C = 301.30000000000000;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER,100,0.000001);
	CvSVM svm;


	printf("%s\n","Training SVM classifier");

	bool res = svm.train(trainingData, labels, cv::Mat(), cv::Mat(), params);

	cout << endl << "Processing evaluation data..." << endl;
	cout << endl;

	Mat groundTruth(0, 1, CV_32FC1);
	Mat evalData(0, dictionarySize, CV_32FC1);
	vector<KeyPoint> keypoints2;
	Mat bowDescriptor2;

	Mat results(0, 1, CV_32FC1);
	Mat img2;

	dir = "test/";

	for(int i = 20; i < 21; i++){
		filepath = testPaths[i];
		dp1 = opendir( filepath.c_str() );

		while (dirp1 = readdir( dp1 ))
		{
			// checking files
			if (dirp1->d_type != isFile) continue;

			imgpath = filepath + "/" + dirp1->d_name;
			img2 = imread(imgpath);

			// check for invalid input
			if(img2.data )         
			{
				// detect feature points
			    detector.detect(img2, keypoints2);
				bowide.compute(img2, keypoints2, bowDescriptor2);
				if(!bowDescriptor2.empty()){

					evalData.push_back(bowDescriptor2);
					// right answer
					groundTruth.push_back((float) i);
					float response = svm.predict(bowDescriptor2);
					// predicted results
					results.push_back(response);
				}
			}
			    

		}
		closedir(dp1);
	} 

	// calculate the number of unmatched classes 
	double errorRate = (double) countNonZero(groundTruth - results) / evalData.rows;
	printf("%s%f","Error rate is ",errorRate);
	cout << endl << endl;

}

return 0;
}