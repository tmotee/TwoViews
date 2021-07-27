#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <regex>

using namespace cv;
using namespace std;

string inputDir = "../../MiddEval3/testH/Crusade";

const int patchSteps = 2;
const int patchSize = 2 * patchSteps + 1;
const int samplesPerPatch = patchSize * patchSize;
const float samplePitch = 1.6f; // Approx pixels per step

const float expansionGap = 1.0f; // Multiples of the step size

const int maxKeypoints = 10000;

const float epipoleTolerance = 2;
const float disparityOutlierPct = 0.01f;
const float reprojectionOutlierPct = 0.1f;

const char* window_name = "TESTING WINDOW";

Ptr<DownhillSolver> optimizer = DownhillSolver::create();

struct Patch {
    Vec3f center;
    // Normal direction of the patch encoded as Euler angles (y-x-z) but with zero
    // rotation about the z axis, to reduce degrees of freedom for the optimizer.
    // We don't expect z axis rotation to have much effect on the reprojection.
    float alpha, beta;
};

Vec3f patchXAxis(const Patch &patch) {
    return Vec3f(cos(patch.alpha), 0, -sin(patch.alpha));
}

Vec3f patchYAxis(const Patch &patch) {
    return Vec3f(sin(patch.beta) * sin(patch.alpha), cos(patch.beta), sin(patch.beta) * cos(patch.alpha));
}

float sampleStepDist(const Patch &patch, float f) {
    return samplePitch * patch.center[2] / f;
}

Mat parseCameraMatrix(string line) {
    size_t start = line.find('[') + 1;
    size_t end = line.rfind(']') - 1;
    string nums = line.substr(start, end - start);
    Mat cameraMatrix(3, 3, CV_32F, Scalar(0.0));
    sscanf_s(nums.c_str(), "%f %*f %f; %*f %f %f;", 
        &cameraMatrix.at<float>(0, 0), &cameraMatrix.at<float>(0, 2),
        &cameraMatrix.at<float>(1, 1), &cameraMatrix.at<float>(1, 2));
    cameraMatrix.at<float>(2, 2) = 1; 
    return cameraMatrix;
}

float parseFloat(string line) {
    float value;
    sscanf_s(line.c_str(), "%*[^=]=%f", &value);
    return value;
}

void triangulatePatches(const Mat &camMatrix1, const vector<KeyPoint> &keypoints1, 
    const Mat &camMatrix2, const vector<KeyPoint> &keypoints2, 
    const vector<DMatch> &matches, float baseline, vector<Patch> &patches) {
    // Assuming focal lengths are the same for both cameras and in both dimensions
    float f = camMatrix1.at<float>(0, 0);
    float cx1 = camMatrix1.at<float>(0, 2);
    float cy1 = camMatrix1.at<float>(1, 2);
    float cx2 = camMatrix2.at<float>(0, 2);
    float cy2 = camMatrix2.at<float>(1, 2);

    vector<float> disparities;
    disparities.reserve(matches.size());
    for (DMatch match : matches) {
        const Point2f &p1 = keypoints1[match.queryIdx].pt;
        const Point2f &p2 = keypoints2[match.trainIdx].pt;
        if (abs((p1.y - cy1) - (p2.y - cy2)) > epipoleTolerance) {
            continue;
        }
        disparities.push_back((p1.x - cx1) - (p2.x - cx2));
    }
    sort(disparities.begin(), disparities.end());
    float lowPct = disparities[(int)floor(disparities.size() * disparityOutlierPct)];
    float highPct = disparities[(int)floor(disparities.size() * (1 - disparityOutlierPct))];

    cout << "Keeping disparities between " << lowPct << " and " << highPct << endl;

    for (DMatch match : matches) {
        const Point2f &p1 = keypoints1[match.queryIdx].pt;
        const Point2f &p2 = keypoints2[match.trainIdx].pt;

        // Reject matches not along epipole
        if (abs((p1.y - cy1) - (p2.y - cy2)) > epipoleTolerance) {
            //cout << "Exceeded tolerance: " << p1 << ", " << p2 << " with cy1=" << cy1 << ", cy2=" << cy2 << endl;
            continue;
        }

        // Images are rectified, so depth is given by stereo disparity
        float disparity = (p1.x - cx1) - (p2.x - cx2);
        if (disparity < lowPct || disparity > highPct) { // Reject outliers
            continue;
        }
        //cout << "Disparity between " << p1 << " and " << p2 << " is " << disparity << endl;
        float depth = baseline * f / disparity;

        Vec3f center;
        center[0] = (p1.x - cx1) * depth / f;
        center[1] = (p1.y - cy1) * depth / f;
        center[2] = depth;
        patches.push_back(Patch{center, 0, 0});

        //cout << "Pixel " << p1 << " triangulated to " << center << endl;
    }
}

void getSampleLocations(float f, const Patch &patch, Vec3f sampleLocations[samplesPerPatch]) {
    float pitch = sampleStepDist(patch, f);
    Vec3f patchX = patchXAxis(patch) * pitch;
    Vec3f patchY = patchYAxis(patch) * pitch;

    // cout << "Using patch step " << step << " and patchX " << patchX << " and patchY " << patchY << endl;
    // cout << "Patch center is " << patch.center << endl;
    for (int i = -patchSteps; i <= patchSteps; i++) {
        for (int j = -patchSteps; j <= patchSteps; j++) {
            sampleLocations[(i + patchSteps) * patchSize + (j + patchSteps)] = patch.center + i * patchX + j * patchY;
        }
    }
}

void sample(const Mat &camMatrix, const Mat &image, float baseline, 
    const Vec3f sampleLocations[samplesPerPatch], Vec3f sampledColors[samplesPerPatch]) {
    Vec3f baselineOffset(baseline, 0, 0);
    Size imageSize = image.size();

    for (int i = 0; i < samplesPerPatch; i++) {
        const Vec3f &X = sampleLocations[i] - baselineOffset;
        Mat p = camMatrix * X;
        float u = p.at<float>(0, 0) / p.at<float>(2, 0);
        float v = p.at<float>(1, 0) / p.at<float>(2, 0);

        //cout << "Point " << X << " projected to " << u << ", " << v << endl;

        // Linear interpolation
        float u1 = floor(u);
        float u2 = ceil(u);
        float v1 = floor(v);
        float v2 = ceil(v);
        float a = u - u1;
        float b = v - v1;

        // If bounds are exceeded, wrap around to other side of image. This should create
        // a large color difference when comparing the two views.
        if (u1 < 0) { u1 += imageSize.width; }
        else if (u1 >= imageSize.width) { u1 -= imageSize.width; }

        if (u2 < 0) { u2 += imageSize.width; }
        else if (u2 >= imageSize.width) { u2 -= imageSize.width; }

        if (v1 < 0) { v1 += imageSize.height; }
        else if (v1 >= imageSize.height) { v1 -= imageSize.height; }

        if (v2 < 0) { v2 += imageSize.height; }
        else if (v2 >= imageSize.height) { v2 -= imageSize.height; }

        //cout << "Interp pixels " << u1 << "," << v1 << "," << u2 << "," << v2 << " a = " << a << " b = " << b << endl;

        sampledColors[i] = image.at<Vec3b>((int)floor(v1), (int)floor(u1)) * (1 - b) * (1 - a) + 
            image.at<Vec3b>((int)floor(v2), (int)floor(u1)) * b * (1 - a) + 
            image.at<Vec3b>((int)floor(v1), (int)floor(u2)) * (1 - b) * a + 
            image.at<Vec3b>((int)floor(v2), (int)floor(u2)) * b * a;
            //cout << "Point " << patch.center << " projected to " << p1 << endl;
    }
}

class ReprojectionErrorF:public MinProblemSolver::Function{
    const Mat &leftCamMatrix, &leftImage, &rightCamMatrix, &rightImage;
    float baseline, f;
public: 
    ReprojectionErrorF(const Mat &lm, const Mat &li, const Mat &rm, const Mat &ri, float bl) 
    : leftCamMatrix(lm), leftImage(li), rightCamMatrix(rm), rightImage(ri), baseline(bl) { 
        f = leftCamMatrix.at<float>(0, 0);
    }

    int getDims() const { 
        return 5; 
    }

    double calc(const Patch &patch) const {
        Vec3f sampleLocations[samplesPerPatch];
        Vec3f sampledColorsLeft[samplesPerPatch];
        Vec3f sampledColorsRight[samplesPerPatch];

        getSampleLocations(f, patch, sampleLocations);
        sample(leftCamMatrix, leftImage, 0, sampleLocations, sampledColorsLeft);
        sample(rightCamMatrix, rightImage, baseline, sampleLocations, sampledColorsRight);

        //cout << "Sampled left: " << endl;
        // for (int i = 0; i < samplesPerPatch; i++) {
        //     cout << sampledColorsLeft[i] << endl;
        // }
        //cout << "Sampled right: " << endl;
        // for (int i = 0; i < samplesPerPatch; i++) {
        //     cout << sampledColorsRight[i] << endl;
        // }

        double result = 0;
        for (int i = 0; i < samplesPerPatch; i++) {
            result += norm(sampledColorsLeft[i] - sampledColorsRight[i]);
            //cout << "Vector norm " << i << " = " << sqrt(sampledColorsLeft[i].dot(sampledColorsRight[i])) << endl;
        }

        //cout << "Calc result: " << result << endl;
        return result;
    }

    double calc(const double* x) const {
        Patch patch{Vec3f((float)x[0], (float)x[1], (float)x[2]), (float)x[3], (float)x[4]};
        return calc(patch);        
    }
};

void optimizePatch(Ptr<DownhillSolver> optimizer, float f, Patch &patch) {
    float linearStep = patch.center[2] / f;
    float depthStep = patch.center[2] / 1000;
    Mat initStep(5, 1, CV_64F);
    initStep.at<double>(0,0) = linearStep;
    initStep.at<double>(1,0) = linearStep;
    initStep.at<double>(2,0) = depthStep;
    initStep.at<double>(3,0) = 0.1;
    initStep.at<double>(4,0) = 0.1;
    optimizer->setInitStep(initStep);

    Mat x(5, 1, CV_64F);
    x.at<double>(0, 0) = patch.center[0];
    x.at<double>(1, 0) = patch.center[1];
    x.at<double>(2, 0) = patch.center[2];
    x.at<double>(3, 0) = patch.alpha;
    x.at<double>(4, 0) = patch.beta;

    // cout << "Optimizing patch initial " << x << endl;
    optimizer->minimize(x);
    // cout << "Final " << x << endl;

    patch.center[0] = (float)x.at<double>(0, 0);
    patch.center[1] = (float)x.at<double>(1, 0);
    patch.center[2] = (float)x.at<double>(2, 0);
    patch.alpha = (float)x.at<double>(3, 0);
    patch.beta = (float)x.at<double>(4, 0);
}

vector<Patch> filterPatches(ReprojectionErrorF errFn, vector<Patch> patches) {
    vector<double> reprojErrs;
    reprojErrs.reserve(patches.size());

    for (Patch patch : patches) {
        reprojErrs.push_back(errFn.calc(patch));
    }
    sort(reprojErrs.begin(), reprojErrs.end());
    double highPct = reprojErrs[(int)floor(reprojErrs.size() * (1 - reprojectionOutlierPct))];

    vector<Patch> filteredPatches;
    for (Patch patch : patches) {
        if (errFn.calc(patch) <= highPct) {
            filteredPatches.push_back(patch);
        }
    }

    // TODO: Neighborhood "flatness" constraints? Could try a PCA on the positions of
    // nearby points to try and determine the principal plane (if any) and reject off-plane
    // points.

    return filteredPatches;
}

vector<Patch> expandPatches(vector<Patch> patches, float f) {
    vector<Patch> candidatePatches;
    candidatePatches.reserve(patches.size() * 8);
    
    // Generate a new patch in each of the eight cardinal directions from the originals
    for (Patch patch : patches) {
        float offset = sampleStepDist(patch, f) * (patchSize + expansionGap);
        Vec3f X = patchXAxis(patch) * offset, Y = patchYAxis(patch) * offset;

        candidatePatches.push_back(Patch{patch.center + X + Y, patch.alpha, patch.beta});
        candidatePatches.push_back(Patch{patch.center + X, patch.alpha, patch.beta});
        candidatePatches.push_back(Patch{patch.center + X - Y, patch.alpha, patch.beta});
        candidatePatches.push_back(Patch{patch.center - Y, patch.alpha, patch.beta});
        candidatePatches.push_back(Patch{patch.center - X - Y, patch.alpha, patch.beta});
        candidatePatches.push_back(Patch{patch.center - X, patch.alpha, patch.beta});
        candidatePatches.push_back(Patch{patch.center - X + Y, patch.alpha, patch.beta});
        candidatePatches.push_back(Patch{patch.center + Y, patch.alpha, patch.beta});
    }

    // Filter out patches that are too close to an existing one
    // TODO: space partitioning to make this efficient
    vector<Patch> newPatches;
    auto sufficientlySpaced = [&](const Patch& candidate) {
        float minDist = sampleStepDist(candidate, f) * patchSize;
        
        for (Patch original : patches) {
            if (norm(candidate.center - original.center) < minDist) {
                // cout << "Candidate too close to original at " << original.center << endl;
                return false;
            } 
        }
        for (Patch added : newPatches) {
            if (norm(candidate.center - added.center) < minDist) {
                // cout << "Candidate too close to added at " << added.center << endl;
                return false;
            } 
        }

        return true;
    };

    for (Patch candidate : candidatePatches) {
        // cout << "Checking patch at " << candidate.center << endl;
        if (sufficientlySpaced(candidate)) {
            newPatches.push_back(candidate);
        }
    }
    return newPatches;
}

void exportPLY(const Mat &camMatrix1, const Mat &image1, const Mat &camMatrix2, const Mat &image2, 
    float baseline, vector<Patch> patches, ofstream &outFile) {
    float f = camMatrix1.at<float>(0, 0);

    outFile << "ply" << endl
    << "format ascii 1.0" << endl
    << "element vertex " << samplesPerPatch * patches.size() << endl
    << "property float x" << endl
    << "property float y" << endl
    << "property float z" << endl
    << "property uchar red" << endl
    << "property uchar green" << endl
    << "property uchar blue" << endl
    << "end_header" << endl;

    Vec3f sampledColors[samplesPerPatch];
    Vec3f sampleLocations[samplesPerPatch];
    for (const Patch &patch : patches) {
        getSampleLocations(f, patch, sampleLocations);
        //sample(camMatrix1, image1, 0, sampleLocations, sampledColors);
        sample(camMatrix2, image2, baseline, sampleLocations, sampledColors);

        for (int i = 0; i < samplesPerPatch; i++) {
            const Vec3f &X = sampleLocations[i];
            const Vec3b &c = sampledColors[i];
            outFile << X[0] << " " << -X[1] << " " << -X[2] << " " << (int)round(c[2]) << " " << (int)round(c[1]) <<  " " << (int)round(c[0]) << endl;
        }
    }
}

int main( int argc, const char** argv )
{
    Mat leftImage, rightImage;
    leftImage = imread(samples::findFile(inputDir + "/im0.png"), IMREAD_COLOR);
    rightImage = imread(samples::findFile(inputDir + "/im1.png"), IMREAD_COLOR);
    if(leftImage.empty() || rightImage.empty())
    {
        cout << "Cannot read image files" << endl;
        return -1;
    }

    ifstream calibFile(inputDir + "/calib.txt");
    if (!calibFile.is_open()) {
        cout << "Cannot read calibration" << endl;
        return -1;
    }
    string line;
    getline(calibFile, line);
    Mat leftCameraMatrix = parseCameraMatrix(line);
    getline(calibFile, line);
    Mat rightCameraMatrix = parseCameraMatrix(line);
    cout << "Read cam0 " << leftCameraMatrix << " cam1 " << rightCameraMatrix << endl;
    // Assuming focal lengths are the same for both cameras and in both dimensions
    float f = leftCameraMatrix.at<float>(0, 0);

    getline(calibFile, line); // Skip one line
    getline(calibFile, line);
    float baseline = parseFloat(line);
    cout << "Read baseline " << baseline << endl;

    Ptr<FeatureDetector> detector = ORB::create(maxKeypoints);
    vector<KeyPoint> leftKeypoints, rightKeypoints;
    detector->detect(leftImage, leftKeypoints);
    detector->detect(rightImage, rightKeypoints);
    cout << "Detected " << leftKeypoints.size() << " kepoints in left image and " << rightKeypoints.size() << " in right image" << endl; 

    Mat leftDescriptors, rightDescriptors;
    detector->compute(leftImage, leftKeypoints, leftDescriptors);
    detector->compute(rightImage, rightKeypoints, rightDescriptors);

    cout << "Computed desciptors: " << leftDescriptors.size() << " and " << rightDescriptors.size() << endl;
    
    // TODO: Search along epipolar line
    Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_HAMMING);
    vector<DMatch> matches;
    matcher->match(leftDescriptors, rightDescriptors, matches);

    cout << "Found " << matches.size() << " matches" << endl;

    vector<Patch> patches;
    triangulatePatches(leftCameraMatrix, leftKeypoints, rightCameraMatrix, rightKeypoints, matches, baseline, patches);

    cout << "Triangulated " << patches.size() << " points" << endl;

    auto exportWithName = [&](const vector<Patch> &toExport, string filename) {
        ofstream outFile(filename);
        if (!outFile.is_open()) {
            cout << "Couldn't open output file for writing" << endl;
        }
        exportPLY(leftCameraMatrix, leftImage, rightCameraMatrix, rightImage, baseline, toExport, outFile);
        outFile.close();
        cout << "Exported to " << filename << endl;
    };

    exportWithName(patches, "initial.ply");

    ReprojectionErrorF reprojErr(leftCameraMatrix, leftImage, rightCameraMatrix, rightImage, baseline);
    optimizer->setFunction(makePtr<ReprojectionErrorF>(reprojErr));

    auto optimizePatches = [&](vector<Patch> &toOptimize) {
        for (int i = 0; i < toOptimize.size(); i++) {
            optimizePatch(optimizer, f, toOptimize[i]);
            cout << "\rOptimized patch " << i + 1 << " of " << toOptimize.size();
        }
        cout << endl;
    };

    optimizePatches(patches);

    patches = filterPatches(reprojErr, patches);
    cout << "Filtered to " << patches.size() << " remaining patches" << endl;

    exportWithName(patches, "optimized1.ply");

    vector<Patch> expandedPatches = expandPatches(patches, f);
    cout << "Expansion added " << expandedPatches.size() << " patches" << endl;
    //patches.insert(patches.end(), expandedPatches.begin(), expandedPatches.end());
    exportWithName(expandedPatches, "expanded.ply");

    optimizePatches(expandedPatches);

    expandedPatches = filterPatches(reprojErr, expandedPatches);
    patches.insert(patches.end(), expandedPatches.begin(), expandedPatches.end());
    cout << "Filtered to " << patches.size() << " remaining patches" << endl;

    exportWithName(patches, "optimized2.ply");

    Mat annotatedImage;
    //drawKeypoints(rightImage, rightKeypoints, annotatedImage);
    drawMatches(leftImage, leftKeypoints, rightImage, rightKeypoints, matches, annotatedImage);

    // // Create a window
    namedWindow(window_name, 1);
    imshow(window_name, annotatedImage);    
    waitKey(0);

    return 0;
}
