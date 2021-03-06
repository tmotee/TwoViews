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

const string inputDir = "../../MiddEval3/testH/Crusade";

// Number of samples in each direction from the middle of a patch.
const int patchSteps = 2;
// Number of samples along one edge of a patch.
const int patchSize = 2 * patchSteps + 1;
const int samplesPerPatch = patchSize * patchSize;
// Approx image space pixels between sample points. A value slightly greater
// than 1 seems to work well, as each sample interpolates additional information
// from neighbors.
const float samplePitch = 1.6f;

// Gap (in multiples of samplePitch) to add between a patch and its children
// during the expansion step.
const float expansionGap = 1.0f;

// Maximum number of feature points per image to use for initial triangulation.
const int maxKeypoints = 10000;

// Tolerance (pixels) for matching features between images.
const float epipolarTolerance = 2;
// The extreme percentiles of disparity values to discard.
const float disparityOutlierPct = 0.01f;
// The upper percentile of reprojection error values to discard.
const float reprojectionOutlierPct = 0.1f;

// A patch is an oriented rectangular grid of sample points.
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

// For format specification, see: https://vision.middlebury.edu/stereo/data/scenes2014/
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

// Parse a floating point value given as a string "some-attribute=123.45"
float parseFloat(string line) {
    float value;
    sscanf_s(line.c_str(), "%*[^=]=%f", &value);
    return value;
}

bool readInputFiles(Mat &leftImage, Mat &rightImage, Mat& leftCameraMatrix, Mat& rightCameraMatrix, float &baseline) {
    leftImage = imread(samples::findFile(inputDir + "/im0.png"), IMREAD_COLOR);
    rightImage = imread(samples::findFile(inputDir + "/im1.png"), IMREAD_COLOR);
    if(leftImage.empty() || rightImage.empty())
    {
        cout << "Cannot read image files" << endl;
        return false;
    }

    ifstream calibFile(inputDir + "/calib.txt");
    if (!calibFile.is_open()) {
        cout << "Cannot read calibration" << endl;
        return false;
    }

    string line;
    getline(calibFile, line);
    leftCameraMatrix = parseCameraMatrix(line);
    getline(calibFile, line);
    rightCameraMatrix = parseCameraMatrix(line);

    cout << "cam0:" << endl << leftCameraMatrix << endl << "cam1:" << endl << rightCameraMatrix << endl;

    getline(calibFile, line); // Skip one line
    getline(calibFile, line);
    baseline = parseFloat(line);
    cout << "Camera baseline: " << baseline << endl;

    return true;
}

void detectAndMatchFeatures(const Mat &leftImage, const Mat &rightImage, 
  vector<KeyPoint> &leftKeypoints, vector<KeyPoint> &rightKeypoints, vector<DMatch> &matches) {
    Ptr<FeatureDetector> detector = ORB::create(maxKeypoints);
    detector->detect(leftImage, leftKeypoints);
    detector->detect(rightImage, rightKeypoints);
    cout << "Detected " << leftKeypoints.size() << " kepoints in left image and " << rightKeypoints.size() << " in right image" << endl; 

    Mat leftDescriptors, rightDescriptors;
    detector->compute(leftImage, leftKeypoints, leftDescriptors);
    detector->compute(rightImage, rightKeypoints, rightDescriptors);

    cout << "Computed desciptors: " << leftDescriptors.size() << " and " << rightDescriptors.size() << endl;
    
    // TODO: Search along epipolar line
    Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_HAMMING);
    matcher->match(leftDescriptors, rightDescriptors, matches);
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

    auto alignment = [&](const DMatch &match) {
        const Point2f &p1 = keypoints1[match.queryIdx].pt;
        const Point2f &p2 = keypoints2[match.trainIdx].pt;
        return abs((p1.y - cy1) - (p2.y - cy2));
    };

    auto disparity = [&](const DMatch &match) {
        const Point2f &p1 = keypoints1[match.queryIdx].pt;
        const Point2f &p2 = keypoints2[match.trainIdx].pt;
        return (p1.x - cx1) - (p2.x - cx2);
    };

    vector<float> disparities;
    disparities.reserve(matches.size());
    for (DMatch match : matches) {
        if (abs(alignment(match)) > epipolarTolerance) { // Reject matches not along epipolar line
            continue;
        }
        disparities.push_back(disparity(match));
    }
    sort(disparities.begin(), disparities.end());
    float lowPct = disparities[(int)floor(disparities.size() * disparityOutlierPct)];
    float highPct = disparities[(int)floor(disparities.size() * (1 - disparityOutlierPct))];

    cout << "Keeping disparities between " << lowPct << " and " << highPct << endl;

    for (DMatch match : matches) {
        float d = disparity(match);
        if (abs(alignment(match)) > epipolarTolerance || d < lowPct || d > highPct) {
            continue;
        }

        const Point2f &p1 = keypoints1[match.queryIdx].pt;
        const Point2f &p2 = keypoints2[match.trainIdx].pt;
        float z = baseline * f / d;

        Vec3f center;
        center[0] = (p1.x - cx1) * z / f;
        center[1] = (p1.y - cy1) * z / f;
        center[2] = z;
        patches.push_back(Patch{center, 0, 0});
    }
}

// Generate sampling points in world space for the given patch.
void getSampleLocations(float f, const Patch &patch, Vec3f sampleLocations[samplesPerPatch]) {
    float pitch = sampleStepDist(patch, f);
    Vec3f patchX = patchXAxis(patch) * pitch;
    Vec3f patchY = patchYAxis(patch) * pitch;

    for (int i = -patchSteps; i <= patchSteps; i++) {
        for (int j = -patchSteps; j <= patchSteps; j++) {
            sampleLocations[(i + patchSteps) * patchSize + (j + patchSteps)] = patch.center + i * patchX + j * patchY;
        }
    }
}

// Sample pixel colors by projecting the given sample locations onto the given camera image.
void sample(const Mat &camMatrix, const Mat &image, float baseline, 
    const Vec3f sampleLocations[samplesPerPatch], Vec3f sampledColors[samplesPerPatch]) {
    Vec3f baselineOffset(baseline, 0, 0);
    Size imageSize = image.size();

    for (int i = 0; i < samplesPerPatch; i++) {
        const Vec3f &X = sampleLocations[i] - baselineOffset;
        Mat p = camMatrix * X;
        float u = p.at<float>(0, 0) / p.at<float>(2, 0);
        float v = p.at<float>(1, 0) / p.at<float>(2, 0);

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

        sampledColors[i] = image.at<Vec3b>((int)floor(v1), (int)floor(u1)) * (1 - b) * (1 - a) + 
            image.at<Vec3b>((int)floor(v2), (int)floor(u1)) * b * (1 - a) + 
            image.at<Vec3b>((int)floor(v1), (int)floor(u2)) * (1 - b) * a + 
            image.at<Vec3b>((int)floor(v2), (int)floor(u2)) * b * a;
    }
}

// Calculates the left/right difference in pixel colors for the sample points on a patch by
// projecting them onto the left and right images.
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

        double result = 0;
        for (int i = 0; i < samplesPerPatch; i++) {
            result += norm(sampledColorsLeft[i] - sampledColorsRight[i]);
        }

        return result;
    }

    double calc(const double* x) const {
        Patch patch{Vec3f((float)x[0], (float)x[1], (float)x[2]), (float)x[3], (float)x[4]};
        return calc(patch);        
    }
};

// Set up and run nonlinear optimization on a single patch, updating the patch with
// the final optimized result.
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

    optimizer->minimize(x);

    patch.center[0] = (float)x.at<double>(0, 0);
    patch.center[1] = (float)x.at<double>(1, 0);
    patch.center[2] = (float)x.at<double>(2, 0);
    patch.alpha = (float)x.at<double>(3, 0);
    patch.beta = (float)x.at<double>(4, 0);
}

// Return a new list of patches after removing patches that have high reprojection error.
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

// Generate new patches close to the given ones, under the assumption that the surface is
// locally flat.
vector<Patch> expandPatches(vector<Patch> patches, float f) {
    vector<Patch> candidatePatches;
    candidatePatches.reserve(patches.size() * 8);
    
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
                return false;
            } 
        }
        for (Patch added : newPatches) {
            if (norm(candidate.center - added.center) < minDist) {
                return false;
            } 
        }

        return true;
    };

    for (Patch candidate : candidatePatches) {
        if (sufficientlySpaced(candidate)) {
            newPatches.push_back(candidate);
        }
    }
    return newPatches;
}

// Export patches as a point cloud file.
void exportPLY(const Mat &leftCameraMatrix, const Mat &leftImage, vector<Patch> patches, ofstream &outFile) {
    float f = leftCameraMatrix.at<float>(0, 0);

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
        // TODO: We could sample both images and take an average.
        sample(leftCameraMatrix, leftImage, 0, sampleLocations, sampledColors);

        for (int i = 0; i < samplesPerPatch; i++) {
            const Vec3f &X = sampleLocations[i];
            const Vec3b &c = sampledColors[i];
            outFile << X[0] << " " << -X[1] << " " << -X[2] << " " << (int)round(c[2]) << " " << (int)round(c[1]) <<  " " << (int)round(c[0]) << endl;
        }
    }
}

int main( int argc, const char** argv )
{
    Mat leftImage, rightImage, leftCameraMatrix, rightCameraMatrix;
    float baseline;
    if (!readInputFiles(leftImage, rightImage, leftCameraMatrix, rightCameraMatrix, baseline)) {
        return -1;
    }
    // Assuming focal lengths are the same for both cameras and in both dimensions
    float f = leftCameraMatrix.at<float>(0, 0);

    vector<KeyPoint> leftKeypoints, rightKeypoints;
    vector<DMatch> matches;
    detectAndMatchFeatures(leftImage, rightImage, leftKeypoints, rightKeypoints, matches);
    cout << "Found " << matches.size() << " keypoint matches" << endl;

    vector<Patch> patches;
    triangulatePatches(leftCameraMatrix, leftKeypoints, rightCameraMatrix, rightKeypoints, matches, baseline, patches);
    cout << "Triangulated " << patches.size() << " points" << endl;

    auto exportWithName = [&](const vector<Patch> &toExport, string filename) {
        ofstream outFile(filename);
        if (!outFile.is_open()) {
            cout << "Couldn't open output file " << filename << " for writing" << endl;
            return;
        }
        exportPLY(leftCameraMatrix, leftImage, toExport, outFile);
        outFile.close();
        cout << "Exported to " << filename << endl;
    };

    exportWithName(patches, "initial.ply");

    Ptr<DownhillSolver> optimizer = DownhillSolver::create();
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

    exportWithName(expandedPatches, "expanded.ply");

    optimizePatches(expandedPatches);

    expandedPatches = filterPatches(reprojErr, expandedPatches);
    patches.insert(patches.end(), expandedPatches.begin(), expandedPatches.end());
    cout << "Filtered to " << patches.size() << " remaining patches" << endl;

    exportWithName(patches, "optimized2.ply");

    return 0;
}
