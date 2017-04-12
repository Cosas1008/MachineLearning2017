import os
import cv2
import sys
import csv
import shutil
import random
import numpy as np

"""
Example Call:
    $> python2.7 eigenfaces.py att_faces demo_faces
"""
class Eigenfaces(object):
    faces_count = 3

    faces_dir = '.'                                                             # directory path to the  faces

    train_faces_count = 600                                                       # number of faces used for training
    test_faces_count = 400                                                        # number of faces used for testing

    l = train_faces_count * faces_count                                         # training images count
    m = 30                                                                      # number of columns of the image
    n = 30                                                                     # number of rows of the image
    mn = m * n                                                                  # length of the column vector

    """
    Initializing the Eigenfaces model.
    """
    def __init__(self, _faces_dir = '.', _energy = 0.85):
        print '> Initializing started'

        self.faces_dir = _faces_dir
        self.energy = _energy
        self.training_ids = []                                                  # train image id's for every  face

        L = np.empty(shape=(self.mn, self.l), dtype='float64')                  # each row of L represents one train image

        cur_img = 0
        for face_id in xrange(1, self.faces_count + 1):

            training_ids = random.sample(range(1, 1001), self.train_faces_count)  # the id's of the 6 random training images
            self.training_ids.append(training_ids)                              # remembering the training id's for later

            for training_id in training_ids:
                path_to_img = os.path.join(self.faces_dir,'Class' + str(face_id), 'faceTrain' + str(face_id) + '_' + str(training_id) + '.bmp')          # relative path
                #print '> reading file: ' + path_to_img

                img = cv2.imread(path_to_img, 0)                                # read a grayscale image
                img_col = np.array(img, dtype='float64').flatten()              # flatten the 2d image into 1d

                L[:, cur_img] = img_col[:]                                      # set the cur_img-th column to the current training image
                cur_img += 1

        self.mean_img_col = np.sum(L, axis=1) / self.l                          # get the mean of all images / over the rows of L

        for j in xrange(0, self.l):                                             # subtract from all training images
            L[:, j] -= self.mean_img_col[:]

        C = np.matrix(L.transpose()) * np.matrix(L)                             # instead of computing the covariance matrix as
        C /= self.l                                                             # L*L^T, we set C = L^T*L, and end up with way
                                                                                # smaller and computentionally inexpensive one
                                                                                # we also need to divide by the number of training
                                                                                # images


        self.evalues, self.evectors = np.linalg.eig(C)                          # eigenvectors/values of the covariance matrix
        sort_indices = self.evalues.argsort()[::-1]                             # getting their correct order - decreasing
        self.evalues = self.evalues[sort_indices]                               # puttin the evalues in that order
        self.evectors = self.evectors[sort_indices]                             # same for the evectors

        evalues_sum = sum(self.evalues[:])                                      # include only the first k evectors/values so
        evalues_count = 0                                                       # that they include approx. 85% of the energy
        evalues_energy = 0.0
        for evalue in self.evalues:
            evalues_count += 1
            evalues_energy += evalue / evalues_sum

            if evalues_energy >= self.energy:
                break

        self.evalues = self.evalues[0:evalues_count]                            # reduce the number of eigenvectors/values to consider
        self.evectors = self.evectors[0:evalues_count]

        self.evectors = self.evectors.transpose()                               # change eigenvectors from rows to columns
        self.evectors = L * self.evectors                                       # left multiply to get the correct evectors
        norms = np.linalg.norm(self.evectors, axis=0)                           # find the norm of each eigenvector
        self.evectors = self.evectors / norms                                   # normalize all eigenvectors

        self.W = self.evectors.transpose() * L                                  # computing the weights

        print '> Initializing ended'

    def classify(self, path_to_img):
        img = cv2.imread(path_to_img, 0)                                        # read as a grayscale image
        img_col = np.array(img, dtype='float64').flatten()                      # flatten the image
        img_col -= self.mean_img_col                                            # subract the mean column
        img_col = np.reshape(img_col, (self.mn, 1))                             # from row vector to col vector

        S = self.evectors.transpose() * img_col                                 # projecting the normalized probe onto the
                                                                                # Eigenspace, to find out the weights

        diff = self.W - S                                                       # finding the min ||W_j - S||
        norms = np.linalg.norm(diff, axis=0)

        closest_face_id = np.argmin(norms)                                      # the id [0..240) of the minerror face to the sample
        return (closest_face_id / self.train_faces_count) + 1                   # return the faceid (1..40)

    def evaluate(self):
        print '> Evaluating faces started'
        results_file = os.path.join('results', 'att_results.txt')               # filename for writing the evaluating results in
        f = open(results_file, 'w')                                             # the actual file

        test_count = self.test_faces_count * self.faces_count                   # number of all test images/faces
        test_correct = 0
        for face_id in xrange(1, self.faces_count + 1):
            for test_id in xrange(1, 1001):
                if (test_id in self.training_ids[face_id-1]) == False:          # we skip the image if it is part of the training set
                    path_to_img = os.path.join(self.faces_dir,'Class' + str(face_id), 'faceTrain' + str(face_id) + '_' + str(test_id) + '.bmp')          # relative path

                    result_id = self.classify(path_to_img)
                    result = (result_id == face_id)

                    if result == True:
                        test_correct += 1
                        f.write('image: %s\nresult: correct\n\n' % path_to_img)
                    else:
                        f.write('image: %s\nresult: wrong, got %2d\n\n' %
                                (path_to_img, result_id))

        print '> Evaluating faces ended'
        self.accuracy = float(100. * test_correct / test_count)
        print 'Correct: ' + str(self.accuracy) + '%'
        f.write('Correct: %.2f\n' % (self.accuracy))
        f.close()                                                               # closing the file

    def evaluate_demo(self, demo_dir='.'):
        print '> Evaluating demo matches started'
        result_dir_all = os.path.join('results','all.csv')
        f_all = open(result_dir_all,'wb')
        for img_name in os.listdir(demo_dir):                              # go through all the demo images in the folder
            path_to_img = os.path.join(demo_dir, img_name)

            img = cv2.imread(path_to_img, 0)                                    # read as a grayscale image
            img_col = np.array(img, dtype='float64').flatten()                  # flatten the image
            img_col -= self.mean_img_col                                        # subract the mean column
            img_col = np.reshape(img_col, (self.mn, 1))                         # from row vector to col vector

            S = self.evectors.transpose() * img_col                             # projecting the normalized probe onto the
                                                                                # Eigenspace, to find out the weights

            diff = self.W - S                                                   # finding the min ||W_j - S||
            norms = np.linalg.norm(diff, axis=0)
            top5_ids = np.argpartition(norms, 10)[:10]                           # first five elements: indices of top 5 matches in  set
            
            anum = [0] * 3
            appends = [0] * 3
            writer = csv.writer(f_all)
            for top_id in top5_ids:
                face_id = (top_id / self.train_faces_count) + 1                 # getting the face_id of one of the closest matches
                if top_id in range(0,1000):
                    anum[0] += 1
                elif top_id in range(1000,2000):
                    anum[1] += 1
                elif top_id in range(2000,3000):
                    anum[2] += 1
            max_class = anum.index(max(anum))
            if max_class == 0:
                appends[0] = 1
            elif max_class == 1:
                appends[1] = 1
            elif max_class == 2:
                appends[2] = 1
            writer.writerow(appends)
        f_all.close()
        print '> Evaluating demo matches ended'

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print 'Usage: python2.7 eigenfaces.py ' + '<att faces dir> [<demo faces dir>]'
        sys.exit(1)

    if not os.path.exists('results'):                                           # create a folder where to store the results
        os.makedirs('results')
    else:
        shutil.rmtree('results')                                                # clear everything in the results folder
        os.makedirs('results')

    efaces = Eigenfaces(str(sys.argv[1]))                                       # create the Eigenfaces object with the data dir
    efaces.evaluate()                                                           # evaluate our model

    if len(sys.argv) == 3:                                                      # if we have third argument (demo folder)
        efaces.evaluate_demo(str(sys.argv[2]))                           # find best matches for the demo
