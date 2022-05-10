# ComicBookClassification Spiderman? Batman? None?!

TEAM MEMBERS: Mahdi Rahbar Hayden Stephan Pankil Shaw

Project specs:

For option C of the Final Project, let's take our exploration of feature
detection and matching further.  You may have noticed that the feature
detection and matching from Homework #4 didn't always work that well.  This is
largely because the keypoints and their descriptors are too specific to a given
image region or pattern.  In order to generalize and make the process as a
whole work better, we need to add machine learning.  We'll do that in this
project option.

For our test subject(s), given their recent big box-office battle, let's take
the fight between Spiderman and Batman to computer vision and machine learning,
and have them tell us who's who.  Since comic book covers are the largest
source of easily accessible images (via eBay, comic book sites, etc.), we'll
use comic book covers for our image data.

Your task for this option is to collect enough images to define the three
classes for Spiderman, Batman, and Neither (like those images shown below), and
then adapt a machine learning image classification pipeline to detect which of
the three classes is contained in a set of separate test images.

Chapter 7 in our OpenCV textbook, "Learning OpenCV 4 Computer Vision with
Python 3" discusses, with code, how to create a custom object
detector/classifier using the HOG feature descriptor in conjunction with the
two machine learning algorithms, BoW (Bag of Words) and SVM (support vector
machine), in order to create a custom classifier/detector.  In addition to
showing how it can be done to classify an entire image, it also demonstrates
how it can be enhanced with a sliding window to find the location of the object
in the image.  We'll do both for Spiderman and Batman...

1. Read through the discussion and tutorial in Chapter 7.  Although you're not
   going to explicitly use (much of) this code, you may want to examine the
   code available via the book's github.  If you'd like to run the code, you
   also need the CarData dataset, which no longer seems to be available at the
   stated location, so you'll need to download it here:  CarData.tar.gz
   Download CarData.tar.gz

2. Create your own comic book covers dataset.  This is a critical component of
   this project and should adhere to the following guidelines: -you should
   collect at least 30 training images for each of the three classes (90 total)
   -you should collect at least 4 test images for each of the three classes (12
   total) -every cover should have at least the hero (Spiderman or Batman) or
   some other hero (for Neither) -for sake of dataset size, the solo hero
   should appear in roughly the same costume on each cover (i.e. don't use both
   Spiderman's traditional costume as well as the black one he used for a
   while) -the comic book covers you collect should vary according to: --number
   of characters on cover:  ranging from just the solo hero (as shown
   above) to 2, 3, or more characters --background:  all three classes
   should contain similar collections of cover backgrounds --title and
   font:  to prevent the learning algorithm from focusing on the comic book
   title and style, use different titles with different fonts (e.g. Batman
   can be found on both "Batman" and "Detective" comic book covers, among
   others) --age:  have a mix of both older comics and newer comics;  an
   easy way to determine comic book age is via cost, which range from OLD
   comics around 10 cents to todays comics around $4

3. An unfortunate aspect of the code in Chapter 7 is that they use the HOG
   descriptor to find people in the first example (by using the pre-trained
   \_defaultPeopleDescriptor SVM), but do not train or use the HOG descriptor
   with the Bag of Words (BoW) example.  They instead use SIFT (or any of
   FAST-BRIEF, ORB, SIFT, etc.).  I'd like you to use HOG with BoW and SVM, so
   you're going to start with this Traffic-Sign-Detector example code:
   Traffic-Sign-Detection.zip  Download Traffic-Sign-Detection.zip   This is a
   slightly modified version of the code from Thanh Hoang Le Hai's github
   (Links to an external site.).  However, while this code uses and trains a
   HOG with SVM, it doesn't use BoW.  So you'll be adding in support for BoW
   based on your reading of Chapter 07.

4. Get to understand the Traffic-Sign-Detector example.  Some things you may
   want to question/experiment with to understand the code better: -how many
   classes does it support?  -how many classes does it have labels for?  -what
   happens if you try using only a small number of classes?  -how does it
   actually determine whether there's a sign in the scene?    <-- IMPORTANT
   -how are the training images stored, sized, cropped, etc.?  -to classify
   just test images, how would you get rid of object location detection &
   highlighting?

5. Now, modify the Traffic-Sign-Detector program so that it supports the Bag of
   Words (BoW) machine learning algorithm, as discussed in Chapter 07 of our
   OpenCV textbook.  While the addition of this won't affect the results much
   for how the authors do Traffic-Sign detection, the addition of BoW will
   dramatically affect the results for comic book character detection.
   Signs in test images are sufficiently to training images that they can
   do without BoW, but that's far from true for our case.

6. Now, modify the Traffic-Sign-Detector so that it supports your dataset of 3
   classes and 90 training images (30 per class).  In this first version, have
   the program classify the entire test image as Spiderman, Batman, or Neither.
   Then run tests with it against your 12 test images (4 per class).  How well
   did it do?  Discuss the results.

7. Finally, modify the Traffic-Sign-Detector so that it searches for a
   Spiderman, Batman, or other comic book in the test image and highlights its
   approximate location while classifying it as Spiderman, Batman, or Neither.
   I will supply the test dataset of 12 images with comic books contained
   within the image.  You can download them in this file,
   test\_comic\_locations.zip  Download test\_comic\_locations.zip, and feel
   free to re-size them, if desired.  I'll make that available in the last week
   of April.  How well did it do find the comic locations in the test images?
   Discuss the results.
