import cv2
#opencv documentation
#https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html?highlight=detectmultiscale

img = "car.jpg" #Declaring the img to the one located in your python file
Car_Classifier = "cars.xml" #Car classifier xml file-this xml will check our image and check it against the data and if it passed then it will be classed as a car

# Reference https://gist.github.com/199995/37e1e0af2bf8965e8058a9dfa3285bc6#file-cars-xml

CheckImg = cv2.imread(img)#open cv will take the image and read the pixel data which we can then use later imread = image read
Grayscale = cv2.cvtColor(CheckImg,cv2.COLOR_BGR2GRAY)#The reason to convert the file to gray scale is because it uses less data and when scanning the image it can be
#alot easier to detect

CheckCar = cv2.CascadeClassifier(Car_Classifier)#the xml that I am using is in a cascade format which is why its a cascade classifier
#this is just used now to check the image for any cars

CarTracker = CheckCar.detectMultiScale(Grayscale)#this will apply the car classifier to the image and then will give out  coordinates for each car found
#if i were to print this then it would give the coordinates of the car

for(X, Y,W, H) in CarTracker:#This will take the coordinates given in car tracker and then from that it will draw a rectangle to so the cars position
    cv2.rectangle(CheckImg, (X, Y), (X+W, Y+H), (255, 0, 0), 2) #This is drawing the rectangle from the given coordinates


cv2.imshow('Car detection',CheckImg)#opens the file that we just made and calls the window "Car detection"
cv2.waitKey()#stops the window from autoclosing-click a key to close