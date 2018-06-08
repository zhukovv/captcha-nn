import os
import os.path
from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle


MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "data/labeled_captchas"


# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)

# Grab some random CAPTCHA images to test against.
captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
captcha_image_files = np.random.choice(captcha_image_files, size=(50,), replace=False)

total_captchas = 0
correct_captchas = 0

# loop over the image paths
for image_file in captcha_image_files:
    filename = os.path.basename(image_file)
    captcha_correct_text = (os.path.splitext(filename)[0]).upper()

    letters_count = len(captcha_correct_text)

    # Load the image and convert it to grayscale
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add some extra padding around the image
    image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

    # threshold the image
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(thresh,kernel,iterations = 1)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)

    processed = dilation

    #cv2.imshow('image',image)
    #cv2.moveWindow('gray', 100, 100)
    #cv2.imshow('processed',processed)
    #cv2.moveWindow('processed', 500, 100)
    #cv2.imshow('thresh',thresh)
    #cv2.moveWindow('thresh', 800, 100)

    # find the contours (continuous blobs of pixels) the image
    contours = cv2.findContours(processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hack for compatibility with different OpenCV versions
    contours = contours[0] if imutils.is_cv2() else contours[1]

    contours_count = len(contours)

    if (contours_count<1):
        continue

    letter_image_regions = []

    # if letters are overlapped - try to extract them by dividing the region uniformly
    if (contours_count!=letters_count):
        xmin = 1000
        xmax = 0
        ymin = 1000
        ymax = 0

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if (w<30 or h<30):
                continue
            xmin = min(xmin, x)
            xmax = max(xmax, x+w)
            ymin = min(ymin, y)
            ymax = max(ymax, y+h)

        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin

        #print x, y, w, h

        output = cv2.merge([processed] * 3)
        cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.imshow("Output", output)
        cv2.waitKey(1)

        letter_width = int(w / letters_count)

        for i in range(0, letters_count):
            letter_image_regions.append((x+letter_width*i, y, letter_width, h))

    # Now we can loop through each of the four contours and extract the letter
    # inside of each one
    else:
        for contour in contours:
            # Get the rectangle that contains the contour
            (x, y, w, h) = cv2.boundingRect(contour)

            letter_image_regions.append((x, y, w, h))


    # If we found more or less than is actually in the captcha, our letter extraction
    # didn't work correcly. Skip the image and mark as a fail
    if len(letter_image_regions) != letters_count:
        print("skipped - {}, {}".format(len(letter_image_regions), contours_count))
        total_captchas += 1
        continue

    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Create an output image and a list to hold our predicted letters
    output = cv2.merge([image] * 3)
    predictions = []

    # loop over the lektters
    for letter_bounding_box in letter_image_regions:
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = processed[y - 2:y + h + 2, x - 2:x + w + 2]

        #cv2.imshow('letter',letter_image)
        #cv2.waitKey(0)

        # Re-size the letter image to 20x20 pixels to match training data
        letter_image = resize_to_fit(letter_image, 20, 20)

        # Turn the single image into a 4d list of images to make Keras happy
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        # Ask the neural network to make a prediction
        prediction = model.predict(letter_image)

        # Convert the one-hot-encoded prediction back to a normal letter
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

        # draw the prediction on the output image
        cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

    # Print the captcha's text
    captcha_text = "".join(predictions)
    total_captchas += 1
    if (captcha_text == captcha_correct_text):
        correct_captchas += 1
    print("CAPTCHA text is: {} vs {}, acc: {}, {}/{}".format(captcha_text, captcha_correct_text, float(correct_captchas)/float(total_captchas), correct_captchas, total_captchas))

    # Show the annotated image
    cv2.imshow("Output", output)
    cv2.waitKey()
