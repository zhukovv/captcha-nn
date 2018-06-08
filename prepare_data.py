import os
import os.path
import cv2
import glob
import imutils
import numpy as np

CAPTCHA_IMAGE_FOLDER = "data/labeled_captchas"
OUTPUT_FOLDER = "data/letter_images"


# Get a list of all the captcha images we need to process
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}

# loop over the image paths
for (i, captcha_image_file) in enumerate(captcha_image_files):

    # Get captcha text from the filename and capitalize it for unification
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = (os.path.splitext(filename)[0]).upper()

    letters_count = len(captcha_correct_text)

    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files))+" = "+captcha_correct_text)
    #print letters_count

    # Load the image and convert it to grayscale
    image = cv2.imread(captcha_image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add some extra padding around the image
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

    # threshold the image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # suppress the background
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(thresh,kernel,iterations = 1)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    processed = dilation

    # debug visualizaions
    cv2.imshow('gray',gray)
    cv2.moveWindow('gray', 100, 100)
    cv2.imshow('processed',processed)
    cv2.moveWindow('processed', 500, 100)
    cv2.imshow('thresh',thresh)
    cv2.moveWindow('thresh', 800, 100)
    k = cv2.waitKey(1)

    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
        break

    # find the contours
    contours = cv2.findContours(processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hack for compatibility with different OpenCV versions
    contours = contours[0] if imutils.is_cv2() else contours[1]

    contours_count = len(contours)

    if (contours_count<1):
        continue

    letter_image_regions = []

    # if letters are overlapped - try to extract them by dividing the region uniformly
    if (contours_count<letters_count):
        xmin = 1000
        xmax = 0
        ymin = 1000
        ymax = 0

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            xmin = min(xmin, x)
            xmax = max(xmax, x+w)
            ymin = min(ymin, y)
            ymax = max(ymax, y+h)

        print contours_count
        print xmin, xmax, ymin, ymax

        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin

        print x, y, w, h

        output = cv2.merge([processed] * 3)
        cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.imshow("Output", output)
        cv2.waitKey(1)

        letter_width = int(w / letters_count)

        for i in range(0, letters_count):
            print i
            letter_image_regions.append((x+letter_width*i, y, letter_width, h))

    # Now we can loop through each of the four contours and extract the letter
    # inside of each one
    else:
        for contour in contours:
            # Get the rectangle that contains the contour
            (x, y, w, h) = cv2.boundingRect(contour)

            # Compare the width and height of the contour to detect letters that
            # are conjoined into one chunk
            if w / h > 1.25:
                # This contour is too wide to be a single letter!
                # Split it in half into two letter regions!
                half_width = int(w / 2)
                letter_image_regions.append((x, y, half_width, h))
                letter_image_regions.append((x + half_width, y, half_width, h))
            else:
                # This is a normal letter by itself
                letter_image_regions.append((x, y, w, h))


    # If we found more or less than is actually in the captcha, our letter extraction
    # didn't work correcly. Skip the image instead of saving bad training data!
    if len(letter_image_regions) != letters_count:
        continue

    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Save out each letter as a single image
    for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = processed[y - 2:y + h + 2, x - 2:x + w + 2]

        # Get the folder to save the image in
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)

        # if the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # write the letter image to a file
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

        cv2.imshow('letter',letter_image)
        cv2.waitKey(1)

        # increment the count for the current key
        counts[letter_text] = count + 1
