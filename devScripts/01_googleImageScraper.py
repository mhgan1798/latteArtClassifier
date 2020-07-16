# %% Set up the script
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
import time
import requests
import shutil
import os
import numpy as np

# Set number of images to save
numberImages = 500


# %% Functions
def removeElement(xpath):
    element = driver.find_element_by_xpath(xpath)
    driver.execute_script(
        """
    var element = arguments[0];
    element.parentNode.removeChild(element);
    """,
        element,
    )


def save_img(inp, img, i, directory):
    try:
        filename = inp + str(i) + ".jpg"
        response = requests.get(img, stream=True)
        image_path = os.path.join(directory, filename)
        with open(image_path, "wb") as file:
            shutil.copyfileobj(response.raw, file)
    except Exception:
        pass


def find_urls(inp, url, driver, numberImages, directory):
    iterate = numberImages + numberImages // 25
    iterateRange = np.arange(1, iterate + 1)
    iterateRange = iterateRange[iterateRange % 25 != 0]

    driver.get(url)
    time.sleep(0.2)
    for j in iterateRange:
        # global imgurl
        imgurl = driver.find_element_by_xpath(
            "//div//div//div//div//div//div//div//div//div//div["
            + str(j)
            + "]//a[1]//div[1]//img[1]"
        )
        driver.execute_script("arguments[0].click();", imgurl)

        time.sleep(0.3)

        img = driver.find_element_by_xpath(
            "/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div[1]/div[1]/div/div[2]/a/img"
        ).get_attribute("src")

        # print("Successfully obtained url")

        save_img(inp, img, j, directory)
        # print("Image " + str(j) + " successfully saved")


# %% Run the scrape and saving process for good art
directory = r"C:\Users\HowardG\Google Drive\pythonProjects\development\latteArtClassifier\data\good"
inp = "latte art top view"
numberImages = 500

if "driver" in globals():
    del driver

# Initiate a driver
driver = webdriver.Chrome(r"C:\Program Files\chromeDriver\chromedriver.exe")

# Define alternative url here
url = "https://www.google.com/search?q=latte%20art%20top%20view&tbm=isch&tbs=rimg%3ACQqeDX7cfwhqYSEQgiIyiJuU&hl=en-US&ved=0CAIQrnZqGAoTCMiVhKr2yeoCFQAAAAAdAAAAABCxBA&biw=1457&bih=702"

# Find urls and save
find_urls(inp, url, driver, numberImages, directory)

# quit current session
if "driver" in globals():
    driver.quit()


# %% Run the scrape and saving process for bad art
directory = r"C:\Users\HowardG\Google Drive\pythonProjects\development\latteArtClassifier\data\bad"
inp = "bad latte art5"
numberImages = 50

if "driver" in globals():
    del driver

# Initiate a driver
driver = webdriver.Chrome(r"C:\Program Files\chromeDriver\chromedriver.exe")

# Define alternative url here
url = "https://www.google.com/search?q=latte%20art%20looks%20bad&tbm=isch&hl=en&hl=en&tbs=rimg%3ACcGP-8yXVCznYXCocLT23VLu&ved=0CAIQrnZqFwoTCPjNpfn_yeoCFQAAAAAdAAAAABAH&biw=1457&bih=794"

# Find urls and save
find_urls(inp, url, driver, numberImages, directory)

# quit current session
if "driver" in globals():
    driver.quit()
